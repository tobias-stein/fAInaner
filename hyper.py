import sys
import gym
import pandas as pd
import numpy as np
import dask.dataframe as dd
import json
import config

from copy import deepcopy

from typing import Tuple, Union
from pathlib import Path

from pandas.tseries.offsets import QuarterBegin, QuarterEnd, Day
from torch import optim as optimizer

from env.state import State
from env.env_stockmarket import VirtualStockmarket


def _do_test_model(model_prefict_fn: object, env: gym.Env) -> Tuple[pd.DataFrame, State]:
    time_step_memory                    = []
    account_balance_memory              = []
    total_assets                        = []
    reward_memory                       = []
    profit_loss_memory                  = []
    actions_memory                      = []
    actions, _                          = model_prefict_fn(env.reset())
    obs, reward, done, extra            = env.step(actions)

    actions_memory.append(actions)
    reward_memory.append(reward)
    time_step_memory.append(extra['time_step'])
    profit_loss_memory.append(extra['profit_loss'])
    account_balance_memory.append(float(extra['last_state'][str(config.DataColumnNames.BALANCE)]))
    total_assets.append(float(extra['last_state'][str(config.DataColumnNames.BALANCE)] + sum(extra['last_state']['num_shares'] * extra['last_state'][str(config.DataColumnNames.CLOSE)])))

    while not done: 
        actions, _                  = model_prefict_fn(obs)
        obs, reward, done, extra    = env.step(actions)

        actions_memory.append(actions)
        reward_memory.append(reward)
        time_step_memory.append(extra['time_step'])
        profit_loss_memory.append(extra['profit_loss'])
        account_balance_memory.append(float(extra['last_state'][str(config.DataColumnNames.BALANCE)]))
        total_assets.append(float(extra['last_state'][str(config.DataColumnNames.BALANCE)] + sum(extra['last_state']['num_shares'] * extra['last_state'][str(config.DataColumnNames.CLOSE)])))

    return pd.DataFrame(
        data={ 
            str(config.DataColumnNames.BALANCE): account_balance_memory, 
            'total_assets': total_assets,
            'profit_loss': profit_loss_memory,
            'reward': reward_memory, 
        }, 
        index=time_step_memory
    ), extra['last_state']

def handler(event, context):

    # note this is SQS trigger specific event payload
    event_body = json.loads(event['Records'][0]['body'])
    
    model_name                      = event_body['model_name']
    hyper_params                    = deepcopy(event_body['hyper_params'])
    trial_id                        = event_body['trial_id']

    ddf                             = dd.read_parquet(Path.joinpath(config.SAVE_DATA_TRAIN_TEST_PORTFOLIO, "652bb2a8470cf2b30549ac97a11d3eb4"))
    ddf_index                       = ddf.index.compute()

    trade_days                      = ddf_index.map(pd.Timestamp.date).unique()
    num_assets                      = len(ddf[str(config.DataColumnNames.TICKER)].unique())
    last_train_day                  = (pd.Timestamp(trade_days[-1] - QuarterBegin(1)) - config.MODEL_VALIDATION_WINDOW) - config.MODEL_REBALANCE_WINDOW
        
    train_period                    = (
        trade_days[0], 
        last_train_day.date()
    )
    valid_period                    = (
        (train_period[1]  + Day(1)).date(), 
        ((train_period[1] + Day(1)) + config.MODEL_VALIDATION_WINDOW).date()
    )
    trade_period            = (
        (valid_period[1]  + Day(1)).date(), 
        ((valid_period[1] + Day(1)) + config.MODEL_REBALANCE_WINDOW).date()
    )

    _data                           = ddf.loc[train_period[0]:trade_period[1]].compute()
    _data                           = _data.groupby(_data.index, group_keys=False).apply(lambda x: x)

    turbulence_data                 = None
    turbulence_threshold            = float('+inf')
    # if config.TRAIN_USE_TURBULENCE:
    #     turbulence_data     = _compute_portfolio_turbulence(_data)
    #     turbulence_threshold= float(turbulence_data.quantile(config.TRAIN_TURBULENCE_THRESHOLD_QUANTILE))

    train_data                      = _data[train_period[0]:train_period[1]]
    valid_data                      = _data[valid_period[0]:trade_period[1]]
    
    # create train environment
    train_env                       = VirtualStockmarket(
        data=train_data,
        data_extra=turbulence_data[train_period[0]:train_period[1]] if turbulence_data else pd.DataFrame(),
        data_close_column=str(config.DataColumnNames.CLOSE),

        initial_account_balance=config.ENV_INITIAL_ACCOUNT_BALANCE,
        buy_cost=config.ENV_BUY_COST_PERCENT,
        sell_cost=config.ENV_SELL_COST_PERCENT,

        hmax=config.ENV_HMAX,
        turbulence_threshold=float('inf'),
        reward_scaling=config.ENV_REWARD_SCALING,

        state_exclude_features=config.ENV_EXCLUDE_FEATURES,
        state_previous=None
    )

    # create validation environment
    valid_env                       = VirtualStockmarket(
        data=valid_data,
        data_extra=turbulence_data[valid_period[0]:valid_period[1]] if turbulence_data else pd.DataFrame(),
        data_close_column=str(config.DataColumnNames.CLOSE),

        initial_account_balance=config.ENV_INITIAL_ACCOUNT_BALANCE,
        buy_cost=config.ENV_BUY_COST_PERCENT,
        sell_cost=config.ENV_SELL_COST_PERCENT,

        hmax=config.ENV_HMAX,
        reward_scaling=config.ENV_REWARD_SCALING,
        turbulence_threshold=turbulence_threshold,

        state_exclude_features=config.ENV_EXCLUDE_FEATURES,
        state_previous=None
    )

    num_timesteps                   = (config.TRAIN_MAX_STEPS if config.TRAIN_MAX_STEPS > 0 else len(train_data)) // num_assets

    agent_spec                      = config.MODEL_SPECS[model_name]

    # initialize model parameters
    args                            = hyper_params

    # resolve string param to classes
    if 'policy_args' in args:
        if 'activation_fn' in args['policy_args'] and isinstance(args['policy_args']['activation_fn'], str):
            from torch import nn
            args['policy_args']['activation_fn'] = getattr(nn, args['policy_args']['activation_fn'])
        if 'optimizer_class' in args['policy_args'] and isinstance(args['policy_args']['optimizer_class'], str):
            from torch import nn
            args['policy_args']['optimizer_class'] = getattr(optimizer, args['policy_args']['optimizer_class'])

    # create model
    agent_model                 = agent_spec['model'](
            policy="MlpPolicy",
            policy_kwargs=args['policy_args'] if 'policy_args' in args else None,
            env=train_env,
            tensorboard_log=None,
            **args['model_args'] if 'model_args' in args else None,
        )

    # train model
    objective_values                = []

    for _ in range(1, config.TRAIN_EPOCHS + 1):
        # train model
        agent_model                 = agent_model.learn(total_timesteps=num_timesteps, tb_log_name=None, progress_bar=False)
        # test model    
        test_results, _             = _do_test_model(agent_model.predict, valid_env)
        total_profits               = float(test_results['profit_loss'].sum())

        objective_values.append(total_profits)

    return {
        'trial_id':                 event_body['trial_id'],
        'params':                   event_body['hyper_params'],
        'value':                    np.mean(objective_values),
        'values':                   objective_values
    }