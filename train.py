import gym
import pandas as pd
import numpy as np
import config
import shutil
import optuna
import json

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from typing import Tuple, Union
from pathlib import Path
from copy import deepcopy
from pandas.tseries.offsets import QuarterBegin, QuarterEnd, Day
from tqdm.rich import tqdm

from sklearn.preprocessing import MinMaxScaler

from env.state import State
from env.env_stockmarket import VirtualStockmarket

from pipeline.stocks import fetch_stock_data
from pipeline.portfolio import create_train_portfolio

# Turn off optuna log notes.
optuna.logging.set_verbosity(optuna.logging.ERROR)

def optuna_save_best_callback(study, frozen_trial):
    try:
        previous_best_value = study.user_attrs.get("previous_best_value", None)
        if previous_best_value != study.best_value:
            study.set_user_attr("previous_best_value", study.best_value)
            params = {}
            for key, value in study.best_trial.params.items():
                group_name, param_name = key.split('.')
                if group_name not in params:
                    params[group_name] = {}
                params[group_name][param_name] = value

            json_str = json.dumps(params, indent=4)

            hypterparameter_model_save_dir = Path.joinpath(config.SAVE_TUNED_HYPER_PARAMS, "models")
            hypterparameter_model_save_dir.mkdir(parents=True, exist_ok=True)
            with open(hypterparameter_model_save_dir.joinpath(f"{study.study_name}.json"), 'w') as f:
                f.write(json_str)

            print(f"Hyperparameter study '{study.study_name}' trial {frozen_trial.number} finished with new best objective value: {frozen_trial.value}")
            print(f"{json_str}")
    except:
        pass

def _do_test_model(model_prefict_fn: object, env: gym.Env, env_reset_states: dict = {}) -> Tuple[pd.DataFrame, State]:
    time_step_memory                    = []
    account_balance_memory              = []
    total_assets                        = []
    reward_memory                       = []
    profit_loss_memory                  = []
    actions_memory                      = []
    with tqdm(desc="Testing", total=len(env.data), disable=config.HYPER_TUNE_ENABLED) as progress_bar:
        actions, _                      = model_prefict_fn(env.reset(env_reset_states))
        obs, reward, done, extra        = env.step(actions)
        progress_bar.update(1)  

        actions_memory.append(actions)
        reward_memory.append(reward)
        time_step_memory.append(extra['time_step'])
        profit_loss_memory.append(extra['profit_loss'])
        account_balance_memory.append(float(extra['last_state'][str(config.DataColumnNames.BALANCE)]))

        total_assets.append(float(extra['last_state'][str(config.DataColumnNames.BALANCE)] + np.dot(extra['last_state']['num_shares'], extra['last_state'][str(config.DataColumnNames.CLOSE)])))

        while not done: 
            actions, _                  = model_prefict_fn(obs)
            obs, reward, done, extra    = env.step(actions)
            progress_bar.update(1)
            progress_bar.refresh()

            actions_memory.append(actions)
            reward_memory.append(reward)
            time_step_memory.append(extra['time_step'])
            profit_loss_memory.append(extra['profit_loss'])
            account_balance_memory.append(float(extra['last_state'][str(config.DataColumnNames.BALANCE)]))
            total_assets.append(float(extra['last_state'][str(config.DataColumnNames.BALANCE)] + np.dot(extra['last_state']['num_shares'], extra['last_state'][str(config.DataColumnNames.CLOSE)])))

    return pd.DataFrame(
        data={ 
            str(config.DataColumnNames.BALANCE): account_balance_memory, 
            'total_assets': total_assets,
            'profit_loss': profit_loss_memory,
            'reward': reward_memory, 
        }, 
        index=time_step_memory
    ), extra['last_state']

def _plot_agents_results(agent_results: dict, iteration: int, save_file: str) -> None:

    fig, axis = None, None

    for agent_name, results in agent_results.items():
        if iteration not in results:
            continue

        result = results[iteration]

        if fig is None:
            fig, axis = plt.subplots(nrows=len(result.columns), sharex=True, figsize=(8.27, 11.69))

        for index, column in enumerate(result.columns):
            sns.lineplot(result[column], ax=axis[index], label=agent_name)

            if index == 0:
                axis[index].set_title(column)
                
    fig.autofmt_xdate()
    fig.savefig(save_file)
    plt.close(fig)

def _compute_portfolio_turbulence(data) -> None:
    df_tic_close                                    = data.pivot(columns=str(config.DataColumnNames.TICKER), values=str(config.DataColumnNames.CLOSE))
    df_tic_returns                                  = np.log(df_tic_close / df_tic_close.shift(1)).fillna(0)

    index = list(df_tic_returns.index[0:config.TRAIN_TURBULENCE_HISTORY_SIZE])
    turbo = [0.0] * config.TRAIN_TURBULENCE_HISTORY_SIZE
    for i in range(config.TRAIN_TURBULENCE_HISTORY_SIZE, len(df_tic_returns), 1):
        hist                                        = df_tic_returns.iloc[i - config.TRAIN_TURBULENCE_HISTORY_SIZE:i]
        this                                        = df_tic_returns.iloc[i] - hist.mean()
        try:
            dist                                    = this.values.dot(np.linalg.pinv(hist.cov())).dot(this.values.T)
        except:
            dist                                    = 0.0
            
        index.append(df_tic_returns.index[i])
        turbo.append(dist)

    return pd.DataFrame(data={str(config.DataColumnNames.TURBULENCE): turbo}, index=df_tic_returns.index)

def main(args):
    
    # generate a training dataset of a portfolio from the current specified config
    # first fetch required data
    fetch_stock_data(config, config.PORTFOLIO_ASSETS)
    # second build a training dataset from previously fetched data
    ddf                             = create_train_portfolio(config)

    ddf_index                       = ddf.index.compute()

    sample_period                   = ddf_index.inferred_freq
    trade_days                      = ddf_index.map(pd.Timestamp.date).unique()
    num_total_trade_days            = len(trade_days)
    num_assets                      = len(ddf[str(config.DataColumnNames.TICKER)].unique())
    
    print(f"Training dataset ready: {len(ddf_index)} samples at {sample_period} frequency, {num_total_trade_days} trade days, {num_assets} assets")

    first_train_day                 = trade_days[0]
    last_train_day                  = pd.Timestamp(trade_days[int(num_total_trade_days * config.DATA_TRAIN_TEST_SAMPLE_RATIO)])
    if config.TRAIN_MODE == config.ModelTrainMode.ROLLING:
        last_train_day              = first_train_day + config.MODEL_REBALANCE_WINDOW

    #round to next quarter end
    last_train_day                  = last_train_day if last_train_day.is_quarter_end else last_train_day + QuarterEnd(1)

    last_possible_test_day          = (pd.Timestamp(trade_days[-1] - QuarterBegin(1)) - config.MODEL_VALIDATION_WINDOW) - config.MODEL_REBALANCE_WINDOW
    last_train_day                  = min(last_train_day, last_possible_test_day)

    if config.TRAIN_MODE == config.ModelTrainMode.ALL_IN_ONE:
        last_train_day              = last_possible_test_day

    train_start                     = pd.Timestamp.now()
    iteration                       = 1

    # esemblens model states
    last_trade_state                = None
    last_trade_assets_book          = None

    agents_results                  = dict()

    def _save_agents_results(agent_name, result, iteration):
        if agent_name in agents_results.keys():
            agents_results[agent_name][iteration] = result
        else:
            agents_results[agent_name] = { iteration: result }

    # clear old logs
    shutil.rmtree(config.LOGS, ignore_errors=True)
    shutil.rmtree(config.SAVE_RESULTS, ignore_errors=True)
    shutil.rmtree(config.SAVE_MODELS, ignore_errors=True)

    if config.HYPER_TUNE_ENABLED:
        config.SAVE_TUNED_HYPER_PARAMS.mkdir(parents=True, exist_ok=True)
        save_optuna_study_db= config.SAVE_TUNED_HYPER_PARAMS.joinpath("studies.db")

        if not config.HYPER_CONTINUE_LAST_STUDY:
            save_optuna_study_db.unlink(missing_ok=True)

        # make sure that parameters are tuned/evaludated always on the same data
        last_train_day = last_possible_test_day

    while last_train_day <= last_possible_test_day:
        train_period            = (
            first_train_day, 
            last_train_day.date()
        )
        valid_period            = (
            (train_period[1]  + Day(1)).date(), 
            ((train_period[1] + Day(1)) + config.MODEL_VALIDATION_WINDOW).date()
        )
        trade_period            = (
            (valid_period[1]  + Day(1)).date(), 
            ((valid_period[1] + Day(1)) + config.MODEL_REBALANCE_WINDOW).date()
        )

        print(f"Run #{iteration:03}: train period: {train_period[0]} - {train_period[1]} ({len(pd.bdate_range(start=train_period[0], end=train_period[1]))} days), valid period: {valid_period[0]} - {valid_period[1]} ({len(pd.bdate_range(start=valid_period[0], end=valid_period[1]))} days), trade period: {trade_period[0]} - {trade_period[1]} ({len(pd.bdate_range(start=trade_period[0], end=trade_period[1]))} days)")

        _data                   = ddf.loc[train_period[0]:trade_period[1]].compute()

        _data_scaler            = None
        if config.DATA_NORMALIZE_PORTFOLIO:
            _data_scaler        = MinMaxScaler()
            columns             = _data.select_dtypes(np.number).columns
            _data[columns]      = _data_scaler.fit_transform(_data[columns])

        _data                   = _data.groupby(_data.index, group_keys=False).apply(lambda x: x)

        turbulence_data         = None
        turbulence_threshold    = float('+inf')
        if config.TRAIN_USE_TURBULENCE:
            turbulence_data     = _compute_portfolio_turbulence(_data)
            turbulence_threshold= float(turbulence_data.quantile(config.TRAIN_TURBULENCE_THRESHOLD_QUANTILE))

        train_data              = _data[train_period[0]:train_period[1]]
        valid_data              = _data[valid_period[0]:valid_period[1]]
        trade_data              = _data[trade_period[0]:trade_period[1]]

        # prepare 'results' directories
        save_results_valid_dir  = config.SAVE_RESULTS.joinpath("valid")
        save_results_trade_dir  = config.SAVE_RESULTS.joinpath("trade")
        save_results_valid_dir.mkdir(parents=True, exist_ok=True)
        save_results_trade_dir.mkdir(parents=True, exist_ok=True)

        # create train environment
        train_env               = VirtualStockmarket(
            data=train_data,
            data_extra=turbulence_data[train_period[0]:train_period[1]] if turbulence_data is not None else pd.DataFrame(),
            data_close_column=str(config.DataColumnNames.CLOSE),

            initial_account_balance=config.ENV_INITIAL_ACCOUNT_BALANCE,
            buy_cost=config.ENV_BUY_COST_PERCENT,
            sell_cost=config.ENV_SELL_COST_PERCENT,

            hmax=config.ENV_HMAX,
            turbulence_threshold=float('inf'),
            reward_scaling=config.ENV_REWARD_SCALING,

            state_exclude_features=config.ENV_EXCLUDE_FEATURES,
        )

        # create validation environment
        valid_env               = VirtualStockmarket(
            data=valid_data,
            data_extra=turbulence_data[valid_period[0]:valid_period[1]] if turbulence_data is not None else pd.DataFrame(),
            data_close_column=str(config.DataColumnNames.CLOSE),

            initial_account_balance=config.ENV_INITIAL_ACCOUNT_BALANCE,
            buy_cost=config.ENV_BUY_COST_PERCENT,
            sell_cost=config.ENV_SELL_COST_PERCENT,

            hmax=config.ENV_HMAX,
            reward_scaling=config.ENV_REWARD_SCALING,
            turbulence_threshold=turbulence_threshold,

            state_exclude_features=config.ENV_EXCLUDE_FEATURES,
        )    

        # create trade environment
        trade_env               = VirtualStockmarket(
            data=trade_data,
            data_extra=turbulence_data[trade_period[0]:trade_period[1]] if turbulence_data is not None else pd.DataFrame(),
            data_close_column=str(config.DataColumnNames.CLOSE),

            initial_account_balance=config.ENV_INITIAL_ACCOUNT_BALANCE,
            buy_cost=config.ENV_BUY_COST_PERCENT,
            sell_cost=config.ENV_SELL_COST_PERCENT,

            hmax=config.ENV_HMAX,
            reward_scaling=config.ENV_REWARD_SCALING,
            turbulence_threshold=turbulence_threshold,

            state_exclude_features=config.ENV_EXCLUDE_FEATURES,
        )

        # trade with random actions
        def random_prediction(state: State):
            return np.random.uniform(-1.0, 1.0, num_assets), None

        num_timesteps           = (config.TRAIN_MAX_STEPS if config.TRAIN_MAX_STEPS > 0 else len(train_data)) // num_assets
        model_state_name        = f"{iteration:03}"

        agents                  = dict()

        for agent_name, agent_spec in config.MODEL_SPECS.items():
            print(f"Train and validate agent '{agent_name}' ...")

            def objective(trial: Union[optuna.Trial, None]) -> float:
                if trial is not None:
                    print(f"#{trial.number:04} trial '{agent_name}'")

                # initialize model parameters
                args = agent_spec['tuner'](trial) if trial is not None else agent_spec['param']
                if isinstance(args, (Path, str)):
                    print(f"Load agent '{agent_name}' model paramters from file '{args}'")
                    try:
                        with open(args, 'r') as f:
                            args = json.load(f)
                    except Exception as e:
                        print(f"Failed load parameters from file. Reason: {e}")
                        args = {}
                elif isinstance(args, dict):
                    pass
                else:
                    print(f"Agent '{agent_name}' has invalid model arguments: {args}. Must be dict or path to file.")
                    args = {}

                # resolve string param to classes
                if 'policy_args' in args:
                    if 'activation_fn' in args['policy_args'] and isinstance(args['policy_args']['activation_fn'], str):
                        from torch import nn
                        args['policy_args']['activation_fn'] = getattr(nn, args['policy_args']['activation_fn'])

                def _create_model():
                        if config.TRAIN_LOAD_LAST_STATE:
                            model_state_dir = Path.joinpath(config.SAVE_MODELS, agent_name)
                            if model_state_dir.exists():
                                if not config.HYPER_TUNE_ENABLED:
                                    last_model_state = max(list(model_state_dir.iterdir()), key=lambda f: f.lstat().st_ctime)
                                    print(f"Restore previous model state '{last_model_state}'")
                                    return agent_spec['model'].load(last_model_state, env=train_env)
                                else:
                                    print("'TRAIN_WITH_HYPERPARAM_TUNING' flag is set True, skip restoring previous model state!")

                        return agent_spec['model'](
                            policy="MlpPolicy",
                            policy_kwargs=args['policy_args'] if 'policy_args' in args else None,
                            env=train_env,
                            tensorboard_log=None,
                            **args['model_args'] if 'model_args' in args else None,
                        )

                # create/load model
                agent_model                 = _create_model()

                # train model
                objective_values            = []
                
                for epoch in range(1, config.TRAIN_EPOCHS + 1):
                    # train model
                    agent_model             = agent_model.learn(total_timesteps=num_timesteps, tb_log_name=None, progress_bar=False)

                    # test model
                    test_results, last_state= _do_test_model(agent_model.predict, valid_env)
                    agents[agent_name]      = { 'model': agent_model, 'results': test_results }

                    last_total_assets       = float(last_state[str(config.DataColumnNames.BALANCE)] + np.dot(last_state['num_shares'], last_state[str(config.DataColumnNames.CLOSE)]))
                    diff_total_assets       = float(last_total_assets - config.ENV_INITIAL_ACCOUNT_BALANCE)

                    total_profits           = float(test_results['profit_loss'].sum())
                    objective_values.append(total_profits)

                    # print last test result
                    print(f"[Epoch:{epoch:02}]: Total Assets: {last_total_assets:+0.2f} [{diff_total_assets:+0.2f} {'profit' if diff_total_assets > 0.0 else 'loss'}], Acc. profit/loss: {total_profits:0.3f}")

                    _save_agents_results(agent_name, test_results, iteration)
                    _plot_agents_results(agents_results, iteration, save_results_valid_dir.joinpath(f"{iteration:003}.png"))

                    if trial is not None:
                        trial.report(total_profits, epoch)
                        # Handle pruning based on the intermediate value.
                        if config.HYPER_USE_PRUNING and epoch > 5:
                            if trial.should_prune():
                                raise optuna.TrialPruned()

                # if we observe no changes in the objective values what so ever, prune it
                if trial is not None and np.sum(objective_values) == 0.0:
                    raise optuna.TrialPruned()

                # save model state
                model_state_path    = Path.joinpath(config.SAVE_MODELS, agent_name)
                model_state_path.mkdir(parents=True, exist_ok=True)
                agent_model.save(model_state_path.joinpath(model_state_name))

                return objective_values[-1]

            if config.HYPER_TUNE_ENABLED:
                    hyper_param_study = optuna.create_study(
                        study_name=agent_name, 
                        direction=optuna.study.StudyDirection.MAXIMIZE,
                        pruner=optuna.pruners.SuccessiveHalvingPruner(),
                        sampler=optuna.samplers.CmaEsSampler(),
                        storage=f"sqlite:///{save_optuna_study_db}", 
                        load_if_exists=True)

                    c_trial = 0
                    while c_trial < config.HYPER_TRIALS_MULTIPLIER:
                        try:
                            trial = hyper_param_study.ask()
                            value = objective(trial)
                            froze = hyper_param_study.tell(trial, value, optuna.trial.TrialState.COMPLETE)
                            optuna_save_best_callback(hyper_param_study, froze)
                            c_trial += 1
                        except Exception as e:
                            hyper_param_study.tell(trial, None, optuna.trial.TrialState.PRUNED)
                            print(e)
            else:
                while True:
                    objective_value = objective(None)
                    if config.TRAIN_MIN_OBJECTIVE_VALUE is None or objective_value >= config.TRAIN_MIN_OBJECTIVE_VALUE:
                        break

        random_results, _           = _do_test_model(random_prediction, valid_env)
        _save_agents_results('random', random_results, iteration)
        _plot_agents_results(agents_results, iteration, save_results_valid_dir.joinpath(f"{iteration:003}.png"))


        if config.TRAIN_START_TRADING is None or trade_period[0] >= config.TRAIN_START_TRADING:
            print(f"Trading with ensemble strategy '{config.TRADE_ENSEMBLE_STRATEGY}' on data from '{trade_period[0]}_{trade_period[1]}' ...")
            
            # ensemble prediction strategy
            def ensemble_predict_strategy(obs):
                if config.TRADE_ENSEMBLE_STRATEGY == config.EnsembleStrategy.MEAN:
                    actions = np.array([agent['model'].predict(obs)[0] for agent in agents.values() if agent['model'] is not None])
                    return np.mean(actions, axis=0), None

                elif config.TRADE_ENSEMBLE_STRATEGY == config.EnsembleStrategy.HIGHEST_PROFIT:
                    agent = sorted(agents.values(), key=lambda agent: agent['results']['profit_loss'].sum(), reverse=True)[0]
                    return agent['model'].predict(obs)
                else:
                    raise Exception(f"Unsupported ensemble training strategy '{config.TRADE_ENSEMBLE_STRATEGY}'.")               

            if last_trade_assets_book:
                trade_env.assets_book   = last_trade_assets_book

            init_total_assets           = config.ENV_INITIAL_ACCOUNT_BALANCE if last_trade_state is None else float(last_trade_state[str(config.DataColumnNames.BALANCE)] + np.dot(last_trade_state['num_shares'], last_trade_state[str(config.DataColumnNames.CLOSE)]))
            trade_results, last_state   = _do_test_model(ensemble_predict_strategy, trade_env, {'state': last_trade_state, 'assets_book': last_trade_assets_book})
            _save_agents_results('ensemble', trade_results, iteration)
            _plot_agents_results(agents_results, iteration, save_results_valid_dir.joinpath(f"{iteration:003}.png"))

            last_trade_assets_book      = deepcopy(trade_env.assets_book)
            last_trade_state            = last_state

            last_total_assets       = float(last_state[str(config.DataColumnNames.BALANCE)] + np.dot(last_state['num_shares'], last_state[str(config.DataColumnNames.CLOSE)]))
            diff_total_assets       = float(last_total_assets - init_total_assets)
            total_profits           = float(trade_results['profit_loss'].sum())

            balance_change              = trade_results['total_assets'].pct_change()
            sharpe_ratio                = (balance_change.mean() / balance_change.std())

            print(f"Account Balance: {last_total_assets:+0.2f} [{diff_total_assets:+0.2f} {'profit' if diff_total_assets > 0.0 else 'loss'}], Acc. profit/loss: {total_profits:0.3f}, Sharpe: {sharpe_ratio:0.3f}")
        else:
            _save_agents_results('ensemble', None, iteration)

        # next training cycle
        if   config.TRAIN_MODE == config.ModelTrainMode.ROLLING:
            first_train_day         = first_train_day + config.MODEL_REBALANCE_WINDOW
            last_train_day          = last_train_day + config.MODEL_REBALANCE_WINDOW
        elif config.TRAIN_MODE == config.ModelTrainMode.GROWING:
            last_train_day          = last_train_day + config.MODEL_REBALANCE_WINDOW
        elif config.TRAIN_MODE == config.ModelTrainMode.ALL_IN_ONE:
            break

        iteration                   = iteration + 1

    train_end                       = pd.Timestamp.now()
    print(f"Training finsihed. {train_end - train_start}")

    # fetch_stock_data(config, "^DJI")

    fig, ax = plt.subplots(nrows=5, figsize=(12, 16), sharex=False)

   
    _data.pivot(columns=str(config.DataColumnNames.TICKER), values=str(config.DataColumnNames.CLOSE)).plot(ax=ax[0])
    
    # plot profix/loss
    agents_profit_loss_data = None
    for agent_name, agent_results in agents_results.items():
        processed           = pd.concat([results[['profit_loss', 'total_assets']] for results in agent_results.values() if results is not None])
        processed['return'] = processed['total_assets'].pct_change().fillna(0.0).replace([np.inf, -np.inf], 0.0)
        processed['sharpe'] = (processed['return'].mean() / processed['return'].std())

        sns.lineplot(processed['total_assets'], ax=ax[1], label=agent_name)

        processed           = processed.groupby(pd.Grouper(freq="M")).agg({
            'profit_loss':  'sum',
            'return':       'mean',
            'sharpe':       'mean',
        })

        processed['agent']  = agent_name
        processed['date']   = [str(d) for d in processed.index.date]

        agents_profit_loss_data = processed if agents_profit_loss_data is None else pd.concat([agents_profit_loss_data, processed])

    sns.barplot(data=agents_profit_loss_data, x='date', y='profit_loss',   hue='agent', ax=ax[2])
    sns.barplot(data=agents_profit_loss_data, x='date', y='return',        hue='agent', ax=ax[3])
    sns.barplot(data=agents_profit_loss_data, x='date', y='sharpe',        hue='agent', ax=ax[4])


    fig.savefig(config.SAVE_RESULTS.joinpath("profit_loss.png"), bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main(None)