import pandas as pd
import numpy as np

from pandas.tseries.offsets import MonthEnd, QuarterEnd, Day, Second

from enum import Enum
from pathlib import Path

from stable_baselines3 import A2C, PPO, SAC, DDPG
from stable_baselines3.common import noise
from torch import nn
from torch import optim as optimizer


from services.stock_data.Yahoo import YahooFinancialService
from services.stock_data.AlphaVantage import AlphaVantageService


DOW_JONES = [
    "AAPL",
    "AMGN",
    "AXP",
    "BA",
    "CAT",
    "CRM",
    "CSCO",
    "CVX",
    "DIS",
    # "DOW",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "JPM",
    "KO",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "V",
    "VZ",
    "WBA",
    "WMT"
]

def _create_noise_fn(noise_type):
    if isinstance(noise_type, str):
        noise_class = getattr(noise, noise_type)
        return noise_class(mean=np.zeros(len(PORTFOLIO_ASSETS)), sigma=np.ones(len(PORTFOLIO_ASSETS)) * 0.3)
    return None

class DataColumnNames(Enum):
    TIMESTAMP                                   = 'timestamp',

    TICKER                                      = 'ticker',
    BALANCE                                     = 'balanace',
    TURBULENCE                                  = 'turbulence',

    OPEN                                        = 'open',
    HIGH                                        = 'high',
    LOW                                         = 'low',
    CLOSE                                       = 'close',
    VOLUME                                      = 'volume',

    def __str__(self):
        return str(self.value[0])


class ModelTrainMode(Enum):
    ALL_IN_ONE                                  = 1
    ROLLING                                     = 2
    GROWING                                     = 3

class EnsembleStrategy(Enum):
    MEAN                                        = 1
    HIGHEST_PROFIT                              = 2
    


LOGS                                            = Path('logs')
LOGS_TENSORFLOW                                 = LOGS.joinpath('tensorflow')

SAVE_MODELS                                     = Path('models')
SAVE_RESULTS                                    = Path('results')
SAVE_TUNED_HYPER_PARAMS                         = Path('hyperparams')
SAVE_DATA                                       = Path('data')
SAVE_DATA_HISTORICAL_STOCK_PRICES               = SAVE_DATA.joinpath('stocks')
SAVE_DATA_TRAIN_TEST_PORTFOLIO                  = SAVE_DATA.joinpath('train', 'portfolio')

DATA_PROVIDER                                   = AlphaVantageService()
# DATA_PROVIDER                                   = YahooFinancialService()

DATA_HISTORICAL_STOCK_PRICES_PERIOD             = (
    pd.Timestamp('2021-01-01'), 
    # pd.Timestamp('2008-01-01'), 
    # last complete years quarter
    pd.Timestamp.now().date()
)

# adjust provided historical time period to years quarter
DATA_HISTORICAL_STOCK_PRICES_PERIOD_ADJ = (
    DATA_HISTORICAL_STOCK_PRICES_PERIOD[0] - QuarterEnd(1) + Day(1),
    DATA_HISTORICAL_STOCK_PRICES_PERIOD[1] - QuarterEnd(1) + Day(1) - Second(1)
)

DATA_NORMALIZE_PORTFOLIO                        = False
DATA_TRAIN_TEST_SAMPLE_FREQUENCY                = '60T'
DATA_TRAIN_TEST_SAMPLE_RATIO                    = 0.8
DATA_ENRICHMENT_TECH_INDICATORS_ADJ_TIMEPERIOD  = True
DATA_ENRICHMENT_TECH_INDICATORS                 = {
    'sma_200d': 
    {
        'indicator': 'sma',
        'parameter':
        {
            'timeperiod': 200,
        }
    },

    # business week
    # **dict([(f"{indi}_{5}d", { 'indicator': indi, 'parameter': { 'timeperiod': 5 } })
    # business month
    **dict([(f"{indi}_{21}d", { 'indicator': indi, 'parameter': { 'timeperiod': 21 } }) 
    # business quarter
    # **dict([(f"{indi}_{63}d", { 'indicator': indi, 'parameter': { 'timeperiod': 63 } })
    # **dict([(f"{indi}_{100}d", { 'indicator': indi, 'parameter': { 'timeperiod': 100 } })
        for indi in [
            'rsi', 
            'macd', 
            'plus_di', 
            'minus_di', 
            'bbands', 
            'cci', 
            'dx', 
            'adx', 
            'adxr', 
            'ad'
        ]
    ]),
    
}

# PORTFOLIO_ASSETS                                = ['AAPL'] 
PORTFOLIO_ASSETS                                = ['AAPL', 'MSFT', 'INTC', 'AMZN', 'AMD', 'NVDA', 'GOOG'] 
# PORTFOLIO_ASSETS                                = DOW_JONES

MODEL_REBALANCE_WINDOW                          = MonthEnd(3)
MODEL_VALIDATION_WINDOW                         = MODEL_REBALANCE_WINDOW
# MODEL_REBALANCE_WINDOW                          = QuarterEnd(1)
# MODEL_VALIDATION_WINDOW                         = QuarterEnd(1)

MODEL_SPECS                                     = {
    'a2c': 
    {
        'model': A2C,
        # 'param': Path.joinpath(SAVE_TUNED_HYPER_PARAMS, "models", "a2c.json"),
        'param': 
        {
            'model_args': 
            {
                'learning_rate':                1e-5,
                'n_steps':                      8,
                'gamma':                        0.9,
                'gae_lambda':                   0.9,
                # 'ent_coef':                     0.0,
                # 'vf_coef':                      1.0,
                'use_sde':                      False,
            },      
            'policy_args':      
            {       
                'activation_fn':                nn.Tanh,
                'optimizer_class':              optimizer.AdamW,
                'net_arch':                     [512, 512]
            }
        },
        'tuner': lambda trial: 
        {
            'model_args': 
            {
          
            },      
            'policy_args':      
            {       
                
            }
        }
    },

    'sac': 
    {
        'model': SAC,
        # 'param': Path.joinpath(SAVE_TUNED_HYPER_PARAMS, "models", "sac.json"),
        'param':  
        {
            'model_args':
            {
                'learning_rate':                0.00003,
                'gamma':                        0.99,
                'tau':                          0.0025,
                'batch_size':                   128,
                "buffer_size":                  50000,
                'learning_starts':              100,
                'train_freq':                   1,
                'use_sde':                      False,
            },      
            'policy_args':      
            {       
                'activation_fn':                nn.Tanh,
                'optimizer_class':              optimizer.AdamW,
                'n_critics':                    2,
                'net_arch':                     [512, 512]
            }
        },
        'tuner': lambda trial: 
        {
            'model_args':
            {
            },      
            'policy_args':      
            {       
            }
        }
    },

    'ppo': 
    {
        'model': PPO,
        # 'param': Path.joinpath(SAVE_TUNED_HYPER_PARAMS, "models", "ppo.json"),
        'param':  
        {
            'model_args':
            {
                'learning_rate':                0.0006,
                'n_steps':                      2048,
                'batch_size':                   64,
                'gae_lambda':                   0.975,
                'ent_coef':                     0.0,
                'vf_coef':                      0.5,
                'clip_range':                   0.15,
                'max_grad_norm':                0.75,
                'use_sde':                      False,
                'normalize_advantage':          True,
            },      
            'policy_args':      
            {       
                'activation_fn':                nn.Tanh,
                'optimizer_class':              optimizer.AdamW,
                'net_arch':                     [512, 512],
            }
        },
        'tuner': lambda trial: 
        {
            'model_args':
            {
            },      
            'policy_args':      
            {       
            }
        }
    },
}

TRAIN_MODE                                      = ModelTrainMode.ALL_IN_ONE
# number to epoch to re-train/valid a model
TRAIN_EPOCHS                                    = 1
# if specified, the training continues until model validation yields a objective value greater than this
TRAIN_MIN_OBJECTIVE_VALUE                       = None

# number of samples used per train iteration
TRAIN_MAX_STEPS                                 = 0 # 0 for entire train dataset
TRAIN_LOAD_LAST_STATE                           = True
TRAIN_USE_TURBULENCE                            = False
# number of history sampes used to compute the turbulence value 
TRAIN_TURBULENCE_HISTORY_SIZE                   = 100
TRAIN_TURBULENCE_THRESHOLD_QUANTILE             = 0.99
# date when start using ensemble model for trading (use None to start right from the beginning)
TRAIN_START_TRADING                             = pd.Timestamp('2020-01-01').date()

ENV_NORMALIZE_ACTIONS                           = True
ENV_INITIAL_ACCOUNT_BALANCE                     = 2e5
ENV_HMAX                                        = 3
ENV_SELL_COST_PERCENT                           = 1e-3
ENV_BUY_COST_PERCENT                            = 1e-3
ENV_REWARD_SCALING                              = 1e-0
ENV_EXCLUDE_FEATURES                            = [
    str(DataColumnNames.TICKER),
    str(DataColumnNames.OPEN),
    str(DataColumnNames.HIGH),
    str(DataColumnNames.LOW),
    str(DataColumnNames.VOLUME),
]

TRADE_ENSEMBLE_STRATEGY                         = EnsembleStrategy.MEAN

# if True, performs hyperparameter tuning throught the train process
HYPER_TUNE_ENABLED                              = False
# continue a previously conducted study, when avaialble
HYPER_CONTINUE_LAST_STUDY                       = False
# the number of optuna trials run per train iteration
HYPER_TRIALS_MULTIPLIER                         = 10
HYPER_USE_PRUNING                               = False