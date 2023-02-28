import dask.dataframe as dd
import pandas as pd
import numpy as np
import talib.abstract as ta
import hashlib
import json

from copy import deepcopy
from pandas.tseries import frequencies

from pathlib import Path

def create_train_portfolio(config):

    # derive portfolio MD5 hash from provided by config
    portfolio_spec = {
        'assets':               config.PORTFOLIO_ASSETS,
        'sample_date_start':    str(config.DATA_HISTORICAL_STOCK_PRICES_PERIOD[0]),
        'sample_date_end':      str(config.DATA_HISTORICAL_STOCK_PRICES_PERIOD[1]),
        'sample_period':        config.DATA_TRAIN_TEST_SAMPLE_FREQUENCY,
        'technical_indicators': config.DATA_ENRICHMENT_TECH_INDICATORS
    }

    portfolio_hash = hashlib.md5("".join(json.dumps(portfolio_spec)).encode('utf-8')).hexdigest()
    portfolio_dir  = Path.joinpath(config.SAVE_DATA_TRAIN_TEST_PORTFOLIO, portfolio_hash)

    if Path.exists(portfolio_dir):
        print("Portfolio train data already generated. Load from file.")
        return dd.read_parquet(portfolio_dir)

    ddf_portfolio = None

    for ticker in config.PORTFOLIO_ASSETS:
        # load all ticker data
        ddf_ticker = dd.read_parquet(Path.joinpath(config.SAVE_DATA_HISTORICAL_STOCK_PRICES, ticker, config.DATA_PROVIDER.data_frequency, "*.parquet.gz"), calculate_divisions=True)
        # only keep desired range
        ddf_ticker = ddf_ticker.loc[config.DATA_HISTORICAL_STOCK_PRICES_PERIOD_ADJ[0]:config.DATA_HISTORICAL_STOCK_PRICES_PERIOD_ADJ[1]]

        if len(ddf_ticker.index) == 0:
            print(f"Ticker '{ticker}' has no data.")
            continue
        
        # compute final pandas dataframe
        df = ddf_ticker.compute()
            
        # if 
        desired_data_freq = frequencies.to_offset(config.DATA_TRAIN_TEST_SAMPLE_FREQUENCY)
        service_data_freq = frequencies.to_offset(config.DATA_PROVIDER.data_frequency)
        onedays_data_freq = frequencies.to_offset('1D')

        if desired_data_freq > onedays_data_freq:
            print(f"DATA_TRAIN_TEST_SAMPLE_FREQUENCY value ('{config.DATA_TRAIN_TEST_SAMPLE_FREQUENCY}') cannot be higher than '1D'. Value will be clamped.")
            desired_data_freq = onedays_data_freq

        # desired data sample frequency does not match the data prodiders sample frequency, try to resample data
        if desired_data_freq != service_data_freq:

            if service_data_freq < onedays_data_freq:
                df = pd.concat([
                    trade_day.resample(config.DATA_TRAIN_TEST_SAMPLE_FREQUENCY).agg(
                    {
                        str(config.DataColumnNames.OPEN)    :   'first', 
                        str(config.DataColumnNames.HIGH)    :   'max', 
                        str(config.DataColumnNames.LOW)     :   'min', 
                        str(config.DataColumnNames.CLOSE)   :   'last', 
                        str(config.DataColumnNames.VOLUME)  :   'sum'
                    }) 
                    for _, trade_day 
                    in df.groupby(df.index.date)
                ])
                num_samples_per_day = df.groupby(df.index.date)
                num_samples_per_day = len(num_samples_per_day.get_group(list(num_samples_per_day.groups)[0]))

            else:
                print(f"Skip data resampling. Data provider's ({config.DATA_PROVIDER.data_provider_name}) sample frequency '{config.DATA_PROVIDER.data_frequency}' is insufficient for desired sample frequency '{config.DATA_TRAIN_TEST_SAMPLE_FREQUENCY}'.")
        
        for _, indicator_spec in config.DATA_ENRICHMENT_TECH_INDICATORS.items():
            try:
                if 'parameter' not in indicator_spec:
                    indicator_spec['parameter'] = {}

                indicator = getattr(ta, indicator_spec['indicator'].upper())
                parameter = deepcopy(indicator_spec['parameter'])

                # adjust timeperiod value, if desired sample freq is intraday
                adj_timeperiod = desired_data_freq < onedays_data_freq and service_data_freq < onedays_data_freq
                if adj_timeperiod and config.DATA_ENRICHMENT_TECH_INDICATORS_ADJ_TIMEPERIOD and 'timeperiod' in indicator.parameters:
                    parameter['timeperiod'] = (parameter['timeperiod'] if 'timeperiod' in parameter else indicator.parameters['timeperiod']) * num_samples_per_day
                    
                result = indicator(df, **parameter)

                if len(indicator.info['output_names']) > 1:
                    for output_name in indicator.info['output_names']:
                        column_name = "_".join([output_name] + [f"{k}_{v}" for k, v in parameter.items()])
                        df[column_name] = result[output_name]
                else:
                    column_name = "_".join([indicator.info['name'].lower()] + [f"{k}_{v}" for k, v in parameter.items()])
                    df[column_name] = result
                

            except Exception as e:
                print(e)
                print(f"Technical indicator '{indicator_spec['indicator']}' is not supported for data enrichment.")

        # drop popential NaN row after enrichment
        df = df.fillna(0)
        df = df.replace(np.inf, 0)
        
        df['weekday']                               = df.index.dayofweek
        df[str(config.DataColumnNames.TICKER)]      = f"{ticker}"
        
        ddf_portfolio = dd.concat([ddf_portfolio, dd.from_pandas(df, npartitions=32)]) if ddf_portfolio is not None else dd.from_pandas(df, npartitions=32)

    print("Portfolio train data generated. Save to file.")
    Path(portfolio_dir).mkdir(parents=True, exist_ok=True)
    # write the portfolio specs to json file
    with open(portfolio_dir.joinpath('portfolio_spec.json'), 'w') as f:
        f.write(json.dumps(portfolio_spec, indent = 4))


    # save the portfolio train data
    dd.to_parquet(ddf_portfolio, portfolio_dir, engine='pyarrow', compression='gzip')

    return ddf_portfolio