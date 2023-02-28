import pandas as pd

from time import sleep
from pandas.tseries import frequencies
from pandas.tseries.offsets import QuarterEnd, Day, Second

from pathlib import Path
from tqdm.rich import tqdm

def fetch_stock_data(config, ticker_list):
    for ticker in ticker_list:
        try:
            save_dir = Path(config.SAVE_DATA_HISTORICAL_STOCK_PRICES).joinpath(ticker, config.DATA_PROVIDER.data_frequency)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            
            this_quarter = pd.Timestamp(config.DATA_HISTORICAL_STOCK_PRICES_PERIOD_ADJ[0])
            last_quarter = pd.Timestamp(config.DATA_HISTORICAL_STOCK_PRICES_PERIOD_ADJ[1])
            print(f"fetch '{ticker}' historical intraday data between {this_quarter} - {last_quarter}")

            with tqdm(total=(last_quarter - this_quarter).total_seconds()) as ticker_data_progress_bar:
                while last_quarter > this_quarter:
                    next_quarter = this_quarter + QuarterEnd(1) + Day(1) - Second(1)
                    print(f"fetch '{ticker}' intraday data from period {this_quarter} - {next_quarter}")

                    save_file = save_dir.joinpath(f"{this_quarter.date()}_{next_quarter.date()}.parquet.gz")
                    if save_file.exists():
                        pass
                    else:

                        df = config.DATA_PROVIDER.fetch(ticker, start=this_quarter, end=next_quarter)
                        if df.empty:
                            print(f"No data fetched for period {this_quarter} - {next_quarter}")
                            continue

                        # reindex data (adds potential missing data samples (rows) with 'NaN' values)
                        this_quarter_index_range = pd.date_range(start=this_quarter, end=next_quarter, freq=config.DATA_PROVIDER.data_frequency, name=str(config.DataColumnNames.TIMESTAMP))
                        df = df.reindex(this_quarter_index_range, axis='index')
                        
                        # only keep rows that correspond to trade hours
                        # bussiness days
                        df = df.loc[df.index.dayofweek < 5]
                        # when dealing with intraday data, only keep US market trade hours
                        if frequencies.to_offset(config.DATA_PROVIDER.data_frequency) < frequencies.to_offset('1D'):
                            df = df.loc[(df.index.time.astype(str) >= "09:30") & (df.index.time.astype(str) <= "16:01")]

                        # # ffill missing data values    
                        df[[
                            str(config.DataColumnNames.OPEN) ,
                            str(config.DataColumnNames.HIGH) ,
                            str(config.DataColumnNames.LOW)  ,
                            str(config.DataColumnNames.CLOSE),
                        ]]                                      = df[[
                                                                    str(config.DataColumnNames.OPEN) ,
                                                                    str(config.DataColumnNames.HIGH) ,
                                                                    str(config.DataColumnNames.LOW)  ,
                                                                    str(config.DataColumnNames.CLOSE),
                                                                ]].bfill().ffill()
                        df[str(config.DataColumnNames.VOLUME)]  = df[str(config.DataColumnNames.VOLUME)].fillna(0)

                        # save data to file
                        df.to_parquet(path=save_file, compression='gzip', engine='pyarrow')
                            
                    ticker_data_progress_bar.update((next_quarter - this_quarter).total_seconds())
                    this_quarter = this_quarter + QuarterEnd(1) + Day(1)
                    ticker_data_progress_bar.refresh()

        except Exception as e:
            print(e)
