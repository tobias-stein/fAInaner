from services.stock_data.AbstractStockDataProvider import AbstractStockDataProvider

import requests
import pandas as pd
import io
import config

from time import sleep
from pandas.tseries.offsets import Day
from fake_useragent import UserAgent


ALPHA_VANTAGE_API_KEYS = [
    'NYJWZAKW8MQ8J2T2'
]

FAKE_UA = UserAgent()

def rotating_api_key(keys):
    i = 0
    N = len(keys)
    while True:
        yield keys[i % N]
        i += 1

class AlphaVantageService(AbstractStockDataProvider):
    def __init__(self):
        super().__init__()
        self.api_key = rotating_api_key(ALPHA_VANTAGE_API_KEYS)

        # https://www.alphavantage.co/documentation/#intraday-extended
        self.slices = dict(zip(
            [f"year{y}month{m}" for y in range(1,3) for m in range(1,13)][::-1],
            list(pd.date_range(end=pd.Timestamp.now().date() - Day(1), periods=25, freq='30D')[:-1])
        ))


    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        df_acc = self._create_empty()

        for slice in self._get_slices(start, end): 
            for retry in range(1, 5, 1):
                try:
                    df = self._download(ticker, slice)
                    break
                except:
                    wait = 60.0
                    print(f"Failed to fetch slice. Retrying in {int(wait)} seconds.")
                    sleep(wait)
                    df = None
                    continue

            if df is None:
                return self._create_empty()

            df = pd.DataFrame(
                data={
                    str(config.DataColumnNames.OPEN)    : list(df['open']),
                    str(config.DataColumnNames.HIGH)    : list(df['high']),
                    str(config.DataColumnNames.LOW)     : list(df['low']),
                    str(config.DataColumnNames.CLOSE)   : list(df['close']),
                    str(config.DataColumnNames.VOLUME)  : list(df['volume'])
                },
                index=list(df.index)
            )

            df_acc = pd.concat([df_acc, df]) 

        return df_acc

    def _get_slices(self, start: pd.Timestamp, end: pd.Timestamp):
        slices = []

        for slice, t in self.slices.items():
            if t < start:
                continue
            if t > end:
                break
            slices.append(slice)

        return slices

    def _create_empty(self):
        return pd.DataFrame({
            str(config.DataColumnNames.OPEN)    : [],
            str(config.DataColumnNames.HIGH)    : [],
            str(config.DataColumnNames.LOW)     : [],
            str(config.DataColumnNames.CLOSE)   : [],
            str(config.DataColumnNames.VOLUME)  : []
        })

    def _download(self, ticker: str, slice: str):
        req = requests.get(
            f"https://www.alphavantage.co/query/", 
            headers={
                'User-Agent': FAKE_UA.random
            },
            params = { 
                'apikey': next(self.api_key), 
                'function': 'TIME_SERIES_INTRADAY_EXTENDED',
                'interval': '1min', 
                'slice': slice,
                'symbol': ticker
            })

        if req.status_code == 200:
            try:
                return pd.read_csv(io.StringIO(req.text), parse_dates=['time'], index_col='time')
            except:
                raise Exception(req.text)

    @property
    def data_provider_name(self):
        return "Alpha Vantage Service"

    @property
    def data_frequency(self):
        return '1T'