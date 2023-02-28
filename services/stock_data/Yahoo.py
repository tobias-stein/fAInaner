from services.stock_data.AbstractStockDataProvider import AbstractStockDataProvider

import pandas as pd
import yfinance as yf

import config

class YahooFinancialService(AbstractStockDataProvider):
    def __init__(self):
        super().__init__()

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        df = yf.download(tickers = ticker, start=str(start.date()), end=str(end.date()), progress=False)
        return pd.DataFrame(
            data={
                str(config.DataColumnNames.OPEN)    : list(df['Open']),
                str(config.DataColumnNames.HIGH)    : list(df['High']),
                str(config.DataColumnNames.LOW)     : list(df['Low']),
                str(config.DataColumnNames.CLOSE)   : list(df['Close']),
                str(config.DataColumnNames.VOLUME)  : list(df['Volume'])
            },
            index=list(df.index.date)
        )


    @property
    def data_provider_name(self):
        return "Yahoo Financial Service"

    @property
    def data_frequency(self):
        return '1D'