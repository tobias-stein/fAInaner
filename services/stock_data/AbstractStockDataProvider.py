import pandas as pd

class AbstractStockDataProvider(object):
    def __init__(self):
        pass

    def fetch(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        raise NotImplementedError("")

    @property
    def data_provider_name(self) -> str:
        raise NotImplementedError("")

    @property
    def data_frequency(self) -> str:
        raise NotImplementedError("")