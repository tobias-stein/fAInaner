import argparse
import dask.dataframe as dd
import pandas as pd
import plotly.graph_objects as go

from pathlib import Path

parser = argparse.ArgumentParser(description="")
parser.add_argument("ticker", type=str)
parser.add_argument("--start", type=str, default=None)
parser.add_argument("--end", type=str, default=None)
parser.add_argument("--freq", type=str, default='1D')

def main(args):
    ddf     = dd.read_parquet(Path("data/stocks").joinpath(args.ticker, '1D', "*.parquet.gz"), calculate_divisions=True)
    idx     = ddf.index.compute()
    
    start   = idx[0]  if args.start is None else pd.Timestamp(args.start)
    end     = idx[-1] if args.end   is None else pd.Timestamp(args.end)
    ddf     = ddf[start:end]
    ddf     = ddf.compute()

    ddf     = pd.concat([
        trade_day.resample(args.freq).agg(
        {
            'open':     'first', 
            'high':     'max', 
            'low':      'min', 
            'close':    'last', 
            'volume':   'sum'
        }) 
        for _, trade_day in ddf.groupby(ddf.index.date)
    ])

    figure_ohlc = go.Figure(data=go.Candlestick(open=ddf.open, high=ddf.high, low=ddf.low, close=ddf.close, x=ddf.index))
    figure_ohlc.show()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)