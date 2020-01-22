# %%
# importing libraries
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
import cufflinks as cf
import seaborn as sns
import pandas as pd
import numpy as np
import quandl
import plotly
import time

from pandas_datareader import data as pdr

import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from IPython.display import Markdown, display
from matplotlib.ticker import FuncFormatter
from pandas.core.base import PandasObject
from datetime import datetime

# Setting pandas dataframe display options
pd.set_option("display.max_rows", 20)
pd.set_option('display.width', 800)
pd.set_option('max_colwidth', 800)

# Set plotly offline
init_notebook_mode(connected=True)

# Set matplotlib style
plt.style.use('seaborn')

# Set cufflinks offline
cf.go_offline()

def preview(df):
    return pd.concat([df.head(3), df.tail(3)])

def compute_time_series(dataframe):

#    INPUT: Dataframe of returns
#    OUTPUT: Growth time series starting in 100

    return (np.exp(np.log1p(dataframe).cumsum())) * 100

# %%
# download quotes
tickers = ['SPY', 'TLT']
Quotes = pd.DataFrame()
Start ='2017-12-29'
End = "2019-01-01"

for ticker in tickers:
    Quotes[ticker] = pdr.get_data_yahoo(ticker, start=Start, end=End)['Adj Close']

Returns = Quotes.pct_change().dropna()

# %%
Returns

# %%
quotes_norm = Quotes / Quotes.iloc[0]
weights_all = pd.DataFrame()
n = 0
loop_end = 'No'

weights_limits = [0.5, 0.5]

for i in np.arange(10):
    n = n + 1
    
    weights = pd.DataFrame()

    Portfolio_norm = quotes_norm.sum(axis=1)

    for ticker in Quotes.columns:
        weights[ticker] = quotes_norm[ticker].shift(-1) / Portfolio_norm.shift(-1)

    weights = weights.shift(2)
    

    weights.iloc[1] = weights_limits
    
    weights = weights.dropna()
    
    try:
        break_point = weights.query( ("weights['SPY'] > 0.54"))# | (weights['TLT'] > 0.54)].index[0]    
        weights = weights[:break_point]
        quotes_norm = quotes_norm[break_point:]
        quotes_norm = quotes_norm / quotes_norm.iloc[0]
    except:
        loop_end = 'Yes'
    # print(quotes_norm)

    weights_all = weights_all.append(weights)

    if loop_end == "Yes":
        break

contributions = weights_all * Returns
Portfolio_ret = contributions.sum(axis=1)
Portfolio = pd.DataFrame(compute_time_series(Portfolio_ret))

first_day = pd.DataFrame(np.array([[100]]), index=[pd.to_datetime(Start)])
Portfolio = Portfolio.append(first_day)
Portfolio = Portfolio.sort_index()
    
Portfolio

# %%


# %%
