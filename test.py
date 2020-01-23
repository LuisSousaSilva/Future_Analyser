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

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from IPython.display import Markdown, display
from matplotlib.ticker import FuncFormatter
from pandas.core.base import PandasObject
from datetime import datetime

# Setting pandas dataframe display options
pd.set_option("display.max_rows", 20)
pd.set_option('display.width', 800)
pd.set_option('max_colwidth', 800)

pd.options.display.float_format = '{:,.2f}'.format

# Set plotly offline
init_notebook_mode(connected=True)

# Set matplotlib style
plt.style.use('seaborn')

# Set cufflinks offline
cf.go_offline()

# Defining today's Date
from datetime import date
from pandas.tseries.offsets import BDay
today = date.today()

from pandas_datareader import data as pdr

import yfinance as yf
yf.pdr_override() # <== that's all it takes :-)

def merge_time_series(df_1, df_2):
    df = df_1.merge(df_2, how='left', left_index=True, right_index=True)
    return df
    
# %%


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

def compute_portfolio(quotes, weights):

    Nomes = quotes.columns
    
    # Anos do Portfolio
    Years = quotes.index.year.unique()

    # Dicionário com Dataframes anuais das cotações dos quotes
    Years_dict = {}
    k = 0

    for Year in Years:
        # Dynamically create key
        key = Year
        # Calculate value
        value = quotes.loc[str(Year)]
        # Insert in dictionary
        Years_dict[key] = value
        # Counter
        k += 1

    # Dicionário com Dataframes anuais das cotações dos quotes
    global Weights
    
    Quotes_dict = {}
    Portfolio_dict = {}
    Weights = pd.DataFrame()

    k = 0    
    
    for Year in Years:        
        n = 0
        
        #Setting Portfolio to be a Global Variable
        global Portfolio
                
        
        # Dynamically create key
        key = Year

        # Calculate value
        if (Year-1) in Years:
            value = Years_dict[Year].append(Years_dict[Year-1].iloc[[-1]]).sort_index()
        else:
            value = Years_dict[Year].append(Years_dict[Year].iloc[[-1]]).sort_index()

        # Set beginning value to 100
        value = (value / value.iloc[0]) * 100
        # 
        for column in value.columns:
            value[column] = value[column] * weights[n]
            n +=1
        
        # Get Returns
        Returns = value.pct_change()
        # Calculating Portfolio Value
        value['Portfolio'] = value.sum(axis=1)

        # Creating Weights_EOP empty DataFrame
        Weights_EOP = pd.DataFrame()
        # Calculating End Of Period weights
        for Name in Nomes:
            Weights_EOP[Name] = value[Name] / value['Portfolio']
        # Calculating Beginning Of Period weights
        Weights_BOP = Weights_EOP.shift(periods=1)
        Weights = Weights.append(Weights_BOP)

        # Calculatins Portfolio Value
        Portfolio = pd.DataFrame(Weights_BOP.multiply(Returns).sum(axis=1))
        Portfolio.columns=['Simple']
        # Transformar os simple returns em log returns 
        Portfolio['Log'] = np.log(Portfolio['Simple'] + 1)
        # Cumsum() dos log returns para obter o preço do Portfolio 
        Portfolio['Price'] = 100*np.exp(np.nan_to_num(Portfolio['Log'].cumsum()))
        Portfolio['Price'] = Portfolio['Price']   

        # Insert in dictionaries
        Quotes_dict[key] = value
        Portfolio_dict[key] = Portfolio
        # Counter
        k += 1

    # Making an empty Dataframe for Portfolio data
    Portfolio = pd.DataFrame()

    for Year in Years:
        Portfolio = pd.concat([Portfolio, Portfolio_dict[Year]['Log']])

    # Delete repeated index values in Portfolio    
    Portfolio.drop_duplicates(keep='last')

    # Naming the column of log returns 'Log'
    Portfolio.columns= ['Log']

    # Cumsum() dos log returns para obter o preço do Portfolio 
    Portfolio['Price'] = 100*np.exp(np.nan_to_num(Portfolio['Log'].cumsum()))
        
    # Round Portfolio to 2 decimals and eliminate returns
    Portfolio = pd.DataFrame(round(Portfolio['Price'], 2))

    # Naming the column of Portfolio as 'Portfolio'
    Portfolio.columns= ['Portfolio']

    # Delete repeated days
    Portfolio = Portfolio.loc[~Portfolio.index.duplicated(keep='first')]

    return Portfolio

# %%
# download quotes
tickers = ['SPY', 'AGG']
Quotes = pd.DataFrame()
Start ='2005-08-01'
End = "2020-01-15"

for ticker in tickers:
    Quotes[ticker] = pdr.get_data_yahoo(ticker, start=Start, end=End)['Adj Close']

Returns = Quotes.pct_change().dropna()

quotes_norm = Quotes / Quotes.iloc[0]
weights_all = pd.DataFrame()
n = -1
loop_end = 'No'
trigger = 0.99

while loop_end == "No":
    n = n + 1
    
    weights = pd.DataFrame()

    Portfolio_norm = quotes_norm.sum(axis=1)

    for ticker in Quotes.columns:
        weights[ticker] = quotes_norm[ticker].shift(-1) / Portfolio_norm.shift(-1)

    weights = weights.shift(2)
    
    weights.iloc[1] = [0.5, 0.5]
    
    weights = weights.dropna()
    
    try:
        break_point = weights[(weights['SPY'] > trigger) | (weights['AGG'] > trigger)].index[0]    
        weights = weights[:break_point]
        quotes_norm = quotes_norm[break_point:]
        quotes_norm = quotes_norm / quotes_norm.iloc[0]
    except:
        loop_end = 'Yes'
        print('Fez ' + str(n) + ' rebalanceamentos')

    weights_all = weights_all.append(weights)

contributions = weights_all * Returns
Portfolio_ret = contributions.sum(axis=1)
Portfolio = pd.DataFrame(compute_time_series(Portfolio_ret))

first_day = pd.DataFrame(np.array([[100]]), index=[pd.to_datetime(Start)])
Portfolio = Portfolio.append(first_day)

Portfolio_BH = Portfolio.sort_index()

# %%
Portfolio_BH.iplot(color='royalblue', title='B&H')

# %%
weights_all.iplot(title='Pesos de Buy & Hold')

# %%
# download quotes
tickers = ['SPY', 'TLT']
Quotes = pd.DataFrame()
Start ='2005-08-01'
End = "2020-01-15"

for ticker in tickers:
    Quotes[ticker] = pdr.get_data_yahoo(ticker, start=Start, end=End)['Adj Close']

# %%
##########################
Returns = Quotes.pct_change().dropna()

quotes_norm = Quotes / Quotes.iloc[0]
weights_all = pd.DataFrame()
n = -1
loop_end = 'No'

rebalance_dates = pd.bdate_range(start=Start, end=End, freq='BA')

for i in np.arange(len(rebalance_dates) + 1):
    n = n + 1
    
    weights = pd.DataFrame()

    Portfolio_norm = quotes_norm.sum(axis=1)

    for ticker in Quotes.columns:
        weights[ticker] = quotes_norm[ticker].shift(-1) / Portfolio_norm.shift(-1)

    weights = weights.shift(2)
    
    weights.iloc[1] = [0.5, 0.5]
    
    weights = weights.dropna()
    
    try:
        if rebalance_dates[n] in weights.index:
            break_point = weights[(weights.index == rebalance_dates[n])].index[0]
        else:
            break_point = weights[(weights.index == rebalance_dates[n] - BDay(1))].index[0]
            print(break_point)
        weights = weights[:break_point]
        quotes_norm = quotes_norm[break_point:]
        quotes_norm = quotes_norm / quotes_norm.iloc[0]
    except:
        loop_end = 'Yes'
        print('Fez ' + str(n) + ' rebalanceamentos')

    weights_all = weights_all.append(weights)

contributions = weights_all * Returns
Portfolio_ret = contributions.sum(axis=1)
Portfolio = pd.DataFrame(compute_time_series(Portfolio_ret))

first_day = pd.DataFrame(np.array([[100]]), index=[pd.to_datetime(Start)])
Portfolio = Portfolio.append(first_day)

Portfolio_time_rebanced = Portfolio.sort_index()

# %%
Portfolio_time_rebanced.iplot(color='royalblue', title='Relanceamentos temporais')

# %%
weights_all.iplot(title='Pesos de rebalanceamentos temporais')

# %%
Portfolios = merge_time_series(Portfolio_BH, Portfolio_time_rebanced)
Portfolios.columns = ['B&H', 'TR']
Portfolios.iplot()

# %%
def compute_portfolio_2(Quotes, asset_weights, rebalance_periods='years'):

    if rebalance_periods == 'years':
        rebalance_dates = pd.bdate_range(start=Start, end=End, freq='BA')

    if rebalance_periods == 'quarters':
        rebalance_dates = pd.bdate_range(start=Start, end=End, freq='BQ')
    
    if rebalance_periods == 'months':
        rebalance_dates = pd.bdate_range(start=Start, end=End, freq='BM')

    Returns = Quotes.pct_change().dropna()

    quotes_norm = Quotes / Quotes.iloc[0]
    weights_all = pd.DataFrame()
    n = -1
    loop_end = 'No'    

    for i in np.arange(len(rebalance_dates) + 1):
        n = n + 1
        
        weights = pd.DataFrame()

        Portfolio_norm = quotes_norm.sum(axis=1)

        for ticker in Quotes.columns:
            weights[ticker] = quotes_norm[ticker].shift(-1) / Portfolio_norm.shift(-1)

        weights = weights.shift(2)
        
        weights.iloc[1] = asset_weights
        
        weights = weights.dropna()
        
        try:
            if rebalance_dates[n] in weights.index:
                break_point = weights[(weights.index == rebalance_dates[n])].index[0]
            else:
                break_point = weights[(weights.index == rebalance_dates[n] - BDay(1))].index[0]
            weights = weights[:break_point]
            quotes_norm = quotes_norm[break_point:]
            quotes_norm = quotes_norm / quotes_norm.iloc[0]
        except:
            loop_end = 'Yes'
            print('Fez ' + str(n) + ' rebalanceamentos')

        weights_all = weights_all.append(weights)

    contributions = weights_all * Returns
    Portfolio_ret = contributions.sum(axis=1)
    Portfolio = pd.DataFrame(compute_time_series(Portfolio_ret))

    first_day = pd.DataFrame(np.array([[100]]), index=[pd.to_datetime(Start)])
    Portfolio = Portfolio.append(first_day)

    Portfolio = Portfolio.sort_index()
    return Portfolio

# %%
compute_portfolio(Quotes, [0.5, 0.5])

# %%
years = compute_portfolio_2(Quotes, [0.5, 0.5], 'years')
quarters = compute_portfolio_2(Quotes, [0.5, 0.5], 'quarters')
months = compute_portfolio_2(Quotes, [0.5, 0.5], 'months')

# %%
time_series = merge_time_series(years, quarters)
time_series = merge_time_series(time_series, months)
time_series.columns = ['years', 'quarters', 'months']
time_series.iplot()