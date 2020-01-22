###################### BY YEAR ###################################################



# %%
# Quotes.to_excel("Quotes.xlsx")
preview(Quotes)

# %%
Years = list(set(Quotes.index.year))

# %%
Returns = Quotes.pct_change().dropna()
preview(Returns)

# %%
weights = pd.DataFrame()
weights_all = pd.DataFrame()
last_business_day_years = pd.date_range(Start, periods=12, freq='BA')
extra_day = Quotes.iloc[:1,:]
n = 0

for year_period in [2017, 2018, 2019]:
    weights = pd.DataFrame()
    df = Quotes[Quotes.index.year == year_period]
    df = pd.concat([extra_day, df])
    df = df / df.iloc[0]

    if Start not in last_business_day_years:
        df = df.loc[~df.index.duplicated(keep='first')]
        
    df_ret = df.pct_change().dropna()
    Portfolio_year = df.sum(axis=1)
    for ticker in Quotes.columns:
        weights[ticker] = df[ticker].shift(-1) / Portfolio_year.shift(-1)
    weights = weights.shift(2)
    weights.iloc[1] = [0.5, 0.5]
    weights = weights.dropna()
    weights_all = weights_all.append(weights)
    extra_day = pd.DataFrame(Quotes[str(year_period)].iloc[-1]).transpose()
    n = n + 1

contributions = weights_all * Returns
Portfolio_ret = contributions.sum(axis=1)
Portfolio = pd.DataFrame(compute_time_series(Portfolio_ret))

if Start not in last_business_day_years:
    first_day = pd.DataFrame(np.array([[100]]), index=[pd.to_datetime(Start)])
    Portfolio = Portfolio.append(first_day)
    Portfolio = Portfolio.sort_index()

Portfolio

# %%