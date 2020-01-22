Portfolio_norm = quotes_norm.sum(axis=1)
weights = pd.DataFrame()

for ticker in Quotes.columns:
    weights[ticker] = quotes_norm[ticker].shift(-1) / Portfolio_norm.shift(-1)

weights = weights.shift(2)
weights.iloc[1] = [0.5, 0.5]
weights = weights.dropna()

# %%
weights

# %%
break_point = weights[(weights['SPY'] > 0.54) | (weights['TLT'] > 0.54)].index[0]
break_point

# %%
dataframe_1 = quotes_norm[:break_point]
weights_1 = weights[:break_point]
quotes_norm_2 = quotes_norm[break_point:]
quotes_norm_2 = quotes_norm_2 / quotes_norm_2.iloc[0]
quotes_norm_2

# %%
Portfolio_norm_2 = quotes_norm_2.sum(axis=1)
weights_2 = pd.DataFrame()

for ticker in Quotes.columns:
    weights_2[ticker] = quotes_norm_2[ticker].shift(-1) / Portfolio_norm_2.shift(-1)

weights_2 = weights_2.shift(2)
weights_2.iloc[1] = [0.5, 0.5]
weights_2 = weights_2.dropna()

# %%
break_point_2 = weights_2[(weights_2['SPY'] > 0.54) | (weights_2['TLT'] > 0.54)].index[0]
weights_2 = weights_2[:break_point_2]
dataframe_2 = quotes_norm_2[:break_point_2]
quotes_norm_3 = quotes_norm_2[break_point_2:]
quotes_norm_3 = quotes_norm_2 / quotes_norm_2.iloc[0]
quotes_norm_3

# %%
Portfolio_norm_3 = quotes_norm_3.sum(axis=1)
weights_3 = pd.DataFrame()

for ticker in Quotes.columns:
    weights_3[ticker] = quotes_norm_3[ticker].shift(-1) / Portfolio_norm_3.shift(-1)

weights_3 = weights_3.shift(2)
weights_3.iloc[1] = [0.5, 0.5]
weights_3 = weights_3.dropna()

# %%
break_point_3 = weights_3[(weights_3['SPY'] > 0.54) | (weights_3['TLT'] > 0.54)].index[0]
weights_3 = weights_3[break_point_3:]
dataframe_3 = quotes_norm_3[:break_point_3]
quotes_norm_4 = quotes_norm_3[break_point_3:]
quotes_norm_4 = quotes_norm_2 / quotes_norm_2.iloc[0]
quotes_norm_4

# %%
