# -*- coding: utf-8 -*-

# MF 796 - Assignment 4
# Date: 2019-02-17


# (1)
import fix_yahoo_finance as yf
yf.pdr_override()
import pandas_datareader.data as pdr
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def get_stock_prices_yahoo(ticker, start_date, end_date = 0):
    if end_date == 0:
        # if the end_day is not in the input, then choose the current date
        return pdr.get_data_yahoo(ticker, start_date)["Adj Close"]
    else: 
        return pdr.get_data_yahoo(ticker, start_date, end_date)["Adj Close"]

ticker = pd.read_excel(r'E:\BU\19 Spring\MF796 Computational Methods of Mathematical Finance\Assignments\Assignment4\sp_500_stocks.xlsx')['Ticker'][150:250].values
ticker_list = list(ticker)
start_date = datetime.datetime(2014,1,1)
end_date = datetime.datetime(2018,12,31)
price_daily_data = get_stock_prices_yahoo(ticker_list, start_date, end_date)
price_daily_data = price_daily_data.fillna(method='ffill')
price_daily_data = price_daily_data.fillna(method='bfill')

# (2)
return_daily_data = price_daily_data.pct_change().dropna()

# (3)
cov = return_daily_data.cov()
cov_array = cov.values
e_vals,e_vecs = np.linalg.eig(cov_array)
e_vals.sort()
e_vals = e_vals[::-1]
pos = sum(e_vals>0)
neg = sum(e_vals<0)
plt.scatter(np.arange(len(e_vals)), e_vals)
plt.ylim([1.25*np.min(e_vals)-0.25*np.max(e_vals), -0.25*np.min(e_vals)+1.25*np.max(e_vals)])
plt.title('eigenvalues in order')
plt.ylabel('eigenvalue')
plt.show()

# (4)
from sklearn.decomposition import PCA

# define the pca function
n = 20
data = return_daily_data

# drop stocks that are not listed, and save stock index in a dict
data = data.dropna(axis=1,how='any')
ticker_name = data.columns
ticker_dict = {x: y for x, y in enumerate(ticker_name)}

# construct the pca model and fit the model with data
sample = data.values
model = PCA(n_components=20)
model.fit(sample)

# compute PCA components and corresponding variance ratio
pcs = model.components_
pcs_mat = np.matrix(pcs)
var_ratio = model.explained_variance_ratio_
var_ratio_mat = np.matrix(var_ratio)

# compute overall loadings for each stock
load_mat = var_ratio_mat*pcs_mat

# find top 20 stocks with largest loadings
load_arr = np.asarray(load_mat).reshape(-1)
load_dict = {y: x for x, y in enumerate(load_arr)}
sort_load = sorted(load_arr, key=abs, reverse=True)
top_load = sort_load[:n]
ticker_num = [load_dict[x] for x in top_load]
selected_ticker = [ticker_dict[x] for x in ticker_num]

features = range(model.n_components_)
plt.bar(features, var_ratio)
plt.xticks(features)
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()

