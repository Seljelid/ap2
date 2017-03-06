#%%
import numpy as np
import csv
import pandas as pd
from datetime import datetime as dt
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing

DATA_PATH = '../Data/ai_dataset2.csv'

def read_data(file):
    with open(file) as file:
        df = pd.read_csv(file, delimiter=',', low_memory='false')
        n_nan = df.isnull().sum()   #count number of NaN
        #df.fillna(value=0,inplace='true')  #fill NaN with zero. Don't do this for now
        header = list(df)
        data_nan = df
        #data = df.as_matrix()
        #for row in data:
        #   row[1] = dt.strptime(row[1],'%d%b%Y') #create datetime objects
    return data_nan, n_nan, header

data_nan, n_nan, header = read_data(DATA_PATH)

#%%
stocks = data_nan['xref']
stock_counts = stocks.value_counts()
min_weeks = np.percentile(stock_counts,95) #over the 70th percentile
stock_counts_70 = stock_counts[stock_counts >= min_weeks]
stocks_70 = np.array(stock_counts_70.index.values.tolist())

#%%
max_dates = 0;

for stock_i in stocks_70:
    
    data_i = data_nan[data_nan.xref == stock_i]
    current_max_dates = np.shape(data_i)[0]
    if max_dates == 0 or current_max_dates < max_dates:
        max_dates = current_max_dates 

i = 0
for stock_i in stocks_70:
    
    data_i = data_nan[data_nan.xref == stock_i]
    data_i.set_index('date', inplace = True)
    start_idx = np.shape(data_i)[0] - max_dates 
    data_i = data_i.iloc[start_idx:, :]
    if i == 0:
        df_stocks = data_i
    else:
        df_stocks = pd.concat([df_stocks,data_i],axis=1)
    i += 1

