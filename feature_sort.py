import numpy as np
import numpy.matlib
import csv
import pandas as pd
from datetime import datetime as dt
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import scipy
from sklearn import preprocessing

#DATA_PATH = '../Data/ai_dataset2.csv'
DATA_PATH = '../Data/data_pro.csv'

def read_data(file):
    with open(file) as file:
        df = pd.read_csv(file, delimiter=',', low_memory='false')
        n_nan = df.isnull().sum()   #count number of NaN
        header = list(df)
        data = df
    return data, n_nan, header

data, n_nan, header = read_data(DATA_PATH)
data['date'] = pd.to_datetime(data['date'], format='%d%b%Y')#Correct date format
data.set_index('date', inplace = True)#Dates as index

#%%

dates = np.unique(data.index)
features = list(data)[2:23]

data_feature_rank = pd.DataFrame([])
data_rank = pd.DataFrame([])
data_weights = pd.DataFrame([])
for date_i in dates:
    data_date = data[data.index == date_i]
    stocks = data_date.xref
    data_date = data_date[features]

    order = np.argsort(data_date, axis = 0)
    feature_rank = np.argsort(order, axis = 0)
    stock_rank = np.sum(feature_rank, axis = 1)
    
    if data_rank.empty:
        #data_feature_rank = feature_rank
        tmp = pd.concat([stocks, stock_rank],axis = 1)
        tmp = tmp.iloc[np.argsort(tmp.iloc[:,1]),:]#Sort by ranking
        tmp['weights'] = np.arange(1,len(stock_rank)+1)/np.sum(np.arange(1,len(stock_rank)+1))
    
        data_rank = tmp
        
        
    else: 
        
        tmp = pd.concat([stocks, stock_rank],axis = 1)
        tmp = tmp.iloc[np.argsort(tmp.iloc[:,1]),:]
        tmp['weights'] = np.arange(1,len(stock_rank)+1)/np.sum(np.arange(1,len(stock_rank)+1))
        data_rank = pd.concat([data_rank,tmp],axis = 0)



