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
        df.fillna(value=0,inplace='true')  #fill NaN with zero. Don't do this for now
        header = list(df)
        data = df
        #data = df.as_matrix()
        #for row in data:
        #   row[1] = dt.strptime(row[1],'%d%b%Y') #create datetime objects
    return data, n_nan, header

data, n_nan, header = read_data(DATA_PATH)
#%%

good_stocks = np.array(['MS:TS604','MS:TS1000','MS:TS3551','MS:TS3821'])

max_dates = 0;

for stock_i in good_stocks:
    
    data_i = data[data.xref == stock_i]
    current_max_dates = np.shape(data_i)[0]
    if max_dates == 0 or current_max_dates < max_dates:
        max_dates = current_max_dates 

#%%
df_stocks =  pd.DataFrame(data=None, index = None, columns = None)
print(nps(df_stocks))
for stock_i in good_stocks:
    
    data_i = data[data.xref == stock_i]
    start_idx = np.shape(data_i)[0] - max_dates 
    data_i = data_i.iloc[start_idx:, :]
    df_stocks = pd.concat([df_stocks, data_i], join = 'outer' ,axis = 1, ignore_index = True)
    print(nps(df_stocks))

## Varför appendar den på rader oxå?  (1072, 27) -> (2144, 54) osv,
# borde vara (1072, 27) -> (1072, 54)


