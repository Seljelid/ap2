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
data_stock = data_nan[data_nan.xref == 'MS:TS1438' ]
num = 24

dates = np.unique(data_stock.date)
x_values = np.array(range(len(dates)))

y_values = np.array([None]*len(dates))
y_values = np.array( [y_values, ]*num)
#y_values = np.zeros(len(dates))
#y_values(data_stock.F1.isnull()) = 1 
for i in range(num):
    y_current = y_values[i]
    nan_idx  =  np.array(data_stock.iloc[:,i+3].isnull())
    y_values[i,nan_idx] = i+1
#y_values[data_stock]
#%%
plt.clf()

for i in range(num):
    plt.plot(x_values,y_values[i,:], 'ro')
plt.xlim(0,len(dates))
plt.ylim(0,26)

xlabels = header[3:]
plt.yticks(range(1,num+1), xlabels)
plt.xlabel('dates')
plt.title('Missing values for parameters')
#%%





