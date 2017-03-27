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

for date_i in dates:
    data_date = data[data.index == date_i]
    order = np.argsort(data_date, axis = 0)
    rank = np.argsort(order, axis = 0)
    print(rank['F1'].iloc[0:10], data_date['F1'][0:10])
