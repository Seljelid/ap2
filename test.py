import numpy as np
import csv
import pandas as pd
from datetime import datetime as dt
import tensorflow as ts

DATA_PATH = '../Data/ai_dataset2.csv'

def read_data(file):
    with open(file) as file:
        df = pd.read_csv(file, delimiter=',', low_memory='false')
        n_nan = df.isnull().sum()   #count number of NaN
        #df.fillna(value=0,inplace='true')  #fill NaN with zero. Don't do this for now
        data = df.as_matrix()
        for row in data:
            row[1] = dt.strptime(row[1],'%d%b%Y') #create datetime objects
    return data, n_nan

data, n_nan = read_data(DATA_PATH)
print(data[-15000:-14990,:])
print(n_nan/np.shape(data[0::,1]))
unique, counts = np.unique(data[:,2], return_counts='True')
print('Number of unique stocks', len(unique))


