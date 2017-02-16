#%%
import numpy as np
import csv
import pandas as pd
from datetime import datetime as dt
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
from diagnostic_missing import read_data

DATA_PATH = '../Data/ai_dataset2.csv'

data_nan, n_nan, header = read_data(DATA_PATH)

#%%
good_stocks = ['MS:TS604','MS:TS1000','MS:TS3551','MS:TS3546'] #3821

#stock_factors = data_nan[header[2:23]]

df_604 = data_nan[data_nan.xref == 'MS:TS604']
df_1000 = data_nan[data_nan.xref == 'MS:TS1000']

df_tot = pd.concat([df_604,df_1000])
df_tot = df_tot.sort_values(by = 'date')

