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

def rank_by_features(data):
    
    """ Creates a dataframe with stocks ranked based on performance in each
        feature (like AP2 does). Highest total rank, summed over all of
        the features recieves the best rank and therefore the highest weight.
        Dataframe includes stock, return scores, rank and weight.
    """

    dates = np.unique(data.index)
    features = list(data)[2:23]
    
    data_feature_rank = pd.DataFrame([])
    data_rank = pd.DataFrame([])
    data_weights = pd.DataFrame([])
    for date_i in dates:
        data_date = data[data.index == date_i]
        stocks = data_date.xref
        z_4w = data_date.z_score_4w
        z_8w = data_date.z_score_8w
        z_12w = data_date.z_score_12w
        data_date = data_date[features]
    
        order = np.argsort(data_date, axis = 0)
        feature_rank = np.argsort(order, axis = 0)
        stock_rank = np.sum(feature_rank, axis = 1)
        
        if data_rank.empty:
            #data_feature_rank = feature_rank
            tmp = pd.concat([stocks, z_4w, z_8w, z_12w, stock_rank],axis = 1)
            tmp = tmp.iloc[np.argsort(tmp.iloc[:,-1]),:]#Sort by ranking
            tmp['weight'] = np.arange(1,len(stock_rank)+1)/np.sum(np.arange(1,len(stock_rank)+1))
        
            data_rank = tmp
            
        else: 
            
            tmp = pd.concat([stocks, z_4w, z_8w, z_12w, stock_rank],axis = 1)
            tmp = tmp.iloc[np.argsort(tmp.iloc[:,-1]),:]
            tmp['weight'] = np.arange(1,len(stock_rank)+1)/np.sum(np.arange(1,len(stock_rank)+1))
            data_rank = pd.concat([data_rank,tmp],axis = 0)
        
    data_rank.columns = ['xref', 'z_score_4w', 'z_score_8w', 'z_score_12w', 'rank', 'weight']
        
    return data_rank

data_rank = rank_by_features(data)
#%%
n_pred_weeks = 230
dates = np.unique(data_rank.index)

start_date = dates[-n_pred_weeks]
end_date = dates[-1]

def compute_return(df, start_date, end_date):
    
    """ Creates a dataframe with return values for each available date between
        start_date and end_date. df should be a dataframe produced by rank_by_features().
    """
    
    pred_df = df.loc[start_date:end_date]
    pred_dates = np.unique(pred_df.index)
    returns = [np.dot(pred_df.loc[date]['weight'], pred_df.loc[date]['z_score_4w']) for date in pred_dates]
    df_return = pd.DataFrame(data = [pred_dates,returns]).T
    df_return.columns = ['date','return']
    df_return.set_index('date', inplace = True)
    
    return df_return, pred_dates

df_return, pred_dates = compute_return(data_rank, start_date, end_date)

#%%
trade_dates = pred_dates[0:-1:4]

def compute_value(df_return, trade_dates):
    val = 1
    val_vec = np.ones((len(trade_dates)+1))
    i = 1
    for date in trade_dates:
        ret = df_return.loc[date]
        val += val*ret/100
        val_vec[i] = val
        i += 1
        
    return val_vec
        
feat_val_vec = compute_value(df_return, trade_dates)
plt.clf()
plt.ylabel('Return')
plt.xlabel('Week')
plt.plot(np.arange(0,len(feat_val_vec)*4,4),feat_val_vec,label='Feature ranked')
plt.legend()







