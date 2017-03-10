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
data_stock = data_nan[data_nan.xref == 'MS:TS480']#'MS:TS1438' 
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
plt.title('Missing values. dates: %0.0f' %(len(dates)))


#%%

stocks = data_nan['xref']
stock_counts = stocks.value_counts()
unique_weeks = stock_counts.value_counts()


stock_count = np.sum([n_stocks if weeks > 1100 else 0 for n_stocks,weeks 
                      in zip(unique_weeks,unique_weeks.index.values)])


 
#%%    
def get_stocks_min_weeks(data, min_number_of_weeks):
    stocks = data['xref']
    stock_counts = stocks.value_counts()
    stocks_of_min_weeks = stock_counts[stock_counts.values >= min_number_of_weeks]
    
    return(stocks_of_min_weeks)
def plot_missing_values(data, stock_name):

    data_stock = data[data.xref == stock_name]#'MS:TS1438' 
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

    for i in range(num):
        plt.plot(x_values,y_values[i,:], 'ro')
    #plt.plot(x_values,y_values, 'ro')
        
    plt.xlim(0,len(dates))
    plt.ylim(0,26)
    
    xlabels = header[3:]
    plt.yticks(range(1,num+1), xlabels)
    plt.xlabel('dates')
    plt.title(stock_name + ' missing values. Dates: %0.0f' %(len(dates)))
    
    
    #plt.draw()
    return(None)
    
def get_missing_x_y(data, stock_name):
    data_stock = data[data.xref == stock_name]#'MS:TS1438' 
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
    return(x_values, y_values)
    
    
#%%

store_good_stocks = pd.Series(data = None, index = [None])
stocks_min_weeks = get_stocks_min_weeks(data_nan, 1100)

plt.ion()
fig = plt.figure()

for stock_name_i in stocks_min_weeks.index:
    
    plt.clf()
    #plot_missing_values(data_nan, stock_name_i)
    #plt.draw()
    
    x,y = get_missing_x_y(data_nan, stock_name_i)
    
    plt.axis([0,len(x),0,y.shape[0]])
    
    for i in range(y.shape[0]):
        plt.scatter(x, y[i,:],color = 'r')
        plt.show()
        plt.pause(0.0001)
        #plt.flush_events()
    
    ylabels = header[3:]
    plt.yticks(range(1,y.shape[0]+1), ylabels)
    plt.xlabel('dates')
    plt.title(stock_name_i + ' missing values. Dates: %0.0f' %(stocks_min_weeks[stock_name_i]))    

    good_or_bad = []

    while good_or_bad != 1 and good_or_bad != 0:
        good_or_bad = input(stock_name_i + ', good(1) or bad (0): ')
        
        try:
            good_or_bad = int(good_or_bad)
        except ValueError:
            good_or_bad = []
    
    if good_or_bad == 1:
        
        good_stock = stocks_min_weeks[stocks_min_weeks.index == stock_name_i]
        store_good_stocks = store_good_stocks.append(good_stock)

      
path = '../Data/'
store_good_stocks.to_csv(path + 'good_stocks.csv')

#%%

data_good = pd.DataFrame(data = None, columns = data_nan.columns)
for stock_i in store_good_stocks.index:
    data_stock_i = data_nan[data_nan.xref == stock_i]
    data_good = data_good.append(data_stock_i)

path = '../Data/'
data_good.to_csv(path + 'data_good.csv',index = False)
    
#%%
x, y = get_missing_x_y(data_nan, 'MS:TS712')

plt.ion()
fig = plt.figure()
plt.axis([0,len(x),0,y.shape[0]])

for i in range(y.shape[0]):
    plt.scatter(x, y[i,:],color = 'r')
    plt.show()
    plt.pause(0.05)
    #plt.flush_events()

xlabels = header[3:]
plt.yticks(range(1,num+1), xlabels)
plt.xlabel('dates')
plt.title(stock_name + ' missing values. Dates: %0.0f' %(len(dates)))    
plt.title(stock_name + ' missing values. Dates: %0.0f' %(len(dates)))



#%%
plt.clf()
for i in range(1,21):
    print(i)
    
    y_current = data_stock.iloc[:,i+2]
    
    plt.subplot(5,4,i)
    plt.plot(x_values,y_current, color = 'b')
    mean = np.mean(y_current)
    sd = np.std(y_current)
    plt.axhline(mean, color = 'k', linestyle= '--')
    plt.axhline(mean + sd, color = 'r', linestyle = '--')
    plt.axhline(mean - sd, color = 'r', linestyle = '--')
    plt.title('F' + str(i))



