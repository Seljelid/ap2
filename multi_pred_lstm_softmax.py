#%%
import numpy as np
import numpy.matlib
import csv
import pandas as pd
from datetime import datetime as dt
import tensorflow as tf
import random
import matplotlib.pyplot as plt
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
data['date'] = pd.to_datetime(data['date'], format='%d%b%Y')

#%%
def concat_features(data):
    good_stocks = np.unique(data['xref'])

    max_dates = 0;

    for stock_i in good_stocks:
        
        data_i = data[data.xref == stock_i]
        current_max_dates = np.shape(data_i)[0]
        if max_dates == 0 or current_max_dates < max_dates:
            max_dates = current_max_dates 
    
    i = 0
    for stock_i in good_stocks:
        
        data_i = data[data.xref == stock_i]
        data_i.set_index('date', inplace = True)
        start_idx = np.shape(data_i)[0] - max_dates 
        data_i = data_i.iloc[start_idx:, :]
        if i == 0:
            df_stocks = data_i
        else:
            df_stocks = pd.concat([df_stocks,data_i],axis=1)
        i += 1
        
    return df_stocks

df_stocks = concat_features(data)

#%% Fill in missing values
def fill_missing(data,header):
    factors = header[3:]

    for factor in factors:
        df_factor = df_stocks[factor]
        df_factor = df_factor.T
        df_factor = df_factor.fillna(df_factor.mean())
        df_factor = df_factor.T
        df_stocks[factor] = df_factor
                 
    df_stocks.fillna(value=0, inplace = True)
    
    return df_stocks

df_stocks = fill_missing(data,header)

#%%    
X = df_stocks[header[3:24]]
Y = df_stocks[header[-3]].as_matrix() # y-value of the first stock

# Add previous returns, remove first 4
Z4 = df_stocks[header[24]]
Z4  = Z4.iloc[:-4,:]
X = X.iloc[4:,:]
Y = Y[4:]
Z4 =  Z4.set_index( X.index )
X = pd.concat([X,Z4],axis = 1)

#X = preprocessing.scale(X)
#X = pd.DataFrame(X)
#Y = np.expand_dims(Y,axis = 1)
X, Y = X.iloc[:-4,:], Y[:-4]

#%%
def sequence_data(X,Y,time_steps):
    start = np.shape(X)[0] % time_steps
    X = X.iloc[start:,]
    Y = Y[start:]
    rnn_data = []
    rnn_targets = []
    for i in range(int(np.shape(X)[0]/time_steps)):
        X_current = X.iloc[i*time_steps:(i+1)*time_steps,].as_matrix()
        Y_current = Y[i*time_steps:(i+1)*time_steps]
        rnn_data.append(X_current)
        rnn_targets.append(Y_current)
    
    return np.array(rnn_data), np.array(rnn_targets)
    
    
# PARA AND DATASETS
#parameters
batch_size = 20
backprop_length = 10
input_size = np.shape(X)[-1]
output_size = np.shape(Y)[-1]
state_size = 512
num_layers = 4
learning_rate = 0.0005
#dropout_prob = 0.5
n_epochs = 200
dropout_prob = 0.8

X, Y = sequence_data(X,Y,backprop_length)

input_size = np.shape(X)[2]
training_size = int(np.shape(X)[0]*0.90)
test_size = np.shape(X)[0]-training_size
train_data = X[:training_size]
train_target = Y[:training_size]
test_data = X[training_size:]
test_target = Y[training_size:]

#%% MODEL, TENSORFLOW
tf.reset_default_graph()
#placeholders, things that we want to feed between runs

inputs = tf.placeholder(tf.float32, [None, backprop_length , input_size], name = 'x')
outputs = tf.placeholder(tf.float32, [None, backprop_length, output_size], name = 'y')
keep_prob = tf.placeholder(tf.float32)

#Initialize
init_state = tf.placeholder(tf.float32, [num_layers, 2, None, state_size])
state_per_layer_list = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple(
    [tf.contrib.rnn.core_rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(num_layers)])
cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(state_size, state_is_tuple = True)
cell = tf.contrib.rnn.core_rnn_cell.DropoutWrapper(cell,output_keep_prob = keep_prob)
cell = tf.contrib.rnn.core_rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
state_outputs, current_state =  tf.nn.dynamic_rnn(cell, inputs, initial_state = rnn_tuple_state)#init_state
weight = tf.Variable(tf.truncated_normal([state_size, output_size]))
bias = tf.Variable(tf.truncated_normal([output_size]))

#the model
state_outputs_reshaped = tf.reshape(state_outputs,[tf.shape(inputs)[0]*backprop_length,state_size])
prediction_orig = tf.matmul(state_outputs_reshaped, weight) + bias
                           
prediction_softmax = tf.nn.softmax(prediction_orig)

#one_hot_out = tf.one_hot(out_matrix ,depth = output_size , )

out_retard = tf.reshape(outputs,[-1, output_size], name = 'Retard')
out_softmax = tf.nn.softmax(out_retard)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction_orig, labels = out_softmax))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
minimize = optimizer.minimize(cost)

#Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_error_store = []
test_error_store = []

zero_state_batch = np.zeros((num_layers, 2, batch_size, state_size))
zero_state_train = np.zeros((num_layers, 2, training_size, state_size))
zero_state_test = np.zeros((num_layers, 2, test_size, state_size))
num_batches = int(training_size / batch_size)

best_model_error = -1;
for epoch_idx in range(n_epochs):
    
    for batch_idx in range(num_batches):
        #
        _current_state = zero_state_batch
        start_idx = (batch_idx) * batch_size
        end_idx =  (batch_idx + 1) * batch_size
        x = train_data[start_idx:end_idx, : , :]
        y = train_target[start_idx:end_idx, : ]

        _current_state,_,_state_out,_pred_out = sess.run([current_state,minimize,state_outputs,prediction_orig], 
                                               feed_dict = {inputs: x, outputs: y,init_state: _current_state,keep_prob: dropout_prob})
        
    _epoch_error = sess.run(cost, feed_dict={inputs: train_data, outputs: train_target,
                            init_state: zero_state_train , keep_prob: dropout_prob})
    _test_error,_test_pred,_test_out = sess.run([cost,prediction_softmax,out_softmax], feed_dict={inputs: test_data, outputs: test_target, 
                                              init_state: zero_state_test, keep_prob: 1})
    train_error_store.append(_epoch_error)
    test_error_store.append(_test_error)
    print('Epoch: ', epoch_idx , ', training:  %.2f  validation: %.2f' %(_epoch_error, _test_error))
    
    if best_model_error > _test_error or best_model_error < 0:
        best_model_error = _test_error
        best_prediction = _test_pred
        best_epoch = epoch_idx

#%%
past_return , new_target = predict_past_return(test_target)
mse_past_return = mse_error(past_return,new_target)

past_mean, new_target = predict_past_mean(test_target)
mse_past_mean = mse_error(past_mean, new_target)

stock_to_plot = 4
n_pred_weeks = np.shape(test_target)[0]*np.shape(test_target)[1]
    
plt.clf()
plt.subplot(2,1,1)
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.title('learn rate = %s, batch = %s, time steps = %s'
          %(learning_rate,batch_size,backprop_length) )
plt.plot(train_error_store, linewidth=1, label = "Train")
plt.plot(test_error_store, linewidth=1, label = "Validation", color = "green")
plt.plot(best_epoch, best_model_error, 'ro')
plt.axhline(mse_past_return,color = 'orange',label = "Past return", linestyle = "--")
plt.axhline(mse_past_mean, color = 'lightcoral', label = "Past mean", linestyle = "--")
plt.grid()
plt.legend()
plt.show()
plt.subplot(2,1,2)
# = range(len(best_prediction)/232)
x = range(n_pred_weeks)
plt.plot(x, best_prediction[stock_to_plot*n_pred_weeks:stock_to_plot*n_pred_weeks+n_pred_weeks], color = 'g', label = "Best-pred")
plt.plot(x, np.reshape(test_target[:,:,stock_to_plot],[-1]), color = 'steelblue', label = "Target")
#plt.plot(x, best_prediction,color = 'g',label = "Best-pred")
#plt.plot(x, _test_pred,color = 'black',label = "Last-pred")
#plt.plot(x, np.reshape(np.reshape(test_target,[60,232]).T,[-1]),color = 'steelblue', label = "Target")
#plt.plot(x, np.append([None,None,None,None],past_mean), color = 'lightcoral', label = "Past mean")
plt.axhline(0,color = 'y', linestyle = "--")
plt.title('best model prediction')
plt.legend()

#%%
def mse_error(x1,x2):
    dx = x1-x2
    mse = np.sum(np.power(dx,2)) / len(dx)
    return mse
    
def predict_past_return(target, weeks = 4):
    if len(np.shape(target)) > 1:
        target = np.reshape(target,[-1])
    past_return = target[:-weeks]
    new_target = target[weeks:]
    return past_return, new_target
    
def predict_past_mean(target, weeks = 4):
    if len(np.shape(target)) > 1:
        target = np.reshape(target,[-1])
    past_return = target[:-weeks]
    past_mean = np.cumsum(past_return)/np.linspace(1,len(past_return),len(past_return))
    new_target = target[weeks:]
    return past_mean, new_target

#%%
weekly_returns = np.reshape(best_prediction,[n_pred_weeks,-1])
real_returns = np.reshape(test_target,[n_pred_weeks,-1])
sorted_returns = np.argsort(weekly_returns)
ratings = np.arange(1,233)
ratings = ratings/np.sum(ratings)
#ratings = best_prediction
#ratings = np.matlib.repmat(ratings,60,1)'

#i = 0

#for ratings_row, returns_row in zip(ratings,sorted_returns):
#   ratings[i,:] = ratings_row[returns_row]
#   i += 1
    
real_returns_per_week = np.mean(real_returns, axis = 1)

#%%
value = 1
soft_value_vector = np.ones([int(n_pred_weeks/4)])

i = 0
j = 1
k = 0
for weekly_return in real_returns_per_week:
    if np.mod(i,4) == 0:
        value = value + real_returns_per_week[j-4]*value/100
        soft_value_vector[k] = value
        k += 1
    i += 1
    j += 1
 
#val = 1
#val_vector = np.ones([15])
#i = 0
#j = 1
#k = 0
#weighted = np.array([])
#ret_vector = np.zeros([n_pred_weeks+1])
#for weekly_return, sorted_return in zip(real_returns,sorted_returns):
#    return_vector = weekly_return[sorted_return]
#    ret = np.dot(return_vector,ratings)
#    weighted = np.append(weighted,ret)
#    ret_vector[j] = ret
#    if np.mod(i,4) == 0:
#        val += ret_vector[j-4]*val/100
#        val_vector[k] = val
#        k += 1
#    #else:
#        #val_vector[i+1] = val_vector[i]
#    i += 1
#    j+= 1

val = 1
soft_val_vector = np.ones([int(n_pred_weeks/4)])
i = 0
j = 1
k = 0
weighted = np.array([])
ret_vector = np.zeros([n_pred_weeks+1])
for weekly_return, best_pred in zip(real_returns,best_prediction):
    #return_vector = weekly_return[sorted_return]
    ret = np.dot(weekly_return,best_pred)
    weighted = np.append(weighted,ret)
    ret_vector[j] = ret
    if np.mod(i,4) == 0:
        val += ret_vector[j-4]*val/100
        soft_val_vector[k] = val
        k += 1
    #else:
        #val_vector[i+1] = val_vector[i]
    i += 1
    j+= 1
    
opt_val = 1
soft_opt_val_vector = np.ones([int(n_pred_weeks/4)])
i = 0
j = 1
k = 0
weighted = np.array([])
ret_vector = np.zeros([n_pred_weeks+1])
for weekly_return, opt_weight in zip(real_returns,_test_out):
    ret = np.dot(weekly_return,opt_weight)
    weighted = np.append(weighted,ret)
    ret_vector[j] = ret
    if np.mod(i,4) == 0:
        val += ret_vector[j-4]*val/100
        soft_opt_val_vector[k] = val
        k += 1
    i += 1
    j += 1

cray_val = 1
cray_vector = np.ones([n_pred_weeks+1])
i = 0
for weekly_return, sorted_return in zip(real_returns, sorted_returns):
    cray_val = cray_val + weekly_return[sorted_return[-1]]*cray_val/100
    cray_vector[i+1] = cray_val
    i += 1
    
plt.clf()
plt.ylabel('Value')
plt.xlabel('Week')
plt.plot(np.arange(0,n_pred_weeks,4),soft_value_vector/soft_value_vector[0],label='Uniform')
plt.plot(np.arange(0,n_pred_weeks,4),soft_val_vector/soft_val_vector[0], label='LSTM softmax dist')
#plt.plot(np.arange(0,n_pred_weeks,4),soft_opt_val_vector/soft_opt_val_vector[0], label = 'Optimal portfolio (softmax)')
#plt.plot(cray_vector,label='Only best stock')
plt.legend()
