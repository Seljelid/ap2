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
data['date'] = pd.to_datetime(data['date'], format='%d%b%Y')#Correct date format
data.set_index('date', inplace = True)#Dates as index

#%%

def concat_data(data): #concats, by column, stock data. Missing dates gives NA.
    stock_ids = np.unique(data.xref)
    
    i = 0
    for stock_i in stock_ids:
        data_i = data[data.xref==stock_i]
        
        if i == 0:
            df = data_i
        else: 
            df = pd.concat([df,data_i], axis = 1)
        i = 1
        
    return(df)
    
def fill_missing(df,header):#Fill with mean if available otherwise 0
    factors = header[3:]

    for factor in factors:
        df_factor = df[factor]
        df_factor = df_factor.T
        df_factor = df_factor.fillna(df_factor.mean())
        df_factor = df_factor.T
        df[factor] = df_factor
                 
    df.fillna(value=0, inplace = True)
    
    return df


def create_X_Y(data_conc, header): #Creates X and Y data, also returns the order of stocks
    #TODO: add the last known return value to X
    features = header[3:24]
    X = data_conc[features]
    
    targets = header[24]
    Y = data_conc[targets]

    stock_order = data_conc['xref'].iloc[0,:]
    
    return X,Y,stock_order
    
    
data_concat = concat_data(data)
df = fill_missing(data_concat, header)
ix,why,stock_order = create_X_Y(df, header)
#%%

X = preprocessing.scale(ix)
X = pd.DataFrame(X)
y = np.expand_dims(why,axis = 1)
#X, Y = X.iloc[:-4,:], Y[:-4]

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
batch_size = 10
backprop_length = 10
input_size = np.shape(X)[-1]
output_size = np.shape(y)[-1]
state_size = 256
num_layers = 2
learning_rate = 0.0001
#dropout_prob = 0.5
n_epochs = 100
dropout_prob = 0.7

X, Y = sequence_data(X,y,backprop_length)
Y = np.squeeze(Y)
input_size = np.shape(X)[2]
training_size = int(np.shape(X)[0]*0.80)
test_size = int((np.shape(X)[0]-training_size)/2)
validation_size = test_size
train_data = X[:training_size]
train_target = Y[:training_size]
validation_data = X[training_size:training_size+validation_size]
validation_target = Y[training_size:training_size+validation_size]
test_data = X[training_size+validation_size:training_size+validation_size+test_size]
test_target = Y[training_size+validation_size:training_size+validation_size+test_size]

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
weight = tf.Variable(tf.truncated_normal([state_size, output_size], mean = 0.0, stddev = 0.3))
bias = tf.Variable(tf.truncated_normal([output_size], mean = 0.0, stddev = 0.3))

#the model
state_outputs_reshaped = tf.reshape(state_outputs,[tf.shape(inputs)[0]*backprop_length,state_size])
prediction_orig = tf.matmul(state_outputs_reshaped, weight) + bias
prediction = tf.reshape(prediction_orig,[-1])

l2 = tf.nn.l2_loss(weight)

out_reshaped = tf.reshape(outputs, [-1])
cost = tf.reduce_sum(tf.pow(prediction-out_reshaped, 2)/(2*training_size*backprop_length*output_size)) + 0.001*l2
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
minimize = optimizer.minimize(cost)
error_validation = tf.reduce_sum(tf.pow(prediction-out_reshaped,2)/(2*validation_size*backprop_length*output_size)) + 0.001*l2
error_test = tf.reduce_sum(tf.pow(prediction-out_reshaped,2)/(2*test_size*backprop_length*output_size)) + 0.001*l2

#Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_error_store = []
validation_error_store = []

zero_state_batch = np.zeros((num_layers, 2, batch_size, state_size))
zero_state_train = np.zeros((num_layers, 2, training_size, state_size))
zero_state_validation = np.zeros((num_layers, 2, validation_size, state_size))
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

        _current_state,_,_state_out,_pred_out,_out = sess.run([current_state,minimize,state_outputs,prediction_orig,out_reshaped], 
                                               feed_dict = {inputs: x, outputs: y, init_state: _current_state,keep_prob: dropout_prob})
        
    _epoch_error = sess.run(cost, feed_dict={inputs: train_data, outputs: train_target, 
                            init_state: zero_state_train , keep_prob: dropout_prob})
    _validation_error,_validation_pred = sess.run([error_validation,prediction], feed_dict={inputs: validation_data, outputs: validation_target,
                                              init_state: zero_state_validation, keep_prob: 1})
    train_error_store.append(_epoch_error)
    validation_error_store.append(_validation_error)
    print('Epoch: ', epoch_idx , ', training:  %.2f  validation: %.2f' %(_epoch_error, _validation_error))
    
    if best_model_error > _validation_error or best_model_error < 0:
        best_model_error = _validation_error
        best_prediction = _validation_pred
        best_epoch = epoch_idx
        _test_error,_test_pred = sess.run([error_test,prediction], feed_dict={inputs: test_data, outputs: test_target,
                                              init_state: zero_state_test, keep_prob: 1})


print('Test error: %.2f' %(_test_error))
#%%
past_return , new_target = predict_past_return(test_target)
mse_past_return = mse_error(past_return,new_target)

past_mean, new_target = predict_past_mean(test_target)
mse_past_mean = mse_error(past_mean, new_target)

stock_to_plot = 0
n_pred_weeks = np.shape(test_target)[0]*np.shape(test_target)[1]
    
plt.clf()
plt.subplot(2,1,1)
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.title('learn rate = %s, batch = %s, time steps = %s'
          %(learning_rate,batch_size,backprop_length) )
plt.plot(train_error_store, linewidth=1, label = "Train")
plt.plot(validation_error_store, linewidth=1, label = "Validation", color = "green")
plt.plot(best_epoch, best_model_error, 'ro')
plt.axhline(mse_past_return,color = 'orange',label = "Past return", linestyle = "--")
plt.axhline(mse_past_mean, color = 'lightcoral', label = "Past mean", linestyle = "--")
plt.grid()
plt.legend()
plt.show()
plt.subplot(2,1,2)
# = range(len(best_prediction)/232)
x = range(n_pred_weeks)
plt.plot(x, _test_pred[stock_to_plot*n_pred_weeks:stock_to_plot*n_pred_weeks+n_pred_weeks], color = 'g', label = "Test-pred")
plt.plot(x, np.reshape(test_target[:,:,stock_to_plot],[-1]), color = 'steelblue', label = "Target")
#plt.plot(x, best_prediction,color = 'g',label = "Best-pred")
#plt.plot(x, _test_pred,color = 'black',label = "Last-pred")
#plt.plot(x, np.reshape(np.reshape(test_target,[60,232]).T,[-1]),color = 'steelblue', label = "Target")
#plt.plot(x, np.append([None,None,None,None],past_mean), color = 'lightcoral', label = "Past mean")
plt.axhline(0,color = 'y', linestyle = "--")
plt.title('Test model prediction')
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
weekly_returns = np.reshape(_test_pred,[n_pred_weeks,-1])
real_returns = np.reshape(test_target,[n_pred_weeks,-1])

pred_order = np.argsort(weekly_returns, axis = 1)
pred_rank = np.argsort(pred_order, axis = 1)
weights = np.arange(1,233)/np.sum(np.arange(1,233))

i = 0
returns = np.zeros([n_pred_weeks])
for week, real_return_vec in zip(pred_rank, real_returns):
    weight_vec = weights[week]
    ret = np.dot(weight_vec, real_return_vec)
    returns[i] = ret
    i += 1
    
returns = returns[0:-1:4]
def compute_pred_value(returns):
    val = 1
    val_vec = np.ones([len(returns)+1])
    i = 1
    for ret in returns:
        val += val*ret/100
        val_vec[i] = val
        i += 1
        
    return val_vec
    
val_vec = compute_pred_value(returns)

#%%
i = 0
uni_ret = np.zeros([n_pred_weeks])
for real_return in real_returns:
    ret = np.mean(real_return)
    uni_ret[i] = ret
    i += 1

uni_ret = uni_ret[0:-1:4]
    
uni_val_vec = compute_pred_value(uni_ret)

#%%
    
plt.clf()
plt.ylabel('Return')
plt.xlabel('Week')
plt.plot(np.arange(0,len(feat_val_vec)*4,4),feat_val_vec,label='Feature ranked')
plt.plot(np.arange(0,len(val_vec)*4,4),val_vec, label='Linear rating by LSTM return')
#plt.plot(np.arange(0,len(soft_val_vec)*4,4),soft_val_vec, label='Softmax dist')
plt.plot(np.arange(0,len(uni_val_vec)*4,4),uni_val_vec, label='Uniform')
plt.grid()
plt.legend()
plt.show()
