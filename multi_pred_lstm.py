#%%
import numpy as np
import csv
import pandas as pd
from datetime import datetime as dt
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing
from IPython.display import display
from IPython.display import Image

#DATA_PATH = '../Data/ai_dataset2.csv'
DATA_PATH = '../Data/data_pro.csv'

def read_data(file):
    with open(file) as file:
        df = pd.read_csv(file, delimiter=',', low_memory='false')
        n_nan = df.isnull().sum()   #count number of NaN
        #df.fillna(value=0,inplace='true')  #fill NaN with zero. Don't do this for now
        header = list(df)
        data = df
        #data = df.as_matrix()
        #for row in data:
        #   row[1] = dt.strptime(row[1],'%d%b%Y') #create datetime objects
    return data, n_nan, header

data, n_nan, header = read_data(DATA_PATH)

#%% DO NOT RUN!!!!
stocks = data['xref']
stock_counts = stocks.value_counts()
min_weeks = np.percentile(stock_counts,95) #over the 70th percentile
stock_counts_70 = stock_counts[stock_counts >= min_weeks]
stocks_70 = np.array(stock_counts_70.index.values.tolist())


#%%

#good_stocks = np.array(['MS:TS604','MS:TS1000','MS:TS3551','MS:TS3821',
                        #'MS:TS2014', 'MS:TS2808', 'MS:TS2278', 'MS:TS3262'])
#good_stocks = stocks_70[0:5]
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

#%% Fill in missing values

factors = header[3:]

for factor in factors:
    df_factor = df_stocks[factor]
    df_factor = df_factor.T
    df_factor = df_factor.fillna(df_factor.mean())
    df_factor = df_factor.T
    df_stocks[factor] = df_factor
             
df_stocks.fillna(value=0, inplace = True)

#%%    
X = df_stocks[header[3:24]]
Y = df_stocks[header[-3]].as_matrix() # y-value of the first stock

# Add previous returns, remove first 4
Z4 = df_stocks[header[25]]
Z4  = Z4.iloc[:-4,:]
X = X.iloc[4:,:]
Y = Y[4:]
Z4 =  Z4.set_index( X.index )
X = pd.concat([X,Z4],axis = 1)

X = preprocessing.scale(X)
X = pd.DataFrame(X)
#Y = np.expand_dims(Y,axis = 1)
X, Y = X.iloc[:-4,:], Y[:-4]


def sequence_data(X,Y,time_steps):
    start = np.shape(X)[0] % time_steps
    X = X.iloc[start:,]
    Y = Y[start:]
    
    #rnn_data = np.zeros(np.shape(X)[0]/time_steps, time_steps, np.shape(X)[1])
    rnn_data = []
    rnn_targets = []
    for i in range(int(np.shape(X)[0]/time_steps)):
        X_current = X.iloc[i*time_steps:(i+1)*time_steps,].as_matrix()
        Y_current = Y[i*time_steps:(i+1)*time_steps]
        rnn_data.append(X_current)
        rnn_targets.append(Y_current)
        #print(Y_current[-1])
        #print(rnn_targets[-1])
        #rnn_data=  X[i*time_steps:(i+1)*time_steps, ]
    
    return np.array(rnn_data), np.array(rnn_targets)
    
    
# PARA AND DATASETS
#parameters
batch_size = 10
backprop_length = 10
input_size = np.shape(X)[-1]
output_size = np.shape(Y)[-1]
state_size = 512
num_layers = 3
learning_rate = 0.001
#dropout_prob = 0.5
n_epochs = 200
dropout_prob = 0.8

# Training and test/validation
#X = rnn_data(X,backprop_length)
#Y = Y[backprop_length:,:]
#Y_original = Y

X, Y = sequence_data(X,Y,backprop_length)

input_size = np.shape(X)[2]
training_size = int(np.shape(X)[0]*0.95)
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
#cell_state = tf.placeholder(tf.float32, [None, state_size])
#hidden_state = tf.placeholder(tf.float32, [None, state_size])

#Initialize
init_state = tf.placeholder(tf.float32, [num_layers, 2, None, state_size])
#init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)
state_per_layer_list = tf.unpack(init_state, axis=0)
rnn_tuple_state = tuple(
    [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(num_layers)])
cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple = True)
cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob = keep_prob)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
state_outputs, current_state =  tf.nn.dynamic_rnn(cell, inputs, initial_state = rnn_tuple_state)#init_state
weight = tf.Variable(tf.truncated_normal([state_size, output_size]))
bias = tf.Variable(tf.truncated_normal([output_size]))

#the model
state_outputs_reshaped = tf.reshape(state_outputs,[tf.shape(inputs)[0]*backprop_length,state_size])
#state_outputs_reshaped = tf.transpose(state_outputs, [1, 0, 2])
#state_outputs_reshaped = tf.gather(state_outputs_reshaped, int(state_outputs_reshaped.get_shape()[0]) - 1)
prediction_orig = tf.matmul(state_outputs_reshaped, weight) + bias
prediction = tf.reshape(prediction_orig,[-1])
#prediction = tf.nn.dropout(prediction,keep_prob) #konsitgt?

#cost_vec = prediction_orig - outputs
#cost_vec = tf.reshape(cost_vec,[-1], name = 'cost')

out_reshaped = tf.reshape(outputs, [-1])
cost = tf.reduce_sum(tf.pow(prediction-out_reshaped, 2)/(2*training_size*backprop_length*output_size))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
minimize = optimizer.minimize(cost)
error_test = tf.reduce_sum(tf.pow(prediction-out_reshaped,2)/(2*test_size*backprop_length*output_size))

#Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_error_store = []
test_error_store = []

#zero_state_train = np.zeros(( training_size,state_size  ))
#zero_state_test = np.zeros(( test_size,state_size ))
#zero_state_batch = np.zeros((batch_size, state_size))
zero_state_batch = np.zeros((num_layers, 2, batch_size, state_size))
zero_state_train = np.zeros((num_layers, 2, training_size, state_size))
zero_state_test = np.zeros((num_layers, 2, test_size, state_size))
num_batches = int(training_size / batch_size)

best_model_error = -1;
for epoch_idx in range(n_epochs):
    #initialize new state each epoch
    #_current_cell_state = zero_state_batch
    #_current_hidden_state = zero_state_batch
    
    for batch_idx in range(num_batches):
        #
        _current_state = zero_state_batch
        start_idx = (batch_idx) * batch_size
        end_idx =  (batch_idx + 1) * batch_size
        x = train_data[start_idx:end_idx, : , :]
        y = train_target[start_idx:end_idx, : ]

        _current_state,_,_state_out,_pred_out,_out = sess.run([current_state,minimize,state_outputs,prediction_orig,out_reshaped], 
                                               feed_dict = {inputs: x, outputs: y, init_state: _current_state,keep_prob: dropout_prob})
        # cell_state: _current_cell_state, hidden_state: _current_hidden_state
        #_current_cell_state, _current_hidden_state = _current_state
        
    _epoch_error = sess.run(cost, feed_dict={inputs: train_data, outputs: train_target, 
                            init_state: zero_state_train , keep_prob: dropout_prob})
    _test_error,_test_pred = sess.run([error_test,prediction], feed_dict={inputs: test_data, outputs: test_target,
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
#past_mean, new_target = predict_past_return
stock_to_plot = 89
n_pred_weeks = 60
    
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
x = range(60)
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

