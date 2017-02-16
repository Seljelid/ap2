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
#%% Data pre-processing

data_stock = data[data.xref == 'MS:TS604' ]
X = data_stock[header[3:24]]
Y = data_stock[header[-3]]
X = preprocessing.scale(X)
X = pd.DataFrame(X)
Y = np.expand_dims(Y,axis = 1)
print(nps(Y))

def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [2, 3, 4, 5]
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
    return np.array(rnn_df)
    
def sequence_data(X,Y,time_steps):
    start = np.shape(X)[1] % time_steps
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
        #rnn_data=  X[i*time_steps:(i+1)*time_steps, ]
    
    return np.array(rnn_data), np.array(rnn_targets)
    
    
# PARA AND DATASETS
#parameters
batch_size = 10
backprop_length = 10
input_size = np.shape(X)[-1]
output_size = 1
state_size = 32
num_layers = 3
learning_rate = 0.0001
dropout_prob = 0.5
n_epochs = 200
keep_prob = 0.8

# Training and test/validation
#X = rnn_data(X,backprop_length)
#Y = Y[backprop_length:,:]
#Y_original = Y

X, Y = sequence_data(X,Y,backprop_length)

input_size = np.shape(X)[2]
training_size = int(np.shape(X)[0]*0.8)
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
bias = tf.Variable(tf.constant(0.1, output_size))

#the model
state_outputs_reshaped = tf.reshape(state_outputs,[tf.shape(inputs)[0]*backprop_length,state_size])
#state_outputs_reshaped = tf.transpose(state_outputs, [1, 0, 2])
#state_outputs_reshaped = tf.gather(state_outputs_reshaped, int(state_outputs_reshaped.get_shape()[0]) - 1)
prediction = tf.matmul(state_outputs_reshaped, weight) + bias
prediction = tf.reshape(prediction,[-1])
#prediction = tf.nn.dropout(prediction,keep_prob) #konsitgt?

out_reshaped = tf.reshape(outputs, [-1])
cost = tf.reduce_sum(tf.pow(prediction - out_reshaped, 2)/(2*training_size*backprop_length))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
minimize = optimizer.minimize(cost)
error_test = tf.reduce_sum(tf.pow(prediction - out_reshaped,2)/(2*test_size*backprop_length))

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

        _current_state,_,_state_out,_pred_out,_out = sess.run([current_state,minimize,state_outputs,prediction,out_reshaped], 
                                               feed_dict = {inputs: x, outputs: y, init_state: _current_state})
        # cell_state: _current_cell_state, hidden_state: _current_hidden_state
        #_current_cell_state, _current_hidden_state = _current_state
        
    _epoch_error = sess.run(cost, feed_dict={inputs: train_data, outputs: train_target, 
                            init_state: zero_state_train })
    train_error_store.append(_epoch_error)
    _test_error = sess.run(error_test, feed_dict={inputs: test_data, outputs: test_target,
                                              init_state: zero_state_test})
    test_error_store.append(_test_error)
    print('Epoch: ', epoch_idx , ', training:  %.2f  test: %.2f' %(_epoch_error, _test_error))
        

#%%
    
plt.clf()

plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.title('learn rate = %s, batch = %s, time steps = %s'
          %(learning_rate,batch_size,backprop_length) )
plt.plot(train_error_store, linewidth=1, label = "Train")
plt.plot(test_error_store, linewidth=1, label = "Validation")

plt.grid()
plt.legend()
plt.show()




