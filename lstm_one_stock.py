#%% -----------DATA-------------
import numpy as np
import csv
import pandas as pd
from datetime import datetime as dt
import tensorflow as tf
import random
import matplotlib.pyplot as plt

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
#print(data[-15000:-14990,:])
#print(n_nan/np.shape(data[0::,1]))
#unique, counts = np.unique(data[:,2], return_counts='True')
#print('Number of unique stocks', len(unique))
#%%
#data_t = data[0:10000,:] 

rows = np.random.choice(data.index.values, 10000)
df = data.ix[rows]
COLS = np.array(list(df))
CAT_COLS = np.array(['portfolio','xref'])
CONT_COLS = np.array(['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11'
                      ,'F12','F13','F14','F15','F16','F17','F18','F19','F20'
                      ,'F21'])#,'z_score_4w','z_score_8w','z_score_12w'
TARGET_COLS = ['z_score_4w']

def input_fn(df): 
    cont_cols = {k: tf.constant(df[k].values)
      for k in CONT_COLS}
    cat_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                    for k in CAT_COLS}
    feature_cols =  dict(list(cat_cols.items()) + list(cont_cols.items()) )
    target = tf.constant(df[TARGET_COLS].values)
    return feature_cols, target

#%%
tf.reset_default_graph()
X,Y = input_fn(df)

sess = tf.Session()
print(sess.run(X['F1'][10]))

#init = tf.global_variables_initializer

#%% -----------PARA AND DATA-------------
data_stock = data[data.xref == 'MS:TS10']
header = np.array(list(data_stock))
X = data_stock[header[3:24]]
Y = data_stock[header[-3]]

X = np.expand_dims(X,axis = 1)
Y = np.expand_dims(Y,axis = 1)

seq_len = 1
input_size = np.shape(X)[2]
output_size = 1
hidden_size = 16
learning_rate = 0.001

training_size = int(np.shape(X)[0]*0.5)
test_size = np.shape(X)[0]-training_size


train_data = X[:training_size]
train_target = Y[:training_size]
test_data = X[training_size:]
test_target = Y[training_size:]

#%% MODEL   ---------MODEl-------------
tf.reset_default_graph()

x = tf.placeholder(tf.float32,[None,seq_len, input_size])
y_target = tf.placeholder(tf.float32,[None, output_size])

#cell = tf.nn.rnn_cell.LSTMCell(hidden_size,state_is_tuple=True)
num_layers=2
cell = tf.nn.rnn_cell.LSTMCell(hidden_size,state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

val, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([hidden_size, int(y_target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[y_target.get_shape()[1]]))

prediction = tf.matmul(last, weight) + bias 
cost = tf.reduce_sum(tf.pow(prediction - y_target, 2)/(2*training_size))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
minimize = optimizer.minimize(cost)
test_error = tf.reduce_sum(tf.pow(prediction - y_target,2)/(2*test_size))


# SESSION

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 10
no_of_batches = int(training_size / batch_size)
epoch = 100
print('APA')
store_train_c = np.zeros(shape=(epoch,1))
store_test_c = np.zeros(shape = (epoch,1))
for i in range(epoch):
    ptr = 0#?
    for j in range(no_of_batches):
        inp, out = train_data[ptr: ptr+batch_size], train_target[ptr: ptr+batch_size]
        ptr += batch_size
        sess.run(minimize,{x: inp, y_target: out})
    print('Epoch ',str(i))#,': Cost ', c )
    train_c = sess.run(cost,{x: train_data, y_target: train_target})
    test_c = sess.run(test_error,{x: test_data, y_target: test_target})
    store_train_c[i] = train_c
    store_test_c[i] = test_c
    print(train_c)
    print(test_c)
#test_error = sess.run(test_error, {x: test_data, y_target: test_target })
#print(test_error)
sess.close()

#%% ------PLOT-------------

plt.plot(store_train_c, linewidth=1, label = "Train")
plt.plot(store_test_c, linewidth=1, label = "Validation")
plt.grid()
plt.legend()
plt.show()

