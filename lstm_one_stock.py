#%% -----------DATA-------------
import numpy as np
import csv
import pandas as pd
from datetime import datetime as dt
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from sklearn import preprocessing

DATA_PATH = '../Data/data_good.csv'

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

factors = header[3:]

data_missing = data.copy(deep = True)

for factor in factors:
    data_factor = data[factor]
    data_factor = data_factor.T
    data_factor = data_factor.fillna(data_factor.mean())
    data_factor = data_factor.T
    data[factor] = data_factor
             
data.fillna(value=0, inplace = True)

data.to_csv('../Data/data_pro.csv', index = False )

#%% -----------PARA AND DATA-------------
data_stock = data[data.xref == 'MS:TS69451' ] #'MS:TS10' #'MS:TS69451''MS:TS3019'
header = np.array(list(data_stock))
X = data_stock[header[3:24]]
Y = data_stock[header[-3]]


#X = preprocessing.scale(X)
#X = pd.DataFrame(X)

#X = np.expand_dims(X,axis = 1)
Y = np.expand_dims(Y,axis = 1)
Y_original = Y

seq_len = 10
output_size = 1
dropout_para = 0.5
hidden_size = 32
learning_rate = 0.0001

y_idiot = Y[(seq_len-4):-4,:]
Y = Y[seq_len:,:]


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

X = rnn_data(X,seq_len)
input_size = np.shape(X)[2]

training_size = int(np.shape(X)[0]*0.8)
test_size = np.shape(X)[0]-training_size


train_data = X[:training_size]
train_target = Y[:training_size]
test_data = X[training_size:]
test_target = Y[training_size:]

#%% MODEL   ---------MODEl-------------
tf.reset_default_graph()

x = tf.placeholder(tf.float32,[None,seq_len, input_size])
y_target = tf.placeholder(tf.float32,[None, output_size])
keep_prob = tf.placeholder(tf.float32)

#cell = tf.nn.rnn_cell.LSTMCell(hidden_size,state_is_tuple=True)
num_layers=2
cell = tf.nn.rnn_cell.LSTMCell(hidden_size,state_is_tuple=True)
#cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

val, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([hidden_size, int(y_target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[y_target.get_shape()[1]]))

prediction = tf.matmul(last, weight) + bias
prediction = tf.nn.dropout(prediction,keep_prob)
cost = tf.reduce_sum(tf.pow(prediction - y_target, 2)/(2*training_size))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
minimize = optimizer.minimize(cost)
test_error = tf.reduce_sum(tf.pow(prediction - y_target,2)/(2*test_size))
idiot_error = tf.reduce_sum(tf.pow(np.mean(Y)-y_target,2)/(2*np.shape(Y)[0]))

# SESSION

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 10
no_of_batches = int(training_size / batch_size)
epoch = 100
store_train_c = np.zeros(shape=(epoch,1))
store_test_c = np.zeros(shape = (epoch,1))
#store_idiot_c = np.zeros(shape = (epoch,1))
for i in range(epoch):
    ptr = 0#?
    for j in range(no_of_batches):
        inp, out = train_data[ptr: ptr+batch_size], train_target[ptr: ptr+batch_size]
        ptr += batch_size
        sess.run(minimize,feed_dict={x: inp, y_target: out, keep_prob: dropout_para})
    print('Epoch ',str(i))#,': Cost ', c )
    train_c = sess.run(cost,feed_dict={x: train_data, y_target: train_target, keep_prob:1})
    test_c = sess.run(test_error,feed_dict={x: test_data, y_target: test_target, keep_prob:1})
    idiot_c = sess.run(idiot_error,feed_dict={x: X, y_target: Y })
    store_train_c[i] = train_c
    store_test_c[i] = test_c
    #store_idiot_c[i] = idiot_c
    print(train_c)
    print(test_c)
    #print(idiot_c)
#test_error = sess.run(test_error, {x: test_data, y_target: test_target })
#print(test_error)
sess.close()

#%%
#%% ------PLOT-------------
plt.clf()
idiot1 = np.sum(np.power(np.mean(Y)-Y,2)/(2*np.shape(Y)[0]) )
idiot2 = np.sum(np.power(y_idiot - Y, 2)/(2*np.shape(Y)[0]) )

def Past_mean(y):
    y_mean = np.cumsum(y)/np.linspace(1,len(y),len(y))
    return(np.expand_dims(y_mean,axis=1))

y_mean_past_4w = Past_mean(Y_original)[seq_len-4:-4]
idiot3 = np.sum(np.power(y_mean_past_4w - Y, 2)/(2*np.shape(Y)[0]) )  
   


plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.title('learn rate = %s, batch = %s, time steps = %s'
          %(learning_rate,batch_size,seq_len) )
plt.plot(store_train_c, linewidth=1, label = "Train")
plt.plot(store_test_c, linewidth=1, label = "Validation")
#plt.plot(store_idiot_c, linewidth=1, label = "Idiot")
#plt.axhline(y = idiot1, xmin = 0, xmax = epoch)
#plt.axline(y = idiot2, xmin = 0, xmax = epoch)
plt.axhline(y = idiot3, xmin = 0, xmax = epoch,color = 'r',label = 'Past mean')
plt.grid()
plt.legend()
plt.show()

    
