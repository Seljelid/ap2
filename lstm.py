import numpy as np
import tensorflow as tf
from datetime import datetime as dt
import matplotlib as plt
import pandas as pd
from sklearn import preprocessing

# Path do data file
DATA_PATH = '../Data/ai_dataset2.csv'

def read_data(file):
    with open(file) as file:
        df = pd.read_csv(file,delimiter=',',low_memory='false')
        n_nan = df.isnull().sum()   #count number of NaN
        df.fillna(value=0,inplace='true')  #fill NaN with zero
        data = df.as_matrix()
        for row in data:
            row[1] = dt.strptime(row[1],'%d%b%Y') #create datetime objects, not used atm
    return data, n_nan

# Preprocess data
data, n_nan = read_data(DATA_PATH)
train_x = data[1000000:1001000,3:24]
train_x = preprocessing.scale(train_x)
train_y = data[1000000:1001000,25]
train_y = preprocessing.scale(train_y)

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 1

batch_size = 10
total_batches = (train_x.shape[0]//batch_size)
n_hidden = 2
n_classes = 1
n_steps = 10
n_input = 21

def create_weight(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def create_bias(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)

def lstm(x,weight,bias,n_steps,n_classes):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden,state_is_tuple = True)
    multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)
    output, state = tf.nn.dynamic_rnn(multi_layer_cell, x, dtype=tf.float32)
    output_flattened = tf.reshape(output, [-1, n_hidden])
    output_logits = tf.add(tf.matmul(output_flattened, weight), bias)
    output_reshaped = tf.reshape(output_logits, [-1, n_steps, n_classes])
    output_last = tf.gather(tf.transpose(output_reshaped, [1, 0, 2]), n_steps - 1)
    return output_logits, output_last

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
y_steps = tf.placeholder("float", [None, n_classes])

weight = create_weight([n_hidden,n_classes])
bias = create_bias([n_classes])
y_last, y_all = lstm(x,weight,bias,n_steps,n_classes)

cost = tf.reduce_mean(tf.squared_difference(y_last, y))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)


with tf.Session() as session:
    tf.global_variables_initializer().run()
    for epoch in range(training_epochs):
        for b in range(total_batches):
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :]
            batch_y = train_y[offset:(offset + batch_size)]
            batch_y_steps = np.tile(batch_y,((train_x.shape[1]),1))
            print(np.shape(batch_x))
            print(np.shape(batch_y))
            _, c = session.run([optimizer, cost],feed_dict={x: batch_x, y : batch_y, y_steps: batch_y_steps})
            if (epoch+1) % display_step == 0:
                print ("Epoch:" '%04d' % (epoch+1), " Training cost=", "{:.9f}".format(c))







