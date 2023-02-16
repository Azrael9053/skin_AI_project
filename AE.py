from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

dim = 12000

name = 'tl-br'
data = np.load(f'{name}.npy')
# eval_data = np.load('test.npy')ˋ
train_data, test_data = train_test_split(data, random_state=0, train_size=0.8)
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)
# eval_data = (eval_data - min_val) / (max_val - min_val)

train_data = train_data.reshape((len(train_data), np.prod(train_data.shape[1:])))
test_data = test_data.reshape((len(test_data), np.prod(test_data.shape[1:])))

print(train_data.shape)
print(test_data.shape)

units = [[4096, 2048, 1024, 2048, 4096],
         [4096, 1024, 512, 1024, 4096],
         [2048, 1024, 512, 1024, 2048],
         [2048, 512, 256, 512, 2048],
         [2048, 256, 128, 256, 2048]]

for unit in units:
    model = Sequential()
    model.add(Dense(unit[0], Activation('relu'), input_dim=dim))
    model.add(Dense(unit[1], Activation('relu')))
    model.add(Dense(unit[2], Activation('relu')))
    model.add(Dense(unit[3], Activation('relu')))
    model.add(Dense(unit[4], Activation('relu')))
    model.add(Dense(dim, Activation('sigmoid')))
    model.compile(optimizer='adam',
                  loss='mae')
    model.fit(train_data, train_data,
              epochs=30,
              # batch_size=256,
              validation_data=(test_data, test_data))
    model.summary()
    result = model.evaluate(test_data, test_data)
    print('test loss: ', result)