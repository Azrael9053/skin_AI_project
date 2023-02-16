import os
import keras
from keras import Input, Model
from keras.layers import Dense,Conv2D,MaxPooling2D,UpSampling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

name = 'right-left 鑷子'
data = np.load(f'{name}.npy')
eval_data = np.load('tl-br.npy')
train_data, test_data = train_test_split(data, random_state=0, train_size=0.8)
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)
eval_data = (eval_data - min_val) / (max_val - min_val)

train_data = train_data.reshape((len(train_data), np.prod(train_data.shape[1:])))
test_data = test_data.reshape((len(test_data), np.prod(test_data.shape[1:])))
eval_data = eval_data.reshape((len(eval_data), np.prod(eval_data.shape[1:])))

print(train_data.shape)
print(test_data.shape)
print(eval_data.shape)



inputs = Input(shape=(12000,))
encoder1 = Dense(4096, activation='relu')(inputs)
# encoder2 = Dense(2048, activation='relu')(encoder1)
# encoder3 = Dense(1024, activation='relu')(encoder2)
# decoder1 = Dense(2048, activation='relu')(encoder3)
# decoder2 = Dense(4096, activation='relu')(decoder1)
decoder3 = Dense(12000, activation='sigmoid')(encoder1)
encoded_input = Input(shape=(4096,))
autoencoder = Model(inputs, decoder3)
decoder_layer = autoencoder.layers[-1]
encoder = Model(inputs, encoder1)
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adam', loss='mae')

history = autoencoder.fit(train_data, train_data,
                epochs=100,
                # batch_size=1,
                validation_data=(test_data, test_data))

autoencoder.summary()
encoder.summary()
decoder.summary()

autoencoder.save(f'{name}.h5')

result = autoencoder.evaluate(test_data, test_data)
er_result = autoencoder.evaluate(eval_data, eval_data)
pred = autoencoder.predict(test_data)
print('test loss: ', result)
print('error loss: ', er_result)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()