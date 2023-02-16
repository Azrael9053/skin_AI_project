import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Reshape, BatchNormalization, MaxPooling1D, Input, Activation, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.models import load_model
from keras.utils import np_utils
import keras
import tensorflow as tf
from tensorflow.keras import models, layers, Model
import hyperas.distributions as hp
from hyperopt import Trials, STATUS_OK, tpe, fmin
from hyperas import optim
from hyperas.distributions import choice, uniform
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import csv
from hyperopt.hp import uniformint

space = {
    'lr': hp.uniform('lr', .00001, .001),
    # 'epoch_times': hp.choice('epoch_times', [100, 300, 500, 700, 1000]),
    'batch_size': hp.choice('batch_size', [32,64,128,256]),
    'numofnode1': uniformint('numofnode1', 800, 1000),
    'numofnode2': uniformint('numofnode2', 500, 700),
    'numofnode3': uniformint('numofnode3', 200, 400),
    'numofnode4': uniformint('numofnode4', 50, 150),
    'numoflayer': hp.choice('numoflayer', [1, 2, 3, 4]),
    'dropout': hp.uniform('dropout', 0.1, 0.5),
    'kernel_size': hp.choice('kernel_size', [3, 5, 7]),
    'numoffilters': hp.choice('numoffilters', [3, 4, 5]),
    'activation' : hp.choice('activation', ['relu', 'tanh', 'softplus', 'softsign']),
    'numofstrides' : hp.choice('numofstrides', [1, 2, 3])
    # 'pool_size': hp.choice('pool_size', [2, 3, 4])
}

def data():
    x_data = np.load('x_data.npy')
    y_data = np.load('y_data.npy')
    # x_data = np.reshape(x_data, (3500, 1, 500, 6))
    print(x_data.shape)
    x_data, y_data = shuffle(x_data, y_data)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test

def create_model(x_train, y_train, x_test, y_test, space):
    numoflayer = space['numoflayer']
    numoffilter = space['numoffilters']
    kernal_size = space['kernel_size']
    batch_size = space['batch_size']
    # pool_size = space['pool_size']
    numofnode1 = space['numofnode1']
    numofnode2 = space['numofnode2']
    numofnode3 = space['numofnode3']
    numofnode4 = space['numofnode4']
    dropout = space['dropout']
    lr = space['lr']
    epoch_times = 500
    activation = space['activation']
    numofstrides = space['numofstrides']
    input_layer = Input(shape=(15, 5, 6))
    all_layer = Conv2D(filters=numoffilter, kernel_size=kernal_size, padding='same', strides=numofstrides)(input_layer)
    all_layer = Activation(activation)(all_layer)
    # all_layer = MaxPooling1D(pool_size=pool_size, strides=None, padding='same', data_format='channels_last')(all_layer)
    all_layer = Flatten()(all_layer)
    if(numoflayer <= 1):
        all_layer = Dense(numofnode1)(all_layer)
        all_layer = BatchNormalization()(all_layer)
        all_layer = Activation(activation)(all_layer)
        all_layer = Dropout(dropout)(all_layer)
    if (numoflayer <= 2):
        all_layer = Dense(numofnode2)(all_layer)
        all_layer = BatchNormalization()(all_layer)
        all_layer = Activation(activation)(all_layer)
        all_layer = Dropout(dropout)(all_layer)
    if (numoflayer <= 3):
        all_layer = Dense(numofnode3)(all_layer)
        all_layer = BatchNormalization()(all_layer)
        all_layer = Activation(activation)(all_layer)
        all_layer = Dropout(dropout)(all_layer)
    if (numoflayer <= 4):
        all_layer = Dense(numofnode4)(all_layer)
        all_layer = BatchNormalization()(all_layer)
        all_layer = Activation(activation)(all_layer)
        all_layer = Dropout(dropout)(all_layer)
    output_layer = Dense(7, activation='softmax')(all_layer)
    model = Model(input_layer, output_layer)
    model.summary()
    save_path = f'.\\model\\model.h5'
    model_checkpoint_callback = [
        tf.keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_accuracy', save_best_only=True)]
    adam_op = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=adam_op,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test),
                        callbacks=model_checkpoint_callback,
                        epochs=epoch_times)
    model = tf.keras.models.load_model(save_path)
    acc = model.evaluate(x_test, y_test, verbose=1)[1]
    with open('output.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([numoffilter, kernal_size, numofstrides, batch_size, activation, numoflayer, numofnode1, numofnode2, numofnode3, numofnode4, dropout, lr, epoch_times, acc])

    return acc

def NN_Training(param_grid):
    acc = create_model(x_train, y_train, x_test, y_test, param_grid)
    # y_predict = cnn.predict(x_test)
    # print(y_predict)
    # print(y_test)
    # acc = cnn.evaluate(x_test, y_test, verbose=1)[1]
    return {'loss': -acc, 'status': STATUS_OK}

if __name__ == '__main__':

    with open('output.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['numoffilter', 'kernal_size', 'numofstrides', 'batch_size', 'activation', 'numoflayer', 'numofnode1', 'numofnode2', 'numofnode3', 'numofnode4', 'dropout', 'lr', 'epoch_times', 'acc'])
    x_train, x_test, y_train, y_test = data()
    trials = Trials()
    best = fmin(NN_Training,
                space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=500)
    print(best)
