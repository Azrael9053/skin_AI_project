import numpy as np
from keras.layers import Dense, Conv2D, Flatten, Reshape, BatchNormalization, MaxPooling2D, Input, Activation, Dropout, \
    Conv2DTranspose, UpSampling2D
from keras.models import Model
import tensorflow as tf
from tensorflow.keras import models, layers, Model
import hyperas.distributions as hp
from hyperopt import Trials, STATUS_OK, tpe, fmin
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import csv
from hyperopt.hp import uniformint
from scipy.signal import stft

space = {
    'lr': hp.uniform('lr', .00001, .001),
    # 'epoch_times': hp.choice('epoch_times', [100, 300, 500, 700, 1000]),
    'batch_size': hp.choice('batch_size', [128, 256, 512]),
    'numofnode1': uniformint('numofnode1', 10, 500),
    'numofnode2': uniformint('numofnode2', 10, 499),
    'numofnode3': uniformint('numofnode3', 10, 498),
    'numofnode4': uniformint('numofnode4', 10, 497),
    'numoflayer': hp.choice('numoflayer', [1, 2, 3]),
    'numofconv': hp.choice('numofconv', [1, 2, 3]),
    'dropout': hp.uniform('dropout', 0.1, 0.5),
    'kernel_size': hp.choice('kernel_size', [3, 5, 7]),
    'numoffilters': hp.choice('numoffilters', [1, 2, 3, 4, 5, 6]),
    'activation': hp.choice('activation', ['relu', 'tanh', 'softplus', 'softsign']),
    'numofstrides': hp.choice('numofstrides', [1, 2]),
    'stft_len': hp.choice('stft_len', [8, 16, 32, 64, 128]),
    'overlap': uniformint('overlap', 2, 5),
    'pool_size': hp.choice('pool_size', [2, 3]),
    # 'use_pooling': hp.choice('use_pooling', [0, 1]),
    # 'zoom': hp.choice('zoom', [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
}


def Min_Max_Normalization(array, min=0, max=1):
    array_max = np.max(array.flatten())
    array_min = np.min(array.flatten())
    ratio = (max - min) / (array_max - array_min)
    return (min + ratio * (array - array_min))


def data(space):
    x_data = np.load('x_data.npy')
    y_data = np.load('y_data.npy')
    # x_data = x_data.swapaxes(1, 2)
    stft_len = space['stft_len']
    overlap = space['overlap']
    data_shape = abs(
        stft(x_data[0][0], 320, nperseg=stft_len, window='boxcar', boundary=None, noverlap=int(stft_len / overlap))[-1][
        1:]).shape
    print(stft_len, overlap, data_shape)
    input_shape = (len(x_data), 6, data_shape[0], data_shape[1])
    fftdata = np.zeros(input_shape)
    # x_data = np.reshape(x_data, (3500, 1, 500, 6))
    for i, data in enumerate(x_data):
        for j, ch in enumerate(data):
            fftdata[i][j] = Min_Max_Normalization(abs(
                stft(ch, 320, nperseg=stft_len, window='boxcar', boundary=None, noverlap=int(stft_len / overlap))[-1][
                1:]))
    fftdata = fftdata.swapaxes(1, 3)
    print(fftdata.shape)
    x_data, y_data = shuffle(fftdata, y_data)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
    return x_train, x_test, y_train, y_test, (fftdata.shape[1], fftdata.shape[2], fftdata.shape[3])


def create_model(x_train, y_train, x_test, y_test, space, input_shape):
    numoflayer = space['numoflayer']
    numoffilter = space['numoffilters']
    kernal_size = space['kernel_size']
    batch_size = space['batch_size']
    pool_size = space['pool_size']
    numofnode1 = space['numofnode1']
    numofnode2 = space['numofnode2']
    numofnode3 = space['numofnode3']
    numofnode4 = space['numofnode4']
    dropout = space['dropout']
    lr = space['lr']
    stft_len = space['stft_len']
    overlap = space['overlap']
    epoch_times = 500
    activation = space['activation']
    numofstrides = space['numofstrides']
    # use_pooling = space['use_pooling']
    # nn_node = 256
    # zoom = space['zoom']
    numofconv = space['numofconv']
    input_layer = Input(shape=input_shape)
    if (input_shape[0] % numofstrides != 0) or (input_shape[1] % numofstrides != 0):
        numofstrides = 1
        numofconv = 1

    while (input_shape[0] / (numofstrides ** numofconv) <= 1) or (input_shape[1] / (numofstrides ** numofconv) <= 1):
        numofconv -= 1
    if numofconv == 0:
        numofconv = 1
    print(numofstrides, numofconv)
    # -----------------------------------------------------------------------------------------------------------------------------------------

    # if (numofconv >= 4):
    #     all_layer = Conv2D(filters=numoffilter, kernel_size=kernal_size, padding='same', strides=numofstrides)(
    #         input_layer)
    #     all_layer = Activation(activation)(all_layer)
    #     all_layer = MaxPooling2D(pool_size=(pool_size, pool_size))(all_layer)
    #     all_layer = Activation(activation)(all_layer)
    if (numofconv == 3):
        all_layer = Conv2D(filters=numoffilter, kernel_size=kernal_size, padding='same', strides=numofstrides)(
            input_layer)
        all_layer = Activation(activation)(all_layer)
        all_layer = Conv2D(filters=numoffilter, kernel_size=kernal_size, padding='same', strides=numofstrides)(
            all_layer)
        all_layer = Activation(activation)(all_layer)
        all_layer = Conv2D(filters=numoffilter, kernel_size=kernal_size, padding='same', strides=numofstrides)(
            all_layer)
        decoder = Activation(activation)(all_layer)
        # all_layer = MaxPooling2D(pool_size=(pool_size, pool_size))(all_layer)
        # all_layer = Activation(activation)(all_layer)
    if (numofconv == 2):
        all_layer = Conv2D(filters=numoffilter, kernel_size=kernal_size, padding='same', strides=numofstrides)(
            input_layer)
        all_layer = Activation(activation)(all_layer)
        all_layer = Conv2D(filters=numoffilter, kernel_size=kernal_size, padding='same', strides=numofstrides)(
            all_layer)
        decoder = Activation(activation)(all_layer)
        # all_layer = MaxPooling2D(pool_size=(pool_size, pool_size))(all_layer)
        # all_layer = Activation(activation)(all_layer)
    if (numofconv == 1):
        all_layer = Conv2D(filters=numoffilter, kernel_size=kernal_size, padding='same', strides=numofstrides)(
            input_layer)
        decoder = Activation(activation)(all_layer)
        # all_layer = MaxPooling2D(pool_size=(pool_size, pool_size))(all_layer)
        # decoder = Activation(activation)(all_layer)

    # -----------------------------------------------------------------------------------------------------------------------------------------

    nn_layer = Flatten()(decoder)
    if (numoflayer >= 4):
        nn_layer = Dense(numofnode1)(nn_layer)
        nn_layer = BatchNormalization()(nn_layer)
        nn_layer = Activation(activation)(nn_layer)
        nn_layer = Dropout(dropout)(nn_layer)
    if (numoflayer >= 3):
        nn_layer = Dense(numofnode2)(nn_layer)
        nn_layer = BatchNormalization()(nn_layer)
        nn_layer = Activation(activation)(nn_layer)
        nn_layer = Dropout(dropout)(nn_layer)
    if (numoflayer >= 2):
        nn_layer = Dense(numofnode3)(nn_layer)
        nn_layer = BatchNormalization()(nn_layer)
        nn_layer = Activation(activation)(nn_layer)
        nn_layer = Dropout(dropout)(nn_layer)
    if (numoflayer >= 1):
        nn_layer = Dense(numofnode4)(nn_layer)
        nn_layer = BatchNormalization()(nn_layer)
        nn_layer = Activation(activation)(nn_layer)
        nn_layer = Dropout(dropout)(nn_layer)

    # -----------------------------------------------------------------------------------------------------------------------------------------

    if (numofconv >= 3):
        # decoder = UpSampling2D(size=(pool_size, pool_size))(decoder)
        # decoder = Activation(activation)(decoder)
        decoder = Conv2DTranspose(filters=numoffilter, kernel_size=kernal_size, padding='same', strides=numofstrides)(
            decoder)
        decoder = Activation(activation)(decoder)
    if (numofconv >= 2):
        # decoder = UpSampling2D(size=(pool_size, pool_size))(decoder)
        # decoder = Activation(activation)(decoder)
        decoder = Conv2DTranspose(filters=numoffilter, kernel_size=kernal_size, padding='same', strides=numofstrides)(
            decoder)
        decoder = Activation(activation)(decoder)
    if (numofconv >= 1):
        # decoder = UpSampling2D(size=(pool_size, pool_size))(decoder)
        # decoder = Activation(activation)(decoder)
        decoder = Conv2DTranspose(filters=6, kernel_size=kernal_size, padding='same', strides=numofstrides)(
            decoder)
        decoder = Activation(activation, name='encoder')(decoder)

    # -----------------------------------------------------------------------------------------------------------------------------------------

    output_layer = Dense(7, activation='softmax', name='class')(nn_layer)
    model = Model(input_layer, outputs=[output_layer, decoder])
    model.summary()
    save_path = f'.\\model\\model.h5'
    model_checkpoint_callback = [
        tf.keras.callbacks.ModelCheckpoint(filepath=save_path, monitor='val_class_accuracy', save_best_only=True)]
    adam_op = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=adam_op,
                  loss={'class': 'categorical_crossentropy',
                        'encoder': 'mse'
                        },
                  metrics=['accuracy'])
    history = model.fit(x_train,
                        [y_train, x_train],
                        batch_size=batch_size,
                        validation_data=(x_test, [y_test, x_test]),
                        callbacks=model_checkpoint_callback,
                        verbose=0,
                        epochs=epoch_times)
    model = tf.keras.models.load_model(save_path)
    result = model.evaluate(x_test, [y_test, x_test], verbose=1)
    loss = result[1]
    acc = result[3]
    # print(len(result), result)
    with open('test.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [numofconv, numoffilter, kernal_size, numofstrides, batch_size, activation, numoflayer, numofnode1,
             numofnode2, numofnode3, numofnode4, dropout, lr, epoch_times, pool_size, stft_len, overlap, acc, loss])

    return loss


def NN_Training(param_grid):
    x_train, x_test, y_train, y_test, input_shape = data(param_grid)
    loss = create_model(x_train, y_train, x_test, y_test, param_grid, input_shape)
    return {'loss': loss, 'status': STATUS_OK}


if __name__ == '__main__':
    with open('test.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['numofconv', 'numoffilter', 'kernal_size', 'numofstrides', 'batch_size', 'activation', 'numoflayer',
             'numofnode1', 'numofnode2', 'numofnode3', 'numofnode4', 'dropout', 'lr', 'epoch_times', 'pool_size',
             'stft_len', 'overlap', 'acc', 'loss'])

    trials = Trials()
    best = fmin(NN_Training,
                space,
                algo=tpe.suggest,
                trials=trials,
                max_evals=1000)
    print(best)
