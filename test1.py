import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Reshape, BatchNormalization, MaxPooling1D, Input, Activation, Dropout, Conv2D, MaxPooling2D, Conv2DTranspose
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
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.signal import stft


plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def Min_Max_Normalization(array, min=0, max=1):    # min,max=輸出資料大小的範圍
    # array = array.flatten()
    array_max = np.max(array.flatten())
    array_min = np.min(array.flatten())
    ratio = (max - min) / (array_max - array_min)
    return (min + ratio * (array - array_min))#.reshape(5, 15)

def data():
    x_data = np.load('x_data.npy')
    y_data = np.load('y_data.npy')
    # x_data = x_data.swapaxes(1, 2)
    stft_len = 128
    overlap = 2
    data_shape = abs(
        stft(x_data[0][0], 80, nperseg=stft_len, window='boxcar', boundary=None, noverlap=int(stft_len / overlap))[-1][
        1:]).shape
    print(stft_len, overlap, data_shape)
    input_shape = (len(x_data), 6, data_shape[0], data_shape[1])
    fftdata = np.zeros(input_shape)
    # x_data = np.reshape(x_data, (3500, 1, 500, 6))
    for i, data in enumerate(x_data):
        for j, ch in enumerate(data):
            fftdata[i][j] = Min_Max_Normalization(abs(
                stft(ch, 80, nperseg=stft_len, window='boxcar', boundary=None, noverlap=int(stft_len / overlap))[-1][
                1:]))
    fftdata = fftdata.swapaxes(1, 3)
    print(fftdata.shape)
    x_data, y_data = shuffle(fftdata, y_data)
    kf = KFold(n_splits=5)
    # for train_index, test_index in kf.split(x_data):
    #     print(train_index, test_index)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=87)
    return x_train, x_test, y_train, y_test, (fftdata.shape[1], fftdata.shape[2], fftdata.shape[3])

def create_model(x_train, y_train, x_test, y_test, index1, input_shape):
    numoflayer = 1
    numoffilter = 2
    kernal_size = 5
    batch_size = 256
    # pool_size = space['pool_size']
    numofnode1 = 200
    numofnode2 = 190
    numofnode3 = 180
    numofnode4 = 421
    dropout = 0.110740773889798
    lr = 0.000724600179916754
    stft_len = 128
    overlap = 2
    epoch_times = 500
    activation = 'softsign'
    numofstrides = 2
    # use_pooling = space['use_pooling']
    # nn_node = 256
    # zoom = space['zoom']
    numofconv = 1
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
                  loss={'class':'categorical_crossentropy',
                        'encoder':'mse'
                        },
                  metrics=['accuracy'])
    history = model.fit(x_train,
                        [y_train, x_train],
                        batch_size=batch_size,
                        validation_data=(x_test, [y_test, x_test]),
                        callbacks=model_checkpoint_callback,
                        verbose=1,
                        epochs=epoch_times)
    model = tf.keras.models.load_model(save_path)
    result = model.evaluate(x_test, [y_test, x_test], verbose=1)
    model.save(f'cnn{index1}.h5')
    y_pred = np.zeros(len(y_test))
    y_test_cm = np.zeros(len(y_test))
    for index, result in enumerate(model.predict(x_test)[0]):
        y_pred[index] = np.argmax(result)
    for index, result in enumerate(y_test):
        y_test_cm[index] = np.argmax(result)
    print(y_pred, y_test_cm)
    labels = ['下到上(手)', '下到上(鑷子)', '上下捏', '左右捏', '螺絲起子按壓', '上到下(手)', '上到下(鑷子)']
    cm = confusion_matrix(y_test_cm, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('confusion_matrix')
    plt.xticks(rotation=45)
    plt.show()
    # plt.savefig(f'confusion_matrix_{index1}.png')
    # plt.close(disp.figure_)

    # model.save(f'cnn{index1}.h5')

    return history, result

if __name__ == '__main__':
    x_train, x_test, y_train, y_test, input_shape = data()
    index1 = 81
    history, result = create_model(x_train, y_train, x_test, y_test, index1, input_shape)
    fig = plt.figure()
    val_acc, = plt.plot(history.history['val_accuracy'])
    acc, = plt.plot(history.history['accuracy'])
    plt.title(f'{index1}')
    plt.xlabel('epoch')
    plt.legend([acc, val_acc], ['acc', 'test_acc'], loc='lower right')
    plt.savefig(f'cv_acc_{index1}.png')
    plt.close(fig)
    # y = [0.926, 0.918, 0.92, 0.926, 0.926]
    # x = [1, 2, 3, 4, 5]
    # plt.title('CV_acc')
    # plt.ylabel('acc')
    # plt.ylim(0, 1)
    # x_major_locator = MultipleLocator(1)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)
    # plt.plot(x, y, marker='o')
    # for a, b in zip(x, y):
    #     plt.text(a, b, b, ha='center', va='bottom', fontsize=12)
    # plt.show()
