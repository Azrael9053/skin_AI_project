import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Reshape, BatchNormalization, MaxPooling1D, Input, Activation, Dropout, Conv2D, MaxPooling2D
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
    x_data = np.load('x_data_filter.npy')
    y_data = np.load('y_data.npy')
    x_data = x_data.swapaxes(1, 2)
    x_data = x_data[:, np.newaxis, :, :]
    # print(x_data.shape)
    x_data, y_data = shuffle(x_data, y_data)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
    print(x_data.shape)
    return x_train, x_test, y_train, y_test, (1, x_data.shape[2], x_data.shape[3])

def create_model(x_train, y_train, x_test, y_test, index1, input_shape):
    numoflayer = 1
    numoffilter = 4
    kernal_size = 5
    batch_size = 256
    # pool_size = space['pool_size']
    numofnode1 = 802
    numofnode2 = 371
    numofnode3 = 258
    numofnode4 = 186
    dropout = 0.346989037703441
    lr = 0.0000883421210939563
    epoch_times = 500
    activation = 'relu'
    numofstrides = 1
    input_layer = Input(shape=input_shape)
    all_layer = Conv2D(filters=numoffilter, kernel_size=kernal_size, padding='same', strides=numofstrides)(input_layer)
    all_layer = Activation(activation)(all_layer)
    # all_layer = MaxPooling1D(pool_size=pool_size, strides=None, padding='same', data_format='channels_last')(all_layer)
    all_layer = Flatten()(all_layer)
    if (numoflayer >= 4):
        all_layer = Dense(numofnode1)(all_layer)
        all_layer = BatchNormalization()(all_layer)
        all_layer = Activation(activation)(all_layer)
        all_layer = Dropout(dropout)(all_layer)
    if (numoflayer >= 3):
        all_layer = Dense(numofnode2)(all_layer)
        all_layer = BatchNormalization()(all_layer)
        all_layer = Activation(activation)(all_layer)
        all_layer = Dropout(dropout)(all_layer)
    if (numoflayer >= 2):
        all_layer = Dense(numofnode3)(all_layer)
        all_layer = BatchNormalization()(all_layer)
        all_layer = Activation(activation)(all_layer)
        all_layer = Dropout(dropout)(all_layer)
    if (numoflayer >= 1):
        all_layer = Dense(numofnode4)(all_layer)
        all_layer = BatchNormalization()(all_layer)
        all_layer = Activation(activation)(all_layer)
        all_layer = Dropout(dropout)(all_layer)
    all_layer = Dense(7)(all_layer)
    output_layer = Activation('softmax')(all_layer)
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
                        validation_data=(x_test, y_test),
                        batch_size=batch_size,
                        callbacks=model_checkpoint_callback,
                        epochs=epoch_times)
    model = tf.keras.models.load_model(save_path)
    result = model.evaluate(x_test, y_test, verbose=1)
    y_pred = np.zeros(len(y_test))
    y_test_cm = np.zeros(len(y_test))
    for index, result in enumerate(model.predict(x_test)):
        y_pred[index] = np.argmax(result)
    for index, result in enumerate(y_test):
        y_test_cm[index] = np.argmax(result)
    labels = ['下到上(手)', '下到上(鑷子)', '上下捏', '左右捏', '螺絲起子按壓', '上到下(手)', '上到下(鑷子)']
    cm = confusion_matrix(y_test_cm, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('confusion_matrix')
    plt.xticks(rotation=45)
    plt.show()
    # plt.savefig(f'confusion_matrix_{index1}.png')
    # plt.close(disp.figure_)

    model.save(f'cnn{index1}.h5')

    return history, result

if __name__ == '__main__':
    x_train, x_test, y_train, y_test, input_shape = data()
    index1 = 3
    history, result = create_model(x_train, y_train, x_test, y_test, index1, input_shape)
    fig = plt.figure()
    val_acc, = plt.plot(history.history['val_accuracy'])
    acc, = plt.plot(history.history['accuracy'])
    plt.title(f'{index1}')
    plt.xlabel('epoch')
    plt.legend([acc, val_acc], ['acc', 'test_acc'], loc='lower right')
    plt.savefig(f'cv_acc_{index1}.png')
    plt.close(fig)