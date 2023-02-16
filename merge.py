import numpy as np
import os
from keras.utils import np_utils
import matplotlib.pyplot as plt
from scipy.signal import stft

import sys
# np.set_printoptions(threshold=sys.maxsize)

def Min_Max_Normalization(array, min=0, max=1):    # min,max=輸出資料大小的範圍
    # array = array.flatten()
    array_max = np.max(array.flatten())
    array_min = np.min(array.flatten())
    ratio = (max - min) / (array_max - array_min)
    return min + ratio * (array - array_min)  #.reshape(5, 15)

flag = 0
dirpath = r'C:\Users\user\PycharmProjects\skin\1226驗證'
all_file = os.listdir(dirpath)
print(all_file)
other = np.load('other.npy')
# alldata = np.array([[[]]])
# print(alldata.shape)
for file in all_file:
    if flag == 0:
        alldata = np.load(f'{dirpath}\\{file}')
        flag = 1
    else:
        data = np.load(f'{dirpath}\\{file}')
        alldata = np.vstack((alldata, data))
        print(data.shape)

lens = len(alldata)
stft_len = 128
overlap = 2
alldata = alldata.swapaxes(1, 2)
data_shape = abs(stft(alldata[0][0], 320, nperseg=stft_len, window='boxcar', boundary=None, noverlap=int(stft_len/overlap))[-1][1:]).shape
fftdata = np.zeros((lens+20, 6, data_shape[0], data_shape[1]))
lens_1 = int(lens/7)
lens_2 = int(lens*2/7)
lens_3 = int(lens*3/7)
lens_4 = int(lens*4/7)
lens_5 = int(lens*5/7)
lens_6 = int(lens*6/7)
lens_7 = lens
y_test = np.ones((lens))
y_test[0:lens_1] = y_test[0:lens_1] * 0
y_test[lens_1:lens_2] = y_test[lens_1:lens_2] * 1
y_test[lens_2:lens_3] = y_test[lens_2:lens_3] * 2
y_test[lens_3:lens_4] = y_test[lens_3:lens_4] * 3
y_test[lens_4:lens_5] = y_test[lens_4:lens_5] * 4
y_test[lens_5:lens_6] = y_test[lens_5:lens_6] * 5
y_test[lens_6:lens_7] = y_test[lens_6:lens_7] * 6
y_train = np_utils.to_categorical(y_test)
# np.save('x_data', alldata)
alldata = alldata.swapaxes(1, 2)
print(other.shape, alldata.shape)
alldata = np.concatenate((alldata, other), axis=0)
alldata = alldata.swapaxes(1, 2)
np.save('y_test', y_train)
# np.save('y_data', y_train)
print(alldata.shape)
# print(alldata)
for i, data in enumerate(alldata):
    for j, ch in enumerate(data):
        # print(abs(stft(ch, 1000, nperseg=100, window='boxcar', boundary=None)[-1][1:]).shape)
        # print(abs(np.fft.rfft(ch[:100])))
        # print(stft(ch, 80, nperseg=10, window='boxcar', boundary=None)[-1][1:].shape)
        fftdata[i][j] = Min_Max_Normalization(abs(stft(ch, 320, nperseg=stft_len, window='boxcar', boundary=None, noverlap=int(stft_len/overlap))[-1][1:]))
        # print(fftdata[i][j])

print(fftdata)
# plt.plot(fftdata[22][0])
alldata = fftdata.swapaxes(1, 3)
# np.save('x_test', alldata)
# np.save('x_data', alldata)
np.save('x_test1', alldata)
# plt.show()
print(alldata.shape)

# alldata = np.asarray(alldata)
# print(alldata)
# print(alldata.shape)