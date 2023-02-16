import os
import numpy as np
from keras.utils import np_utils
from scipy.signal import stft


def Min_Max_Normalization(array, min=0, max=1):    # min,max=輸出資料大小的範圍
    # array = array.flatten()
    array_max = np.max(array.flatten())
    array_min = np.min(array.flatten())
    ratio = (max - min) / (array_max - array_min)
    return (min + ratio * (array - array_min))#.reshape(5, 15)

flag = 0
dirpath = r'C:\Users\user\PycharmProjects\skin\1226新皮膚'
all_folder = os.listdir(dirpath)
print(all_folder)
all_file = os.listdir((dirpath + f'\\{all_folder[0]}'))
for file in all_file:
    for folder in all_folder:
        if flag == 0:
            alldata = np.load(f'{dirpath}\\{folder}\\{file}')
            flag = 1
        else:
            data = np.load(f'{dirpath}\\{folder}\\{file}')
            alldata = np.vstack((alldata, data))
    flag = 0
    np.save(f'{dirpath}\\{file}', alldata)

# print(all_file)
#
# dirpath1 = dirpath + f'\\{all_folder[0]}'
# dirpath2 = dirpath + f'\\{all_folder[1]}'
# all_file1 = os.listdir(dirpath1)
# all_file2 = os.listdir(dirpath2)
# print(all_file1, all_file2)
# for file1, file2 in zip(all_file1, all_file2):
#     if flag == 0:
#         alldata = np.load(f'{dirpath1}\\{file1}')
#         data = np.load(f'{dirpath2}\\{file2}')[:100]
#         alldata = np.vstack((alldata, data))
#         flag = 1
#     else:
#         data = np.load(f'{dirpath1}\\{file1}')
#         alldata = np.vstack((alldata, data))
#         data = np.load(f'{dirpath2}\\{file2}')[:100]
#         alldata = np.vstack((alldata, data))
#         # print(data.shape)
#
# lens = len(alldata)
# alldata = alldata.swapaxes(1, 2)
# data_shape = abs(stft(alldata[0][0], 320, nperseg=32, window='boxcar', boundary=None, noverlap=8)[-1][1:]).shape
# input_shape = (lens, 6, data_shape[0], data_shape[1])
# fftdata = np.zeros(input_shape)
# lens_1 = int(lens/7)
# lens_2 = int(lens*2/7)
# lens_3 = int(lens*3/7)
# lens_4 = int(lens*4/7)
# lens_5 = int(lens*5/7)
# lens_6 = int(lens*6/7)
# lens_7 = lens
# y_test = np.ones((lens))
# y_test[0:lens_1] = y_test[0:lens_1] * 0
# y_test[lens_1:lens_2] = y_test[lens_1:lens_2] * 1
# y_test[lens_2:lens_3] = y_test[lens_2:lens_3] * 2
# y_test[lens_3:lens_4] = y_test[lens_3:lens_4] * 3
# y_test[lens_4:lens_5] = y_test[lens_4:lens_5] * 4
# y_test[lens_5:lens_6] = y_test[lens_5:lens_6] * 5
# y_test[lens_6:lens_7] = y_test[lens_6:lens_7] * 6
# y_train = np_utils.to_categorical(y_test)
# np.save('x_data2', alldata)
# # np.save('y_test', y_train)
# np.save('y_data2', y_train)
# print(alldata.shape)
# # print(alldata)
# for i, data in enumerate(alldata):
#     for j, ch in enumerate(data):
#         fftdata[i][j] = Min_Max_Normalization(abs(stft(ch, 320, nperseg=32, window='boxcar', boundary=None, noverlap=8)[-1][1:]))
#
# # print(fftdata)
# # plt.plot(fftdata[22][0])
# alldata = fftdata.swapaxes(1, 3)
# # np.save('x_test', alldata)
# # np.save('x_data2', alldata)
# # plt.show()
# print(alldata.shape)