import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

dirpath = r'C:\Users\user\PycharmProjects\skin\data'
data = np.load(f'{dirpath}\\pinch2.npy')
# data = np.load('tl-br.npy').tolist()
# test = np.loadtxt('6.txt', delimiter='\t')
# test1 = np.loadtxt('4.txt', delimiter='\t')
# test2 = np.loadtxt('7.txt', delimiter='\t')
# print(test, test.shape)
data = data.swapaxes(1, 2)
print(data.shape)
for item in data[300:301]:   #144: 147
    for i, pd in enumerate(item):
        plt.subplot((321 + i))
        plt.plot(pd)

# index = np.arange(169, 170, 1)
# data = np.delete(data, index, axis=0)
# index = np.arange(219, 220, 1)
# data = np.delete(data, index, axis=0)
# index = np.arange(215, 216, 1)
# data = np.delete(data, index, axis=0)
# print(data.shape)
# np.save('up_metal', data)

# plt.subplot(221)
# plt.title('閥值取樣')
# plt.xlabel('time(ms)')
# plt.ylabel('ADC_VALUE')
# for i in range(10):
#
#     plt.plot(data[i])
#
# plt.subplot(222)
# plt.title('沒有閥值取樣')
# plt.xlabel('time(ms)')
# plt.ylabel('ADC_VALUE')
# plt.plot(test)
# plt.plot(test1)
# plt.plot(test2)

# print_data = data[0].T
# plt.subplot(321)
# plt.plot(print_data[0])
# plt.subplot(322)
# plt.plot(print_data[1])
# plt.subplot(323)
# plt.plot(print_data[2])
# plt.subplot(324)
# plt.plot(print_data[3])
# plt.subplot(325)
# plt.plot(print_data[4])
# plt.subplot(326)
# plt.plot(print_data[5])

plt.show()