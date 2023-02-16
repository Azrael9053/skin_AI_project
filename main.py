import matplotlib.pyplot as plt
import serial
import numpy as np
import sys
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
from scipy.signal import stft

file_name = 'other'
COM_PORT = 'COM4'
BAUD_RATES = 115200

def main():
    try:
        pre_data = np.load(f'{file_name}.npy')
        all_data = pre_data.tolist()
        print(pre_data, pre_data.shape)
    except FileNotFoundError:
        all_data = []

    arr = []
    ser = serial.Serial(COM_PORT, BAUD_RATES)
    while True:
        # data = ser.readline().decode().replace('\n', '')
        # plt.close()
        # # if data == "start":
        # print(data)
        for i in range(320):
            data = ser.readline()
            ap_list = data.decode().replace('\n', '').split('\t')
            ap_list = list(map(int, ap_list))
            arr.append(ap_list)

        all_data.append(arr)
        arr = []
        if len(all_data) >= 20:
            arr_np = np.array(all_data)
            np.save(file_name, arr_np)
            # print(arr_np, arr_np.shape)
            print('finish')
            sys.exit()
        elif len(all_data) % 10 == 1:
            arr_np = np.array(all_data)
            np.save(file_name, arr_np)
            print(arr_np, arr_np.shape)
            print(len(all_data))
        else:
            print(len(all_data))
        print(ser.readline().decode().replace('\n', ''))
        # print_data = np.asarray(all_data[len(all_data) - 1])
        # print_data = print_data.T
        # print(print_data)
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
        # plt.show()

def Min_Max_Normalization(array, min=0, max=1):    # min,max=輸出資料大小的範圍
    array = array.flatten()
    array_max = np.max(array)
    array_min = np.min(array)
    ratio = (max - min) / (array_max - array_min)
    return (min + ratio * (array - array_min))#.reshape(5, 15)

def print_data():
    data = np.load(f'{file_name}.npy').swapaxes(1, 2)
    print(data.shape)
    for i in range(len(data))[::10]:
        plt.suptitle(file_name)
        plt.subplot(321)
        plt.plot(np.arange(0,1000, 3.125), data[i][0], label=i)
        plt.legend()
        plt.subplot(322)
        plt.plot(np.arange(0,1000, 3.125), data[i][1], label=i)
        plt.legend()
        plt.subplot(323)
        plt.plot(np.arange(0,1000, 3.125), data[i][2], label=i)
        plt.legend()
        plt.subplot(324)
        plt.plot(np.arange(0,1000, 3.125), data[i][3], label=i)
        plt.legend()
        plt.subplot(325)
        plt.plot(np.arange(0,1000, 3.125), data[i][4], label=i)
        plt.legend()
        print(data[i][4])
        plt.subplot(326)
        plt.plot(np.arange(0,1000, 3.125), data[i][5], label=i)
        plt.legend()
        plt.show(block=True)
        # plt.pause(1)
        # plt.savefig(f'{file_name}{i}.png')
        plt.close()

def print_stft():
    alldata = np.load(f'.\\泰璇\\{file_name}.npy').swapaxes(1, 2)
    fftdata = np.zeros((11, 6, 40))
    for i, data in enumerate(alldata):
        for j, ch in enumerate(data):
            # print(abs(stft(ch, 1000, nperseg=100, window='boxcar', boundary=None)[-1][1:]).shape)
            # print(abs(np.fft.rfft(ch)[1:]).shape)
            # print(stft(ch, 80, nperseg=10, window='boxcar', boundary=None)[-1][1:].shape)
            fftdata[i][j] = Min_Max_Normalization(abs(np.fft.rfft(ch)[1:]))
    for i in range(161):
        plt.suptitle(file_name)
        plt.subplot(321)
        plt.plot(fftdata[i][0])
        plt.subplot(322)
        plt.plot(fftdata[i][1])
        plt.subplot(323)
        plt.plot(fftdata[i][2])
        plt.subplot(324)
        plt.plot(fftdata[i][3])
        plt.subplot(325)
        plt.plot(fftdata[i][4])
        plt.subplot(326)
        plt.plot(fftdata[i][5])
    plt.show()

def filter():
    data = np.load('x_data.npy')
    print(data.shape)
    for d in range(7000):
        for ch in range(6):
            for i in range(1,319):
                if abs(data[d][ch][i-1]-data[d][ch][i]) > 3000:
                    data[d][ch][i] = (data[d][ch][i-1]+data[d][ch][i+1])/2

    np.save('x_data_filter.npy', data)

def spprint():
    data = np.load(f'{file_name}.npy').swapaxes(1, 2)
    for i in range(len(data))[401:]:
        plt.suptitle(file_name)
        plt.subplot(311)
        plt.plot(np.arange(0,1000, 3.125), data[i][0], label="第一軸")
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.legend()
        # plt.subplot(322)
        # plt.plot(np.arange(0,1000, 3.125), data[i][1], label=i)
        # plt.xlabel('Time (ms)')
        # plt.ylabel('Amplitude')
        # plt.legend()
        plt.subplot(312)
        plt.plot(np.arange(0,1000, 3.125), data[i][2], label="第二軸")
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.legend()
        # plt.subplot(324)
        # plt.plot(np.arange(0,1000, 3.125), data[i][3], label=i)
        # plt.xlabel('Time (ms)')
        # plt.ylabel('Amplitude')
        # plt.legend()
        plt.subplot(313)
        plt.plot(np.arange(0,1000, 3.125), data[i][4], label="第三軸")
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.legend()
        # plt.subplot(326)
        # plt.plot(np.arange(0,1000, 3.125), data[i][5], label=i)
        # plt.xlabel('Time (ms)')
        # plt.ylabel('Amplitude')
        # plt.legend()
        plt.show(block=False)
        plt.pause(3)
        # plt.savefig(f'{file_name}{i}.png')
        plt.close()

def deleate():
    data = np.load(f'{file_name}.npy')
    print(data.shape)
    data = np.delete(data, [20, 30, 50, 75, 100, 370, 560], axis=0)
    print(data.shape)
    np.save(file_name, data)


if __name__ == '__main__':
    main()
    # print_data()
    # print_stft()
    # filter()
    # spprint()
    # deleate()