import serial
import numpy as np
from scipy.signal import stft

COM_PORT = 'COM4'
BAUD_RATES = 115200

def Min_Max_Normalization(array, min=0, max=1):    # min,max=輸出資料大小的範圍
    # array = array.flatten()
    array_max = np.max(array.flatten())
    array_min = np.min(array.flatten())
    ratio = (max - min) / (array_max - array_min)
    print(ratio, array_max, array_min)
    return (min + ratio * (array - array_min))#.reshape(5, 15)

def main():
    arr = []
    ser = serial.Serial(COM_PORT, BAUD_RATES)
    while True:
        for i in range(320):
            data = ser.readline()
            ap_list = data.decode().replace('\n', '').split('\t')
            ap_list = list(map(int, ap_list))
            arr.append(ap_list)
        arr = np.array(arr).swapaxes(0,1)
        # print(arr)
        for a in arr:
            # print(a)
            data = Min_Max_Normalization(abs(stft(a, 320, nperseg=128, window='boxcar', boundary=None, noverlap=64)[-1][1:]))
            # data = abs(stft(a, 320, nperseg=32, window='boxcar', boundary=None, noverlap=8)[-1][1:])
            data = data.swapaxes(0,1)
            data_shape = data.shape
            print(data)
        arr = []

if __name__ == '__main__':
    main()