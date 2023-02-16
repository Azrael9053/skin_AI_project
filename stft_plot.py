from scipy import io as spio
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image

def Min_Max_Normalization(array, min=0, max=1):    # min,max=輸出資料大小的範圍
    # array = array.flatten()
    array_max = np.max(array.flatten())
    array_min = np.min(array.flatten())
    ratio = (max - min) / (array_max - array_min)
    return min + ratio * (array - array_min)  #.reshape(5, 15)


file = 'test100ml3'
data = np.load(f'{file}.npy')#.swapaxes(1, 2)
print(data.shape)
# signal.stft
# i = 0
# for n, d in enumerate(data[:2]):
#     for ch in d:
#         print(len(ch))
#         f, t, X = signal.stft(ch, 320, nperseg=128, window='boxcar', boundary=None, noverlap=64)
#         print(len(f))
#         print(t)
#         plt.subplot(321+i)
#         plt.pcolormesh(t, f[1:], Min_Max_Normalization(np.abs(X)[1:]), shading='auto')
#         plt.colorbar()
#         plt.title('STFT Magnitude')
#         plt.ylabel('Frequency [Hz]')
#         plt.xlabel('Time [sec]')
#         plt.tight_layout()
#         i = i + 1
#     i=0
#     plt.show()
#     # plt.savefig(f'{file}{n+2}.png')
#     plt.close()


f, t, X = signal.stft(data, 10000, nperseg=128, window='boxcar', boundary=None, noverlap=64)
plt.pcolormesh(t, f[1:], Min_Max_Normalization(np.abs(X)[1:]), shading='auto')
plt.colorbar()
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.tight_layout()
plt.show()

