import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from matplotlib import pyplot as plt

def mse_BATCH(y_true, y_pred):
    return np.mean(pow(y_true - y_pred, 2), axis=1)

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

x_data = np.load('x_test.npy')
y_data = np.load('y_test.npy')
x_data1 = np.load('x_test1.npy')

# x_data = x_data.swapaxes(1, 2)
# x_data = x_data[:, np.newaxis, :, :]

index1 = 81
print(x_data.shape)
# x_data = np.reshape(x_data, (84, 1, 1000, 6))
model = tf.keras.models.load_model(f'cnn{index1}.h5')
y_pred = np.zeros(len(y_data))
y_test_cm = np.zeros(len(y_data))
p = model.predict(x_data)
for index, result in enumerate(p[0]):
    y_pred[index] = np.argmax(result)
for index, result in enumerate(y_data):
    y_test_cm[index] = np.argmax(result)
print(y_pred.shape, y_test_cm.shape)
labels = ['下到上(手)', '下到上(鑷子)', '上下捏', '左右捏', '螺絲起子按壓', '上到下(手)', '上到下(鑷子)']
cm = confusion_matrix(y_test_cm, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('confusion_matrix')
plt.xticks(rotation=45)
plt.savefig(f'confusion_matrix_test{index1}.png')
plt.show()
plt.close(disp.figure_)
result = model.evaluate(x_data, [y_data, x_data], verbose=1)
p1 = model.predict(x_data1)[1]
test = [[] for i in range(160)]
data = [[] for i in range(160)]

for i1 , j1 in enumerate(p1):
    for i2, j2 in enumerate(j1):
        for i3, j3, in enumerate(j2):
            for i4, j4, in enumerate(j3):
                # print(test[i1])
                test[i1].append(j4)

for i1 , j1 in enumerate(x_data1):
    for i2, j2 in enumerate(j1):
        for i3, j3, in enumerate(j2):
            for i4, j4, in enumerate(j3):
                # print(test[i1])
                data[i1].append(j4)

test = np.array(test)
data = np.array(data)

loss_all = mse_BATCH(data, test)
labels = ['下到上(手)', '下到上(鑷子)', '上下捏', '左右捏', '螺絲起子按壓', '上到下(手)', '上到下(鑷子)', '其他']
p = [i for i in range(160)]
plt.plot(p[0:20], loss_all[0:20], 'o')
plt.plot(p[21:40], loss_all[21:40], 'o')
plt.plot(p[41:60], loss_all[41:60], 'o')
plt.plot(p[61:80], loss_all[61:80], 'o')
plt.plot(p[81:100], loss_all[81:100], 'o')
plt.plot(p[101:120], loss_all[101:120], 'o')
plt.plot(p[121:140], loss_all[121:140], 'o')
plt.plot(p[141:160], loss_all[141:160], 'o')
plt.legend(labels)
plt.show()

print(result)