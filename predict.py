import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

x_data = np.load('x_test.npy')
y_data = np.load('y_test.npy')

# x_data = x_data.swapaxes(1, 2)
# x_data = x_data[:, np.newaxis, :, :]

index1 = 60
print(x_data.shape)
# x_data = np.reshape(x_data, (84, 1, 1000, 6))
model = tf.keras.models.load_model(f'cnn{index1}.h5')
y_pred = np.zeros(len(y_data))
y_test_cm = np.zeros(len(y_data))
print(model.predict(x_data))
for index, result in enumerate(model.predict(x_data)):
    y_pred[index] = np.argmax(result)
for index, result in enumerate(y_data):
    y_test_cm[index] = np.argmax(result)
labels = ['下到上(手)', '下到上(鑷子)', '上下捏', '左右捏', '螺絲起子按壓', '上到下(手)', '上到下(鑷子)']
cm = confusion_matrix(y_test_cm, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('confusion_matrix')
plt.xticks(rotation=45)
plt.savefig(f'confusion_matrix_test{index1}.png')
plt.show()
plt.close(disp.figure_)
result = model.evaluate(x_data, y_data, verbose=1)
print(result)