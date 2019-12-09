from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
tfconfig = ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
session = InteractiveSession(config=tfconfig)
import numpy as np

print(tf.__version__)
W = np.array([[
    [0, 0, -1],
    [0, 1, 0],
    [-2, 0, 2]
]], dtype=np.float32)
b = np.array([1], dtype=np.float32)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        filters=1,              # 卷积层神经元（卷积核）数目
        kernel_size=[3, 3],     # 感受野大小
        kernel_initializer=tf.constant_initializer(W),
        bias_initializer=tf.constant_initializer(b)
    )]
)

image = np.array([[
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 2, 1, 0],
    [0, 0, 2, 2, 0, 1, 0],
    [0, 1, 1, 0, 2, 1, 0],
    [0, 0, 2, 1, 1, 0, 0],
    [0, 2, 1, 1, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]], dtype=np.float32)
image = np.expand_dims(image, axis=-1)  # 因为是灰度图， 所以是单通道，因此在最后加一维

output = model(image)
# output = one_layer_cnn(image)
print(np.squeeze(output))
