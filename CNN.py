import tensorflow as tf
import numpy as np

W = np.array([[
    [0, 0, -1],
    [0, 1, 0],
    [-2, 0, 2]
]], dtype=np.float32)
b = np.array([1], dtype=np.float32)


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(
            filters=1,  # 卷积核个数
            kernel_size=[3, 3],
            kernel_initializer=tf.constant_initializer(W),
            bias_initializer=tf.constant_initializer(b)
            # padding: 在外一圈补零，使前后shape相同
            # strides: 滑动窗口步长
        )

    def call(self, input):
        output = self.conv(input)
        return output


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

one_layer_cnn = CNN()
output = one_layer_cnn(image)
print(output)
