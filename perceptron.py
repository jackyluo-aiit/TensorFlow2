import time

import tensorflow as tf
import numpy as np


class MNISTLoader(object):
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data("mnist.npz")
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)  # [60000]
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]


class PerceptronModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()  # 折叠矩阵变成一维数组
        self.layer1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.layer2 = tf.keras.layers.Dense(units=10)

    def call(self, input):
        x = self.flatten(input)
        x = self.layer1(x)
        output = self.layer2(x)
        output = tf.nn.softmax(output)
        return output


num_epoch = 100
batch_size = 50

perceptron_model = PerceptronModel()
data_loader = MNISTLoader()
optimizer = tf.optimizers.Adam(learning_rate=0.001)


num_batch = int(data_loader.num_train_data // batch_size * num_epoch)
# print(num_batch)
start = time.time()
for batch_index in range(num_batch):
    train_X, train_X_label = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        pred = perceptron_model(train_X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(train_X_label, pred)
        loss = tf.reduce_mean(loss, axis=0)
        print('batch:', batch_index, 'loss:', loss.numpy())
    grads = tape.gradient(loss, perceptron_model.variables)
    optimizer.apply_gradients(zip(grads, perceptron_model.variables))
end1 = time.time()

# evaluation
num_batch = int(data_loader.num_test_data // batch_size)
sparse_categorical_crossentropy_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
for batch_index in range(num_batch):
    start_index, end_index = batch_index * batch_size, (batch_index+1) * batch_size
    pred = perceptron_model.predict(data_loader.test_data[start_index: end_index, :], batch_size)
    sparse_categorical_crossentropy_accuracy.update_state(data_loader.test_label[start_index: end_index], pred)
end2 = time.time()
print('test accuracy:', sparse_categorical_crossentropy_accuracy.result())
print('training time:', end1-start, 'testing time:', end2-end1)
