import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


class CNN(object):
    def __init__(self):
        # self.filter = cnn_config[0]
        # self.kernel = cnn_config[1]
        model = models.Sequential()
        # 第1层卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第2层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第3层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.summary()

        self.model = model


class CNN1d(tf.keras.Model):
    def __init__(self, layer_nums):
        super(CNN1d, self).__init__()
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Conv1D(32, 5, activation='relu', input_shape=(784, 1), strides=1, padding='SAME'),
        #     tf.keras.layers.MaxPool1D(2, strides=1, padding='SAME'),
        #     tf.keras.layers.Conv1D(64, 5, activation='relu', strides=1, padding='SAME'),
        #     tf.keras.layers.MaxPool1D(2, strides=1, padding='SAME'),
        # ])
        self.layers_nums = layer_nums
        self.model = {}
        for i in range(self.layers_nums):
            model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(32, 5, activation='relu', strides=1, padding='SAME'),
                tf.keras.layers.MaxPool1D(2, strides=1, padding='SAME')])
            self.model[i] = model
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.post_model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def call(self, inputs):
        x = self.flatten(inputs)
        x = tf.expand_dims(x, -1)
        print(x.shape)
        out = x
        for i in range(self.layers_nums):
            out = self.model[i](out)
        out = self.post_model(out)
        return out


class DataSource(object):
    def __init__(self):
        # mnist数据集存储的位置，如何不存在将自动下载
        data_path = os.path.abspath(os.path.dirname(__file__)) + '/../RL_image_classification/mnist.npz'
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path=data_path)
        # 6万张训练图片，1万张测试图片
        # train_images = train_images.reshape((60000, 28, 28, 1))
        # test_images = test_images.reshape((10000, 28, 28, 1))
        # 像素值映射到 0 - 1 之间
        train_images, test_images = train_images / 255.0, test_images / 255.0

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels


class Train:
    def __init__(self):
        self.cnn = CNN1d(4)
        self.data = DataSource()

    def train(self):
        # check_path = './ckpt/cp-{epoch:04d}.ckpt'
        # period 每隔5epoch保存一次
        # save_model_cb = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True, verbose=1, period=5)

        self.cnn.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
        self.cnn.fit(self.data.train_images, self.data.train_labels, epochs=5)

        test_loss, test_acc = self.cnn.evaluate(self.data.test_images, self.data.test_labels)
        print("准确率: %.4f " % (test_acc))
        return test_acc


if __name__ == "__main__":
    app = Train()
    app.train()
