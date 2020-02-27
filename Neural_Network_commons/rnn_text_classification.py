import tensorflow as tf
import tensorflow_datasets as tfds
import sys

print("TensorFlow version: " + tf.__version__)
print("Using %s" % ("GPU" if (tf.test.is_gpu_available()) else "CPU"))
print("Eager execution: ", tf.executing_eagerly())
print("+++++++++++++++++++++++++")

total_words = 10000
max_sequence_len = 80
embed_len = 100
initial_epochs = 10
validation_steps = 20
batchsz = 128

print('loading data...')
# data, info = tfds.load(name="imdb_reviews/subwords8k",
#                        with_info=True,
#                        as_supervised=True, )
# imdb_builder = tfds.builder(name="imdb_reviews/subwords8k")
# imdb_builder.download_and_prepare()
# info = imdb_builder.info
# print("dataset name {} \ndataset size: {}\ndataset features: {}".format(info.name, info.splits, info.features))
# imdb_dataset = imdb_builder.as_dataset()
# test_dataset = imdb_dataset["test"]
# train_dataset = imdb_dataset["train"]
# for train_example in train_dataset.take(1):
#     sentence, label = train_example["text"], train_example["label"]
#     print("sentence: {}".format(sentence.shape))
#     print("label: {}".format(label.shape))
# encoder = info.features['text'].encoder
# # encoder = info.encoder
# # print(test_dataset.info)
# print('Vocabulary size: {}'.format(encoder.vocab_size))
# print("Vocabulary detail: {}".format(encoder.subwords))
# # print("Vocabulary decode example: ", encoder.decode(train_example["text"]))
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=total_words)
print(x_train.shape, len(x_train[0]), y_train.shape)
print(x_test.shape, len(x_test[0]), y_test.shape)
# word_index = tf.keras.datasets.imdb.get_word_index()
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_sequence_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_sequence_len)
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)
print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)


class TextClassification(tf.keras.Model):
    def __init__(self, units, layer_nums):
        super(TextClassification, self).__init__()
        self.units = units
        self.layer_nums = layer_nums
        self.cells = {}

    def build(self, input_shape):
        # self.h0 = [tf.zeros(shape=[batchsz, self.units])]
        # self.h1 = [tf.zeros(shape=[batchsz, self.units])]
        states = getattr(self, 'states', None)
        if states is None:
            states = {}
            for i in range(self.layer_nums):
                init_cell_states = [tf.random.uniform([batchsz, self.units])]
                states[i] = init_cell_states
                cell = tf.keras.layers.SimpleRNNCell(self.units, dropout=0.5)
                self.cells[i] = cell
            self.states = states
        self.embedding = tf.keras.layers.Embedding(total_words, embed_len, input_length=max_sequence_len)
        # self.rnn_cell0 = tf.keras.layers.SimpleRNNCell(self.units, dropout=0.5)
        # self.rnn_cell1 = tf.keras.layers.SimpleRNNCell(self.units, dropout=0.5)
        self.output_layer = tf.keras.layers.Dense(1)  # output is 1 dim
        super(TextClassification, self).build(input_shape)

    def call(self, inputs, training=None):
        x = inputs
        x = self.embedding(x)
        # state0 = self.h0.copy()
        # state1 = self.h1.copy()
        states = self.states.copy()
        for word in tf.unstack(x, axis=1):
            input = word
            for i in range(self.layer_nums):
                out, self.states[i] = self.cells[i](input, self.states[i], training)
                input = out
            # out0, state0 = self.rnn_cell0(word, state0, training)
            # out1, state1 = self.rnn_cell1(out0, state1, training)
        z = self.output_layer(out)
        output = tf.sigmoid(z)

        return output


def main():
    units = 64  # RNN状态向量长度n
    epochs = 10  # 训练 epochs
    layer_nums = 2
    model = TextClassification(units, layer_nums)  # 创建模型
    # 装配
    model.compile(optimizer=tf.optimizers.Adam(0.001),
                  loss="binary_crossentropy", metrics=['accuracy'],
                  experimental_run_tf_function=False)
    # 训练和验证
    model.fit(db_train, epochs=epochs)  # 测试
    model.evaluate(db_test)


if __name__ == '__main__':
    main()
