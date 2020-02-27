import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import numpy as np

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
word_index = tf.keras.datasets.imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0  # 填充标志
word_index["<START>"] = 1  # 起始标志
word_index["<UNK>"] = 2  # 未知单词的标志
word_index["<UNUSED>"] = 3
print("Total %s word in vocabulary." % (len(word_index)))

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_sequence_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_sequence_len)
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)
print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test shape:', x_test.shape)


class GRUClassification(tf.keras.Model):
    def __init__(self, units, embedding_matrix):
        super(GRUClassification, self).__init__()
        self.embedding = tf.keras.layers.Embedding(total_words, embed_len, input_length=max_sequence_len,
                                                   trainable=False)
        self.embedding.build(input_shape=(None, max_sequence_len))
        self.embedding.set_weights([embedding_matrix])
        self.GRU = tf.keras.Sequential([
            tf.keras.layers.GRU(units, dropout=0.5, return_sequences=True),
            tf.keras.layers.GRU(units, dropout=0.5)
        ])
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, input):
        x = input
        x = self.embedding(x)
        z = self.GRU(x)
        out = self.output_layer(z)
        prob = tf.sigmoid(out)
        return prob


class LSTMClassification(tf.keras.Model):
    def __init__(self, units, embedding_matrix):
        super(LSTMClassification, self).__init__()
        self.embedding = tf.keras.layers.Embedding(total_words, embed_len, input_length=max_sequence_len,
                                                   trainable=False)
        self.embedding.build(input_shape=(None, max_sequence_len))
        self.embedding.set_weights([embedding_matrix])
        self.LSTM = tf.keras.Sequential([
            tf.keras.layers.LSTM(units, dropout=0.5, return_sequences=True),
            tf.keras.layers.LSTM(units, dropout=0.5)
        ])
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, input):
        x = input
        x = self.embedding(x)
        z = self.LSTM(x)
        out = self.output_layer(z)
        prob = tf.sigmoid(out)
        return prob

        return prob


def index_word():
    print("Indexing word vectors...")
    embeddings_index = {}
    file_dir = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/glove.6B.100d.txt"
    with open(file_dir, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    print("Found %s word vectors." % (len(embeddings_index)))
    num_words = min(total_words, len(word_index))
    embedding_matrix = np.zeros((num_words, embed_len))
    i = 0;
    for word, i in word_index.items():
        if i >= num_words: continue  # 过滤掉其他词汇
        embedding_vector = embeddings_index.get(word)  # 从 GloVe 查询词向量
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector  # 写入对应位置
    print(embedding_matrix.shape)
    return embedding_matrix


def LSTM_training(model_name):
    units = 64  # RNN状态向量长度n
    epochs = 50  # 训练 epochs
    embedding_matrix = index_word()
    model = LSTMClassification(units, embedding_matrix)  # 创建模型
    # 装配
    model.compile(optimizer=tf.optimizers.Adam(0.001),
                  loss="binary_crossentropy",
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)
    # 训练和验证
    model.fit(db_train, epochs=epochs)  # 测试
    model.evaluate(db_test)
    # model.save(model_name)
    print("+++++++++Finished Training {}+++++++++".format(model_name))


def GRU_training(model_name):
    units = 64
    epochs = 50
    embedding_matrix = index_word()
    model = GRUClassification(units, embedding_matrix)
    model.compile(optimizer=tf.optimizers.Adam(0.001),
                  loss=tf.losses.BinaryCrossentropy(),
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)
    model.fit(db_train, epochs=epochs)
    model.evaluate(db_test)
    # model.save(model_name)
    print("+++++++++Finished Training {}+++++++++".format(model_name))


if __name__ == '__main__':
    GRU_name = "gru_model.h5"
    LSTM_name = "lstm_model.h5"
    # GRU_training(GRU_name)
    LSTM_training(LSTM_name)
