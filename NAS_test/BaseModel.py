#!/usr/bin/env python

import tensorflow as tf
from NAS_test.preprocess import preprocess
from NAS_test.load_data import load_data, load_data2
import os
import numpy as np
from sklearn.model_selection import train_test_split


# embed_depth = 100
# batch_size = 1024
# epoch_size = 25
# state_len = 256
# learning_rate = 0.005
# max_len = 40
# dense_units = 1024


def get_model(actions, x, y, input_vocab, target_vocab):
    embed_depth, state_len, dense_units = actions
    model = EnDeRNNModel(x.shape, y.shape[1], len(input_vocab.word_index) + 1,
                         len(target_vocab.word_index) + 1, embed_depth, state_len, dense_units)
    return model


class EnDeRNNModel(tf.keras.Model):
    def __init__(self, x_shape, output_seq_len, input_vocab_size, target_vocab_size, embed_depth, state_len,
                 dense_units):
        super(EnDeRNNModel, self).__init__()
        self.x_shape = x_shape
        self.output_seq_len = output_seq_len
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.embed_depth = embed_depth
        self.state_len = state_len
        self.dense_units = dense_units
        self.en_de_model = tf.keras.Sequential(
            [tf.keras.layers.Embedding(input_vocab_size, embed_depth, input_length=self.x_shape[1],
                                       input_shape=self.x_shape[1:]),
             tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.state_len)),
             tf.keras.layers.RepeatVector(self.output_seq_len),
             tf.keras.layers.Bidirectional(tf.keras.layers.GRU(self.state_len, return_sequences=True)),
             tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.dense_units, activation='relu')),
             tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.target_vocab_size, activation='softmax'))]
        )

    def call(self, inputs, **kwargs):
        out = self.en_de_model(inputs)
        return out


def ende_training(x, y, x_valid, y_valid, x_test, y_test,
                  input_vocab, target_vocab,
                  embed_depth, state_len, dense_units, model_path,
                  epoch_size, batch_size, learning_rate):
    model = EnDeRNNModel(x.shape, y.shape[1], len(input_vocab.word_index) + 1,
                         len(target_vocab.word_index) + 1, embed_depth, state_len, dense_units)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=['accuracy'])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='callbacks/best_base_model',
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1)
    ]
    model.fit(x, y, epochs=epoch_size, batch_size=batch_size, validation_data=(x_valid, y_valid), callbacks=callbacks)
    model.summary()
    model.load_weights('callbacks/best_base_model')
    model.evaluate(x_test, y_test)
    print("Finished training")
    model.save_weights(model_path, save_format='tf')


def prediction(x, y, x_tk, y_tk, model_path, embed_depth, state_len, dense_units):
    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    x_id_to_word = {value: key for key, value in x_tk.word_index.items()}
    y_id_to_word[0] = '<PAD>'
    x_id_to_word[0] = '<PAD>'
    model = EnDeRNNModel(x.shape, y.shape[1], len(x_tk.word_index) + 1,
                         len(y_tk.word_index) + 1, embed_depth, state_len, dense_units)
    model.load_weights(model_path)
    origin_sentence1 = 'he saw a old yellow truck'
    origin_sentence2 = ' '.join([x_id_to_word[np.max(i)] for i in x[0]])
    sentence = [x_tk.word_index[word] for word in origin_sentence1.split()]
    sentence = tf.keras.preprocessing.sequence.pad_sequences([sentence], maxlen=x.shape[-1], padding='post')
    sentences = np.array([sentence[0], x[0]])
    predictions = model.predict(sentences, len(sentences))

    print('Sample 1:', origin_sentence1)
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
    print('Il a vu un vieux camion jaune')
    print('Sample 2:', origin_sentence2)
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
    print(' '.join([y_id_to_word[np.max(x)] for x in y[0]]))


if __name__ == '__main__':
    embed_depth = 100
    batch_size = 1024
    epoch_size = 25
    state_len = 256
    learning_rate = 0.005
    dense_units = 1024
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    model_path = os.path.join(father_path, 'model/base_model')
    data_path = os.path.join(father_path, 'dataset/data')
    en_dir = os.path.join(data_path, 'small_vocab_en')
    fr_dir = os.path.join(data_path, 'small_vocab_fr')
    english_sentences = load_data(en_dir)
    french_sentences = load_data(fr_dir)
    # english_sentences, french_sentences = load_data2(data_dir)

    preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = \
        preprocess(english_sentences, french_sentences)

    max_english_sequence_length = preproc_english_sentences.shape[1]
    max_french_sequence_length = preproc_french_sentences.shape[1]
    english_vocab_size = len(english_tokenizer.word_index)
    french_vocab_size = len(french_tokenizer.word_index)

    print('Data Preprocessed')
    print("Max English sentence length:", max_english_sequence_length)
    print("Max French sentence length:", max_french_sequence_length)
    print("English vocabulary size:", english_vocab_size)
    print("French vocabulary size:", french_vocab_size)

    x_train, x_test, y_train, y_test = train_test_split(
        preproc_english_sentences, preproc_french_sentences,  # x,y是原始数据
        test_size=0.2  # test_size默认是0.25
    )  # 返回的是 剩余训练集+测试集

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train,  # 把上面剩余的 x_train, y_train继续拿来切
        test_size=0.2  # test_size默认是0.25
    )

    print("Data split")
    print('x_train: ', x_train.shape)
    print('y_train: ', y_train.shape)
    print('x_test: ', x_test.shape)
    print('y_test: ', y_test.shape)
    print('x_valid: ', x_valid.shape)
    print('y_valid: ', y_valid.shape)

    ende_training(x_train, y_train, x_valid, y_valid, x_test, y_test, english_tokenizer, french_tokenizer,
                  embed_depth=embed_depth, state_len=state_len, dense_units=dense_units, model_path=model_path,
                  epoch_size=epoch_size, batch_size=batch_size, learning_rate=learning_rate)
    prediction(preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer, model_path,
               embed_depth=embed_depth, state_len=state_len, dense_units=dense_units)
