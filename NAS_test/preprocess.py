import sys

from NAS_test.load_data import load_data, load_data2
import numpy as np
import tensorflow as tf
import collections
import os
from sklearn.model_selection import train_test_split


# current_path = os.path.abspath(__file__)
# father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
# data_path = os.path.join(father_path, 'dataset/data')
# en_dir = os.path.join(data_path, 'small_vocab_en')
# fr_dir = os.path.join(data_path, 'small_vocab_fr')
# english_sentences = load_data(en_dir)
# # Load French data
# french_sentences = load_data(fr_dir)

# print('Dataset Loaded')
# for sample_i in range(5):
#     print('English sample {}:  {}'.format(sample_i + 1, english_sentences[sample_i]))
#     print('French sample {}:  {}\n'.format(sample_i + 1, french_sentences[sample_i]))
#
# english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
# french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])
#
# print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
# print('{} unique English words.'.format(len(english_words_counter)))
# print('10 Most common words in the English dataset:')
# print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
# print()
# print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
# print('{} unique French words.'.format(len(french_words_counter)))
# print('10 Most common words in the French dataset:')
# print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')


def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    return tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=length, padding='post')


# Tokenize Example output
# text_sentences = [
#     'The quick brown fox jumps over the lazy dog .',
#     'By Jove , my quick study of lexicography won a prize .',
#     'This is a short sentence .']
# text_tokenized, text_tokenizer = tokenize(text_sentences)
# print(text_tokenizer.word_index)
# print()
# for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
#     print('Sequence {} in x'.format(sample_i + 1))
#     print('  Input:  {}'.format(sent))
#     print('  Output: {}'.format(token_sent))


def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk


if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    data_path = os.path.join(father_path, 'dataset/data')
    # en_dir = os.path.join(data_path, 'news-commentary-v14.en')
    # fr_dir = os.path.join(data_path, 'news-commentary-v14.fr')
    en_dir = os.path.join(data_path, 'small_vocab_en')
    fr_dir = os.path.join(data_path, 'small_vocab_fr')
    # data_dir = os.path.join(data_path, 'news-commentary-v14.en-fr.tsv')
    english_sentences = load_data(en_dir)
    french_sentences = load_data(fr_dir)
    # english_sentences, french_sentences = load_data2(data_dir)

    print('Dataset Loaded')
    for sample_i in range(5):
        print('English sample {}:  {}'.format(sample_i + 1, english_sentences[sample_i]))
        print('French sample {}:  {}\n'.format(sample_i + 1, french_sentences[sample_i]))

    english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
    french_words_counter = collections.Counter([word for sentence in french_sentences for word in sentence.split()])

    print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
    print('{} unique English words.'.format(len(english_words_counter)))
    print('10 Most common words in the English dataset:')
    print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
    print()
    print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
    print('{} unique French words.'.format(len(french_words_counter)))
    print('10 Most common words in the French dataset:')
    print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')

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
    print('English sentences: ', preproc_english_sentences.shape[0])
    print('French sentences: ', preproc_french_sentences.shape[0])

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

# tmp_x = pad(preproc_english_sentences, preproc_french_sentences.shape[1])
# print(tmp_x.shape)
# tmp_x = tf.expand_dims(tmp_x, -1)
# print(tmp_x.shape)
