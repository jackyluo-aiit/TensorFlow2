from pprint import pprint

import tensorflow_datasets as tfds
import tensorflow as tf
import collections
import pandas as pd
import os
import numpy as np
import sys

# tmp_builder = tfds.builder("wmt19_translate/zh-en")
# pprint(tmp_builder.subsets)
download_dir = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/dataset"
en_sentences_file = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/data/en_sentences.txt"
zh_sentences_file = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/data/zh_sentences.txt"
en_filtered = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/data/en_filtered.txt"
zh_filtered = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/data/zh_filtered.txt"
en_test_sentences_file = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/data/en_test_sentences.txt"
zh_test_sentences_file = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/data/zh_test_sentences.txt"
en_vocab_file = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/data/en_dict"
zh_vocab_file = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/data/zh_dict"
en_encoded_file = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/data/encoded_en.npy"
zh_encoded_file = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/data/encoded_zh.npy"
en_test_encoded_file = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/data/encoded_test_en.npy"
zh_test_encoded_file = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/data/encoded_test_zh.npy"
config = tfds.translate.wmt.WmtConfig(
    version=tfds.core.Version('0.0.3', experiments={tfds.core.Experiment.S3: False}),
    language_pair=("zh", "en"),
    subsets={
        tfds.Split.TRAIN: ["newscommentary_v14"],
        tfds.Split.VALIDATION: ["newstest2018"],
    },
)
builder_name = "wmt_translate"
embed_depth = 100
batch_size = 128
epoch_size = 10
state_len = 256
learning_rate = 0.001
max_len = 40
subword_encoder_en = None
subword_encoder_zh = None
num_train_examples = 0
num_example = 0


def fetch_data(download_dir, builder_name, config):
    """
    Download the dataset using tfds.builder
    :param download_dir:
    :return: tfds.builder
    """
    builder = tfds.builder(builder_name, config=config)
    builder.download_and_prepare(download_dir=download_dir)
    # zh_file_path = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/dataset/news-commentary-v14.zh"
    # en_file_path = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/dataset/news-commentary-v14.en"
    # file_path = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/dataset/news-commentary-v14.en-zh.tsv"
    # with open(zh_file_path, encoding='utf-8') as f:
    #     zh_sentences = f.readlines()
    # with open(en_file_path, encoding='utf-8') as f:
    #     en_sentences = f.readlines()
    # with open(file_path, encoding="utf-8") as f:
    #     sentences = f.readlines()
    # for sample in range(5):
    #     print("English sample {}: {}".format(sample, sentences[sample][0]))
    #     print("Chinese sample {}: {}".format(sample, sentences[sample][1]))
    print(builder.info)
    return builder


def build_vocabulary(vocab_file, sentences, zh=False):
    """
    load the vocabulary if exist, build the vocabulary if not exist
    :param vocab_file: a path of the vocabulary file
    :param sentences: a list of sequences of text
    :return: vocabulary of the sentences
    """
    if zh:
        max_char_len = 1
    else:
        max_char_len = 20
    try:
        subword_encoder = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_file)
        print(f"Loaded existed vocabulary： {vocab_file}")
    except:
        print("Can not found vocabulary{}，building now.".format(vocab_file))
        subword_encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            sentences,
            target_vocab_size=2 ** 13,
            max_subword_length=max_char_len)  # 有需要可以調整字典大小
        subword_encoder.save_to_file(vocab_file)
    return subword_encoder


def encode(en, zh):
    en_sen = subword_encoder_en.encode(str(en.numpy(), encoding='utf-8', errors="ignore"))
    zh_sen = subword_encoder_zh.encode(str(zh.numpy(), encoding='utf-8', errors="ignore"))
    # element['en'] = en_sen
    # element['zh'] = zh_sen
    return en_sen, zh_sen


def encode_all(data):
    """
    encode all the sentences to indices
    :param sentences: list of strings of text
    :param subword_encoder: vocabulary
    :param encoded_file: vocabulary file
    :return: return a list of encoded sentences
    """
    # for each in iter(data):
    #     each = encode(each, subword_encoder_en, subword_encoder_zh)
    print(data)
    zh = data['en']
    en = data['zh']
    return tf.py_function(encode, inp=[en, zh], Tout=[tf.int64, tf.int64])


def save_sentence(pair_sentences, file1, file2, test=False):
    ss1 = []
    ss2 = []
    if not os.path.exists(file1 and file2):
        for ex in pair_sentences:
            ss1.append(str(ex["en"], encoding="utf-8"))
            ss2.append(str(ex["zh"], encoding="utf-8"))
        with open(file1, 'w', encoding='utf-8') as f:
            for en_sentence in ss1:
                f.writelines(en_sentence)
                f.write('\n')
        with open(file2, 'w', encoding='utf-8') as f:
            for zh_sentence in ss2:
                f.writelines(zh_sentence)
                f.write('\n')
    else:
        if test:
            f1 = open(file1, 'r')
            ss1 = f1.readlines()
            f2 = open(file2, 'r')
            ss2 = f2.readlines()
        else:
            f1 = open(file1, 'r')
            ss2 = f1.readlines()
            f2 = open(file2, 'r')
            ss1 = f2.readlines()
    return ss1, ss2


def save_numpy_array(sentences, file):
    # with open(file, 'w', encoding='utf-8') as f:
    #     for each in sentences:
    #         f.write(str(each))
    #         f.write('\n')
    np.save(file, sentences)
    print("Save {} file success.".format(file))


def logit_to_text(logits, subword_encoder):
    return ' '.join(subword_encoder.decode(prediction) for prediction in np.argmax(logits, 1))


def filter_max_length(element, max_length=max_len):
    return tf.logical_and(tf.strings.length(element["en"]) <= max_length,
                          tf.strings.length(element["zh"]) <= max_length)


class SimpleRNN(tf.keras.Model):
    def __init__(self, input_vocab_size, target_vocab_size, embed_depth):
        super(SimpleRNN, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.embed_depth = embed_depth
        self.embedding = tf.keras.layers.Embedding(self.input_vocab_size, self.embed_depth)
        self.rnn = tf.keras.Sequential([
            tf.keras.layers.GRU(state_len, dropout=0.5, return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='relu')),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.target_vocab_size, activation='softmax'))
        ])

    def call(self, inputs):
        embeded_inputs = self.embedding(inputs)
        out = self.rnn(embeded_inputs)
        return out


def simpleRNN_training(data, test, input_vocab_size, target_vocab_size):
    model = SimpleRNN(input_vocab_size, target_vocab_size, embed_depth)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.optimizers.Adam(learning_rate),
                  metrics=['accuracy'])
    model.fit(data, epochs=epoch_size,
              validation_data=test,
              steps_per_epoch=num_train_examples//batch_size,
              validation_steps=num_examples//batch_size)
    model.evaluate(test)
    print('Finish training')


class BidirectionRNN(tf.keras.Model):
    def __init__(self, input_vocab_size, target_vocab_size, embed_depth):
        super(BidirectionRNN, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.embed_depth = embed_depth
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, embed_depth)
        self.rnn = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(state_len, dropout=0.5, return_sequences=True)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation='relu')),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.target_vocab_size, activation='softmax'))
        ])

    def call(self, inputs):
        embed_input = self.embedding(inputs)
        out = self.rnn(embed_input)
        return out


def bd_rnn_training(data, test, input_vocab_size, target_vocab_size):
    model = BidirectionRNN(input_vocab_size, target_vocab_size, embed_depth)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate),
                  metrics=['accuracy'])
    model.fit(data, epochs=epoch_size,
              validation_data=test,
              steps_per_epoch=num_train_examples//batch_size,
              validation_steps=num_examples//batch_size)
    model.evaluate(test)
    print("Finished training")


def evaluation(input_sentence, subword_encoder_en, subword_encoder_zh, model):
    encoded_input = subword_encoder_en.encode(input_sentence)
    encoded_input = tf.expand_dims(encoded_input, 0)  # add one more dimension meaning the batch_size
    output = model.predict(encoded_input)
    output = tf.squeeze(0, output)
    print(output.shape)
    decoded_output = subword_encoder_zh.decode(predition for predition in np.argmax(output, axis=1))
    print(decoded_output)


if __name__ == '__main__':
    builder = fetch_data(download_dir, builder_name, config)
    dataset = builder.as_dataset()
    train_data = dataset["train"]
    test_data = dataset["test"]
    index = 0
    train_sentences = tfds.as_numpy(train_data)
    test_sentences = tfds.as_numpy(test_data)
    en_sentences = []
    zh_sentences = []
    # for ex in train_sentences:
    #     if index > 3:
    #         break
    #     index += 1
    #     print("en train sentence {}: {}".format(index, str(ex["en"], encoding="utf-8")))
    #     print("zh train sentence {}: {}".format(index, str(ex["zh"], encoding="utf-8")))
    # index = 0
    # for ex in test_sentences:
    #     if index > 3:
    #         break
    #     index += 1
    #     print("en test sentence {}: {}".format(index, str(ex["en"], encoding="utf-8")))
    #     print("zh test sentence {}: {}".format(index, str(ex["zh"], encoding="utf-8")))

    # if not os.path.exists(en_sentences_file and zh_sentences_file):
    #     for ex in train_sentence:
    #         en_sentences.append(str(ex["en"], encoding="utf-8"))
    #         zh_sentences.append(str(ex["zh"], encoding="utf-8"))
    #     with open(en_sentences_file, 'w', encoding='utf-8') as f:
    #         for en_sentence in en_sentences:
    #             f.writelines(en_sentence)
    #             f.write('\n')
    #     with open(zh_sentences_file, 'w', encoding='utf-8') as f:
    #         for zh_sentence in zh_sentences:
    #             f.writelines(zh_sentence)
    #             f.write('\n')
    # else:
    #     f1 = open(en_sentences_file, 'r')
    #     zh_sentences = f1.readlines()
    #     f2 = open(zh_sentences_file, 'r')
    #     en_sentences = f2.readlines()
    en_train_sentences, zh_train_sentences = save_sentence(train_sentences, en_sentences_file, zh_sentences_file)
    en_test_sentences, zh_test_sentences = save_sentence(test_sentences, en_test_sentences_file,
                                                         zh_test_sentences_file, test=True)
    print("Total en_sentences: {}".format(len(en_train_sentences)))
    print("Total zh_sentences: {}".format(len(zh_train_sentences)))
    print("Total en_test_sentences: {}".format(len(en_test_sentences)))
    print("Total zh_test_sentences: {}".format(len(zh_test_sentences)))

    en_sentences = en_train_sentences + en_test_sentences
    zh_sentences = zh_train_sentences + zh_test_sentences
    subword_encoder_en = build_vocabulary(en_vocab_file, en_sentences)
    subword_encoder_zh = build_vocabulary(zh_vocab_file, zh_sentences, zh=True)
    # english_words_counter = collections.Counter([word for sentence in en_sentences for word in sentence.split()])
    # chinese_words_counter = collections.Counter([word for sentence in zh_sentences for word in sentence.split()])
    # print('{} English words.'.format(len([word for sentence in en_sentences for word in sentence.split()])))
    # print('{} unique English words.'.format(len(english_words_counter)))
    # print('10 Most common words in the English dataset:')
    # print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
    # print('{} Chinese characters.'.format(len([word for sentence in zh_sentences for word in sentence.split()])))
    # print('{} unique French words.'.format(len(chinese_words_counter)))
    # print('10 Most common words in the French dataset:')
    # print('"' + '" "'.join(list(zip(*chinese_words_counter.most_common(10)))[0]) + '"')
    # try:
    #     subword_encoder_en = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_file)
    #     print(f"Loaded existed vocabulary： {vocab_file}")
    # except:
    #     print("Can not found vocabulary，building now.")
    #     subword_encoder_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    #         en_sentences,
    #         target_vocab_size=2 ** 13)  # 有需要可以調整字典大小
    #     subword_encoder_en.save_to_file(vocab_file)
    #
    # try:
    #     subword_encoder_zh = tfds.features.text.SubwordTextEncoder.load_from_file(zh_vocab_file)
    #     print(f"Loaded existed vocabulary： {zh_vocab_file}")
    # except:
    #     print("Can not found vocabulary，building now.")
    #     subword_encoder_zh = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    #         zh_sentences,
    #         target_vocab_size=2 ** 13, max_subword_length=1)  # 有需要可以調整字典大小
    #     subword_encoder_zh.save_to_file(zh_vocab_file)

    print(f"Vocabulary size：{subword_encoder_en.vocab_size}")
    print(f"Top 10 subwords：{subword_encoder_en.subwords[:10]}")
    print(f"Vocabulary size：{subword_encoder_zh.vocab_size}")
    print(f"Top 10 subwords：{subword_encoder_zh.subwords[:10]}")

    en_example = "Making Do With More"
    zh_example = "多劳应多得"
    encode_en = subword_encoder_en.encode(en_example)
    for idx in encode_en:
        subword = subword_encoder_en.decode([idx])
        print("{}:{}".format(idx, subword))
    encode_zh = subword_encoder_zh.encode(zh_example)
    for idx in encode_zh:
        subword = subword_encoder_zh.decode([idx])
        print("{}:{}".format(idx, subword))

    filter_data = train_data.filter(filter_max_length)
    filter_tmp_data = test_data.filter(filter_max_length)
    max = 0
    for element in filter_data:
        cond1 = tf.strings.length(element['en']) <= max_len
        cond2 = tf.strings.length(element['zh']) <= max_len
        assert cond1 and cond2
        num_train_examples += 1
        if (tf.strings.length(element['en']) >= max):
            max = tf.strings.length(element['en'])

    print(f"All the Eng and Zh sentences has {max_len} 個 tokens")
    print(f"Train dataset total has {num_train_examples} data, max_len {max}")
    num_examples = 0
    max = 0
    for element in filter_tmp_data:
        cond1 = tf.strings.length(element['en']) <= max_len
        cond2 = tf.strings.length(element['zh']) <= max_len
        assert cond1 and cond2
        num_examples += 1
        if tf.strings.length(element['en']) >= max:
            max = tf.strings.length(element['en'])

    print(f"All the Eng and Zh sentences has {max_len} 個 tokens")
    print(f"Test dataset total has {num_examples} data, max_len {max}")
    # filter_data = tfds.as_numpy(tmp_data)
    tmp_data = filter_data.map(lambda x: encode_all(x))
    test_tmp_data = filter_data.map(lambda x: encode_all(x))
    element = next(iter(tmp_data))
    print(element[0])
    print(element[1])

    db_train = tmp_data.padded_batch(batch_size, padded_shapes=([max_len], [max_len])).repeat()
    db_test = test_tmp_data.padded_batch(batch_size, padded_shapes=([max_len], [max_len]))

    # en_train_sentences, zh_train_sentences = save_sentence(filter_data, en_filtered, zh_filtered)

    # encoded_train_en = encode_all(en_train_sentences, subword_encoder_en, en_encoded_file)
    # encoded_train_zh = encode_all(zh_train_sentences, subword_encoder_zh, zh_encoded_file)
    # encoded_test_en = encode_all(en_test_sentences, subword_encoder_en, en_test_encoded_file)
    # encoded_test_zh = encode_all(zh_test_sentences, subword_encoder_zh, zh_test_encoded_file)

    # en_train = tf.keras.preprocessing.sequence.pad_sequences(encoded_train_en, maxlen=max_len, padding="post")
    # zh_train = tf.keras.preprocessing.sequence.pad_sequences(encoded_train_zh, maxlen=max_len, padding="post")
    # en_test = tf.keras.preprocessing.sequence.pad_sequences(encoded_test_en, maxlen=max_len, padding="post")
    # zh_test = tf.keras.preprocessing.sequence.pad_sequences(encoded_test_zh, maxlen=max_len, padding="post")
    filtered = next(iter(filter_data))
    en_batched, zh_batched = next(iter(db_train))
    print(
        "Plain text: %s\nEncoded: %s\n" % (str(filtered['zh'].numpy(), encoding='utf-8'), en_batched[0, :]))
    print(
        "Plain text: %s\nEncoded: %s\n" % (str(filtered['en'].numpy(), encoding='utf-8'), zh_batched[0, :]))
    # sys.exit()

    # db_train = tf.data.Dataset.from_tensor_slices((en_train, zh_train))
    # db_train = db_train.shuffle(1000).batch(batch_size, drop_remainder=True)
    # db_test = tf.data.Dataset.from_tensor_slices((en_test, zh_test))
    # db_test = db_test.batch(batch_size, drop_remainder=True)
    # print(list(db_train.as_numpy_iterator()))
    bd_rnn_training(db_train, db_test, subword_encoder_en.vocab_size, subword_encoder_zh.vocab_size)
