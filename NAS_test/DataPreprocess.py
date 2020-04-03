#!/usr/bin/env python

import tensorflow_datasets as tfds
import tensorflow as tf
import os
import numpy as np
import sys


# current_path = os.path.abspath(__file__)
# # 获取当前文件的父目录
# father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
# dataset_path = os.path.join(father_path, 'dataset')
# data_path = os.path.join(father_path, 'data')
# download_dir = dataset_path
# en_vo_dir = os.path.join(data_path, 'en_vo')
# zh_vo_dir = os.path.join(data_path, 'zh_vo')
# en_vo_dir = "/Users/luoxuqi/PycharmProjects/TensorFlow2/NAS_test/data/en_vo"
# zh_vo_dir = "/Users/luoxuqi/PycharmProjects/TensorFlow2/NAS_test/data/zh_vo"


# builder_name = "wmt_translate"
# max_len = 40
# batch_size = 64


class WMTDataSetPipeline():
    def __init__(self):
        current_path = os.path.abspath(__file__)
        father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
        dataset_path = os.path.join(father_path, 'dataset')
        data_path = os.path.join(father_path, 'data')
        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        self.download_dir = dataset_path
        self.data_path = data_path
        self.en_vo_dir = os.path.join(data_path, 'en_vo')
        self.zh_vo_dir = os.path.join(data_path, 'zh_vo')
        self.builder_name = "wmt_translate"
        self.max_len = 40
        self.batch_size = 64
        self.config = tfds.translate.wmt.WmtConfig(
            version=tfds.core.Version('0.0.3', experiments={tfds.core.Experiment.S3: False}),
            language_pair=("zh", "en"),
            subsets={
                tfds.Split.TRAIN: ["newscommentary_v14"],
            },
        )

    def setup_DataBuilder(self):
        """
        Download the dataset using tfds.builder
        :param download_dir:
        :return: tfds.builder
        """
        builder = tfds.builder(self.builder_name, config=self.config)
        builder.download_and_prepare(download_dir=self.download_dir)
        print(builder.info)
        return builder

    def build_vocabulary_en(self, train_dataset, vocab_file, max_subword_length):
        """

        :param train_dataset:
        :param test_dataset:
        :param vocab_file:
        :param key1:
        :param key2:
        :return:
        """
        try:
            subword_encoder = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_file)
            print(f"Loading built vocabulary： {vocab_file}")
        except:
            train_pair = tfds.as_numpy(train_dataset)
            print("Can not find built vocabulary %s, re-building" % vocab_file)
            subword_encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (en.numpy() for en, _ in train_dataset),
                target_vocab_size=2 ** 13, max_subword_length=max_subword_length)  # 有需要可以調整字典大小

            # 將字典檔案存下以方便下次 warmstart
            subword_encoder.save_to_file(vocab_file)

        print(f"vocabulary size：{subword_encoder.vocab_size}")
        print(f"Top 10 subwords：{subword_encoder.subwords[:10]}")
        print()
        return subword_encoder

    def build_vocabulary_zh(self, train_dataset, vocab_file, max_subword_length):
        """

        :param train_dataset:
        :param test_dataset:
        :param vocab_file:
        :param key1:
        :param key2:
        :return:
        """
        try:
            subword_encoder = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_file)
            print(f"Loading built vocabulary： {vocab_file}")
        except:
            train_pair = tfds.as_numpy(train_dataset)
            print("Can not find built vocabulary %s, re-building" % vocab_file)
            subword_encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                (zh.numpy() for _, zh in train_dataset),
                target_vocab_size=2 ** 13, max_subword_length=max_subword_length)  # 有需要可以調整字典大小

            # 將字典檔案存下以方便下次 warmstart
            subword_encoder.save_to_file(vocab_file)

        print(f"vocabulary size：{subword_encoder.vocab_size}")
        print(f"Top 10 subwords：{subword_encoder.subwords[:10]}")
        print()
        return subword_encoder

    def filter_max_length(self, en, zh):
        return tf.logical_and(tf.size(en) <= self.max_len,
                              tf.size(zh) <= self.max_len)

    def encode(self, en, zh):
        en_sen = self.subword_encoder_en.encode(str(en.numpy(), encoding='utf-8', errors="ignore"))
        zh_sen = self.subword_encoder_zh.encode(str(zh.numpy(), encoding='utf-8', errors="ignore"))
        # element['en'] = en_sen
        # element['zh'] = zh_sen
        return en_sen, zh_sen

    def encode_all(self, en, zh, test=False):
        """
        encode all the sentences to indices
        :param sentences: list of strings of text
        :param subword_encoder: vocabulary
        :param encoded_file: vocabulary file
        :return: return a list of encoded sentences
        """
        return tf.py_function(self.encode, inp=[en, zh], Tout=[tf.int64, tf.int64])

    def execution(self):
        builder = self.setup_DataBuilder()
        # dataset = builder.as_dataset()
        # train_data = dataset["train"]
        train_perc = 20
        val_prec = 1
        drop_prec = 100 - train_perc - val_prec

        split = tfds.Split.TRAIN.subsplit([train_perc, val_prec, drop_prec])
        examples = builder.as_dataset(split=split, as_supervised=True)
        train_data, val_examples, _ = examples
        print(train_data)
        print(val_examples)
        # test_data = dataset["test"]
        for en, zh in train_data.take(3):
            print(en)
            print(zh)
            print('-' * 10)

        self.subword_encoder_zh = self.build_vocabulary_zh(train_data, self.zh_vo_dir, 1)
        self.subword_encoder_en = self.build_vocabulary_en(train_data, self.en_vo_dir, 20)
        sample_string = 'China is beautiful.'
        indices = self.subword_encoder_en.encode(sample_string)
        print("Sample sentence: ", sample_string)
        print("Sample indices: ", indices)
        train_data = train_data.map(self.encode_all)
        train_data = train_data.filter(self.filter_max_length)
        # nums_train_data = 0
        # for en, zh in train_data:
        #     cond1 = len(en) <= self.max_len
        #     cond2 = len(zh) <= self.max_len
        #     assert cond1 and cond2
        #     nums_train_data += 1
        #
        # print(f"All the Eng and Zh sentences has {self.max_len} tokens")
        # print(f"Train dataset total has {nums_train_data} data")
        #
        # num_test_data = 0
        # for en, zh in test_data:
        #     cond1 = len(en) <= self.max_len
        #     cond2 = len(zh) <= self.max_len
        #     assert cond1 and cond2
        #     num_test_data += 1
        #
        # print(f"All the Eng and Zh sentences has {self.max_len} tokens")
        # print(f"Test dataset total has {num_test_data} data")

        db_train = train_data.padded_batch(self.batch_size, padded_shapes=([self.max_len], [self.max_len]))
        en_batch, zh_batch = next(iter(db_train))
        print("英文索引序列的 batch")
        print(en_batch)
        print('-' * 20)
        print("中文索引序列的 batch")
        print(zh_batch)

        en_batched, zh_batched = next(iter(db_train))
        print(
            "Encoded: %s\n" % en_batched[0, :])
        print(
            "Encoded: %s\n" % zh_batched[0, :])

        return db_train


# def setup_DataBuilder(dir, builder_name, config):
#     """
#     Download the dataset using tfds.builder
#     :param download_dir:
#     :return: tfds.builder
#     """
#     builder = tfds.builder(builder_name, config=config)
#     builder.download_and_prepare(download_dir=dir)
#     # zh_file_path = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/dataset/news-commentary-v14.zh"
#     # en_file_path = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/dataset/news-commentary-v14.en"
#     # file_path = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/dataset/news-commentary-v14.en-zh.tsv"
#     # with open(zh_file_path, encoding='utf-8') as f:
#     #     zh_sentences = f.readlines()
#     # with open(en_file_path, encoding='utf-8') as f:
#     #     en_sentences = f.readlines()
#     # with open(file_path, encoding="utf-8") as f:
#     #     sentences = f.readlines()
#     # for sample in range(5):
#     #     print("English sample {}: {}".format(sample, sentences[sample][0]))
#     #     print("Chinese sample {}: {}".format(sample, sentences[sample][1]))
#     print(builder.info)
#     return builder
#
#
# def build_vocabulary(train_dataset, test_dataset, vocab_file, key1, key2, max_subword_length):
#     try:
#         subword_encoder = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_file)
#         print(f"Loading built vocabulary： {vocab_file}")
#     except:
#         train_pair = tfds.as_numpy(train_dataset)
#         test_pair = tfds.as_numpy(test_dataset)
#         print("Can not find built vocabulary, re-building")
#         train_sentences = seperate_sentences(train_pair, key1)
#         test_sentences = seperate_sentences(test_pair, key2)
#         subword_encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
#             train_sentences + test_sentences,
#             target_vocab_size=2 ** 13)  # 有需要可以調整字典大小
#
#         # 將字典檔案存下以方便下次 warmstart
#         subword_encoder.save_to_file(vocab_file)
#
#     print(f"vocabulary size：{subword_encoder.vocab_size}")
#     print(f"Top 10 subwords：{subword_encoder.subwords[:10]}")
#     print()
#     return subword_encoder


# def seperate_sentences(pair_sentences, key):
#     ss = []
#     for ex in pair_sentences:
#         s = str(ex[key], encoding="utf-8")
#         ss.append(s)
#     return ss
#
#
# def filter_max_length(element, max_length=max_len):
#     return tf.logical_and(tf.strings.length(element["en"]) <= max_length,
#                           tf.strings.length(element["zh"]) <= max_length)
#
#
# def encode(en, zh):
#     en_sen = subword_encoder_en.encode(str(en.numpy(), encoding='utf-8', errors="ignore"))
#     zh_sen = subword_encoder_zh.encode(str(zh.numpy(), encoding='utf-8', errors="ignore"))
#     # element['en'] = en_sen
#     # element['zh'] = zh_sen
#     return en_sen, zh_sen
#
#
# def encode_all(data, test=False):
#     """
#     encode all the sentences to indices
#     :param sentences: list of strings of text
#     :param subword_encoder: vocabulary
#     :param encoded_file: vocabulary file
#     :return: return a list of encoded sentences
#     """
#     # for each in iter(data):
#     #     each = encode(each, subword_encoder_en, subword_encoder_zh)
#     if test:
#         zh = data['en']
#         en = data['zh']
#     else:
#         zh = data['zh']
#         en = data['en']
#     return tf.py_function(encode, inp=[en, zh], Tout=[tf.int64, tf.int64])


# builder = setup_DataBuilder(download_dir, builder_name, config)
# dataset = builder.as_dataset()
# nums_train_data = builder.info.splits["train"].num_examples
# train_data = dataset["train"]
# test_data = dataset["test"]
# for each in train_data.take(3):
#     print(each["en"], tf.strings.length(each["en"]))
#     print(each["zh"], tf.strings.length(each["zh"]))
#     print('-' * 10)
#
# subword_encoder_zh = build_vocabulary(train_data, test_data, en_vo_dir, "en", "zh")
# subword_encoder_en = build_vocabulary(train_data, test_data, zh_vo_dir, "zh", "en")
# sample_string = 'China is beautiful.'
# indices = subword_encoder_en.encode(sample_string)
# print(indices)
# train_data = train_data.map(lambda x: encode_all(x))
# test_data = test_data.map(lambda x: encode_all(x, True))
# train_data = train_data.filter(filter_max_length)
# test_data = test_data.filter(filter_max_length)
#
# nums_train_data = 0
# max = 0
# for element in train_data:
#     cond1 = tf.strings.length(element['en']) <= max_len
#     cond2 = tf.strings.length(element['zh']) <= max_len
#     assert cond1 and cond2
#     nums_train_data += 1
#     if (tf.strings.length(element['en']) >= max):
#         max = tf.strings.length(element['en'])
#
# print(f"All the Eng and Zh sentences has {max_len} tokens")
# print(f"Train dataset total has {nums_train_data} data, max_len {max}")
#
# num_examples = 0
# max = 0
# for element in test_data:
#     cond1 = tf.strings.length(element['en']) <= max_len
#     cond2 = tf.strings.length(element['zh']) <= max_len
#     assert cond1 and cond2
#     num_examples += 1
#     if tf.strings.length(element['en']) >= max:
#         max = tf.strings.length(element['en'])
#
# print(f"All the Eng and Zh sentences has {max_len} tokens")
# print(f"Test dataset total has {num_examples} data, max_len {max}")
#
# db_train = train_data.padded_batch(batch_size, padded_shapes=([max_len], [max_len]))
# db_test = test_data.padded_batch(batch_size, padded_shapes=([max_len], [max_len]))
if __name__ == '__main__':
    pipeline = WMTDataSetPipeline()
    pipeline.execution()
