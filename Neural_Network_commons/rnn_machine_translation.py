from pprint import pprint

import tensorflow_datasets as tfds
import tensorflow as tf
import collections
import pandas as pd
import os

# tmp_builder = tfds.builder("wmt19_translate/zh-en")
# pprint(tmp_builder.subsets)
download_dir = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/dataset"
en_sentences_file = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/data/en_sentences.txt"
zh_sentences_file = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/data/zh_sentences.txt"
en_vocab_file = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/data/en_dict"
zh_vocab_file = "/Users/luoxuqi/PycharmProjects/TensorFlow2/Neural_Network_commons/data/zh_dict"
config = tfds.translate.wmt.WmtConfig(
    version=tfds.core.Version('0.0.3', experiments={tfds.core.Experiment.S3: False}),
    language_pair=("zh", "en"),
    subsets={
        tfds.Split.TRAIN: ["newscommentary_v14"],
        tfds.Split.VALIDATION: ["newstest2018"],
    },
)
builder = tfds.builder("wmt_translate", config=config)
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
dataset = builder.as_dataset()
train_data = dataset["train"]
test_data = dataset["test"]
index = 0
train_sentence = tfds.as_numpy(train_data)
en_sentences = []
zh_sentences = []
for ex in train_sentence:
    if (index > 5):
        break
    index += 1
    print("en sentence {}: {}".format(index, str(ex["en"], encoding="utf-8")))
    print("zh sentence {}: {}".format(index, str(ex["zh"], encoding="utf-8")))
if not os.path.exists(en_sentences_file and zh_sentences_file):
    for ex in train_sentence:
        # if (index > 5):
        #     break
        # index += 1
        # print("en sentence {}: {}".format(index, str(ex["en"], encoding="utf-8")))
        # print("zh sentence {}: {}".format(index, str(ex["zh"], encoding="utf-8")))
        en_sentences.append(str(ex["en"], encoding="utf-8"))
        zh_sentences.append(str(ex["zh"], encoding="utf-8"))
    with open(en_sentences_file, 'w', encoding='utf-8') as f:
        for en_sentence in en_sentences:
            f.writelines(en_sentence)
            f.write('\n')
    with open(zh_sentences_file, 'w', encoding='utf-8') as f:
        for zh_sentence in zh_sentences:
            f.writelines(zh_sentence)
            f.write('\n')
else:
    f1 = open(en_sentences_file, 'r')
    zh_sentences = f1.readlines()
    f2 = open(zh_sentences_file, 'r')
    en_sentences = f2.readlines()
print("Total en_sentences: {}".format(len(en_sentences)))
print("Total zh_sentences: {}".format(len(zh_sentences)))
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
try:
    subword_encoder_en = tfds.features.text.SubwordTextEncoder.load_from_file(en_vocab_file)
    print(f"Loaded existed vocabulary： {en_vocab_file}")
except:
    print("Can not found vocabulary，building now.")
    subword_encoder_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        en_sentences,
        target_vocab_size=2 ** 13)  # 有需要可以調整字典大小
    subword_encoder_en.save_to_file(en_vocab_file)

try:
    subword_encoder_zh = tfds.features.text.SubwordTextEncoder.load_from_file(zh_vocab_file)
    print(f"Loaded existed vocabulary： {zh_vocab_file}")
except:
    print("Can not found vocabulary，building now.")
    subword_encoder_zh = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        zh_sentences,
        target_vocab_size=2 ** 13,max_subword_length=1)  # 有需要可以調整字典大小
    subword_encoder_zh.save_to_file(zh_vocab_file)

print(f"Vocabulary size：{subword_encoder_en.vocab_size}")
print(f"Top 10 subwords：{subword_encoder_en.subwords[:10]}")
print(f"Vocabulary size：{subword_encoder_zh.vocab_size}")
print(f"Top 10 subwords：{subword_encoder_zh.subwords[:10]}")

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(zh_sentences[0])
print(zh_sentences[0])
print(tokenizer.word_index)
