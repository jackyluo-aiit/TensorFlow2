import tensorflow as tf
from tensorflow.keras.datasets import imdb
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import sys

cnn_model_name = 'cnn_text_classification.h5'
rnn_model_name = 'rnn_text_classification.h5'
print(tf.executing_eagerly())
max_features = 10000
max_len = 200
initial_epochs = 10
validation_steps = 20

print('loading data...')
data, info = tfds.load(name="imdb_reviews/subwords8k",
                       with_info=True,
                       as_supervised=True, )

test_dataset = data['test']
train_dataset = data['train']
print(train_dataset)
sys.exit()
encoder = info.features['text'].encoder
print('Vocabulary size: {}'.format(encoder.vocab_size))
# imdb_builder = tfds.builder(name="imdb_reviews/subwords8k")
# imdb_builder.download_and_prepare()
# info = imdb_builder.info
# print("dataset name {} \ndataset size: {}\ndataset features: {}".format(info.name, info.splits, info.features))
# test_dataset = imdb_builder.as_dataset(split="test")
# train_dataset = imdb_builder.as_dataset(split="train")
# for train_example in train_dataset.take(1):
#     sentence, label = train_example["text"], train_example["label"]
#     print("sentence: {}".format(sentence.shape))
#     print("label: {}".format(label.shape))
# encoder = info.features['text'].encoder
# encoder = info.encoder
# print(test_dataset.info)
# print('Vocabulary size: {}'.format(encoder.vocab_size))
# print("Vocabulary detail: {}".format(encoder.subwords))
# print("Vocabulary decode example: ", encoder.decode(train_example["text"]))
BATCH_SIZE = 128
SHUFFLE_SIZE = 10000
for sentence, label in train_dataset.take(1):
    print(sentence)
    print(label)
print(test_dataset)
print(train_dataset)


def filter_max_length(x, y, max_length=max_len):
    return tf.logical_and(tf.size(x) <= max_length,
                          tf.size(y) <= max_length)


train_dataset = train_dataset.filter(filter_max_length)
test_dataset = test_dataset.filter(filter_max_length)
train_dataset = train_dataset.shuffle(SHUFFLE_SIZE)
train_batches = train_dataset.padded_batch(batch_size=BATCH_SIZE, padded_shapes=((max_len,), []))
test_batches = test_dataset.padded_batch(batch_size=BATCH_SIZE, padded_shapes=((max_len,), []))
# x_train, y_train = next(iter(train_batches))
# x_test, y_test = next(iter(test_batches))
# print('train sentence example: {}'.format(x_train[0]))
# print('train label example: {}'.format(y_test[0]))
print(train_batches)
print(test_batches)
for sentence, label in train_batches.take(2):
    print(sentence)
    print(label)


# tokenize and padding 0 behind
# tokenizer = Tokenizer(num_words=10000)
# tokenizer.fit_on_texts((iter(train_batches)))


model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(128, 10, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_acc')
loss_avg = tf.keras.metrics.Mean()


@tf.function
def train_process(sentence, label):
    with tf.GradientTape() as tape:
        prediction = model(sentence)
        loss = tf.keras.losses.BinaryCrossentropy()(label, prediction)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_accuracy(label, prediction)
    loss_avg(loss)


for epoch in range(initial_epochs):
    for sentences, labels in train_batches:
        train_process(sentences, labels)
    print('epoch: {}: Loss: {:.3f}, Acc: {:.3%}'.format(epoch, loss_avg.result(), train_accuracy.result()))

# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(encoder.vocab_size, 64),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])
# model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               optimizer=tf.keras.optimizers.Adam(1e-4),
#               metrics=['accuracy'])
#
# history = model.fit(train_batches, epochs=10, validation_data=test_batches, validation_steps=30)


model.save(cnn_model_name)

test_model = tf.keras.models.load_model(cnn_model_name)
test_acc = tf.keras.metrics.BinaryAccuracy(name='test_acc')
for sentences, labels in test_batches:
    preds = test_model.predict(sentences)
    test_acc(labels, preds)

print('Test acc: {:.3%}'.format(test_acc.result()))
