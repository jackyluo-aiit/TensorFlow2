import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np
import random
from tensorflow.keras.optimizers import RMSprop

max_features = 10000
max_len = 500


# def shuffle_set(train_image, train_label, test_image, test_label):
#     train_row = range(len(train_label))
#     random.shuffle(train_row)
#     train_image = train_image[train_row]
#     train_label = train_label[train_row]
#
#     test_row = range(len(test_label))
#     random.shuffle(test_row)
#     test_image = test_image[test_row]
#     test_label = test_label[test_row]
#     return train_image, train_label, test_image, test_label
#
#
# def get_batch(image, label, batch_size, now_batch, total_batch):
#     if now_batch < total_batch - 1:
#         image_batch = image[now_batch * batch_size:(now_batch + 1) * batch_size]
#         label_batch = label[now_batch * batch_size:(now_batch + 1) * batch_size]
#     else:
#         image_batch = image[now_batch * batch_size:]
#         label_batch = label[now_batch * batch_size:]
#     return image_batch, label_batch
# def get_Batch(data, label, batch_size):
#     print(data.shape, label.shape)
#     input_queue = tf.train.slice_input_producer([data, label], num_epochs=1, shuffle=True, capacity=32 )
#     x_batch, y_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=32, allow_smaller_final_batch=False)
#     return x_batch, y_batch
def get_batch(train_data, train_label, batch_size):
    index = np.random.randint(0, np.shape(train_data)[0], batch_size)
    return train_data[index, :], train_label[index]


print('loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)


print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(128, 10, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
epoch_accuracy = tf.keras.metrics.BinaryCrossentropy(name='train_acc')

# def train_process(sentence, label):
#     with tf.GradientTape()as tape:
#         prediction = model(sentence)
#         loss = tf.keras.losses.binary_crossentropy(y_pred=prediction, y_true=label)
#     grads = tape.gradient(loss, model.variables)
#     optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
#     train_acc(label, prediction)


num_epoch = 10
num_batch = int(20000 * num_epoch // 128)
loss_avg = tf.keras.metrics.Mean()
for batch in range(num_batch):
    sentence, label = get_batch(x_train[:20000, :], y_train[:20000], 128)
    with tf.GradientTape()as tape:
        prediction = model(sentence)
        loss = tf.keras.losses.binary_crossentropy(y_pred=prediction, y_true=label)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    print('step:', optimizer.iterations.numpy(), 'loss: {:.3f}'.format(loss_avg(loss)))


# model = Sequential()
# model.add(layers.Embedding(max_features, 128, input_length=max_len))
# model.add(layers.Conv1D(40, 7, activation='relu'))
# model.add(layers.MaxPooling1D(5))
# model.add(layers.Conv1D(40, 7, activation='relu'))
# model.add(layers.GlobalMaxPooling1D())
# model.add(layers.Dense(1))
# model.summary()
# model.compile(optimizer=RMSprop(lr=1e-4),
#               loss='binary_crossentropy',
#               metrics=['acc'])
# history = model.fit(x_train, y_train,
#                     epochs=10,
#                     batch_size=128,
#                     validation_split=0.2)
num_batch = int(5000 // 128)
for batch in range(num_batch):
    sentence, label = get_batch(x_train[20000:,:], y_train[20000:], 128)
    pred = model.predict(sentence)
    epoch_accuracy.update_state(label, pred)
print('test acc:', epoch_accuracy.result())