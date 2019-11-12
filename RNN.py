import tensorflow as tf
import numpy as np


class DataLoader():
    def __init__(self):
        path = tf.keras.utils.get_file('nietzsche.txt',
                                       origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        with open(path, encoding='utf-8') as f:
            self.raw_text = f.read().lower()
        self.chars = sorted(list(set(self.raw_text)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.text = [self.char_indices[c] for c in self.raw_text]

    def get_batch(self, seq_length, batch_size):
        seq = []
        next_char = []
        for i in range(batch_size):
            index = np.random.randint(0, len(self.text) - seq_length)
            seq.append(self.text[index:index + seq_length])
            next_char.append(self.text[index + seq_length])
        return np.array(seq), np.array(next_char)


class RNN(tf.keras.Model):
    def __init__(self, num_chars, batch_size, seq_length):
        super().__init__()
        self.num_chars = num_chars
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.lstm_cell = tf.keras.layers.LSTMCell(units=256)
        self.dense = tf.keras.layers.Dense(units=self.num_chars)

    def call(self, input, from_logits = False):
        inputs = tf.one_hot(input, depth=self.num_chars)
        state = self.lstm_cell.get_initial_state(inputs, self.batch_size, dtype=tf.float32)
        for char in range(self.seq_length):
            output, state = self.lstm_cell(inputs[:, char, :], state)
        logits = self.dense(output)
        if from_logits:
            return logits
        else:
            return tf.nn.softmax(logits)

    def predict(self, inputs, temperature=1.):
        batch_size, _ = tf.shape(inputs)
        logits = self(inputs, from_logits=True)
        prob = tf.nn.softmax(logits / temperature).numpy()
        return np.array([np.random.choice(self.num_chars, p=prob[i, :])
                         for i in range(batch_size.numpy())])


batch_size = 50
num_batch = 50
seq_length = 50

data_loader = DataLoader()
print(data_loader.text)
rnn_model = RNN(len(data_loader.chars), batch_size, seq_length)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
for batch_index in range(batch_size):
    X, Y = data_loader.get_batch(seq_length, batch_size)
    with tf.GradientTape() as tape:
        pred = rnn_model(X)
        loss = tf.losses.sparse_categorical_crossentropy(y_true=Y, y_pred=pred)
        loss = tf.reduce_mean(loss, axis=0)
    grads = tape.gradient(target=loss, sources=rnn_model.variables)
    optimizer.apply_gradients(zip(grads, rnn_model.variables))

# evaluation
X, _ = data_loader.get_batch(seq_length, 1)
print(X)
for i in X[0]:
    print(data_loader.indices_char[i], end='', flush=True)
for _ in range(100):
    pred = rnn_model.predict(X, 1.0)
    X = np.concatenate([X[:, :], np.expand_dims(pred, axis=1)], axis=-1)
    # pred = rnn_model(X)
    # # print(pred[0])
    # pred = np.squeeze(pred)
    # pred_index = np.random.choice(rnn_model.num_chars, p=pred)
    # # pred_index = tf.argmax(pred)
    # # pred_index = pred_index.numpy()
    # # print(pred_index)
    # # print(data_loader.indices_char[pred_index])
    # # print(pred)
    # # pred_index = np.expand_dims(pred_index, axis=0)
    # # print(np.shape(pred_index))
    # # print(np.shape(X))
    # X = np.concatenate([X[0, :], np.expand_dims(pred_index, axis=1)], axis=-1)
    # X = np.expand_dims(X, axis=0)
    print(X)
for i in X[0]:
    print(data_loader.indices_char[i], end='', flush=True)


