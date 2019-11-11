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

