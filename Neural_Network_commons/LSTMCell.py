import tensorflow as tf
import numpy as np

x = tf.random.normal([2, 80, 100])
xt = x[:, 0, :]
print(xt.shape)
cell = tf.keras.layers.LSTMCell(64)
state = [tf.zeros([2, 64]), tf.zeros([2, 64])]  # initiate h, and c hidden vectors
out, h = cell(xt, state)
print(np.shape(h))
# print("output: ", id(out), "state h: ", id(state[0]), "state c: ", id(state[1]))

model = tf.keras.layers.RNN(cell=cell,
                            dtype=tf.float32,
                            return_state=True)

out, h, c = model(x, state)
print(np.shape(out))
print(np.shape(h))
print("output: ", id(out), "state: ", id(h), "state: ", id(c))
print(out==c)
# print("output: ", out.shape, "state: ", state.shape)
