import tensorflow as tf

x = tf.random.normal([2, 80, 100])
xt = x[:, 0, :]
cell = tf.keras.layers.LSTMCell(64)
state = [tf.zeros([2, 64]), tf.zeros([2, 64])]  # initiate h, and c hidden vectors
out, state = cell(xt, state)
print("output: ", id(out), "state h: ", id(state[0]), "state c: ", id(state[1]))

