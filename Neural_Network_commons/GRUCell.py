import tensorflow as tf

cell = tf.keras.layers.GRUCell(64)
x = tf.random.normal([2, 80, 100])
h = [tf.zeros([2, 64])]
out, hx = cell(x[:, 0, :], h)
print(out.shape)
print(id(out), id(hx))
print(out == hx)