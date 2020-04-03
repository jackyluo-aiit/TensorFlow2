import tensorflow as tf

x = tf.random.normal([2, 100])
layer = tf.keras.layers.RepeatVector(3)
out = layer(x)
print(out)