import tensorflow as tf

layer = tf.keras.layers.SimpleRNN(64, name="outputLastId")  # hidden layer length 64
x = tf.random.normal([4, 80, 100])
out = layer(x)  # return the output of last timestamp
print(out.shape)

layer = tf.keras.layers.SimpleRNN(64, return_sequences=True, name="outputAllId")
out = layer(x)  # return all the outputs from all the timestamps
print(out.shape)