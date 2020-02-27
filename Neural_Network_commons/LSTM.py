import tensorflow as tf

x = tf.random.normal([2, 80, 100])
layer = tf.keras.layers.LSTM(64, return_sequences=True)
out = layer(x)
print(out)


net = tf.keras.Sequential()
net.add(tf.keras.layers.LSTM(64, return_sequences=True))
net.add(tf.keras.layers.LSTM(64))
out = net(x)
print(out)