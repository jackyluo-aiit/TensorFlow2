import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


class policy_network(tf.keras.Model):
    def __init__(self, max_layers):
        super(policy_network, self).__init__()
        self.max_layers = max_layers
        nas_cell = tfa.rnn.NASCell(4 * max_layers)
        self.model = tf.keras.layers.RNN(cell=nas_cell)

    def call(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, -1)
        outputs = self.model(inputs)
        bias = tf.Variable([0.05] * 4 * self.max_layers)
        outputs = tf.nn.bias_add(outputs, bias)
        # print("outputs: ", outputs, outputs[:, -1:, :],
        #       tf.slice(outputs, [0, 4 * self.max_layers - 1, 0], [1, 1, 4 * self.max_layers]))
        # return tf.slice(outputs, [0, 4*max_layers-1, 0], [1, 1, 4*max_layers]) # Returned last output of rnn
        return outputs


x = tf.keras.Input((8))
state = [tf.zeros([2, 64]), tf.zeros([2, 64])]
model = policy_network(2)
print(x.shape)
# print(state.shape)
out = model(x)
print(out.shape)
action = [out[0][x:x+4] for x in range(0, out.shape[1], 4)]
print(action[0])
cnn_config = action
cnn = [c[0] for c in cnn_config]
cnn_num_filters = [c[1] for c in cnn_config]
max_pool_ksize = [c[2] for c in cnn_config]
for idd, filter_size in enumerate(cnn):
    print(idd, filter_size)



