import tensorflow as tf
import tensorflow_addons as tfa

class CNN():
    def __init__(self, num_input, num_classes, cnn_config, cnn_drop_out_rate):
        cnn = [c[0] for c in cnn_config]
        cnn_num_filters = [c[1] for c in cnn_config]
        max_pool_ksize = [c[2] for c in cnn_config]
        self.layer = {}
        self.cnn_drop_out_rate = cnn_drop_out_rate
        for idd, filter_size in enumerate(cnn):
            layer = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=cnn_num_filters[idd],
                                       kernel_size=(int(filter_size)),
                                       strides=1,
                                       padding="SAME",
                                       activation='relu',
                                       kernel_initializer=tf.initializers.GlorotUniform()
                                       ),
                tf.keras.layers.MaxPool1D(
                    pool_size=(int(max_pool_ksize[idd])),
                    strides=1,
                    padding='SAME'
                ),
                tf.keras.layers.Dropout(rate=self.cnn_drop_out_rate)
            ])
            self.layer[idd] = layer
        self.flatten = tf.keras.layers.Flatten()

