import tensorflow as tf
import numpy as np

# element = [[(1, 1, 1), (1, 1)],
#                   [(2, 2), (2, 2, 2)]]
element = [([1, 2, 3], [10]),
           ([4, 5], [11, 12])]
print(element)
dataset = tf.data.Dataset.from_generator(lambda: iter(element), (tf.int32, tf.int32))
print(dataset)
dataset = dataset.padded_batch(batch_size=2, padded_shapes=([4], [None]))
print(list(dataset.as_numpy_iterator()))
