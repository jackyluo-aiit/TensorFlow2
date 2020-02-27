import tensorflow as tf

cell = tf.keras.layers.SimpleRNNCell(3)  # 内存向量长度
cell.build(input_shape=(None, 4))  # input depth of each word
for each in cell.trainable_variables:
    print(each.name, "shape: ", each.shape)
    # output three vecters: kernel is the Wxh, recurrent_kernel is the Whh, bias is bias

h0 = [tf.zeros([4, 64])]  # Memory vector needs the user to specify, packed by a list[]
x = tf.random.normal(shape=[4, 80, 100])  # generate 4 sentence with 80 words, embedded with 100 depth
xt = x[:,0,:]  # extract the first word from each sentence
cell = tf.keras.layers.SimpleRNNCell(64)
out, h1 = cell(xt, h0)
print(out.shape, h1[0].shape)
for each in cell.trainable_variables:
    print(each.name, "shape: ", each.shape)
print("id of out and h1: {}, {}".format(id(out), id(h1)))  # print out their id to see they are the same

h = h0  # initiate the h
for xt in tf.unstack(x, axis=1):  # unstack the x at the dimension 1, which is the length of each sentence
    print(xt.shape)
    out, h = cell(xt, h)  # the xt is now in a shape of [batch, depth]
out = out  # take the last output as the final output, but actually every output from each timestamp could be saved

cell0 = tf.keras.layers.SimpleRNNCell(64)
cell1 = tf.keras.layers.SimpleRNNCell(64)
h0 = tf.zeros(shape=[4, 64])
h1 = tf.zeros(shape=[4, 64])
for xt in tf.unstack(x, axis=1):
    out0, h0 = cell(xt, h0)
    out1, h1 = cell(out0, h1)  # the second layer takes the output of first layer as input

middle_sequences = []
for xt in tf.unstack(x, axis=1):
    out0, h0 = cell(xt, h0)
    middle_sequences.append(out0)  # also can store all the output of first layer into a list first

for xt in middle_sequences:
    out1, h1 = cell(xt, h1)  # then do the second layer computation after.
