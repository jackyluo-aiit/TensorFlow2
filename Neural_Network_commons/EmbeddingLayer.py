import tensorflow as tf

x = tf.range(10)
print(x)
x = tf.random.shuffle(x)
print(x)
net = tf.keras.layers.Embedding(10, 4)  # depth is 4 for every word
out = net(x)  # output a (10, 4) shape of matrix that contains values are random initiated
# Embedding 层实现起来非常简单，构建一个 shape 为[𝑁vocab, 𝑛]的查询表对象 table
# 任意的单词编号𝑖，只需要查询到对应位置上的向量并返回即可: 𝒗 = 𝑡𝑎𝑏𝑙𝑒[𝑖]
print("output from embedding net: ", out)
print("Embedding internal table: ", net.embeddings)

# # 调用训练好的词向量模型
# embed_glove = load_embed('glove.6B.50d.txt')
# net.set_weights(embed_glove)

