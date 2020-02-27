import tensorflow as tf
import matplotlib.pyplot as plt

W = tf.ones([2, 2])
eigenvalues = tf.linalg.eigh(W)[0]  # calculate the eigenvalues（特征值）
print(eigenvalues)

val = [W]
for i in range(10):  # 矩阵相乘 n 次方
    val.append([val[-1] @ W])
# 计算L2范数
norm = list(map(lambda x: tf.norm(x).numpy(), val))
print(norm)  # gradient become very huge

W = tf.ones([2, 2]) * 0.4  # 任意创建某矩阵
eigenvalues = tf.linalg.eigh(W)[0]  # 计算特征值
print(eigenvalues)
val = [W]
for i in range(10):
    val.append([val[-1] @ W])
norm = list(map(lambda x: tf.norm(x).numpy(), val))  # gradient become very small when the gradient is smaller than 1
print(norm)

a=tf.random.uniform([2,2])
print(tf.clip_by_value(a, 0.4, 0.6))
