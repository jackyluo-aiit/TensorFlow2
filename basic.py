import tensorflow as tf
import numpy as np

# """
# 基础运算
# """
# # 定义一个随机数（标量）
# random_float = tf.random.uniform(shape=())
#
# # 定义一个有2个元素的零向量
# zero_vector = tf.zeros(shape=(2))
#
# # 定义两个2×2的常量矩阵
# A = tf.constant([[1., 2.], [3., 4.]])
# B = tf.constant([[5., 6.], [7., 8.]])
#
# C = tf.add(A, B)  # 计算矩阵A和B的和
# D = tf.matmul(A, B)  # 计算矩阵A和B的乘积
#
# # 查看矩阵A的形状、类型和值
# print(A.shape)  # 输出(2, 2)，即矩阵的长和宽均为2
# print(A.dtype)  # 输出<dtype: 'float32'>
# print(A.numpy())
#
# print(C.numpy())
# print(D)
#
# """
# 自动求导机制
# """
# x = tf.Variable(initial_value=3.)  # initial a var, 3.
# with tf.GradientTape() as tape:
#     y = tf.square(x)  # target function
# y_grad = tape.gradient(y, x)
# print([y.numpy(), y_grad.numpy()])
#
#
# def loss_func(w, b, X, Y):
#     n = X.shape[0]
#     loss = tf.reduce_mean(tf.square(tf.matmul(X, w) + b - Y), axis=0)
#     return loss
#
#
# X = tf.constant([[1., 2.], [3., 4.]])
# Y = tf.constant([[1.], [2.]])
# w = tf.Variable(initial_value=[[1.], [2.]])
# b = tf.Variable(initial_value=[[1.], [1.]])
# with tf.GradientTape() as tape:
#     L = loss_func(w, b, X, Y)
# w_grad, b_grad = tape.gradient(L, [w, b])
# print([L.numpy(), w_grad.numpy(), b_grad.numpy()])
#
"""
基础线性回归
"""
print('basic linear logistic:')
X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

# 归一化操作
X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
# print(X)
# print(y)

X = tf.constant(X)
Y = tf.constant(y)
w = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)


def logistic_Loss(w, b, X, Y):
    loss = tf.reduce_mean(tf.square(X * w + b - Y), axis=0)
    return loss


num_epoch = 1000
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
for e in range(num_epoch):
    with tf.GradientTape() as tape:
        L = logistic_Loss(w, b, X, Y)
    grads = tape.gradient(L, [w, b])
    # print(grads)
    grads_and_var = zip(grads, [w, b])
    optimizer.apply_gradients(grads_and_var)  # require both grads and vars as parameters

print(w.numpy(), b.numpy())
# print(0.5 * w + b)

"""
model construction and training
"""


class LogisticModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.linear_layer = tf.keras.layers.Dense(
            units=1,  # dim of output tensor
            activation=None,
            kernel_initializer=tf.zeros_initializer(),  # shape will be [input_dim, units]
            bias_initializer=tf.zeros_initializer()  # shape will be [units]
        )

    def call(self, input, **kwargs):
        output = self.linear_layer(input)
        return output


if __name__ == '__main__':
    X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
    y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

    # 归一化操作
    X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
    Y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
    X = tf.reshape(X, shape=(X.shape[0], 1))
    Y = tf.reshape(Y, shape=(Y.shape[0], 1))

    linear_model = LogisticModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    num_epoch = 1000
    for e in range(num_epoch):
        with tf.GradientTape() as tape:
            Y_pred = linear_model(X)
            loss = tf.reduce_mean(tf.square(Y_pred - Y), axis=0)
        grads = tape.gradient(loss, linear_model.variables)
        optimizer.apply_gradients(zip(grads, linear_model.variables))
    print(linear_model.variables[0].numpy(), linear_model.variables[1].numpy())
    print(linear_model.variables)
