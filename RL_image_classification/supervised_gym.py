import gym  # 0.12.5
import random
import time
import tensorflow as tf
import numpy as np

"""
state: list：状态，[车位置, 车速度, 杆角度, 杆速度]
action: int：动作(0向左/1向右)
reward: float：奖励(每走一步得1分)
done: bool：是否结束(True/False)，上限200回合
"""


def initial_demo():
    env = gym.make("CartPole-v0")  # 加载游戏环境

    state = env.reset()
    score = 0
    while True:
        time.sleep(0.1)
        env.render()  # 显示画面
        action = random.randint(0, 1)  # 随机选择一个动作 0 或 1
        state, reward, done, _ = env.step(action)  # 执行这个动作
        score += reward  # 每回合的得分
        if done:  # 游戏结束
            print('score: ', score)  # 打印分数
            break
    env.close()


"""
我们的目的就是将随机选择 Action 的部分，变为由神经网络模型来选择。
神经网络的输入是State，输出是Action。
在这里，Action 用独热编码来表示，即 [1, 0] 表示向左，[0, 1] 表示向右。
"""
env = gym.make("CartPole-v0")  # 加载游戏环境

STATE_DIM, ACTION_DIM = 4, 2  # State 维度 4, Action 维度 2
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_dim=STATE_DIM, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(ACTION_DIM, activation='softmax')
])
model.summary()  # 打印神经网络

'''
随机生成训练数据，如果产生的训练数据，state, action所得到的score能达到标准的话就保留
'''


def generate_data_one_episode():
    '''
    生成单次游戏的训练数据，并返回相应的score
    '''
    x, y, score = [], [], 0
    state = env.reset()
    while True:
        action = random.randrange(0, 2)
        x.append(state)
        y.append([1, 0] if action == 0 else [0, 1])  # 记录数据
        state, reward, done, _ = env.step(action)  # 执行动作
        score += reward
        if done:
            break
    return x, y, score


def generate_training_data(expected_score=100):
    '''# 生成N次游戏的训练数据，并进行筛选，选择 > 100 的数据作为训练集'''
    data_X, data_Y, scores = [], [], []
    for i in range(10000):
        x, y, score = generate_data_one_episode()
        if score > expected_score:
            data_X += x
            data_Y += y
            scores.append(score)
    print('dataset size: {}, max score: {}'.format(len(data_X), max(scores)))
    return np.array(data_X), np.array(data_Y)


def training():
    data_X, data_Y = generate_training_data()
    model.compile(loss='mse', optimizer='adam')  # loss：均方差
    model.fit(data_X, data_Y, epochs=5)
    model.save('CartPole-v0-nn.h5')  # 保存模型


def predict():
    saved_model = tf.keras.models.load_model('CartPole-v0-nn.h5')  # 加载模型
    env = gym.make("CartPole-v0")  # 加载游戏环境

    for i in range(5):
        state = env.reset()
        score = 0
        while True:
            time.sleep(0.01)
            # env.render()  # 显示画面
            action = np.argmax(saved_model.predict(np.array([state]))[0])  # 预测动作
            state, reward, done, _ = env.step(action)  # 执行这个动作
            score += reward  # 每回合的得分
            if done:  # 游戏结束
                print('using nn, score: ', score)  # 打印分数
                break
    env.close()


training()
predict()
