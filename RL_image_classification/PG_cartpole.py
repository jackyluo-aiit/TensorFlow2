import matplotlib.pyplot as plt
import gym
import numpy as np
from tensorflow.keras import models, layers, optimizers
import tensorflow as tf
import time
import sys

"""
Policy Gradient 网络的输入也是状态(State)，那输出呢？每个动作的概率。
例如 [0.7, 0.3] ，这意味着有70%的几率会选择动作0，30%的几率选择动作1。
一个动作的累加期望很高，自然希望该动作出现的概率变大，这就是学习的目的。
"""

env = gym.make('CartPole-v0')

STATE_DIM, ACTION_DIM = 4, 2
LEARNING_RATE = 0.001
GAMMA = 0.95


class PolicyNet(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.model = models.Sequential([
            layers.Dense(128, input_dim=state_dim, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(action_dim)
        ])

    def call(self, inputs, training=None, mask=None):
        x = self.model(inputs)
        self.x = x
        x = tf.nn.softmax(x, axis=1)
        return x

class PG:
    def __init__(self):
        self.model = PolicyNet(STATE_DIM, ACTION_DIM)
        self.data = []

    def choose_action(self, s):
        prob = self.model.predict(np.array([s]))[0]  # 得到第一个prob，因为没有batch
        return np.random.choice(len(prob), p=prob)

    def put_data(self, item):
        self.data.append(item)

    def action_batch(self):
        indices = [record[1] for record in self.data]
        a_batch = tf.one_hot(indices=indices, depth=ACTION_DIM)
        return a_batch

    def tf_action_batch(self):
        return self.x

    def discount_reward(self, gamma=0.95):  # 衰减reward 通过最后一步奖励反推真实奖励
        rewards = [record[2] for record in self.data]
        out = np.zeros_like(rewards)
        dis_reward = 0
        for i in reversed(range(len(rewards))):
            dis_reward = dis_reward + gamma * rewards[i]  # 前一步的reward等于后一步衰减reward加上即时奖励乘以衰减因子
            out[i] = dis_reward
        return out / np.std(out - np.mean(out))

    def loss(self, label=action_batch, logit=tf_action_batch):
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit)
        return neg_log_prob

    def train(self):
        s_batch = np.array([record[0] for record in self.data])
        a_batch = self.action_batch()
        r_batch = self.discount_reward()
        self.model.compile(loss=tf.losses.categorical_crossentropy, optimizer=tf.optimizers.Adam(LEARNING_RATE))
        self.model.fit(s_batch, a_batch, sample_weight=r_batch)
        self.data = []


pg = PG()
print(pg.choose_action(env.reset()))

# 按照概率选动作


# def discount_rewards(rewards, gamma=0.95):
#     """
#     discount_reward[i] = reward[i] + gamma * discount_reward[i+1]
#     某一步的累加期望等于下一步的累加期望乘衰减系数gamma，加上reward。
#     :param rewards:
#     :param gamma:
#     :return:
#     """
#     prior = 0
#     out = np.zeros_like(rewards)
#     for i in reversed(range(len(rewards))):
#         prior = prior * gamma + rewards[i]
#         out[i] = prior
#     return out / np.std(out - np.mean(out))
#
#
# def train(records):
#     s_batch = np.array([record[0] for record in records])
#     # action 独热编码处理，方便求动作概率，即 prob_batch
#     a_batch = np.array([[1 if record[1] == i else 0 for i in range(ACTION_DIM)]
#                         for record in records])
#     # 假设predict的概率是 [0.3, 0.7]，选择的动作是 [0, 1]
#     # 则动作[0, 1]的概率等于 [0, 0.7] = [0.3, 0.7] * [0, 1]
#     # prob_batch = model.predict(s_batch) * a_batch
#     r_batch = np.array(discount_rewards([record[2] for record in records]))
#
#     # 设置参数sample_weight，即可给loss设权重。
#     model.fit(s_batch, a_batch, sample_weight=r_batch)


# 开始训练：在每一个回合完成后开始训练，储存下每个动作相应的state, action, reword
def training():
    episodes = 2000  # 至多2000次
    score_list = []  # 记录所有分数
    pg = PG()
    for i in range(episodes):
        s = env.reset()
        score = 0
        # replay_records = []
        while True:
            a = pg.choose_action(s)
            next_s, r, done, _ = env.step(a)
            pg.put_data((s, a, r))

            score += r
            s = next_s
            if done:
                pg.train()
                score_list.append(score)
                print('episode:', i, 'score:', score, 'max:', np.mean(score_list))
                break
        # 最后10次的平均分大于 195 时，停止并保存模型
        if np.mean(score_list[-10:]) > 100:
            pg.model.save('CartPole-v0-pg.h5')
            break
    env.close()


def predict():
    saved_model = models.load_model('CartPole-v0-pg.h5')
    env = gym.make("CartPole-v0")

    for i in range(5):
        s = env.reset()
        score = 0
        while True:
            time.sleep(0.01)
            # env.render()
            prob = saved_model.predict(np.array([s]))[0]
            a = np.random.choice(len(prob), p=prob)
            s, r, done, _ = env.step(a)
            score += r
            if done:
                print('using policy gradient, score: ', score)  # 打印分数
                break
    env.close()


