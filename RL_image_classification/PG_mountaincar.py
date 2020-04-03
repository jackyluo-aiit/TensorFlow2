import sys

import gym, time, random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def inital_demo():
    env = gym.make('MountainCar-v0')  # 加载游戏环境

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


class PGModel(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(PGModel, self).__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=state_dim, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(units=action_dim, activation='softmax')
        ])

    def call(self, inputs):
        x = self.net(inputs)
        return x


ACTION_DIM = 3
STATE_DIM = 2


class PG:
    def __init__(self):
        self.model = PGModel(STATE_DIM, ACTION_DIM)
        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                           optimizer=tf.optimizers.Adam(0.001))

    def choose_action(self, s):
        prob = self.model.predict(np.array([s]))[0]  # 得到第一个prob，因为没有batch
        x = np.random.choice(len(prob), p=prob)
        return x

    def discount_rewards(self, rewards, gamma=0.95):
        """
        discount_reward[i] = reward[i] + gamma * discount_reward[i+1]
        某一步的累加期望等于下一步的累加期望乘衰减系数gamma，加上reward。
        :param rewards:
        :param gamma:
        :return:
        """
        prior = 0
        out = np.zeros_like(rewards)
        for i in reversed(range(len(rewards))):
            prior = prior * gamma + rewards[i]
            out[i] = prior
        return out / np.std(out - np.mean(out))

    def train(self, records):
        s_batch = np.array([record[0] for record in records])
        # action 独热编码处理，方便求动作概率，即 prob_batch
        indices = [record[1] for record in records]
        a_batch = tf.one_hot(indices=indices, depth=3)
        # 假设predict的概率是 [0.3, 0.7]，选择的动作是 [0, 1]
        # 则动作[0, 1]的概率等于 [0, 0.7] = [0.3, 0.7] * [0, 1]
        # prob_batch = model.predict(s_batch) * a_batch
        r_batch = np.array(self.discount_rewards([record[2]+200 for record in records]))

        # 设置参数sample_weight，即可给loss设权重。
        self.model.fit(s_batch, a_batch, sample_weight=r_batch)

    def save(self):
        self.model.save('MountainCar-v0-pg.h5')


# 开始训练：在每一个回合完成后开始训练，储存下每个动作相应的state, action, reword
def training():
    env = gym.make('MountainCar-v0')
    episodes = 3000  # 至多2000次
    score_list = []  # 记录所有分数
    pg = PG()
    for i in range(episodes):
        s = env.reset()
        score = 0
        replay_records = []
        while True:
            a = pg.choose_action(s)
            next_s, r, done, _ = env.step(a)
            replay_records.append((s, a, r))

            score += r
            s = next_s
            if done:
                pg.train(replay_records)
                score_list.append(score)
                print('episode:', i, 'score:', score, 'max:', max(score_list))
                # if i == 1:
                #     plt.plot(pg.discount_rewards([record[2] for record in replay_records]))  # plot the episode vt
                #     plt.xlabel('episode steps')
                #     plt.ylabel('normalized state-action value')
                #     plt.show()
                break
        # 最后10次的平均分大于 195 时，停止并保存模型
        if np.mean(score_list[-10:]) > -100:
            pg.save()
            break
    env.close()


training()
