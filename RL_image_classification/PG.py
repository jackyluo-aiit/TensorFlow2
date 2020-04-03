import tensorflow as tf
import numpy as np
from RL_image_classification.train import policy_network


class PG():
    def __init__(self, max_layers, gamma, optimizer):
        self.model = policy_network(max_layers)
        self.gamma = gamma
        self.optimizer = optimizer
        self.state_buffer = []
        self.reward_buffer = []

    def choose_action(self, s):
        action = self.model.predict(s)  # 得到第一个prob，因为没有batch
        return action

    def loss(self):
        states = self.state_buffer[-1:]
        actions = self.model.x
        return tf.nn.softmax_cross_entropy_with_logits(logits=actions, labels=states)

    def put_data(self, state, reward):
        self.state_buffer.append(state)
        self.reward_buffer.append(reward)

    def discount_reward(self):  # 衰减reward 通过最后一步奖励反推真实奖励
        rewards = self.reward_buffer[-1:]
        out = np.zeros_like(rewards)
        dis_reward = 0
        for i in reversed(range(len(rewards))):
            dis_reward = dis_reward + self.gamma * rewards[i]  # 前一步的reward等于后一步衰减reward加上即时奖励乘以衰减因子
            out[i] = dis_reward
        return out / np.std(out - np.mean(out))

    def train_net(self, tape):
        rewards = self.reward_buffer[-1:]
        loss = self.loss()
        pg_loss = tf.reduce_mean(loss)
        reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x) for x in self.model.trainable_variables)])
        loss = pg_loss + 0.001 * reg_loss
        with tape.stop_recording():
            grads = tape.gradient(loss, self.model.trainable_variables)
            for i, (grad, var) in enumerate(grads):
                if grad is not None:
                    grads[i] = (grad * rewards, var)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    # def train(self):
    #     s_batch = np.array([record[0] for record in self.data])
    #     a_batch = np.array([record[1] for record in self.data])
    #     r_batch = self.discount_reward()
    #     self.model.compile(loss=self.loss, optimizer=tf.optimizers.Adam(self.lr))
    #     self.model.fit(s_batch, a_batch, sample_weight=r_batch)
    #     self.data = []


state = np.array([[10.0, 128.0, 1.0, 1.0] * 2], dtype=np.float32)
print(state.shape)
optimizer = tf.optimizers.RMSprop(learning_rate=0.001)
pg = PG(2, 0.95, optimizer)
print(pg.choose_action(state))
model = policy_network(2)
print(model(state))
