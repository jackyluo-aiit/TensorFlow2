from tensorflow import keras, optimizers, losses
import tensorflow as tf
import os
import numpy as np
import time
import pprint
from collections import OrderedDict
import sys


class StateSpace:
    def __init__(self):
        self.states = OrderedDict()
        self.state_count = 0

    def add_state(self, name, values):
        """
        store the hyper-param name into state and return the ID of it
        :param name: hyper-param name
        :param values: some valid value of this hyper-param
        :return:
        """
        index_map = {}
        for i, val in enumerate(values):
            index_map[i] = val

        value_map = {}
        for i, val in enumerate(values):
            value_map[val] = i

        metadata = {
            'id': self.state_count,
            'name': name,
            'values': values,
            'size': len(values),
            'index_map_': index_map,
            'value_map_': value_map,
        }
        self.states[self.state_count] = metadata
        self.state_count += 1

        return self.state_count - 1

    def embedding_encode(self, id, value):
        """
        Embedding index encode the specific state value
        Args:
            id: global id of the state
            value: state value
        Returns:
            embedding encoded representation of the state value
        """
        state = self[id]
        size = state['size']
        value_map = state['value_map_']
        value_idx = value_map[value]

        one_hot = np.zeros((1, size), dtype=np.float32)
        one_hot[np.arange(1), value_idx] = value_idx + 1
        return one_hot

    def get_state_value(self, id, index):
        """
        Retrieves the state value from the state value ID
        Args:
            id: global id of the state
            index: index of the state value (usually from argmax)
        Returns:
            The actual state value at given value index
        """
        state = self[id]
        index_map = state['index_map_']

        if (type(index) == list or type(index) == np.ndarray) and len(index) == 1:
            index = index[0]

        value = index_map[index]
        return value

    def get_random_state_space(self, num_layers):
        """
        Constructs a random initial state space for feeding as an initial value
        to the Controller RNN
        Args:
            num_layers: number of layers to duplicate the search space
        Returns:
            A list of one hot encoded states
        """
        states = []

        for id in range(self.size * num_layers):
            state = self[id]
            size = state['size']

            sample = np.random.choice(size, size=1)
            value = state['index_map_'][sample[0]]
            state = self.embedding_encode(id, value)
            states.append(state)
        return states

    def parse_state_space_list(self, state_list):
        """
        Parses a list of one hot encoded states to retrieve a list of state values
        Args:
            state_list: list of one hot encoded states
        Returns:
            list of state values
        """
        state_values = []
        for id, state_one_hot in enumerate(state_list):
            state_val_idx = np.argmax(state_one_hot, axis=-1)[0]
            value = self.get_state_value(id, state_val_idx)
            state_values.append(value)

        return state_values

    def print_state_space(self):
        """ Pretty print the state space """
        print('*' * 40, 'STATE SPACE', '*' * 40)

        pp = pprint.PrettyPrinter(indent=2, width=100)
        for id, state in self.states.items():
            pp.pprint(state)
            print()

    def print_actions(self, actions):
        """ Print the action space properly """
        print('Actions :')

        for id, action in enumerate(actions):
            if id % self.size == 0:
                print("*" * 20, "Layer %d" % (((id + 1) // self.size) + 1), "*" * 20)

            state = self[id]
            name = state['name']
            vals = [(n, p) for n, p in zip(state['values'], *action)]
            print("%s : " % name, vals)
        print()

    def __getitem__(self, id):
        return self.states[id % self.size]

    @property
    def size(self):
        return self.state_count


class PolicyController(tf.keras.Model):
    def __init__(self, num_layers, state_space, controller_cell_units, embedding_dim=20):
        super(PolicyController, self).__init__()
        self.num_layers = num_layers
        self.state_space = state_space
        self.state_size = self.state_space.size
        self.controller_cell_units = controller_cell_units
        self.embedding_dim = embedding_dim
        self.nas_cell = keras.layers.LSTMCell(self.controller_cell_units)
        self.cell_state = self.nas_cell.get_initial_state(batch_size=1, dtype=tf.float32)
        self.rnn_layers = {}
        self.cell_outputs = []
        self.policy_classifiers = []
        self.policy_actions = []
        self.embedding_weights = []
        for i in range(self.state_size * self.num_layers):
            state_space = self.state_space[i]
            model = keras.layers.RNN(cell=self.nas_cell,
                                     dtype=tf.float32,
                                     return_state=True)
            self.rnn_layers[i] = model
        for i in range(self.state_size):
            state_ = self.state_space[i]
            print(i, ": ", state_)
            size = state_['size']

            # size + 1 is used so that 0th index is never updated and is "default" value
            weights = tf.random.uniform(shape=[size + 1, self.embedding_dim], minval=-1., maxval=1.)
            print(i, ": ", weights)
            self.embedding_weights.append(weights)

    def call(self, inputs, training=True, mask=None):
        pred_actions = []
        # embedding_weights = []
        state_input = inputs
        # print("state_input: ", state_input)
        print('+' * 20, "States space size:", self.state_size, '+' * 20)
        # for each possible state, create a new embedding. Reuse the weights for multiple layers.
        # for i in range(self.state_size):
        #     state_ = self.state_space[i]
        #     print(i, ": ", state_)
        #     size = state_['size']
        #
        #     # size + 1 is used so that 0th index is never updated and is "default" value
        #     weights = tf.random.uniform(shape=[size + 1, self.embedding_dim], minval=-1., maxval=1.)
        #     print(i, ": ", weights)
        #     embedding_weights.append(weights)

        # initially, cell input will be 1st state input
        # this step is to map each state_input like [[1 0]], to the embeddings
        # print("state_input2: ",state_input)  # state_input:  tf.Tensor([[0 2]], shape=(1, 2), dtype=int32)
        embeddings = tf.nn.embedding_lookup(self.embedding_weights[0], state_input)
        cell_input = embeddings
        # print("cell input: ", cell_input)
        for i in range(self.state_size * self.num_layers):
            state_id = i % self.state_size
            state_space = self.state_space[i]
            size = state_space['size']
            # print(self.cell_state.shape)
            outputs, h, final_state = self.rnn_layers[i](cell_input)
            classifier = keras.layers.Dense(units=size)(outputs)
            preds = tf.keras.activations.softmax(classifier, axis=-1)
            cell_input = tf.argmax(preds, axis=-1)
            # print("cell_input1: ", cell_input)
            cell_input = tf.expand_dims(cell_input, -1)
            # print("cell_input2: ", cell_input)
            cell_input = tf.cast(cell_input, tf.int32)
            # print("cell_input3: ", cell_input)
            cell_input = tf.add(cell_input, 1)  # we avoid using 0 so as to have a "default" embedding at 0th index
            # print("cell_input4: ", cell_input)
            cell_input = tf.nn.embedding_lookup(self.embedding_weights[state_id], cell_input)
            # print("cell_input: ", cell_input)
            if training:
                self.cell_state = final_state
                # print(cell_input)
                self.cell_outputs.append(cell_input)
                self.policy_classifiers.append(classifier)
                self.policy_actions.append(preds)
            else:
                pred_actions.append(preds)
        print('+' * 20, "Predicted actions shape: ", np.shape(pred_actions), '+' * 20)
        return pred_actions


class PolicyNet():
    def __init__(self, num_layers, state_space, policy_controller,
                 reg_param=0.001,
                 discount_factor=0.99,
                 exploration=0.8,
                 ):
        self.num_layers = num_layers
        self.state_space = state_space
        self.policy_controller = policy_controller
        self.reg_strength = reg_param
        self.discount_factor = discount_factor
        self.exploration = exploration
        self.state_size = self.state_space.size
        self.reward_buffer = []
        self.state_buffer = []
        self.labels = []
        self.global_step = tf.Variable(0)
        starter_learning_rate = 0.1
        learning_rate = keras.optimizers.schedules.ExponentialDecay(
            starter_learning_rate,
            decay_steps=50,
            decay_rate=0.95,
            staircase=True)
        self.optimizer = optimizers.Adam(learning_rate=learning_rate)

    def get_action(self, state):
        if np.random.random() < self.exploration:
            print('+' * 20, "Generating random action to explore", '+' * 20)
            actions = []

            for i in range(self.state_size * self.num_layers):
                state_ = self.state_space[i]
                size = state_['size']

                sample = np.random.choice(size, size=1)
                sample = state_['index_map_'][sample[0]]
                action = self.state_space.embedding_encode(i, sample)
                actions.append(action)
            return actions

        else:
            print('+' * 20, "Prediction action from Policy Net", '+' * 20)
            initial_state = self.state_space[0]
            size = initial_state['size']
            pred_action_list = []

            if state[0].shape != (1, size):
                state = state[0].reshape((1, size)).astype('int32')
            else:
                state = state[0].astype('int32')
            print(state.shape)
            pred_action = self.policy_controller(state, training=False)
            print('+' * 20, "State input to Policy Net for predicting action : ", state.flatten(), '+' * 20)
            temp_list = self.state_space.parse_state_space_list(pred_action)
            for id, state_value in enumerate(temp_list):
                state_one_hot = self.state_space.embedding_encode(id, state_value)
                pred_action_list.append(state_one_hot)
            return pred_action_list

    def loss(self):
        cross_entropy_loss = 0
        print('+' * 20, "Logits: ", self.policy_controller.policy_classifiers, '+' * 20)
        print('+' * 20, "Labels: ", self.labels, '+' * 20)
        for i in range(self.state_size * self.num_layers):
            classifier = self.policy_controller.policy_classifiers[i]
            state_space = self.state_space[i]
            size = state_space['size']
            ce_loss = tf.nn.softmax_cross_entropy_with_logits(logits=classifier, labels=self.labels[i])
            cross_entropy_loss += ce_loss
        policy_gradient_loss = tf.reduce_mean(cross_entropy_loss)
        reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in self.policy_controller.trainable_variables])
        total_loss = policy_gradient_loss + self.reg_strength * reg_loss
        return total_loss

    # def build_net(self):
    #     embedding_weights = []
    #     state_input = self.state_input
    #     print("state_input: ", state_input)
    #     print("state_input_size: ", state_input.size)
    #     print("state_space_size:", self.state_size)
    #     # for each possible state, create a new embedding. Reuse the weights for multiple layers.
    #     for i in range(self.state_size):
    #         state_ = self.state_space[i]
    #         size = state_['size']
    #
    #         # size + 1 is used so that 0th index is never updated and is "default" value
    #         weights = tf.random.uniform(shape=[size + 1, self.embedding_dim], minval=-1., maxval=1.)
    #         print(i, ": ", weights)
    #         embedding_weights.append(weights)
    #
    #     # initially, cell input will be 1st state input
    #     # this step is to map each state_input like [[1 0]], to the embeddings
    #     embeddings = tf.nn.embedding_lookup(embedding_weights[0], state_input)
    #
    #     cell_input = embeddings
    #     print("cell input: ", cell_input)
    #
    #     for i in range(self.state_size * self.num_layers):
    #         state_id = i % self.state_size
    #         state_space = self.state_space[i]
    #         size = state_space['size']
    #         print("state_id: ", state_id, "size: ", size)

    def store_rollout(self, state, reward):
        self.reward_buffer.append(reward)
        self.state_buffer.append(state)

        # dump buffers to file if it grows larger than 50 items
        if len(self.reward_buffer) > 20:
            with open('buffers.txt', mode='a+') as f:
                for i in range(20):
                    state_ = self.state_buffer[i]
                    state_list = self.state_space.parse_state_space_list(state_)
                    state_list = ','.join(str(v) for v in state_list)

                    f.write("%0.4f,%s\n" % (self.reward_buffer[i], state_list))

                print('+' * 20, "Saved buffers to file `buffers.txt` !", '+' * 20)

            self.reward_buffer = [self.reward_buffer[-1]]
            self.state_buffer = [self.state_buffer[-1]]
            print('+' * 20, "self.state_buffer[-1]: ", self.state_buffer[-1], '+' * 20)

    def discount_rewards(self):
        """
        Compute discounted rewards over the entire reward buffer

        Returns:
            Discounted reward value
        """
        rewards = np.asarray(self.reward_buffer)
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards[-1]

    def train_step(self, tape):
        print('*' * 40, "Train Controller begin:", '*' * 40)
        states = self.state_buffer[-1]
        print('+' * 20, "States: ", states, '+' * 20)
        label_list = []
        starter_learning_rate = 0.1
        # learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, self.global_step,
        #                                                      500, 0.95, staircase=True)
        # learning_rate = keras.optimizers.schedules.ExponentialDecay(
        #     starter_learning_rate,
        #     decay_steps=self.global_step / 500,
        #     decay_rate=0.95,
        #     staircase=True)
        # self.optimizer = optimizers.Adam(learning_rate=learning_rate)

        # parse the state space to get real value of the states,
        # then one hot encode them for comparison with the predictions
        state_list = self.state_space.parse_state_space_list(states)
        for id, state_value in enumerate(state_list):
            state_one_hot = self.state_space.embedding_encode(id, state_value)
            label_list.append(state_one_hot)
        self.labels = label_list
        print('+' * 20, "Label_list: ", label_list, '+' * 20)
        state_input_size = self.state_space[0]['size']
        print('+' * 20, "States[0].shape: ", states[0].shape, '+' * 20)
        # state_input = states[0].reshape((1, state_input_size)).astype('int32')
        try:
            state_input = states[0].reshape((1, state_input_size)).astype('int32')
        except AttributeError:
            state_input = states[0].numpy().reshape((1, state_input_size)).astype('int32')
        print('+' * 20, "State input to Controller for training : ", state_input.flatten(), '+' * 20)
        reward = self.discount_rewards()
        reward = np.asarray([reward]).astype('float32')
        self.state_input = state_input
        self.policy_controller(state_input)
        loss = self.loss() * reward
        with tape.stop_recording():
            grads = tape.gradient(loss, self.policy_controller.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.policy_controller.trainable_variables))
        self.global_step = self.optimizer.iterations.numpy()
        print(self.optimizer._decayed_lr(tf.float32))
        print('+' * 20, "Training Controller on: ", state_list, '+' * 20)
        print('+' * 20, "Training Controller reward: ", reward.flatten(), '+' * 20)
        if self.global_step != 0 and self.global_step % 20 == 0 and self.exploration > 0.5:
            self.exploration *= 0.99

        print('+' * 20, "Global step: ", self.global_step, '+' * 20)
        return loss


if __name__ == '__main__':
    state = StateSpace()
    state.add_state(name='kernel', values=[1, 3])
    state.add_state(name='filters', values=[16, 32, 64])
    state.print_state_space()

    pc = PolicyController(2, state, 32)

    state_list = state.get_random_state_space(2)
    print("Testing state:", state_list)
    print("Testing actual state:", state.parse_state_space_list(state_list))
    pn = PolicyNet(2, state, pc)
    action = pn.get_action(state_list)
    state.print_actions(action)
    print("Predicted actions : ", state.parse_state_space_list(action))
    print("action: ", action)
    pn.store_rollout(action, 1)
    with tf.GradientTape() as tape:
        loss = pn.train_step(tape)
    print(loss)
