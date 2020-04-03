import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint


class NetworkManager:
    def __init__(self, x_train, y_train, x_valid, y_valid, x_test, y_test, input_vocab, target_vocab, model_path,
                 learning_rate=0.001, epochs=5, child_batch_size=128, acc_beta=0.8, clip_rewards=0.0):
        """

        :param dataset: a tuple of 4 arrays (X_train, y_train, X_val, y_val)
        :param epochs: number of epochs to train the subnetworks
        :param child_batch_size: batchsize of training the subnetworks
        :param acc_beta: exponential weight for the accuracy
        :param clip_rewards: float - to clip rewards in [-range, range] to prevent large weight updates.
                Use when training is highly unstable.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test
        self.model_path = model_path
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = child_batch_size
        self.beta = acc_beta
        self.beta_bias = acc_beta
        self.moving_acc = 0.0
        self.clip_rewards = clip_rewards

    def get_rewards(self, model, actions):
        """
        creates a child-network given the actions from controller RNN
        trains it on the provided dataset, and then return a reward
        :param model: the child network model
        :param actions: a list of parsed actions/states from StateSpace
        :return: a reward for training a model with given actions
        """
        embed_depth, state_len, dense_units = actions
        run_id = f"cp-{embed_depth}embed_depth_{state_len}state_len_{dense_units}dense_units"
        checkpoint_path = os.path.join(self.model_path, run_id)
        model = model(actions, self.x_train, self.y_train, self.input_vocab, self.target_vocab)
        ckpt = tf.train.Checkpoint(mymodel=model)
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=tf.keras.optimizers.Adam(self.learning_rate),
                      metrics=['accuracy'])
        if ckpt_manager.latest_checkpoint:
            model.load_weights(checkpoint_path)
            print('%' * 20, 'Find trained child model ', run_id, '%' * 20)
        else:
            print('%' * 20, 'Can not find trained child model ', run_id, ', begin training', '%' * 20)
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    save_best_only=True,
                    monitor='val_accuracy',
                    verbose=1)
            ]
            model.fit(self.x_train, self.y_train, validation_data=(self.x_valid, self.y_valid),
                      batch_size=self.batch_size, epochs=self.epochs, callbacks=callbacks)
            model.load_weights(checkpoint_path)
        loss, accuracy = model.evaluate(self.x_test, self.y_test)
        # compute the reward
        reward = (accuracy - self.moving_acc)

        # if rewards are clipped, clip them in the range -0.05 to 0.05
        if self.clip_rewards:
            reward = np.clip(reward, -0.05, 0.05)

        # update moving accuracy with bias correction for 1st update
        if 0.0 < self.beta < 1.0:
            self.moving_acc = self.beta * self.moving_acc + (1 - self.beta) * accuracy
            self.moving_acc = self.moving_acc / (1 - self.beta_bias)
            self.beta_bias = 0

            reward = np.clip(reward, -0.1, 0.1)

        print()
        print('%' * 20, "Manager: EWA Accuracy = ", self.moving_acc, '%' * 20)
        return reward, accuracy

