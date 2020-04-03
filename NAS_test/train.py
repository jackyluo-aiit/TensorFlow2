import tensorflow as tf
import os
from NAS_test.Controller import StateSpace, PolicyController, PolicyNet
from NAS_test.NetworkManager import NetworkManager
from NAS_test.load_data import load_data
from NAS_test.preprocess import preprocess
from sklearn.model_selection import train_test_split
from NAS_test.utils import remove_history, write_history
from NAS_test.BaseModel import get_model

NUM_LAYERS = 1
MAX_TRIALS = 250
EXPLORATION = 0.8  # high exploration for the first 1000 steps
REGULARIZATION = 1e-3  # regularization strength
CONTROLLER_CELLS = 32  # number of cells in RNN controller
EMBEDDING_DIM = 20  # dimension of the embeddings for each state
ACCURACY_BETA = 0.8  # beta value for the moving average of the accuracy
CLIP_REWARDS = 0.0  # clip rewards in the [-0.05, 0.05] range
RESTORE_CONTROLLER = True  # restore controller to continue training

child_batch_size = 1024
epoch_size = 25
learning_rate = 0.005
# dense_units = 1024
# embed_depth = 100
# state_len = 256

current_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
callback_path = os.path.join(father_path, 'checkpoints')
data_path = os.path.join(father_path, 'dataset/data')
en_dir = os.path.join(data_path, 'small_vocab_en')
fr_dir = os.path.join(data_path, 'small_vocab_fr')
train_history_path = os.path.join(father_path, 'train_history')
history_path = os.path.join(train_history_path, 'train_history.csv')
buffer_path = os.path.join(train_history_path, 'buffers.txt')

previous_acc = 0.0
total_reward = 0.0

state_space = StateSpace()
state_space.add_state(name='embed_depth', values=[128, 256, 512])
state_space.add_state(name='state_len', values=[64, 128, 256, 512])
state_space.add_state(name='dense_units', values=[64, 128, 256, 512, 1024])
state_space.print_state_space()

english_sentences = load_data(en_dir)
french_sentences = load_data(fr_dir)

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = \
    preprocess(english_sentences, french_sentences)

max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)

x_train, x_test, y_train, y_test = train_test_split(
    preproc_english_sentences, preproc_french_sentences,  # x,y是原始数据
    test_size=0.2
)

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train,
    test_size=0.2
)

print("Data split")
print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)
print('x_test: ', x_test.shape)
print('y_test: ', y_test.shape)
print('x_valid: ', x_valid.shape)
print('y_valid: ', y_valid.shape)

pc = PolicyController(NUM_LAYERS, state_space, CONTROLLER_CELLS, EMBEDDING_DIM)
manager = NetworkManager(x_train, y_train, x_valid, y_valid, x_test, y_test,
                         input_vocab=english_tokenizer, target_vocab=french_tokenizer,
                         model_path=callback_path, learning_rate=learning_rate, epochs=epoch_size, child_batch_size=child_batch_size,
                         acc_beta=ACCURACY_BETA, clip_rewards=CLIP_REWARDS)
pn = PolicyNet(NUM_LAYERS, state_space, pc, reg_param=REGULARIZATION, exploration=EXPLORATION)

state = state_space.get_random_state_space(NUM_LAYERS)
print(40 * '*', 'Initial Random State: ', state_space.parse_state_space_list(state), 40 * '*')
print()

remove_history([history_path, buffer_path])

for trial in range(MAX_TRIALS):
    actions = pn.get_action(state)
    print('-' * 20, 'Actions from controller: ', '-' * 20)
    print('-' * 20, "Parsed actions : ", state_space.parse_state_space_list(actions), '-' * 20)
    reward, previous_acc = manager.get_rewards(get_model, state_space.parse_state_space_list(actions))
    print('-' * 20, "Rewards : ", reward, "Accuracy : ", previous_acc, '-' * 20)
    total_reward += reward
    print('-' * 20, "Total reward : ", total_reward, '-' * 20)

    # actions and states are equivalent, save the state and reward
    state = actions
    pn.store_rollout(state, reward)

    # train the controller on the saved state and the discounted rewards
    with tf.GradientTape(persistent=True) as tape:
        loss = pn.train_step(tape=tape)
    print('-' * 20, "Trial %d: Controller loss : %0.6f" % (trial + 1, loss), '-' * 20)
    write_history(history_path, trial, previous_acc, reward, state_space.parse_state_space_list(state))
    print()
print('-' * 20, "Total Reward : ", total_reward, '-' * 20)
