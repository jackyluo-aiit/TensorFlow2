import time

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
import pandas as pd

path = "/sentiment labelled sentences"
filepath_dict = {'yelp': 'sentiment labelled sentences/yelp_labelled.txt',
                 'amazon': 'sentiment labelled sentences/amazon_cells_labelled.txt',
                 'imdb': 'sentiment labelled sentences/imdb_labelled.txt'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source
    print(df)
    df_list.append(df)

df = pd.concat(df_list)
print(df.head())
print('Number of sentences: ', len(df))

df_yelp = df[df['source'] == 'yelp']
sentences = df_yelp['sentence'].values
y = df_yelp['label'].values
sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y,
    test_size=0.25,
    random_state=1000)

# tokenized and padding 0 behind
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)
# Adding 1 because of  reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
print('X_train: ', X_train)
print('X_test: ', X_test)

print('Training process...')
model_name = "模型名-{}".format(int(time.time()))
#设定存储位置，每个模型不一样的路径
tensorboard = TensorBoard(log_dir='logs/{}'.format(model_name))

model = Sequential()
model.add(layers.Embedding(vocab_size, 128, input_length=maxlen))
model.add(layers.Conv1D(128, 10, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train,
          epochs=10,
          validation_data=(X_test, y_test),
          batch_size=10,
                    callbacks=[tensorboard])


