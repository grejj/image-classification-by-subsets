# get plotting functions
import plot

# tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

# helper libraries
import numpy as np
import matplotlib.pyplot as plt

# convert dictionary integers to words
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# load movie review from imdb database using keras
# keep only 10,000 most recent words in training data
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# the first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2 # unknown
word_index["<UNUSED>"] = 3
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

# print review as text
#print(decode_review(train_data[0]))

# movie reviews must be the same length
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# build the model
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
#print(model.summary())

# add loss function and optimizer
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

# train the model
history = model.fit(train_data, train_labels, epochs=5)

# evaluate the model
results = model.evaluate(test_data, test_labels)
#print(results)

# create graph of accuracy and loss over time
history_dict = history.history # contains metrics of what happened during fit
history_dict.keys()
acc = history_dict['acc']
loss = history_dict['loss']
epochs = range(1, len(acc) + 1)
fig, (loss_plot, acc_plot) = plt.subplots(2)

# loss over time
loss_plot.plot(epochs, loss, 'b', label='Training loss')
loss_plot.set_title('Training loss')

# accuracy over time
acc_plot.plot(epochs, acc, 'b', label='Training accuracy')
acc_plot.set_title('Training accuracy')

plt.show()
