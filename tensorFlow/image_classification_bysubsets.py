# tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras
import pickle

# helper libraries
import numpy as np
import matplotlib.pyplot as plt
import hilbertCurve
import cv2
import plot

# names of classes of fashion mnist database
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# load data from fashion mnist database using keras
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_images = train_images / 255
test_images = test_images / 255

subsize = 14

train_images_subsets1 = None
train_images_subsets2 = None
train_images_subsets3 = None
train_images_subsets4 = None

with open("./pickle/image_subsets1.pkl", 'rb') as f:
    train_images_subsets1 = pickle.load(f)
with open("./pickle/image_subsets2.pkl", 'rb') as f:
    train_images_subsets2 = pickle.load(f)
with open("./pickle/image_subsets3.pkl", 'rb') as f:
    train_images_subsets3 = pickle.load(f)
with open("./pickle/image_subsets4.pkl", 'rb') as f:
    train_images_subsets4 = pickle.load(f)


'''# divide image into subsets
hilbert_level = 0
bert = hilbertCurve.hilbertCurve(hilbert_level)
bert.calculatePoints()
curve = bert.getDPoints()
for point in curve:
    point[0] = int(subsize*point[0])
    point[1] = int(subsize*point[1])

train_images_subsets1 = np.empty((60000, subsize, subsize))
train_images_subsets2 = np.empty((60000, subsize, subsize))
train_images_subsets3 = np.empty((60000, subsize, subsize))
train_images_subsets4 = np.empty((60000, subsize, subsize))



for image in range(60000):
    for i in range(subsize):
        for j in range(subsize):
            train_images_subsets1[image][i][j] = train_images[image][i+curve[0][0]][j+curve[0][1]]
            train_images_subsets2[image][i][j] = train_images[image][i+curve[1][0]][j+curve[1][1]]
            train_images_subsets3[image][i][j] = train_images[image][i+curve[2][0]][j+curve[2][1]]
            train_images_subsets4[image][i][j] = train_images[image][i+curve[3][0]][j+curve[3][1]]

with open("./image_subsets4.pkl", 'wb') as f:
    pickle.dump(train_images_subsets1, f)
with open("./image_subsets1.pkl", 'wb') as f:
    pickle.dump(train_images_subsets2, f)
with open("./image_subsets2.pkl", 'wb') as f:
    pickle.dump(train_images_subsets3, f)
with open("./image_subsets3.pkl", 'wb') as f:
    pickle.dump(train_images_subsets4, f)


plt.figure()
plt.imshow(train_images[0])

plt.figure(figsize=(6,3))
plt.subplot(2,2,1)
plt.imshow(train_images_subsets2[0])
plt.subplot(2,2,2)
plt.imshow(train_images_subsets3[0])
plt.subplot(2,2,3)
plt.imshow(train_images_subsets1[0])
plt.subplot(2,2,4)
plt.imshow(train_images_subsets4[0])
plt.show()

'''
train_images_subsets = [train_images_subsets1, train_images_subsets2, train_images_subsets3, train_images_subsets4]
for subset in train_images_subsets:
    subset = subset.reshape((60000, 14, 14, 1))

# model for feature extraction
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(10, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(25, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(50, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(100, (1, 1), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Reshape((1,100), input_shape=(100,)))
model.add(keras.layers.LSTM(257))

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model on train dataset
for subset in train_images_subsets:
    model.fit(subset, , epochs=1)

# now that the model is trained, can make predictions about new images
predictions = model.predict(test_images)

print(model.summary())'''
