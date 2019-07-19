# get plotting functions
import plot

# tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

# helper libraries
import numpy as np
import matplotlib.pyplot as plt
from hilbertcurve.hilbertcurve import HilbertCurve

# names of classes of fashion mnist database
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# load data from fashion mnist database using keras
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale images from 0 to 1
train_images = train_images / 255
test_images = test_images / 255


# display first 25 images in the fashion_mnist database
'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''
# build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model on train dataset
model.fit(train_images, train_labels, epochs=5)

# now that the model is trained, can make predictions about new images
predictions = model.predict(test_images)

print(model.summary())

# print predictions
i = 2 # test image number i
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot.plot_image(i, predictions, test_labels, test_images, class_names)
plt.subplot(1,2,2)
plot.plot_value_array(i, predictions, test_labels, class_names)
plt.show()
