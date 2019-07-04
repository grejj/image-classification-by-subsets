# helper libraries
import numpy as np
import matplotlib.pyplot as plt

def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.imshow(img, cmap=plt.cm.binary)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color=color)

def plot_value_array(i, predictions_array, true_label, class_names):
    predictions_array, true_label = predictions_array[i], true_label[i]
    predicted_label = np.argmax(predictions_array)
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    plt.xticks(np.arange(10), class_names, rotation=90)
    plt.yticks([])
    plt.ylim([0, 1])

    thisplot[predicted_label].set_color('red')
    thisplot[predicted_label].set_color('blue')
