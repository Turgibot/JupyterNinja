# The MIT License (MIT)
# Copyright (c) 2018 Guy Tordjman. All Rights Reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Some of this code was adopted from the stanford-tensorflow-tutorials github

import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_images_grid(images, img_shape, given_class, predicted_class=None):
    assert len(images) == 32
    assert len(given_class) == 32

    fig, axes = plt.subplots(4, 8)
    fig.subplots_adjust(hspace=0.5, wspace=0.05, left=0, right=2.3)

    for i, ax in enumerate(axes.flat):
        # Plot each image
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show given and predicted classes if exists.
        if predicted_class is None:
            xlabel = "Class: {0}".format(given_class[i])
            ax.set_xlabel(xlabel)

        else:
            xlabel = "Class: {0}, Predicted: {1}".format(given_class[i], predicted_class[i])
            if given_class[i] == predicted_class[i]:
                ax.set_xlabel(xlabel, color='green')
            else:
                ax.set_xlabel(xlabel)
                ax.set_xlabel(xlabel, color='red')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

    
def create_directory(dir_path):
    """ Create a directory but only if it doesnt exist. """
    try:
        os.mkdir(dir_path)
    except OSError:
        pass


def plot_example_errors(cls_pred, correct, data_test_images, data_test_classes):

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data_test_images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data_test_classes[incorrect]

    # Plot the first 9 images.
    plot_images_grid(images=images[0:32],
                     img_shape=(28, 28),
                     given_class=cls_true[0:32],
                     predicted_class=cls_pred[0:32])


def plot_confusion_matrix(cls_pred , data_test_cls, num_classes):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data_test_cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
