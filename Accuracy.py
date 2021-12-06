import numpy as np
from matplotlib import pyplot as plt, colors

import Evaluate
import Utils
import config


def plot_accuracy_image_with_image_index(image_index, model):
    real_disparity = Utils.get_disp_image(image_index)
    predicted_disparity = Evaluate.predict_disparity_map(image_index, model)
    accuracy_image = __calculate_accuracy_image(real_disparity, predicted_disparity)
    __plot_accuracy_image(accuracy_image)


def __calculate_accuracy_image(real_disparity, predicted_disparity, allowed_pixel_error=3):
    """
    :param allowed_pixel_error:
    :param real_disparity: numpy array with correct disparities [HxW]
    :param predicted_disparity: numpy array with predicted disparities [HxW]
    :return: numpy array [HxW], which has 3 possible values inside.
        0 if there is no real disparity at that pixel.
        1 if prediction is correct (predicted disparity is <3px from real disparity).
        -1 if prediction is false.
    """
    accuracy = np.zeros(real_disparity.shape)
    accuracy[real_disparity != 0] = -1
    accuracy[(real_disparity != 0) & (np.abs(real_disparity - predicted_disparity) < allowed_pixel_error)] = 1
    return accuracy


def __plot_accuracy_image(accuracy_image):
    """
    Crta točnost modela. Pikseli se iscrtavaju sa 3 boje (crvena, crna i žuta).
    Crna ako ne postoji disparitet.
    Crvena ako je izračunati disparitet netočan.
    Žuta ako je izračunati disparitet točan.

    :param accuracy_image:
    :return:
    """
    plt.figure(figsize=(20, 10))
    cmap = colors.ListedColormap(['red', 'black', 'yellow'])
    plt.imshow(accuracy_image, cmap=cmap)
