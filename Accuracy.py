import numpy as np
import torch
from matplotlib import pyplot as plt, colors

import Evaluate
import Utils
import config
import hyperparameters


def calculate_accuracy_percentage(model, train=False):
    def __number_of_correct_disparities_and_number_of_total_disparities(
            real_disparity, predicted_disparity, allowed_pixel_error=hyperparameters.ALLOWED_PIXEL_ERROR):
        """
        Method returns number of correct in predicted disparity as it should be in real disparity.
        Also returns total number of nonzero disparities in real disparity. Used to calculate accuracy over
        all images for test.
        :param real_disparity:
        :param predicted_disparity:
        :param allowed_pixel_error: KITTI test allows disparity error of 3 pixels
        :return:
        """
        count_correct_disparities = np.count_nonzero((real_disparity != 0) & (np.abs(real_disparity - predicted_disparity) < allowed_pixel_error))
        count_non_zero_disparities = np.count_nonzero(real_disparity)

        return count_correct_disparities, count_non_zero_disparities

    # Iterate over all images that are reserved for test. Accumulate number of correct and total disparities.
    sum_correct_disparities = 0
    sum_total_disparities = 0
    for image_index in range(config.TRAIN_IMAGES_START_INDEX if train else config.VALIDATION_IMAGES_START_INDEX,
                             config.TRAIN_IMAGES_END_INDEX if train else config.VALIDATION_IMAGES_END_INDEX):

        real_disparity = Utils.get_disp_image(image_index)
        predicted_disparity = Evaluate.predict_disparity_map(image_index, model)

        count_correct_disparities, count_total_disparities = \
            __number_of_correct_disparities_and_number_of_total_disparities(real_disparity, predicted_disparity)
        sum_correct_disparities += count_correct_disparities
        sum_total_disparities += count_total_disparities

    # Return accumulated ratio of correct over total number of disparities (this is the end accuracy)
    return sum_correct_disparities / sum_total_disparities


def plot_accuracy_image_with_image_index(image_index, model):
    real_disparity = Utils.get_disp_image(image_index)
    predicted_disparity = Evaluate.predict_disparity_map(image_index, model)
    accuracy_image = __calculate_accuracy_image(real_disparity, predicted_disparity)
    __plot_accuracy_image(accuracy_image)


def __calculate_accuracy_image(real_disparity, predicted_disparity, allowed_pixel_error=hyperparameters.ALLOWED_PIXEL_ERROR):
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
