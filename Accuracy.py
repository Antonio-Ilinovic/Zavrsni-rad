import numpy as np
from matplotlib import pyplot as plt, colors


def calculate_accuracy(real_disp, predicted_disp):
    '''
    :param real_disp: numpy array with correct disparities [HxW]
    :param predicted_disp: numpy array with predicted disparities [HxW]
    :return: numpy array [HxW], which has 3 possible values inside.
        0 if there is no real disparity at that pixel.
        1 if prediction is correct (predicted disparity is <3px from real disparity).
        -1 if prediction is false.
    '''
    accuracy = np.zeros(real_disp.shape)
    accuracy[real_disp != 0] = -1
    accuracy[(real_disp != 0) & (np.abs(real_disp - predicted_disp) < 3)] = 1
    return accuracy


def plot_accuracy(accuracy):
    '''
    Crta točnost modela. Pikseli se iscrtavaju sa 3 boje (crvena, crna i žuta).
    Crna ako ne postoji disparitet.
    Crvena ako je izračunati disparitet netočan.
    Žuta ako je izračunati disparitet točan.

    :param accuracy:
    :return:
    '''
    plt.figure(figsize=(20, 10))
    cmap = colors.ListedColormap(['red', 'black', 'yellow'])
    plt.imshow(accuracy, cmap=cmap)
