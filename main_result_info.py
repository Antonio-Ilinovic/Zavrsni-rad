import numpy as np
import torch
from matplotlib import pyplot as plt

import Evaluate
import Utils
from Accuracy import plot_accuracy_image_with_image_index, calculate_accuracy_percentage


model = torch.load('D:/ZAVRSNI/Zavrsni-rad/trained_model_13.pth')
num_image = 199


def show_end_result_of_trained_model(model, num_image):
    # train_accuracy = calculate_accuracy_percentage(model, train=True)
    # test_accuracy = calculate_accuracy_percentage(model, train=False)
    # print(f"train accuracy = {train_accuracy}\ntest accuracy = {test_accuracy}")

    real_disparity = Utils.get_disp_image(num_image)
    plt.subplot(2, 1, 1)
    plt.imshow(real_disparity)
    predicted_disparity = Evaluate.predict_disparity_map(num_image, model)
    plt.subplot(2, 1, 2)
    plt.imshow(predicted_disparity)
    plt.show()

    plot_accuracy_image_with_image_index(num_image, model)
    plt.show()

    plt.subplot(2, 1, 1)
    plt.imshow(Utils.get_left_image(num_image))
    plt.subplot(2, 1, 2)
    plt.imshow(Utils.get_right_image(num_image))
    plt.show()

    train_loss = np.load("train_loss_per_epoch.npy")
    validation_loss = np.load("validation_loss_per_epoch.npy")
    plt.plot(train_loss)
    plt.plot(validation_loss)
    plt.show()


show_end_result_of_trained_model(model, num_image)
