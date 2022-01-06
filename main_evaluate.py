import numpy as np
import torch
from matplotlib import pyplot as plt

import Evaluate
import Utils
from Accuracy import plot_accuracy_image_with_image_index, calculate_accuracy_percentage


def evaluate_model(model):
    train_accuracy = calculate_accuracy_percentage(model, train=True)
    test_accuracy = calculate_accuracy_percentage(model, train=False)
    print(f"train accuracy = {train_accuracy}\ntest accuracy = {test_accuracy}")
    with open("model_evaluation.txt", "w") as f:
        f.write(f"train accuracy = {train_accuracy}\ntest accuracy = {test_accuracy}")


def evaluate_image_with_model(model, num_image):
    real_disparity = Utils.get_disp_image(num_image)
    plt.subplot(2, 1, 1)
    plt.imshow(real_disparity)
    plt.title(f'real disparity {num_image}')
    predicted_disparity = Evaluate.predict_disparity_map(num_image, model)
    plt.subplot(2, 1, 2)
    plt.imshow(predicted_disparity)
    plt.title(f'predicted disparity {num_image}')
    plt.show()

    plot_accuracy_image_with_image_index(num_image, model)
    plt.title(f"accuracy image {num_image}")
    plt.show()

    plt.subplot(2, 1, 1)
    plt.imshow(Utils.get_left_image(num_image))
    plt.title(f"left image {num_image}")
    plt.subplot(2, 1, 2)
    plt.imshow(Utils.get_right_image(num_image))
    plt.title(f"right image {num_image}")
    plt.show()

    train_loss = np.load("train_loss_per_epoch.npy")
    validation_loss = np.load("validation_loss_per_epoch.npy")
    plt.plot(train_loss, label='train loss')
    plt.plot(validation_loss, label='validation loss')
    plt.legend()
    plt.show()


def plot_train_and_validation_loss_per_epoch(train_loss, validation_loss):
    pass


if __name__ == '__main__':
    model = torch.load('trained_model_13.pth')
    num_image = 199
    #evaluate_image_with_model(model, num_image)
    #evaluate_model(model)
    Evaluate.save_validation_loss_per_epoch()
