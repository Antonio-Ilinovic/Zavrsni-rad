import numpy as np
from matplotlib import pyplot as plt

import ImagePatch
from ImagePatch import extract_patch
from WholeImageDataset import WholeImageDataset


def test_whole_image_dataset():
    image_dataset = WholeImageDataset()
    left_0, right_0, disp_0 = image_dataset[0]
    plt.subplots(3, 1)
    plt.subplot(3, 1, 1)
    plt.imshow(left_0)
    plt.subplot(3, 1, 2)
    plt.imshow(right_0)
    plt.subplot(3, 1, 3)
    plt.imshow(disp_0)
    plt.show()


def test_extract_image():
    image_dataset = WholeImageDataset()
    left, right, disp = image_dataset[0]
    image_patch = extract_patch(left, 205, 950)
    plt.imshow(image_patch)
    plt.show()


def test_extract_patches(row=205, col=950):
    image_dataset = WholeImageDataset()
    left, right, disp = image_dataset[0]
    disp_at = disp[row][col]
    reference_patch, positive_patch, negative_patch = ImagePatch.extract_reference_positive_negative(left, right, row, col, disp_at)

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.title.set_text("Reference Patch")
    ax2.title.set_text("Positive Patch")
    ax3.title.set_text("Negative Patch")
    ax1.imshow(reference_patch)
    ax2.imshow(positive_patch)
    ax3.imshow(negative_patch)
    plt.show()
