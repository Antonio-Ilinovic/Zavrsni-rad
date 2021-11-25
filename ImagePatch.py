import numpy as np
import torch


def extract_patch(image, row, col, patch_dim=(9, 9)):
    """
    Metoda koja vadi patch dimenzije patch_dim iz predane slike, centrirane u koordinatama (row, col).

    :param image:
    :param row:
    :param col:
    :param patch_dim:
    :return:
    """

    row_from = row - patch_dim[0] // 2
    row_to = row_from + patch_dim[0]
    col_from = col - patch_dim[1] // 2
    col_to = col_from + patch_dim[1]

    patch = torch.tensor(image[row_from:row_to, col_from:col_to, :])
    return patch


def extract_learning_example(left, right, row, col, col_positive, col_negative):
    """
    Metoda koja vadi 3 patch-a iz predane lijeve i desne slike. Ta 3 patch-a su jedan primjer za učenje.

    :param left: left image ndarray[HxWxC]
    :param right: right image ndarray[HxWxC]
    :param row:
    :param col:
    :param col_positive:
    :param col_negative:
    :return: tuple( ndarray[HxWxC], ndarray[HxWxC], ndarray[HxWxC] )
    """

    anchor_patch = extract_patch(left, row, col)
    positive_patch = extract_patch(right, row, col_positive)
    negative_patch = extract_patch(right, row, col_negative)

    return anchor_patch, positive_patch, negative_patch
