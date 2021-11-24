import numpy as np
import torch


def extract_patch(image, row, col, patch_dim=(9, 9)):
    # metoda koja vadi patch dimenzije patch_dim iz predane slike, centrirane u koordinatama (row, col)
    row_from = row - patch_dim[0] // 2
    row_to = row_from + patch_dim[0]
    col_from = col - patch_dim[1] // 2
    col_to = col_from + patch_dim[1]

    patch = torch.tensor(image[row_from:row_to, col_from:col_to, :])
    return patch


def extract_reference_positive_negative(left, right, row, col, col_positive, col_negative, channel_in_front=False):
    # metoda koja vadi 3 patch-a iz predane lijeve i desne slike. Ta 3 patch-a su jedan primjer za učenje.
    # channel_in_front: parametar koji govori trebaju li se channeli slika staviti na prvu dimenziju,
    # jer nn.Conv2D traži da je Channel dimenzija prva.
    anchor_patch = extract_patch(left, row, col)
    positive_patch = extract_patch(right, row, col_positive)
    negative_patch = extract_patch(right, row, col_negative)

    if channel_in_front:
        anchor_patch = anchor_patch.permute(2, 0, 1)
        positive_patch = positive_patch.permute(2, 0, 1)
        negative_patch = negative_patch.permute(2, 0, 1)

    return anchor_patch, positive_patch, negative_patch
