from matplotlib import pyplot as plt
import numpy as np
import ImagePatch
from PatchesDataset import PatchesDataset
from WholeImageDataset import WholeImageDataset
import torch


def sum_of_absolute_differences(left, right):
    return torch.sum(np.abs(np.subtract(left, right)))


def positive_and_negative_similarity_cost(reference, positive, negative):
    positive_cost = sum_of_absolute_differences(reference, positive)
    negative_cost = sum_of_absolute_differences(reference, negative)
    print(f"positive cost = {positive_cost}")
    print(f"negative cost = {negative_cost}")
