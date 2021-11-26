from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import Utils
import config


class PatchesDataset(Dataset):

    def __init__(self, train=True, transform=ToTensor()):
        # Implementation of ToTensor() automatically permutes image from (H x W x C) -> (C x H x W)
        self.transform = transform
        self.disparity_data = Utils.load_disparity_data(train=train)
        self.len = self.disparity_data.size
        self.left_images = {}
        self.right_images = {}

        for index in range(config.TRAIN_IMAGES_START_INDEX if train else config.VALIDATION_IMAGES_START_INDEX,
                           config.TRAIN_IMAGES_END_INDEX if train else config.VALIDATION_IMAGES_END_INDEX):
            self.left_images[index] = Utils.get_left_image(index)
            self.right_images[index] = Utils.get_right_image(index)

    def __len__(self):
        return self.len

    def __getitem__(self, patch_index):
        patch_info = self.disparity_data[patch_index]
        image_index = patch_info['image_index']
        row = patch_info['row']
        col = patch_info['col']
        col_positive = patch_info['col_positive']
        col_negative = patch_info['col_negative']

        anchor_patch = self.extract_patch(self.left_images[image_index], row, col)
        positive_patch = self.extract_patch(self.right_images[image_index], row, col_positive)
        negative_patch = self.extract_patch(self.right_images[image_index], row, col_negative)
        if self.transform:
            anchor_patch = self.transform(anchor_patch)
            positive_patch = self.transform(positive_patch)
            negative_patch = self.transform(negative_patch)

        return anchor_patch, positive_patch, negative_patch


    def extract_patch(self, image, row, col, patch_size=config.PATCH_SIZE):
        """
        Metoda koja vadi patch dimenzije patch_dim iz predane slike, centrirane u koordinatama (row, col).
        """
        row_from = row - patch_size // 2
        row_to = row_from + patch_size
        col_from = col - patch_size // 2
        col_to = col_from + patch_size

        patch = image[row_from:row_to, col_from:col_to, :]
        return patch

