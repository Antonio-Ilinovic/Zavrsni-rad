from torch.utils.data import Dataset

import ImagePatch
import Utils
import config


class PatchesDataset(Dataset):
    def __init__(self):
        self.train_disparity_data = Utils.load_train_disparity_data()
        self.len = self.train_disparity_data.size
        self.left_images = {}
        self.right_images = {}

        for index in range(config.NUM_TRAIN_IMAGES):
            self.left_images[index] = Utils.get_left_image(index)
            self.right_images[index] = Utils.get_right_image(index)

    def __len__(self):
        return self.len

    def __getitem__(self, patch_index):
        patch_info = self.train_disparity_data[patch_index]
        image_index = patch_info['image_index']
        row = patch_info['row']
        col = patch_info['col']
        col_positive = patch_info['col_positive']
        col_negative = patch_info['col_negative']

        return ImagePatch.extract_learning_example(self.left_images[image_index], self.right_images[image_index],
                                                   row, col, col_positive, col_negative)
