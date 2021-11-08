from torch.utils.data import Dataset

import ImagePatch
import Utils


class PatchesDataset(Dataset):
    def __init__(self):
        self.disp_data = Utils.load_disp_data_of_all_images()
        self.len = self.disp_data.size
        self.left_images = {}
        self.right_images = {}

        for index in range(180):
            self.left_images[index] = Utils.get_left_image(index)
            self.right_images[index] = Utils.get_right_image(index)

    def __len__(self):
        return self.len

    def __getitem__(self, patch_index):
        patch_info = self.disp_data[patch_index]
        image_index = patch_info['image_index']
        left_image = self.left_images[image_index]
        right_image = self.right_images[image_index]
        row = patch_info['row']
        col = patch_info['col']
        col_positive = patch_info['col_positive']
        col_negative = patch_info['col_negative']
        # return reference_patch, positive_patch, negative_patch
        return ImagePatch.extract_reference_positive_negative(
            left_image, right_image, row, col, col_positive, col_negative, channel_in_front=True)
