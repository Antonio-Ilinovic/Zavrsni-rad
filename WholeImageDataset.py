from torch.utils.data import Dataset
import Utils


class WholeImageDataset(Dataset):

    def __init__(self):
        self.len = 200

    def __getitem__(self, index):
        if index >= self.len:
            raise IndexError

        left_img = Utils.get_left_image(index)
        right_img = Utils.get_right_image(index)
        disp_img = Utils.get_disp_image(index)

        return left_img, right_img, disp_img

    def __len__(self):
        return self.len
