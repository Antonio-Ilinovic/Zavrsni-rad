import os

import numpy as np
import skimage
import torch
from matplotlib import image as mpimg

import config


def get_image_file_name_from_index(image_index):
    # za zadani index dobivamo ime fajla. Npr za index 10, ime fajla je '000010_10.png'
    # Popunjavaju se nule, tako da bude 6 znamenki u prefiksu.
    return str.zfill(str(image_index), 6) + "_10.png"


def get_disp_image(image_index):
    # metoda vraća mapu točnih dispariteta sa zadanim indexom, dimenzija (HxW)
    return skimage.util.img_as_ubyte(mpimg.imread(config.DISPARITY_IMAGE_ROOT_PATH + get_image_file_name_from_index(image_index)))


def get_left_image(image_index):
    # metoda vraća lijevu sliku sa zadanim indexom, dimenzija (HxWxC)
    return mpimg.imread(config.LEFT_IMAGE_ROOT_PATH + get_image_file_name_from_index(image_index))


def get_right_image(image_index, right_root='D:/ZAVRSNI/Kodovi/Kitti/stereo/image_3/'):
    # metoda vraća desnu sliku sa zadanim indexom, dimenzija (HxWxC)
    return mpimg.imread(config.RIGHT_IMAGE_ROOT_PATH + get_image_file_name_from_index(image_index))


def load_train_disparity_data():
    # učitavanje podataka za učenje. Učitava se ndarray 'disp_data.npy'
    return np.load(config.DISPARITY_DATA_PATH)


def load_model(path=config.TRAINED_MODEL_PATH):
    model = torch.load(path)
    model.eval()
    return model
