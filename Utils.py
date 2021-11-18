import os

import numpy as np
import skimage
import torch
from matplotlib import image as mpimg


def get_image_file_name_from_index(image_index):
    # za zadani index dobivamo ime fajla. Npr za index 10, ime fajla je '000010_10.png'
    # Popunjavaju se nule, tako da bude 6 znamenki u prefiksu.
    return str.zfill(str(image_index), 6) + "_10.png"


def get_disp_image(image_index, disp_root='Kitti/stereo/disp_noc_0/'):
    # metoda vraća mapu točnih dispariteta sa zadanim indexom, dimenzija (HxW)
    return skimage.util.img_as_ubyte(mpimg.imread(disp_root + get_image_file_name_from_index(image_index)))


def get_left_image(image_index, left_root='Kitti/stereo/image_2/'):
    # metoda vraća lijevu sliku sa zadanim indexom, dimenzija (HxWxC)
    return mpimg.imread(os.path.join(left_root, get_image_file_name_from_index(image_index)))


def get_right_image(image_index, right_root='Kitti/stereo/image_3/'):
    # metoda vraća desnu sliku sa zadanim indexom, dimenzija (HxWxC)
    return mpimg.imread(os.path.join(right_root, get_image_file_name_from_index(image_index)))


def get_filtered_disp(image_index, filter_distance=4):
    # metoda prima index slike. Vraća custom numpy array koji u svojim stupcima ima ove vrijednosti:
    # [image_index, row, col, col_positive, col_negative]
    # dtype krajnjeg arraya: image_index=uint8, row=uint16, row=uint16, col_positive=uint16, col_negative=uint16
    #
    # image_index: index slike, za trening se koriste indeksi od 0 - 179
    # row: redak u kojem se nalazi piksel u referentnoj (lijevoj) slici
    # col: stupac u kojem se nalazi piksel s referentnoj (lijevoj) slici
    # col_positive: stupac u kojem se nalazi piksel s točnim disparitetom, u desnoj slici
    # col_negative: stupac u kojem se nalazi piksel s netočnim disparitetom, u desnoj slici
    #
    # Sve vrijednosti se filtriraju tako da kad se budu vadili patchevi slika da ne bude IndexOutOfBounds exceptiona
    # filter_distance je vrijednost koliko rub slike minimalno mora biti udaljen od centra piksela.
    # U defaultnom slučaju je postavljen na 4, što bi značilo da u obzir dolaze samo pikseli koji su udaljeni
    # 4 ili više piksela od vrha i od dna slike. Analogno se računaju stupci za pozitivne i negativne
    # primjere za učenje. Ti pikseli također moraju biti udaljeni minimalno za defaultnih 4 piksela
    # od lijevog i desnog ruba slike. Oni se također filtriraju.

    disp_image = get_disp_image(image_index)

    # izaberi vrijednosti koje nisu 0
    rows, cols = disp_image.nonzero()
    rows = rows.astype(np.uint16)
    cols = cols.astype(np.uint16)
    disp_values = disp_image[rows, cols]
    # izračunaj stupce za pozitivne primjere -> samo posmakni piksele za točan disparitet
    positive_cols = cols - disp_values
    # izračunaj stupce za negativne primjere -> posmakni piskele za točan disparitet i dodaj neki pomak, da bude netočan disparitet
    negative_cols = positive_cols + np.random.choice([-8, -7, -6, -5, -4, 4, 5, 6, 7, 8],
                                                     size=positive_cols.size).astype(np.uint16)

    # izracunaj filter koje vrijednosti i koordinate dispariteta će se razmatrati
    filter_rows = (rows >= filter_distance) & (rows < disp_image.shape[0] - filter_distance)
    filter_cols = (cols >= filter_distance) & (cols < disp_image.shape[1] - filter_distance)
    filter_positive_cols = (positive_cols >= filter_distance) & (positive_cols < disp_image.shape[1] - filter_distance)
    filter_negative_cols = (negative_cols >= filter_distance) & (negative_cols < disp_image.shape[1] - filter_distance)
    all_filters = filter_rows & filter_cols & filter_positive_cols & filter_negative_cols

    rows = rows[all_filters]
    cols = cols[all_filters]
    disp_values = disp_values[all_filters]
    positive_cols = positive_cols[all_filters]
    negative_cols = negative_cols[all_filters]
    img_ids = np.full(rows.shape, image_index, dtype=np.uint8)

    result_dtype = np.dtype([('image_index', 'uint8'),
                             ('row', 'uint16'), ('col', 'uint16'),
                             ('col_positive', 'uint16'),
                             ('col_negative', 'uint16'), ])
    result_structured_array = np.empty(len(rows), dtype=result_dtype)

    result_structured_array['image_index'] = img_ids
    result_structured_array['row'] = rows
    result_structured_array['col'] = cols
    result_structured_array['col_positive'] = positive_cols
    result_structured_array['col_negative'] = negative_cols

    return result_structured_array


def save_disp_data_of_all_images():
    # metoda sprema u jedan numpy array sve podatke potrebne za učenje. Za svaku sliku stvara podatke o
    # disparitetima i sve ih sprema u jedan ndarray. To se sprema u 'disp_data.npy' datoteku.
    disp_data_for_all_images = np.concatenate([get_filtered_disp(x) for x in range(180)])
    np.save('disp_data.npy', disp_data_for_all_images)


def load_disp_data_of_all_images():
    # učitavanje podataka za učenje. Učitava se ndarray 'disp_data.npy'
    return np.load('disp_data.npy')


