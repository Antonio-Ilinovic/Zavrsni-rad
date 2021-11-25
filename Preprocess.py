import numpy as np

import Utils
import config


def main():
    # program se pokreće na kraju datoteke
    save_disparity_data_for_train_images()


def save_disparity_data_for_train_images():
    # metoda sprema u jedan numpy array sve podatke potrebne za učenje. Za svaku sliku stvara podatke o
    # disparitetima i sve ih sprema u jedan ndarray. To se sprema u 'disp_data.npy' datoteku.
    disparity_data_for_train_images = np.concatenate([get_filtered_disparity_data(x) for x in range(config.NUM_TRAIN_IMAGES)])
    np.save(config.DISPARITY_DATA_PATH, disparity_data_for_train_images)


def get_filtered_disparity_data(image_index):
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

    filter_distance = config.PATCH_SIZE // 2

    disparity_image = Utils.get_disp_image(image_index)

    # izaberi vrijednosti koje nisu 0
    rows, cols = disparity_image.nonzero()
    rows = rows.astype(np.uint16)
    cols = cols.astype(np.uint16)
    disparity_values = disparity_image[rows, cols]
    # izračunaj stupce za pozitivne primjere -> samo posmakni piksele za točan disparitet
    positive_cols = cols - disparity_values
    # izračunaj stupce za negativne primjere -> posmakni piskele za točan disparitet i dodaj neki pomak, da bude netočan disparitet
    negative_cols = positive_cols + np.random.choice([-8, -7, -6, -5, -4, 4, 5, 6, 7, 8],
                                                     size=positive_cols.size).astype(np.uint16)

    # izracunaj filter koje vrijednosti i koordinate dispariteta će se razmatrati
    filter_rows = (rows >= filter_distance) & (rows < disparity_image.shape[0] - filter_distance)
    filter_cols = (cols >= filter_distance) & (cols < disparity_image.shape[1] - filter_distance)
    filter_positive_cols = (positive_cols >= filter_distance) & (positive_cols < disparity_image.shape[1] - filter_distance)
    filter_negative_cols = (negative_cols >= filter_distance) & (negative_cols < disparity_image.shape[1] - filter_distance)
    all_filters = filter_rows & filter_cols & filter_positive_cols & filter_negative_cols

    rows = rows[all_filters]
    cols = cols[all_filters]
    positive_cols = positive_cols[all_filters]
    negative_cols = negative_cols[all_filters]
    img_indexes = np.full(rows.shape, image_index, dtype=np.uint8)

    # stvori i napuni custom numpy array (radi uštede prostora i bržeg učitavanja)
    result_dtype = np.dtype([('image_index', 'uint8'),
                             ('row', 'uint16'), ('col', 'uint16'),
                             ('col_positive', 'uint16'),
                             ('col_negative', 'uint16'), ])
    result_structured_array = np.empty(len(rows), dtype=result_dtype)

    result_structured_array['image_index'] = img_indexes
    result_structured_array['row'] = rows
    result_structured_array['col'] = cols
    result_structured_array['col_positive'] = positive_cols
    result_structured_array['col_negative'] = negative_cols

    return result_structured_array


main()
