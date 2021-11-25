import numpy as np
import torch

import Utils
import Network
import config


def transform_image_for_model(image):
    # ndarray[HxWxC] -> tensor[1xCxHxW]

    # metoda prima jednu sliku (koja se učitava pomoću Utils.get_left/right_image) i pretvara je u tensor.
    # Također sliku oblika HxWxC pretvara u BxCxHxW (H=height, W=width, C=channel, B=batch_size)
    # Model očekuje takav oblika podataka na ulazu.
    image_for_model = torch.as_tensor(image).detach().permute(2, 0, 1).unsqueeze(0)
    return image_for_model


def transform_model_output_to_ndarray(model_output):
    # tensor[1xCxHxW] -> ndarray[HxWxC]

    # Model na izlazu vraća BxCxHxW, to mi pretvaramo u numpy array za prikazivanje tj. u HxWxC
    # Izlaz se pretvara iz tensora u ndarray
    model_output_converted_image = model_output.squeeze(0).permute(1, 2, 0).detach().numpy()
    return model_output_converted_image


def similarity_at_d(left_output, right_output, d):
    # ndarray[HxWxC] -> ndarray[HxW]

    # metoda prima outpute modela lijeve i desne slike.
    # Vraća sličnost lijeve i desne slike. Desna slika je pomaknuta za disparitet d.
    shifted_right_output = np.roll(right_output, d, axis=1)
    return np.sum(left_output * shifted_right_output, axis=2)


def predict_disparity_map(image_num, model, max_disparity=config.MAX_DISPARITY):
    # metoda vraća predikciju mape dispariteta
    model.to('cpu').eval()
    # dohvati slike
    left = Utils.get_left_image(image_num)
    right = Utils.get_right_image(image_num)
    # pretvori slike za ulaz u model
    left_model_input = transform_image_for_model(left)
    right_model_input = transform_image_for_model(right)
    # provuci slike kroz model i pretvori ih u prikladni ndarray
    left_output = transform_model_output_to_ndarray(model(left_model_input))
    right_output = transform_model_output_to_ndarray(model(right_model_input))

    # ndarray[HxWxD]
    similarities_at_all_D_disparities = np.stack([similarity_at_d(left_output, right_output, d) for d in range(max_disparity)], axis=2)

    # ndarray[HxW]
    predicted_disparity_map = np.argmin(similarities_at_all_D_disparities, axis=2)
    return predicted_disparity_map


