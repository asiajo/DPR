import numpy as np
import cv2
from testNetwork_demo_512 import get_shadowed_photos
from PIL import Image
from torchvision import transforms
import torch
from glob import glob
import argparse
import os

from utils.utils_shadow import load_model, get_segmented_mask, \
    get_image_padding_data

pre_trained = "./model/face_seg.pt"
device = torch.device("cpu")
ext = ['png', 'jpg', 'jpeg']

parser = argparse.ArgumentParser(
    description='Removes face shadow. Image shall be of a passport photo type.')
parser.add_argument(
    '--input_data_folder', type=str,
    help='folder which shall be searched for files of types: ' + ' '.join(ext),
    default='./data/shadowed')
parser.add_argument(
    '--output_folder', type=str,
    help='folder where the output will be saved.',
    default='./result/deshadowed')
args = parser.parse_args()

DATA_FOLDER = args.input_data_folder
OUT_FOLDER = args.output_folder
LIGHT_DIRECTION_RIGHT = 0
LIGHT_DIRECTION_LEFT = 3


def get_binary_image(seg_mask, y_cb_cr_img):
    """
    Thresholds the image by computed mean. Applies dark pixels only on face
    and hair regions, leaving background white, independently from its color.

    Args:
        seg_mask: mask from the face segmentation.
        y_cb_cr_img: image in YCbCr color space

    Returns: Black and white image with shadows on the face black.

    """
    binary_mask = np.copy(y_cb_cr_img)
    y_mean = np.mean(cv2.split(y_cb_cr_img)[0])
    y_std = np.std(cv2.split(y_cb_cr_img)[0])
    for i in range(y_cb_cr_img.shape[0]):
        for j in range(y_cb_cr_img.shape[1]):

            if y_cb_cr_img[i, j, 0] < y_mean - (y_std / 3) \
                    and seg_mask[i, j] >= 0.4 * 255:
                # paint it white (shadow)
                binary_mask[i, j] = [255, 255, 255]
            else:
                # paint it black (non-shadow)
                binary_mask[i, j] = [0, 0, 0]
    return binary_mask


def get_light_index(seg_mask, face_img):
    """
    Verifies from which side the face is shaded and returns the number
    indicating which light shall be used for its removal.

    Args:
        seg_mask: mask from the face segmentation. Makes sure that only
    shadow on the skin is verified and not for example background color
        face_img: image containing the face

    Returns: Index of the light that shall be used for shadow removal
    """
    h, w = seg_mask.shape
    seg_mask = seg_mask[padding_x:(w - padding_x), padding_y:(h - padding_y)]
    y_cb_cr_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
    binary_mask = get_binary_image(seg_mask, y_cb_cr_img)
    mask_l = binary_mask[:,:175]
    mask_r = cv2.flip(binary_mask, 1)[:,:175]

    if np.mean(mask_l) > np.mean(mask_r):
        return LIGHT_DIRECTION_LEFT
    elif np.mean(mask_l) < np.mean(mask_r):
        return LIGHT_DIRECTION_RIGHT
    return None


def remove_shadow(image, light_direction, seg_mask):
    """
    Removes the shadow from the image by calling light modification network
    and overlapping the original image with newly created one and choosing
    in each case only brighter pixel.

    Args:
        image: original image
        light_direction: direction of the light that shall be applied by DPR
        seg_mask: mask containing segmented image

    Returns: Image with removed or softened shadow

    """
    deshadowed = get_shadowed_photos(image)[light_direction]
    deshadowed = cv2.addWeighted(deshadowed, 0.6, image, 0.4, 0)
    r_channel, g_channel, b_channel = cv2.split(deshadowed)
    h, w = seg_mask.shape
    seg_mask = seg_mask[padding_x:(w - padding_x), padding_y:(h - padding_y)]
    res = np.asarray(cv2.merge((r_channel, g_channel, b_channel, seg_mask)))
    dst = Image.fromarray(image)
    dst.paste(Image.fromarray(res), (0, 0), Image.fromarray(res))
    return dst


if __name__ == '__main__':

    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    images = sorted(glob(DATA_FOLDER + '/*.' + e) for e in ext)
    images = [item for sublist in images for item in sublist]

    for img_path in images:
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        max_size, padding_x, padding_y = get_image_padding_data(img_rgb)
        padded_image = cv2.copyMakeBorder(
            img_rgb, padding_x, padding_x, padding_y, padding_y,
            cv2.BORDER_CONSTANT, None, [0, 0, 0])
        mask = get_segmented_mask(padded_image, max_size, .4, model, transform)
        light = get_light_index(mask, img_bgr)

        if light is not None:
            dst = remove_shadow(img_rgb, light, mask)
        else:
            dst = Image.fromarray(img_rgb)

        if not os.path.exists(OUT_FOLDER):
            os.makedirs(OUT_FOLDER)

        img_name = img_path.split("/")[-1]
        name = OUT_FOLDER + '/' + img_name
        dst.save(name)
