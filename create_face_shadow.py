from glob import glob
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from testNetwork_demo_512 import get_shadowed_photos
import os
import argparse

from utils.utils_shadow import load_model, get_segmented_mask, \
    get_image_padding_data

ext = ['png', 'jpg', 'jpeg']

parser = argparse.ArgumentParser(
    description='Creates face shadow. Image shall be of a passport photo type.')
parser.add_argument(
    '--input_data_folder', type=str,
    help='folder which shall be searched for files of types: ' + ' '.join(ext),
    default='./data')
parser.add_argument(
    '--output_folder', type=str,
    help='folder where the output will be saved.',
    default='./result')
args = parser.parse_args()

DATA_FOLDER = args.input_data_folder
OUT_FOLDER = args.output_folder


def paste_shadow_to_original_img(shaded, image, mask, padding_x, padding_y):
    """
    Pastes shadow on the face and hair to the original photo - with
    different intensity. Face receives twice as much shading as the hair.
    Ignores the enlightened image parts.

    Args:
        shaded: image with modified lightening
        image: original image
        mask: mask with segmented face and the hair
        padding_x: image will be cropped in width on both sides by this value
        padding_y: image will be cropped in height on both sides by this value

    Returns: original image with pasted shadow

    """
    shaded = np.asarray(shaded)
    shaded = np.minimum(image, shaded)
    b_channel, g_channel, r_channel = cv2.split(shaded)
    res = np.asarray(
        cv2.merge((b_channel, g_channel, r_channel, mask)))
    dst = Image.fromarray(image)
    dst.paste(Image.fromarray(res), (0, 0), Image.fromarray(res))
    h, w, channels = image.shape
    dst = dst.crop((padding_y, padding_x, h - padding_y, w - padding_x))
    return dst


def process_face_shading(image_paths):
    """
    The main loop of the script. For every single image creates 7 shaded images
    with different light angles.
    Args:
        image_paths: list with paths to the images that shall be shaded
    """
    for image_file in image_paths:
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        max_size, padding_x, padding_y = get_image_padding_data(image)

        image = cv2.copyMakeBorder(
            image, padding_x, padding_x, padding_y, padding_y,
            cv2.BORDER_CONSTANT, None, [0, 0, 0])

        mask = get_segmented_mask(image, max_size, .6, model, transform)
        shadowed_photos = get_shadowed_photos(image)

        img_name = image_file.split("/")[-1]
        for j, current in enumerate(shadowed_photos):
            dst = paste_shadow_to_original_img(
                current, image, mask, padding_x, padding_y)
            name = OUT_FOLDER + '/' + str(j) + '/' + img_name
            dst.save(name)


def create_output_folders():
    """
    Creates output folders if they don't exist.
    """
    for i in range(7):
        folder = OUT_FOLDER + '/' + str(i)
        if not os.path.exists(folder):
            os.makedirs(folder)


if __name__ == '__main__':
    images = sorted(glob(DATA_FOLDER + '/*.' + e) for e in ext)
    images = [item for sublist in images for item in sublist]
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    print('Model loaded')
    print(len(images), ' files in folder ', DATA_FOLDER)

    if len(images) > 0:
        create_output_folders()
        process_face_shading(images)
