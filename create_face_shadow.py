from glob import glob
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from nets.MobileNetV2_unet import MobileNetV2_unet
from testNetwork_demo_512 import get_shadowed_photos
import os
import argparse

FACE_SEG = "./model/face_seg.pt"
device = torch.device("cpu")
ext = ['png', 'jpg', 'jpeg']

parser = argparse.ArgumentParser(description='Create face shadow.')
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


def load_model():
    """
    Loads pre-trained model and weights for face segmentation
    Returns: model
    """
    unet = MobileNetV2_unet(None).to(device)
    state_dict = torch.load(FACE_SEG, map_location='cpu')
    unet.load_state_dict(state_dict)
    unet.eval()
    return unet


def get_segmented_mask(image, size):
    """
    Segments received image for: face (with a neck), hair and the rest.

    Args:
        image: image to be segmented. shall actually contain the face
        size: size of the image

    Returns: mask with the values: 0 for background 1 for skin and 2 for hair

    """
    pil_img = Image.fromarray(image)
    torch_img = transform(pil_img)
    torch_img = torch_img.unsqueeze(0)
    torch_img = torch_img.to(device)
    # Forward Pass
    logits = model(torch_img)
    mask = np.argmax(logits.data.cpu().numpy(), axis=1).squeeze()
    mask = np.where((mask == 1), .6, mask)
    mask = np.where((mask == 2), .4, mask)
    mask *= 255
    mask = mask.astype(np.uint8)
    mask = np.array(Image.fromarray(mask).resize((size, size)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    mask = cv2.medianBlur(mask, 5)
    return mask


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
        h, w, channels = image.shape
        padding_x = 0
        padding_y = 0
        if h > w:
            padding_y = int((h - w) / 2)
            max_size = h
        else:
            padding_x = int((w - h) / 2)
            max_size = w

        image = cv2.copyMakeBorder(
            image, padding_x, padding_x, padding_y, padding_y,
            cv2.BORDER_CONSTANT, None, [0, 0, 0])

        mask = get_segmented_mask(image, max_size)
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
