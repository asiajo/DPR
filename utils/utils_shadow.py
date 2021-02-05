from nets.MobileNetV2_unet import MobileNetV2_unet
import torch
from PIL import Image
import numpy as np
import cv2

FACE_SEG = "./model/face_seg.pt"
device = torch.device("cpu")


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


def get_segmented_mask(
        image, size, intensity, model, transform):
    """
    Segments received image for: face (with a neck), hair and the rest.

    Args: image: image to be segmented. shall actually contain the face
    size: size of the image
    intensity: intensity of mask over the face, where 0 indicates no impact
    and 1 - full impact
    model: model for segmenting operation
    transform: required transformation for the segmenting model

    Returns: mask with the values: 0 for background 1 for skin and 2 for hair

    """
    pil_img = Image.fromarray(image)
    torch_img = transform(pil_img)
    torch_img = torch_img.unsqueeze(0)
    torch_img = torch_img.to(device)
    # Forward Pass
    logits = model(torch_img)
    mask = np.argmax(logits.data.cpu().numpy(), axis=1).squeeze()
    mask = np.where((mask == 1), intensity, mask)
    mask = np.where((mask == 2), intensity / 2., mask)
    mask *= 255
    mask = mask.astype(np.uint8)
    mask = np.array(Image.fromarray(mask).resize((size, size)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    mask = cv2.medianBlur(mask, 5)
    return mask


def get_image_padding_data(image):
    """
    Args: image: image to have calculated required padding and retrieved the
    longer edge size. It is needed for padding the image to the square.

    Returns: Length of the longer edge of the image and required padding.

    """
    h, w, channels = image.shape
    padding_x = 0
    padding_y = 0
    if h > w:
        padding_y = int((h - w) / 2)
        max_size = h
    else:
        padding_x = int((w - h) / 2)
        max_size = w
    return max_size, padding_x, padding_y