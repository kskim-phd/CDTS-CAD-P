import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor

from scipy.ndimage import gaussian_filter

def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

def show_cam_on_image_nonmask(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    heat_img = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)

    return heat_img

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # heatmap = np.float32(heatmap) / 255

    # if np.max(img) > 1:
    #     raise Exception(
    #         "The input image should np.float32 in the range [0, 1]")

    heatmap[:,:,2][heatmap[:, :, 0]==0]=0
    heatmap[:, :, 1][heatmap[:, :, 0] == 0] =0
    heatmap[:,:,0] = gaussian_filter(heatmap[:,:,0], sigma=4)
    heatmap[:,:,1] = gaussian_filter(heatmap[:,:,1], sigma=4)
    heatmap[:,:,2] = gaussian_filter(heatmap[:,:,2], sigma=4)
    # heatmap[ :, :,2] = np.zeros(heatmap[ :, :,2].shape)
    # heatmap[:, :, 1] = np.zeros(heatmap[:, :, 1].shape)
    # cam = heatmap + img
    heat_img = cv2.addWeighted(img, 0.7, heatmap, 0.3, 0)
    # cam = cam / np.max(cam)
    return heat_img#np.uint8(255 * cam)
