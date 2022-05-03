# -*- coding: utf-8 -*-
import os

import numpy as np
import cv2
from torch.utils.data import Dataset
import glob,torch



class MyInferenceClass(Dataset):

    def __init__(self, image_path, position,save_dir=None):
        self.position = position
        self.save_dir=save_dir
        norm=glob.glob(str(image_path) + "/Normal/*" + position + "*.png")
        pne=glob.glob(str(image_path) + "/Pneumonia/*" + position + "*.png")
        tb=glob.glob(str(image_path) + "/Tuberculosis/*" + position + "*.png")

        self.images = norm+pne+tb

        self.data_len = len(self.images)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        images = cv2.imread(self.images[index], 0)

        original_image_size = images.shape

        # preprocessing
        images = cv2.equalizeHist(images)
        images = cv2.resize(images, (256, 256))
        images = np.array(images, dtype='float32')

        return {'input': np.expand_dims(images, 0),
                'im_size': original_image_size, 'label': self.images[index].split('/')[-2], 'name':self.images[index]}


def one_hot(x, class_count):
    return torch.eye(class_count)[:, x]
