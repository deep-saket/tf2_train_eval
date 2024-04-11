import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os

import tensorflow as tf
from tensorflow import keras


class TensorboardCallback(keras.callbacks.Callback):
    def __init__(self, log_dir, basis_dict, test_image_dir=None, image_shape = (224, 224, 3), start_epoch=0) -> None:
        super().__init__()
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(log_dir)
        self.test_image_dir = test_image_dir
        self.image_shape = image_shape
        self.start_epoch = start_epoch        

        if self.test_image_dir is not None:
            with self.writer.as_default():
                self.images = self.write_images()
        else:
            self.images = None

        self.min_basis_dict = {basis : 9999999 for basis, watch in basis_dict.items() if watch == 'min'}
        self.max_basis_dict = {basis : 0 for basis, watch in basis_dict.items() if watch == 'max'}
        self.basis_list = list(basis_dict.keys()) ## on basis of which checkpoints will be saved
                            ## list of strings, which should also be present in the logs
                            ## param used in on_epoch_end() method
        self.is_called_before = False

    def write_scalars(self, epoch, logs):
        for key, value in logs.items():
            if isinstance(value, tf.Tensor):
                value = value.numpy()
                if sum(value.shape) == 1:
                    value = value[0]
            elif isinstance(value, np.ndarray):
                if sum(value.shape) == 1:
                    value = value[0]
            tf.summary.scalar(key, value, step=epoch)


    def write_images(self):
        images_summeary = []
        images = []
        print('test images =', os.listdir(self.test_image_dir))
        s = 0
        for image_name in os.listdir(self.test_image_dir):
            image_path = os.path.join(self.test_image_dir, image_name)
            image = load_image(image_path, self.image_shape)
            images.append(image)
            images_summeary.append(cv2.resize(cv2.cvtColor(cv2.imread(image_path), \
                cv2.COLOR_BGR2RGB), (self.image_shape[0], self.image_shape[1])))
            tf.summary.image("sample", images_summeary, step=s)
            s += 1
            images_summeary = []
        return images

    def on_epoch_end(self, epoch, logs=None):
        epoch += self.start_epoch
        keys = list(logs.keys())

        if self.is_called_before:
            for basis in self.basis_list:
                if basis in list(self.min_basis_dict.keys()):
                    if logs[basis] < self.min_basis_dict[basis]:
                        self.min_basis_dict[basis] = logs[basis]
                elif basis in list(self.max_basis_dict.keys()):
                    if logs[basis] > self.max_basis_dict[basis]:
                        self.max_basis_dict[basis] = logs[basis]
        else:
            for basis in self.basis_list:
                if basis in list(self.min_basis_dict.keys()):
                    self.min_basis_dict[basis] = logs[basis]
                elif basis in list(self.max_basis_dict.keys()):
                    self.max_basis_dict[basis] = logs[basis]
            self.is_called_before = True
        
        for basis in self.basis_list:
            if basis in list(self.min_basis_dict.keys()):
                logs[f'min_{basis}'] = self.min_basis_dict[basis]
            elif basis in list(self.max_basis_dict.keys()):
                logs[f'max_{basis}'] = self.max_basis_dict[basis]
        
        with self.writer.as_default():
            ## write scalars
            self.write_scalars(epoch, logs)

            ## write sample inferences
            if self.images is not None:
                ## TODO: log segmentation outputs
                pass
        
        self.writer.flush()
