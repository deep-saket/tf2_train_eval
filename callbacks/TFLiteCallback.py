import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os

import tensorflow as tf
from tensorflow import keras

from cvml.tflite.tflite_utils import *

class TFLiteCallback(keras.callbacks.Callback):
    def __init__(self, save_dir, basis_dict, skip_from_start=4, frequency=5, on_epoch_end=True, start_epoch=0) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.frequency = frequency
        self.skip_from_start = skip_from_start
        self.exec_epoch_end = on_epoch_end
        self.start_epoch = start_epoch

        self.min_basis_dict = {basis : 9999999 for basis, watch in basis_dict.items() if watch == 'min'}
        self.max_basis_dict = {basis : 0 for basis, watch in basis_dict.items() if watch == 'max'}
        self.basis_list = list(basis_dict.keys()) ## on basis of which checkpoints will be saved
                            ## list of strings, which should also be present in the logs
                            ## param used in on_epoch_end() method
        self.is_called_before = False

    def on_epoch_end(self, epoch, logs=None):
        epoch += self.start_epoch
        if self.exec_epoch_end:
            keys = list(logs.keys())
            if epoch > self.skip_from_start-1: 
                save_tflites(self.model, self.save_dir, epoch)
    
    def on_train_end(self, logs=None):
        keys = list(logs.keys())

        for basis in self.basis_list:
            if basis in list(self.min_basis_dict.keys()):
                if os.path.exists(f'{self.save_dir}/min_{basis}'):
                    save_tflites_ckpt(self.model, ckpt_path=f'{self.save_dir}/min_{basis}', \
                         tflite_path=f'{self.save_dir}/min_{basis}/HandSeg.tflite')
            elif logs[basis] > self.max_basis_dict[basis]:
                if os.path.exists(f'{self.save_dir}/max_{basis}'):
                    save_tflites_ckpt(self.model, ckpt_path=f'{self.save_dir}/max_{basis}', \
                         tflite_path=f'{self.save_dir}/max_{basis}/HandSeg.tflite')