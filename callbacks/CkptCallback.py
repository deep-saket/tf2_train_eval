import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os

import tensorflow as tf
from tensorflow import keras

class CkptCallback(keras.callbacks.Callback):
    def __init__(self, save_dir, basis_dict, skip_from_start=4, frequency=5, start_epoch=0) -> None:
        '''
        basis_dict is a python dictionary containing keys as metric/loss name and
        value as min or max. min indicates that checkpoints will be saved based on
        the decrease in value and max indicates the checkpoints will be saved based
        on increase in value
        '''
        super().__init__()
        self.save_dir = save_dir
        self.frequency = frequency
        self.skip_from_start= skip_from_start
        self.min_basis_dict = {basis : 9999999 for basis, watch in basis_dict.items() if watch == 'min'}
        self.max_basis_dict = {basis : 0 for basis, watch in basis_dict.items() if watch == 'max'}
        self.is_called_before = False
        self.start_epoch = start_epoch
        self.basis_list = list(basis_dict.keys()) ## on basis of which checkpoints will be saved
                            ## list of strings, which should also be present in the logs
                            ## param used in on_epoch_end() method

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        epoch += self.start_epoch

        if epoch > self.skip_from_start-1:   
            if self.is_called_before: 
                for basis in self.basis_list:
                    if basis in list(self.min_basis_dict.keys()):
                        if logs[basis] < self.min_basis_dict[basis]:
                            self.min_basis_dict[basis] = logs[basis]
                            if os.path.isdir(f'{self.save_dir}/min_{basis}'):
                                os.system(f'rm -rf {self.save_dir}/min_{basis}')
                                os.makedirs(f'{self.save_dir}/min_{basis}')
                                self.model.save_weights(f'{self.save_dir}/min_{basis}/min_{basis}_{epoch}')
                            else:
                                os.makedirs(f'{self.save_dir}/min_{basis}')
                                self.model.save_weights(f'{self.save_dir}/min_{basis}/min_{basis}_{epoch}')
                    elif basis in list(self.max_basis_dict.keys()):
                        if logs[basis] > self.max_basis_dict[basis]:
                            self.max_basis_dict[basis] = logs[basis]
                            if os.path.isdir(f'{self.save_dir}/max_{basis}'):
                                os.system(f'rm -rf {self.save_dir}/max_{basis}')
                                os.makedirs(f'{self.save_dir}/max_{basis}')
                                self.model.save_weights(f'{self.save_dir}/max_{basis}/max_{basis}_{epoch}') 
                            else:
                                os.makedirs(f'{self.save_dir}/max_{basis}')
                                self.model.save_weights(f'{self.save_dir}/max_{basis}/max_{basis}_{epoch}')  

                if epoch % self.frequency == 0:
                    if os.path.isdir(f'{self.save_dir}/{epoch}'):
                        self.model.save_weights(f'{self.save_dir}/{epoch}/ckpt')
                    else:
                        os.makedirs(f'{self.save_dir}/{epoch}')
                        self.model.save_weights(f'{self.save_dir}/{epoch}/ckpt')
            else:
                for basis in self.basis_list:
                    if basis in list(self.min_basis_dict.keys()):
                        self.min_basis_dict[basis] = logs[basis]
                    elif basis in list(self.max_basis_dict.keys()):
                        self.max_basis_dict[basis] = logs[basis]
                self.is_called_before = True
        # print("End epoch {} of training; got log keys: {}".format(epoch, keys))
