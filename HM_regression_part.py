"""
in this class, we just work on generator. We want to figure out the power of generator alone.
"""

from configuration import DatasetName, IbugConf, LearningConfig, CofwConf, WflwConf, InputDataSize
from tf_record_utility import TFRecordUtility
from cnn_model import CNNModel
from custom_Losses import Custom_losses
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda

import numpy as np
from sklearn.utils import shuffle
import os
from sklearn.model_selection import train_test_split
from numpy import save, load
import os.path
import csv
from skimage.io import imread
import itertools
from skimage.transform import resize
from image_utility import ImageUtility
import img_printer as imgpr
import datetime
import matplotlib.pyplot as plt

tf.config.set_soft_device_placement(True)
print("IS GPU AVAIL:: ", str(tf.test.is_gpu_available()))


class HmRegression:
    def __init__(self, dataset_name, hm_regressor_arch, hm_regressor_weight, input_shape_hm_reg):
        self.dataset_name = dataset_name
        self.hm_regressor_arch = hm_regressor_arch
        self.hm_regressor_weight = hm_regressor_weight
        self.input_shape_hm_reg = input_shape_hm_reg

        if dataset_name == DatasetName.ibug:
            self.SUM_OF_ALL_TRAIN_SAMPLES = IbugConf.number_of_all_sample
            self.num_landmark = IbugConf.num_of_landmarks * 2
            self.train_images_dir = IbugConf.train_images_dir
            self.train_hm_dir = IbugConf.train_hm_dir
            self.train_point_dir = IbugConf.normalized_points_npy_dir
            self.hm_stride = IbugConf.hm_stride
        elif dataset_name == DatasetName.cofw:
            self.SUM_OF_ALL_TRAIN_SAMPLES = CofwConf.number_of_all_sample
            self.num_landmark = CofwConf.num_of_landmarks * 2
            self.train_images_dir = CofwConf.train_images_dir
            self.train_hm_dir = CofwConf.train_hm_dir
            self.train_point_dir = CofwConf.normalized_points_npy_dir
            self.hm_stride = CofwConf.hm_stride
        elif dataset_name == DatasetName.wflw:
            self.SUM_OF_ALL_TRAIN_SAMPLES = WflwConf.number_of_all_sample
            self.num_landmark = WflwConf.num_of_landmarks * 2
            self.train_images_dir = WflwConf.train_images_dir
            self.train_hm_dir = WflwConf.train_hm_dir
            self.train_point_dir = WflwConf.normalized_points_npy_dir
            self.hm_stride = WflwConf.hm_stride

    def train(self):
        print('')
