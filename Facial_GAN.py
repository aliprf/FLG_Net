from configuration import DatasetName, DatasetType, \
    AffectnetConf, IbugConf, W300Conf, InputDataSize, LearningConfig, CofwConf, WflwConf
from tf_record_utility import TFRecordUtility
from clr_callback import CyclicLR
from cnn_model import CNNModel
from custom_Losses import Custom_losses
from Data_custom_generator import CustomHeatmapGenerator
import tensorflow as tf
import keras

tf.logging.set_verbosity(tf.logging.ERROR)
from keras.callbacks import ModelCheckpoint

from keras.optimizers import adam
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.callbacks import CSVLogger
from datetime import datetime
from sklearn.utils import shuffle
import os
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray
import os.path
from keras import losses
from keras import backend as K
import csv
from skimage.io import imread


class FacialGAN:
    def __init__(self, dataset_name, geo_custom_loss, hm_custom_loss, regressor_arch, discriminator_arch,
                 num_landmark, regressor_weight, discriminator_weight):
        self.dataset_name = dataset_name
        self.geo_custom_loss = geo_custom_loss
        self.hm_custom_loss = hm_custom_loss
        self.regressor_arch = regressor_arch
        self.discriminator_arch = discriminator_arch
        self.num_landmark = num_landmark
        self.regressor_weight = regressor_weight
        self.discriminator_weight = discriminator_weight

        if dataset_name == DatasetName.ibug:
            self.SUM_OF_ALL_TRAIN_SAMPLES = IbugConf.number_of_all_sample
            self.tf_train_path = IbugConf.tf_train_path
            self.tf_eval_path = IbugConf.tf_evaluation_path
            self.output_len = IbugConf.num_of_landmarks * 2
        elif dataset_name == DatasetName.cofw:
            self.SUM_OF_ALL_TRAIN_SAMPLES = CofwConf.number_of_all_sample
            self.tf_train_path = CofwConf.tf_train_path
            self.tf_eval_path = CofwConf.tf_evaluation_path
            self.output_len = CofwConf.num_of_landmarks * 2
        elif dataset_name == DatasetName.wflw:
            self.SUM_OF_ALL_TRAIN_SAMPLES = WflwConf.number_of_all_sample
            self.tf_train_path = WflwConf.tf_train_path
            self.tf_eval_path = WflwConf.tf_evaluation_path
            self.output_len = WflwConf.num_of_landmarks * 2

        self.BATCH_SIZE = LearningConfig.batch_size
        self.STEPS_PER_VALIDATION_EPOCH = LearningConfig.steps_per_validation_epochs
        self.STEPS_PER_EPOCH = self.SUM_OF_ALL_TRAIN_SAMPLES // self.BATCH_SIZE
        self.EPOCHS = LearningConfig.epochs

        c_loss = Custom_losses(dataset_name, accuracy=100)
        if geo_custom_loss:
            self.geo_loss = c_loss.inter_landmark_loss
        else:
            self.geo_loss = losses.mean_squared_error

        # if hm_custom_loss:
        #     self.hm_loss = c_loss.asm_assisted_loss
        # else:
        #     self.hm_loss = losses.mean_squared_error

    def create_regressor_net(self, input_tensor, input_shape):
        '''
        This is the main network, we use for predicting hm as well as points:
        input:
            X: img
            Y: [hm, points]

        :param input_tensor:
        :param input_shape:
        :return: keras model created for the geo-hm regression task.
        '''
        cnn = CNNModel()
        model = cnn.get_model(train_images=input_tensor, arch=self.regressor_arch, output_len=self.num_landmark)
        if self.regressor_weight is not None:
            model.load_weights(self.regressor_weight)
        model.compile(loss=self.geo_loss, optimizer=self._get_optimizer(), metrics=['mse'])

        return model

    def _get_optimizer(self):
        return adam(lr=1e-2, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False)
