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

    def make_hm_generator_model(self):
        cnn = CNNModel()
        model = cnn.get_model(input_tensor=None, arch=self.hm_regressor_arch, num_landmark=self.num_landmark,
                              input_shape=self.input_shape_hm_reg, num_face_graph_elements=None)
        if self.hm_regressor_weight is not None:
            model.load_weights(self.hm_regressor_weight)
        return model

    def _create_ckpt(self, epoch, hm_generator_optimizer, hm_generator):
        checkpoint_dir = './training_checkpoints/'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(hm_generator_optimizer=hm_generator_optimizer,
                                         hm_generator=hm_generator)
        checkpoint.save(file_prefix=checkpoint_prefix)
        '''save model weights'''
        hm_generator.save_weights(checkpoint_dir + 'only_hm_reg_' + str(epoch) + '_.h5')

    def hm_regressor_loss(self, hm_gr, hm_pr_arr):
        """"""
        '''defining hyper parameters'''
        weight_influence = 10
        if self.dataset_name == DatasetName.cofw:
            weight_influence = 100

        weights = tf.cast(hm_gr > 0, dtype=tf.float32) * weight_influence + 1

        loss_reg = tf.math.reduce_mean(tf.math.abs(hm_gr - hm_pr_arr[0]) * weights*0.5)
        loss_reg += tf.math.reduce_mean(tf.math.abs(hm_gr - hm_pr_arr[1]) * weights*0.6)
        loss_reg += tf.math.reduce_mean(tf.math.abs(hm_gr - hm_pr_arr[2]) * weights*0.8)
        loss_reg += tf.math.reduce_mean(tf.math.abs(hm_gr - hm_pr_arr[3]) * weights)

        return loss_reg

    @tf.function
    def train_step(self, epoch, step, images, heatmaps_gr, hm_reg_model, hm_reg_optimizer, summary_writer):

        with tf.GradientTape() as hm_reg_tape:
            '''prediction'''
            heatmaps_pr_arr = hm_reg_model(images)
            heatmaps_pr = heatmaps_pr_arr[3]

            '''showing the results'''
            if step > 0 and step % 200 == 0:
                self.print_hm_cord(epoch, step, images, heatmaps_gr, heatmaps_pr)

            '''loss calculation'''
            '''     hm loss'''
            hm_reg_total_loss = self.hm_regressor_loss(hm_gr=heatmaps_gr,
                                                                                  hm_pr_arr=heatmaps_pr_arr)
        ''' Calculate: Gradients'''
        '''     hm: '''
        gradients_of_hm_reg = hm_reg_tape.gradient(hm_reg_total_loss, hm_reg_model.trainable_variables)
        '''apply Gradients:'''
        '''     hm: '''
        hm_reg_optimizer.apply_gradients(zip(gradients_of_hm_reg, hm_reg_model.trainable_variables))
        '''printing loss Values: '''
        tf.print("->EPOCH: ", str(epoch), "->STEP: ", str(step),
                 "|||HEATMAP:->", 'hm_reg_total_loss:{', hm_reg_total_loss, '}')

        with summary_writer.as_default():
            tf.summary.scalar('hm_reg_total_loss', hm_reg_total_loss, step=epoch)

    def train(self):
        summary_writer = tf.summary.create_file_writer(
            "./train_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        hm_reg_model = self.make_hm_generator_model()
        hm_reg_optimizer = self._get_optimizer(lr=1e-4)

        x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = self._create_generators()

        step_per_epoch = len(x_train_filenames) // LearningConfig.batch_size

        for epoch in range(LearningConfig.epochs):
            for batch_index in range(step_per_epoch):
                images, heatmaps_gr = self._get_batch_sample(batch_index, x_train_filenames, y_train_filenames)

                images = tf.cast(images, tf.float32)

                self.train_step(epoch=epoch, step=batch_index, images=images, heatmaps_gr=heatmaps_gr, hm_reg_model=hm_reg_model,
                                hm_reg_optimizer=hm_reg_optimizer, summary_writer=summary_writer)
            if (epoch + 1) % 2 == 0:
                self._create_ckpt(epoch=epoch, hm_generator_optimizer=hm_reg_optimizer, hm_generator=hm_reg_model)

        # -----------------------------------------------------
    def _get_optimizer(self, lr=1e-2, beta_1=0.9, beta_2=0.999, decay=1e-5):
        return tf.keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    def _create_generators(self):
        """
        check if we have the img & lbls name. and create in case we need it.
        :return:
        """
        fn_prefix = './file_names/' + self.dataset_name + '_'
        x_trains_path = fn_prefix + 'x_train_fns.npy'
        x_validations_path = fn_prefix + 'x_val_fns.npy'
        y_trains_path = fn_prefix + 'y_train_fns.npy'
        y_validations_path = fn_prefix + 'y_val_fns.npy'

        tf_utils = TFRecordUtility(number_of_landmark=self.num_landmark)

        filenames, labels = tf_utils.create_image_and_labels_name(dataset_name=self.dataset_name)
        filenames_shuffled, y_labels_shuffled = shuffle(filenames, labels)
        x_train_filenames, x_val_filenames, y_train, y_val = train_test_split(
            filenames_shuffled, y_labels_shuffled, test_size=0.05, random_state=1)

        save(x_trains_path, x_train_filenames)
        save(x_validations_path, x_val_filenames)
        save(y_trains_path, y_train)
        save(y_validations_path, y_val)

        return x_train_filenames, x_val_filenames, y_train, y_val

    def _get_batch_sample(self, batch_index, x_train_filenames, y_train_filenames):
        img_path = self.train_images_dir
        hm_tr_path = self.train_hm_dir

        batch_x = x_train_filenames[
                  batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
        batch_y = y_train_filenames[
                  batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]

        img_batch = np.array([imread(img_path + file_name) for file_name in batch_x])
        hm_batch = np.array([load(hm_tr_path + file_name) for file_name in batch_y])

        return img_batch, hm_batch