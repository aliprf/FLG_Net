from configuration import DatasetName, IbugConf, LearningConfig, CofwConf, WflwConf
from tf_record_utility import TFRecordUtility
from cnn_model import CNNModel
from custom_Losses import Custom_losses
import tensorflow as tf
import keras
from keras.optimizers import adam
from keras.layers import Input

import numpy as np
from datetime import datetime
from sklearn.utils import shuffle
import os
from sklearn.model_selection import train_test_split
from numpy import save, load, asarray
import os.path
from keras import losses
import csv
from skimage.io import imread
tf.logging.set_verbosity(tf.logging.ERROR)


class FacialGAN:
    def __init__(self, dataset_name, geo_custom_loss, regressor_arch,
                 discriminator_arch, regressor_weight, discriminator_weight, input_shape):
        self.dataset_name = dataset_name
        self.geo_custom_loss = geo_custom_loss
        self.regressor_arch = regressor_arch
        self.discriminator_arch = discriminator_arch
        self.regressor_weight = regressor_weight
        self.regressor_weight = regressor_weight
        self.input_shape = input_shape
        if dataset_name == DatasetName.ibug:
            self.SUM_OF_ALL_TRAIN_SAMPLES = IbugConf.number_of_all_sample
            self.num_landmark = IbugConf.num_of_landmarks * 2
            self.train_images_dir = IbugConf.train_images_dir
            self.train_hm_dir = IbugConf.train_hm_dir
            self.train_point_dir = IbugConf.normalized_points_npy_dir
            self.num_face_graph_elements = IbugConf.num_face_graph_elements
        elif dataset_name == DatasetName.cofw:
            self.SUM_OF_ALL_TRAIN_SAMPLES = CofwConf.number_of_all_sample
            self.num_landmark = CofwConf.num_of_landmarks * 2
            self.train_images_dir = CofwConf.train_images_dir
            self.train_hm_dir = CofwConf.train_hm_dir
            self.train_point_dir = CofwConf.normalized_points_npy_dir
            self.num_face_graph_elements = CofwConf.num_face_graph_elements
        elif dataset_name == DatasetName.wflw:
            self.SUM_OF_ALL_TRAIN_SAMPLES = WflwConf.number_of_all_sample
            self.num_landmark = WflwConf.num_of_landmarks * 2
            self.train_images_dir = WflwConf.train_images_dir
            self.train_hm_dir = WflwConf.train_hm_dir
            self.train_point_dir = WflwConf.normalized_points_npy_dir
            self.num_face_graph_elements = WflwConf.num_face_graph_elements
        c_loss = Custom_losses(dataset_name, accuracy=100)
        if geo_custom_loss:
            self.geo_loss = c_loss.inter_landmark_loss
        else:
            self.geo_loss = losses.mean_squared_error

    def train_network(self):
        """
        Training Network:
        :return:
        """

        '''Creating models:'''
        reg_model = self._create_regressor_net(input_tensor=None, input_shape=self.input_shape)
        disc_model = self._create_discriminator_net(input_tensor=None, input_shape=self.input_shape)

        '''create train, validation, test data iterator'''
        x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = self._create_generators()

        '''Setting up GAN here:'''
        # reg_model.trainable = False

        '''Save both model metrics in  a CSV file'''
        log_file_name = './train_logs/log_' + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
        metrics_names = []
        metrics_names.append('geo_' + metric for metric in reg_model.metrics_names)
        metrics_names.append('disc_' + metric for metric in disc_model.metrics_names)
        metrics_names.append('epoch')
        self._write_loss_log(log_file_name, metrics_names)

        '''Start training on batch here:'''
        step_per_epoch = len(x_train_filenames) // LearningConfig.batch_size
        for epoch in range(LearningConfig.epochs):
            loss = []
            for batch_index in range(step_per_epoch):
                try:
                    images, heatmaps, points = self._get_batch_sample(batch_index, x_train_filenames, y_train_filenames)

                    predicted_heatmaps, predicted_points = reg_model.predict_on_batch(images)

                    disc_x, disc_y = self._prepare_discriminator_model_input(heatmaps, predicted_heatmaps,
                                                                             points, predicted_points)

                    d_loss = reg_model.train_on_batch(disc_x, disc_y)

                    g_loss = seq_model.train_on_batch(imgs, y_gen)

                    print(f'Epoch: {epoch} \t \t batch:{batch_index} of {step_per_epoch}\t\n '
                          f' Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')
                except Exception as e:
                    print('Faced Exception in train: ' + str(e))

            loss.append(epoch)
            self._write_loss_log(log_file_name, loss)
            model.save_weights('weight_ep_'+str(epoch)+'_los_'+str(loss)+'.h5')


    def _create_regressor_net(self, input_tensor, input_shape):
        """
        This is the main network, we use for predicting hm as well as points:
        input:
            X: img
            Y: [hm, points]

        :param input_tensor:
        :param input_shape:
        :return: keras model created for the geo-hm regression task.
        """
        cnn = CNNModel()
        model = cnn.get_model(input_tensor=input_tensor, arch=self.regressor_arch, num_landmark=self.num_landmark,
                              input_shape=input_shape)
        if self.regressor_weight is not None:
            model.load_weights(self.regressor_weight)
        model.compile(loss=self.geo_loss, optimizer=self._get_optimizer(), metrics=['mse'])

        return model

    def _create_discriminator_net(self, input_tensor, input_shape):
        """
        This is the discriminator network, being used at the second stage when we want to discriminate
        the real and fake data, generated by the RegressorNetwork
        :param input_tensor:
        :param input_shape:
        :return:
        """

        cnn = CNNModel()
        model = cnn.get_model(input_tensor=input_tensor, arch=self.discriminator_arch, input_shape=input_shape,
                              num_landmark=self.num_landmark)
        if self.discriminator_weight is not None:
            model.load_weights(self.discriminator_weight)
        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=self._get_optimizer(),
                      metrics=['accuracy'])
        return model

    def _get_optimizer(self, lr=1e-2, beta_1=0.9, beta_2=0.999, decay=1e-5):
        return adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    def _get_batch_sample(self, batch_index, x_train_filenames, y_train_filenames):
        img_path = self.train_images_dir
        hm_tr_path = self.train_hm_dir
        pn_tr_path = self.train_point_dir

        batch_x = x_train_filenames[batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
        batch_y = y_train_filenames[batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]

        img_batch = np.array([imread(img_path + file_name) for file_name in batch_x])
        hm_batch = np.array([load(hm_tr_path + file_name) for file_name in batch_y])
        pn_batch = np.array([load(pn_tr_path + file_name) for file_name in batch_y])
        return img_batch, hm_batch, pn_batch


    def _create_generators(self):
        """

        :return:
        """
        fn_prefix = './file_names/' + self.dataset_name + '_'
        x_trains_path = fn_prefix + 'x_train_fns.npy'
        x_validations_path = fn_prefix + 'x_train_fns.npy'
        y_trains_path = fn_prefix + 'x_train_fns.npy'
        y_validations_path = fn_prefix + 'x_train_fns.npy'

        tf_utils = TFRecordUtility(number_of_landmark=self.num_landmark)

        if os.path.isfile(x_trains_path) and os.path.isfile(x_validations_path) \
                and os.path.isfile(y_trains_path) and os.path.isfile(y_validations_path):
            x_train_filenames = load(x_trains_path)
            x_val_filenames = load(x_validations_path)
            y_train = load(y_trains_path)
            y_val = load(y_validations_path)
        else:
            filenames, labels = tf_utils.create_image_and_labels_name(dataset_name=self.dataset_name)
            filenames_shuffled, y_labels_shuffled = shuffle(filenames, labels)
            x_train_filenames, x_val_filenames, y_train, y_val = train_test_split(
                filenames_shuffled, y_labels_shuffled, test_size=0.05, random_state=1)

            save(x_trains_path, x_train_filenames)
            save(x_validations_path, x_val_filenames)
            save(y_trains_path, y_train)
            save(y_validations_path, y_val)

        return x_train_filenames, x_val_filenames, y_train, y_val

    def _write_loss_log(self, file_name, row_data):
        with open(file_name, 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(row_data)