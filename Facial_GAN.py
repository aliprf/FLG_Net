from configuration import DatasetName, IbugConf, LearningConfig, CofwConf, WflwConf, InputDataSize
from tf_record_utility import TFRecordUtility
from cnn_model import CNNModel
from custom_Losses import Custom_losses
import tensorflow as tf
import keras
import keras.backend as K
from keras.optimizers import adam
from keras.layers import Input, Lambda

import numpy as np
from datetime import datetime
from sklearn.utils import shuffle
import os
from sklearn.model_selection import train_test_split
from numpy import save, load
import os.path
from keras import losses
import csv
from skimage.io import imread
from keras.models import Model
from keras.utils.vis_utils import plot_model
import itertools
from skimage.transform import resize
from image_utility import ImageUtility
import img_printer as imgpr


tf.logging.set_verbosity(tf.logging.ERROR)


class FacialGAN:
    def __init__(self, dataset_name, geo_custom_loss, regressor_arch,
                 discriminator_arch, regressor_weight, discriminator_weight, input_shape_reg, input_shape_disc):
        self.dataset_name = dataset_name
        self.geo_custom_loss = geo_custom_loss
        self.regressor_arch = regressor_arch
        self.discriminator_arch = discriminator_arch
        self.regressor_weight = regressor_weight
        self.discriminator_weight = discriminator_weight
        self.input_shape_reg = input_shape_reg
        self.input_shape_disc = input_shape_disc
        if dataset_name == DatasetName.ibug:
            self.SUM_OF_ALL_TRAIN_SAMPLES = IbugConf.number_of_all_sample
            self.num_landmark = IbugConf.num_of_landmarks * 2
            self.num_face_graph_elements = IbugConf.num_face_graph_elements
            self.train_images_dir = IbugConf.train_images_dir
            self.train_hm_dir = IbugConf.graph_face_dir
            self.train_point_dir = IbugConf.normalized_points_npy_dir
            self.num_face_graph_elements = IbugConf.num_face_graph_elements
        elif dataset_name == DatasetName.cofw:
            self.SUM_OF_ALL_TRAIN_SAMPLES = CofwConf.number_of_all_sample
            self.num_landmark = CofwConf.num_of_landmarks * 2
            self.num_face_graph_elements = CofwConf.num_face_graph_elements
            self.train_images_dir = CofwConf.train_images_dir
            self.train_hm_dir = CofwConf.graph_face_dir
            self.train_point_dir = CofwConf.normalized_points_npy_dir
            self.num_face_graph_elements = CofwConf.num_face_graph_elements
        elif dataset_name == DatasetName.wflw:
            self.SUM_OF_ALL_TRAIN_SAMPLES = WflwConf.number_of_all_sample
            self.num_landmark = WflwConf.num_of_landmarks * 2
            self.num_face_graph_elements = WflwConf.num_face_graph_elements
            self.train_images_dir = WflwConf.train_images_dir
            self.train_hm_dir = WflwConf.graph_face_dir
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

        '''create train, validation, test data iterator'''
        x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = self._create_generators()

        '''Creating models:'''
        regressor_model = self._create_regressor_net(input_tensor=None, input_shape=self.input_shape_reg)
        discriminator_model = self._create_discriminator_net(input_tensor=None, input_shape=self.input_shape_disc)

        '''Setting up GAN here:'''
        # i_tensor = K.variable(np.zeros([LearningConfig.batch_size, 224, 224, 3]))
        # gan_model_input = Input(shape=self.input_shape_reg, tensor=i_tensor)

        regressor_model.trainable = False
        discriminator_model.trainable = True

        gan_input = Input(shape=self.input_shape_reg)
        reg_out = self._fuse_hm_and_points(regressor_model(gan_input), gan_input)
        # reg_out = regressor_model(gan_input)
        gan_output = discriminator_model(reg_out)
        gan_model = Model(gan_input, outputs=gan_output)
        gan_model.compile(loss=keras.losses.binary_crossentropy,
                          optimizer=self._get_optimizer())
        ''''''
        gan_model.summary()
        '''save GAN Model'''
        # gan_model.save_model('gw.h5')
        plot_model(gan_model, to_file='gan_model.png', show_shapes=True, show_layer_names=True)

        # xx = tf.keras.models.load_model(gan_model, 'gan_model.h5')
        # tf.keras.models.save_model(gan_model, 'gan_model.h5')
        model_json = gan_model.to_json()
        with open("gan_model.json", "w") as json_file:
            json_file.write(model_json)

        '''Save both model metrics in  a CSV file'''
        log_file_name = './train_logs/log_' + datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"
        metrics_names = ['epoch']
        for metric in regressor_model.metrics_names: metrics_names.append('reg_' + metric)
        for metric in discriminator_model.metrics_names: metrics_names.append('disc_' + metric)
        self._write_loss_log(log_file_name, metrics_names)

        '''Start training on batch here:'''
        step_per_epoch = len(x_train_filenames) // LearningConfig.batch_size
        for epoch in range(LearningConfig.epochs):
            loss = []
            for batch_index in range(step_per_epoch):
                try:
                    images, heatmaps_gr, points_gr = self._get_batch_sample(batch_index, x_train_filenames, y_train_filenames)
                    heatmaps_pr, points_pr = regressor_model.predict_on_batch(images)
                    disc_x, disc_y = self._prepare_discriminator_model_input(heatmaps_gr, heatmaps_pr, points_gr, points_pr, images)
                    d_loss = discriminator_model.train_on_batch(disc_x, disc_y)
                    y_gen = np.ones(LearningConfig.batch_size)
                    g_loss = gan_model.train_on_batch(images, y_gen)

                    print(f'Epoch: {epoch} \t \t batch:{batch_index} of {step_per_epoch}\t\n '
                          f' Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')
                except Exception as e:
                    print('Faced Exception in train: ' + str(e))

            loss.append(epoch)
            self._write_loss_log(log_file_name, loss)
            gan_model.save_weights('weight_ep_' + str(epoch) + '_los_' + str(loss) + '.h5')

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
                              input_shape=input_shape, num_face_graph_elements=self.num_face_graph_elements)
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
                              num_landmark=self.num_landmark, num_face_graph_elements=self.num_face_graph_elements)
        if self.discriminator_weight is not None:
            model.load_weights(self.discriminator_weight)
        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=self._get_optimizer(),
                      metrics=['accuracy'])
        return model

    def _prepare_discriminator_model_input(self, heatmaps_gr, heatmaps_pr, points_gr, points_pr, img):
        """
        at the first step, we fuse concat heatmaps + 2d_points.Then create real/fake labels
        :param heatmaps_gr:
        :param heatmaps_pr:
        :param points_gr:
        :param points_pr:
        :param img:
        :return:
        """
        '''resize img to hm_size:'''
        # img = tf.image.resize(img, [InputDataSize.hm_size, InputDataSize.hm_size]) --> the result will be a tensor
        img = resize(img, (LearningConfig.batch_size, InputDataSize.hm_size, InputDataSize.hm_size, 3))
        '''convert points to 2d facial graph:'''
        fg_gt = self._points_to_2d_face_graph(points_gr)
        fg_pr = self._points_to_2d_face_graph(points_pr)

        '''for testing :'''
        # imgpr.print_image_arr_heat(100, fg_gt[0], print_single=False)
        # imgpr.print_image_arr_heat(101, heatmaps_gr[0], print_single=False)
        # imgpr.print_image_arr_heat(200, fg_gt[1], print_single=False)
        # imgpr.print_image_arr_heat(201, heatmaps_gr[1], print_single=False)
        #
        # imgpr.print_image_arr_heat(100, fg_pr[0], print_single=False)
        # imgpr.print_image_arr_heat(101, heatmaps_pr[0], print_single=False)
        # imgpr.print_image_arr_heat(200, fg_pr[1], print_single=False)
        # imgpr.print_image_arr_heat(201, heatmaps_pr[1], print_single=False)

        '''concat hm, 2d_points, img '''
        # real_X = np.concatenate([heatmaps_gr, fg_gt, img], axis=-1)
        # fake_X = np.concatenate([heatmaps_pr, fg_pr, img], axis=-1)
        '''     in case we needed without img'''
        real_X = np.concatenate([heatmaps_gr, fg_gt], axis=-1)
        fake_X = np.concatenate([heatmaps_pr, fg_pr], axis=-1)
        '''create real/fake labels and attach to the input data'''
        X = np.concatenate((fake_X, real_X), axis=0)
        Y = np.zeros(2 * LearningConfig.batch_size)
        Y[:LearningConfig.batch_size] = 0.9 # use smooth labels

        return X, Y

    def _points_to_2d_face_graph(self, _points):
        """
        :param points: [bs, 136]
        :return: [bs, 56, 56, num_fg]
        """
        '''rescale points'''
        points = np.zeros([LearningConfig.batch_size, self.num_landmark])
        image_utility = ImageUtility()
        indx = 0
        for item in _points:
            point_scaled, px_1, py_1 = image_utility.create_landmarks_from_normalized(item, InputDataSize.hm_size,
                                                                                InputDataSize.hm_size,
                                                                                InputDataSize.hm_center,
                                                                                InputDataSize.hm_center)
            # imgpr.print_image_arr('pts_' + str(indx), np.zeros([56,56]), px_1, py_1)

            points[indx, :] = point_scaled
            indx += 1

        '''create partial parts: '''
        partial_points = self._slice_face_graph_np(points)
        '''convert from flatten to 2d'''
        points_2d = []
        for pnt in partial_points:
            points_2d.append(pnt.reshape([LearningConfig.batch_size, len(pnt[1])//2, 2]))
        '''create the spare img for each facial part:'''
        hm_img = np.zeros([LearningConfig.batch_size, InputDataSize.hm_size, InputDataSize.hm_size, self.num_face_graph_elements])
        # bs, 12 * 2
        for i in range(LearningConfig.batch_size):
            for j in range(self.num_face_graph_elements):
                t_hm = np.zeros([InputDataSize.hm_size, InputDataSize.hm_size])
                for x_y in points_2d[j][i]:
                    if not(0 <= x_y[0] <= InputDataSize.hm_size-1):
                        x_y[0] = 0
                    if not(0 <= x_y[1] <= InputDataSize.hm_size-1):
                        x_y[1] = 0
                    t_hm[int(x_y[1]), int(x_y[0])] = 1

                hm_img[i, :, :, j] = t_hm

        return hm_img

    def _fuse_hm_and_points(self, regressor_output, regressor_input):
        """

        :param regressor_output:[ [?, ?, ?, 8], [?, 136] ]
        :return: [bs, 56, 56, num_fg*2] ==> hm, points
        """
        return keras.layers.Concatenate()([regressor_output[0], regressor_output[0]])
        # return keras.layers.Concatenate()([regressor_output, regressor_output])

        # con_1 = keras.layers.Concatenate()([regressor_output, regressor_output])
        # return Lambda(lambda x: keras.layers.Concatenate()([con_1, x]))(t_inp_img)


        t_hm = regressor_output[0]
        t_hm_cp = regressor_output[0]
        t_pn = regressor_output[1]

        t_inp_img = tf.image.resize(regressor_input, [InputDataSize.hm_size, InputDataSize.hm_size])
        t_hm = keras.layers.Concatenate()([t_hm, t_inp_img])

        t_pn_img = Lambda(lambda x: self._convert_to_geometric(t_hm_cp, tf.cast(x, 'int64')))(t_pn)
        t_fused = Lambda(lambda x: self._fuse_tensors(t_hm, x))(t_pn_img)
        return t_fused

    def _fuse_tensors(self, t_hm, t_pn_img):

        # t_hm_shape = t_hm.get_shape().as_list()

        t_pn_img = tf.reshape(tensor=t_pn_img, shape=[tf.shape(t_hm)[0], InputDataSize.hm_size, InputDataSize.hm_size,
                                                      self.num_face_graph_elements])
        t_fused = keras.layers.Concatenate()([t_hm, t_pn_img])

        return t_fused

    def _convert_to_geometric(self, hm_img, coordinates):
        """
        :param img: ? * 56 * 56 * num_face_graph_elements
        :param coordinates: ? * 136
        :return:
        """

        '''create a clean copy of generated image '''
        img_indices = tf.constant([[x] for x in range(LearningConfig.batch_size)])
        img_updates = tf.zeros([LearningConfig.batch_size, InputDataSize.hm_size,
                                InputDataSize.hm_size, self.num_face_graph_elements], dtype=tf.float32)
        hm_img = tf.tensor_scatter_nd_update(hm_img, img_indices, img_updates)

        '''convert two_d_coords to facial part:'''
        sep_1_d_cord = self._slice_face_graph_tensor(coordinates)

        '''convert all points to 2d:'''
        counter = 0
        indices = []
        for cord_item_normal in sep_1_d_cord:
            '''convert to hm_scale: from  { from (-0.5, -0.5): (0.5, 0.5) => (0, 0): (56, 56)}'''
            cord_item = tf.map_fn(fn=lambda landmark: InputDataSize.hm_center + landmark * InputDataSize.hm_size,
                                  elems=cord_item_normal)
            '''convert to 2d'''
            two_d_coords = tf.reshape(tensor=cord_item, shape=[-1, cord_item.shape[1] // 2, 2])  # ?(batch_size)*k*2
            indices.append(self._create_2d_indices(two_d_coords, counter))
            counter += counter

        '''Then, for each 2d layer,scatter it to a 56*56 image (our hm_img)'''
        merged_indices = list(itertools.chain.from_iterable(indices))
        img_updates = tf.ones(shape=len(merged_indices), dtype=tf.float32)
        hm_img_o = tf.tensor_scatter_nd_update(hm_img, merged_indices, img_updates)
        o_T = tf.add(hm_img, hm_img_o)
        return o_T

    def _create_2d_indices(self, two_d_coords, counter):
        # indices = np.zeros([LearningConfig.batch_size, two_d_coords.shape[1], two_d_coords.shape[2], 1])
        indices = []
        for batch_indx in range(LearningConfig.batch_size):  # batch
            for x_indx in range(two_d_coords.shape[1]):  # cordinates
                x = two_d_coords[batch_indx, x_indx][0]
                y = two_d_coords[batch_indx, x_indx][1]
                item = [batch_indx, y, x, counter] # the coordination should be y,x NOT x,y
                # item = [batch_indx, x, y, counter]
                indices.append(item)
        return indices

    def _slice_face_graph_tensor(self, coordinates):
        if self.dataset_name == DatasetName.wflw:
            return 0
        elif self.dataset_name == DatasetName.cofw:
            return 0
        elif self.dataset_name == DatasetName.ibug:

            sep_1_d_cord = [coordinates[:, 0:34],  # face
                            coordinates[:, 34:44],  # li
                            coordinates[:, 44:54],  # ri
                            coordinates[:, 54:62],  # nose_b
                            coordinates[:, 62:72],  # nose
                            coordinates[:, 72:84],  # leye
                            coordinates[:, 84:96],  # reye
                            tf.concat([coordinates[:, 96:110], coordinates[:, 128:130], coordinates[:, 126:128],
                                       coordinates[:, 124:126], coordinates[:, 122:124], coordinates[:, 120:122]],
                                      axis=-1),  # u_lip
                            tf.concat([coordinates[:, 110:120], coordinates[:, 134:136], coordinates[:, 132:134],
                                       coordinates[:, 130:132]], axis=-1)]  # l_lip
        return sep_1_d_cord

    def _slice_face_graph_np(self, coordinates):
        if self.dataset_name == DatasetName.wflw:
            return 0
        elif self.dataset_name == DatasetName.cofw:
            return 0
        elif self.dataset_name == DatasetName.ibug:

            sep_1_d_cord = [coordinates[:, 0:34],  # face
                            coordinates[:, 34:44],  # li
                            coordinates[:, 44:54],  # ri
                            coordinates[:, 54:62],  # nose_b
                            coordinates[:, 62:72],  # nose
                            coordinates[:, 72:84],  # leye
                            coordinates[:, 84:96],  # reye
                            np.concatenate([coordinates[:, 96:110], coordinates[:, 128:130], coordinates[:, 126:128],
                                            coordinates[:, 124:126], coordinates[:, 122:124], coordinates[:, 120:122]],
                                           axis=-1),  # u_lip
                            np.concatenate([coordinates[:, 110:120], coordinates[:, 134:136], coordinates[:, 132:134],
                                            coordinates[:, 130:132]], axis=-1)]  # l_lip
        return sep_1_d_cord

    def _get_optimizer(self, lr=1e-2, beta_1=0.9, beta_2=0.999, decay=1e-5):
        return adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

    def _get_batch_sample(self, batch_index, x_train_filenames, y_train_filenames):
        img_path = self.train_images_dir
        hm_tr_path = self.train_hm_dir
        pn_tr_path = self.train_point_dir

        batch_x = x_train_filenames[
                  batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]
        batch_y = y_train_filenames[
                  batch_index * LearningConfig.batch_size:(batch_index + 1) * LearningConfig.batch_size]

        img_batch = np.array([imread(img_path + file_name) for file_name in batch_x])
        hm_batch = np.array([load(hm_tr_path + file_name) for file_name in batch_y])
        pn_batch = np.array([load(pn_tr_path + file_name) for file_name in batch_y])
        '''test:'''

        # imgpr.print_image_arr_heat('h_0', hm_batch[0])
        # imgpr.print_image_arr_heat('h_1', hm_batch[1])
        # image_utility = ImageUtility()
        # point_scaled, px_1, Py_1 = image_utility.create_landmarks_from_normalized(pn_batch[0], 224, 224, 112, 112)
        # point_scaled, px_2, Py_2 = image_utility.create_landmarks_from_normalized(pn_batch[1], 224, 224, 112, 112)
        # imgpr.print_image_arr('pts_0', img_batch[0], px_1, Py_1)
        # imgpr.print_image_arr('pts_1', img_batch[1], px_2, Py_2)

        return img_batch, hm_batch, pn_batch

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
