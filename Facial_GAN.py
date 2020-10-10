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

tf.config.set_soft_device_placement(True)
print("IS GPU AVAIL:: ", str(tf.test.is_gpu_available()))


# tf.debugging.set_log_device_placement(True)


class FacialGAN:
    def __init__(self, dataset_name, hm_regressor_arch, cord_regressor_arch, hm_discriminator_arch,
                 cord_discriminator_arch, hm_regressor_weight, cord_regressor_weight, hm_discriminator_weight,
                 cord_discriminator_weight, input_shape_hm_reg, input_shape_cord_reg, input_shape_hm_disc,
                 input_shape_cord_disc):
        # tf.executing_eagerly()

        self.dataset_name = dataset_name
        self.hm_regressor_arch = hm_regressor_arch
        self.cord_regressor_arch = cord_regressor_arch
        self.hm_discriminator_arch = hm_discriminator_arch
        self.cord_discriminator_arch = cord_discriminator_arch
        self.hm_regressor_weight = hm_regressor_weight
        self.cord_regressor_weight = cord_regressor_weight
        self.hm_discriminator_weight = hm_discriminator_weight
        self.cord_discriminator_weight = cord_discriminator_weight
        self.input_shape_hm_reg = input_shape_hm_reg
        self.input_shape_cord_reg = input_shape_cord_reg
        self.input_shape_hm_disc = input_shape_hm_disc
        self.input_shape_cord_disc = input_shape_cord_disc

        if dataset_name == DatasetName.ibug:
            self.SUM_OF_ALL_TRAIN_SAMPLES = IbugConf.number_of_all_sample
            self.num_landmark = IbugConf.num_of_landmarks * 2
            self.num_face_graph_elements = IbugConf.num_face_graph_elements
            self.train_images_dir = IbugConf.train_images_dir
            self.train_hm_dir = IbugConf.train_hm_dir
            self.train_point_dir = IbugConf.normalized_points_npy_dir
            self.num_face_graph_elements = IbugConf.num_face_graph_elements
            self.hm_stride = IbugConf.hm_stride
        elif dataset_name == DatasetName.cofw:
            self.SUM_OF_ALL_TRAIN_SAMPLES = CofwConf.number_of_all_sample
            self.num_landmark = CofwConf.num_of_landmarks * 2
            self.num_face_graph_elements = CofwConf.num_face_graph_elements
            self.train_images_dir = CofwConf.train_images_dir
            self.train_hm_dir = CofwConf.train_hm_dir
            self.train_point_dir = CofwConf.normalized_points_npy_dir
            self.num_face_graph_elements = CofwConf.num_face_graph_elements
            self.hm_stride = CofwConf.hm_stride
        elif dataset_name == DatasetName.wflw:
            self.SUM_OF_ALL_TRAIN_SAMPLES = WflwConf.number_of_all_sample
            self.num_landmark = WflwConf.num_of_landmarks * 2
            self.num_face_graph_elements = WflwConf.num_face_graph_elements
            self.train_images_dir = WflwConf.train_images_dir
            self.train_hm_dir = WflwConf.train_hm_dir
            self.train_point_dir = WflwConf.normalized_points_npy_dir
            self.num_face_graph_elements = WflwConf.num_face_graph_elements
            self.hm_stride = WflwConf.hm_stride

        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # assert (gpus)
        # if gpus:
        #     try:
        #         # Currently, memory growth needs to be the same across GPUs
        #         for gpu in gpus:
        #             tf.config.experimental.set_memory_growth(gpu, True)
        #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        #     except RuntimeError as e:
        #         print("NO GPU!!!!")

    def make_hm_generator_model(self):
        cnn = CNNModel()
        model = cnn.get_model(input_tensor=None, arch=self.hm_regressor_arch, num_landmark=self.num_landmark,
                              input_shape=self.input_shape_hm_reg, num_face_graph_elements=None)
        if self.hm_regressor_weight is not None:
            model.load_weights(self.hm_regressor_weight)
        return model

    def make_cord_generator_model(self):
        cnn = CNNModel()
        model = cnn.get_model(input_tensor=None, arch=self.cord_regressor_arch, num_landmark=self.num_landmark,
                              input_shape=self.input_shape_cord_reg, num_face_graph_elements=None)
        if self.cord_regressor_weight is not None:
            model.load_weights(self.cord_regressor_weight)
        return model

    def make_hm_discriminator_model(self):
        cnn = CNNModel()
        model = cnn.get_model(input_tensor=None, arch=self.hm_discriminator_arch, num_landmark=self.num_landmark,
                              input_shape=self.input_shape_hm_disc, num_face_graph_elements=None)
        if self.hm_discriminator_weight is not None:
            model.load_weights(self.hm_discriminator_weight)
        return model

    def make_cord_discriminator_model(self):
        cnn = CNNModel()
        model = cnn.get_model(input_tensor=None, arch=self.cord_discriminator_arch, num_landmark=self.num_landmark,
                              input_shape=self.input_shape_cord_disc, num_face_graph_elements=None)
        if self.cord_discriminator_weight is not None:
            model.load_weights(self.cord_discriminator_weight)
        return model

    def _create_ckpt(self, epoch, hm_generator_optimizer, cord_generator_optimizer, hm_discriminator_optimizer,
                     cord_discriminator_optimizer, hm_generator, cord_generator, hm_discriminator, cord_discriminator):
        checkpoint_dir = './training_checkpoints/'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(hm_generator_optimizer=hm_generator_optimizer,
                                         cord_generator_optimizer=cord_generator_optimizer,
                                         hm_discriminator_optimizer=hm_discriminator_optimizer,
                                         cord_discriminator_optimizer=cord_discriminator_optimizer,
                                         hm_generator=hm_generator,
                                         cord_generator=cord_generator,
                                         hm_discriminator=hm_discriminator,
                                         cord_discriminator=cord_discriminator)
        checkpoint.save(file_prefix=checkpoint_prefix)
        '''save model weights'''
        hm_generator.save_weights(checkpoint_dir + 'hm_reg_' + str(epoch) + '_.h5')
        cord_generator.save_weights(checkpoint_dir + 'cord_reg_' + str(epoch) + '_.h5')
        hm_discriminator.save_weights(checkpoint_dir + 'hm_disc_' + str(epoch) + '_.h5')
        cord_discriminator.save_weights(checkpoint_dir + 'cord_disc_' + str(epoch) + '_.h5')

    def calc_hm_to_pts_MAE(self, hm, pts, epoch):
        weight = 0.2
        # if epoch < 25:
        #     return 0
        # elif 50 < epoch < 100:
        #     weight = 0.3

        x_center = 112
        width = 224
        hm_arr = []
        tf_util = TFRecordUtility(self.num_landmark // 2)

        for i in range(LearningConfig.batch_size):
            hm_t = tf_util.from_heatmap_to_point_tensor(heatmaps=hm[i], number_of_points=5)
            hm_t = tf.reshape(tensor=hm_t, shape=self.num_landmark)
            '''hm is in [0,224] --> should be in [-0.5,+0.5]'''
            hm_t_norm = tf.math.scalar_mul(scalar=1 / width, x=tf.math.subtract(hm_t, np.repeat(x_center, self.num_landmark)))
            hm_arr.append(hm_t_norm)
        '''reshape hm'''
        hm_pts = tf.stack([hm_arr[i] for i in range(LearningConfig.batch_size)], 0)  # bs * self.num_landmark
        mae = tf.reduce_mean(tf.abs(hm_pts - pts))
        return mae*weight

    def calc_pts_to_hm_MAE(self, hm, pts, epoch):
        weight = 0.2
        # if epoch < 25:
        #     return 0
        # elif 50 < epoch < 100:
        #     weight = 0.3
        """
        we want to convert pts to hm{56 * 56 * self.num_landmark//2} and then calculate loss
        :param hm: 56 * 56 * self.num_landmark//2
        :param pts: bs * self.num_landmark: pts are normal between -0.5, +0.5 --> need to upsample
        :return:
        """
        hm_arr = []
        tf_util = TFRecordUtility(self.num_landmark // 2)
        for i in range(LearningConfig.batch_size):
            hm_t = tf_util.generate_hm(height=56, width=56, landmarks=pts[i], s=3.0, upsample=True)
            hm_arr.append(hm_t)
        hm_pts = tf.convert_to_tensor(np.array(hm_arr))
        '''calculate MAE'''
        mae = tf.reduce_mean(tf.abs(hm_pts - hm))
        return mae*weight

    def hm_discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return real_loss, fake_loss, total_loss

    def hm_regressor_loss(self, hm_gr, hm_pr, pnt_gr, pnt_pr, epoch):
        """"""
        '''defining hyper parameters'''
        w_reg_loss = 100
        '''calculating regression loss and discriminator'''
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss_reg = tf.reduce_mean(tf.abs(hm_gr - hm_pr))
        loss_discrimination = cross_entropy(tf.ones_like(hm_pr), hm_pr)

        '''calculating hm_pr VS points_gr loss'''
        if False or epoch < 25:
            loss_regression = w_reg_loss * loss_reg
        else:
            loss_hm_pts = self.calc_pts_to_hm_MAE(hm=hm_pr, pts=pnt_pr, epoch=epoch)
            loss_regression = w_reg_loss * (loss_reg + loss_hm_pts)

        '''creating Total loss'''
        loss_total = loss_regression + loss_discrimination

        return loss_regression, loss_discrimination, loss_total

    def coord_regressor_loss(self, pnt_gr, pnt_pr, hm_gr, hm_pr, epoch):
        """"""
        '''defining hyper parameters'''
        w_reg_loss = 100
        '''calculating regression loss and discriminator'''
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss_reg = tf.reduce_mean(tf.abs(pnt_gr - pnt_pr))
        loss_discrimination = cross_entropy(tf.ones_like(pnt_pr), pnt_pr)

        '''calculating hm_pr VS points_gr loss'''
        if False or epoch < 25:
            loss_regression = w_reg_loss * loss_reg
        else:
            loss_hm_pts = self.calc_hm_to_pts_MAE(hm=hm_pr, pts=pnt_pr, epoch=epoch)
            loss_regression = w_reg_loss * (loss_reg + loss_hm_pts)
        '''creating Total loss'''
        loss_total = loss_regression + loss_discrimination
        return loss_regression, loss_discrimination, loss_total

    def coord_discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return real_loss, fake_loss, total_loss

    # @tf.function
    def train_step(self, epoch, step, images, heatmaps_gr, points_gr, hm_reg_model, hm_disc_model, cord_reg_model,
                   cord_disc_model, hm_reg_optimizer, hm_disc_optimizer, cord_reg_optimizer, cord_disc_optimizer,
                   summary_writer):

        with tf.GradientTape() as hm_reg_tape, tf.GradientTape() as hm_disc_tape, \
                tf.GradientTape() as cord_reg_tape, tf.GradientTape() as cord_disc_tape:
            '''prediction'''
            heatmaps_pr = hm_reg_model(images)
            points_pr = cord_reg_model(images)

            '''hm:     create disc'''
            real_hm = hm_disc_model([images, heatmaps_gr])
            fake_hm = hm_disc_model([images, heatmaps_pr])
            '''cord:     create disc:  for now just sent point, but maybe we need to fuse points with flatten hn'''
            real_cord = cord_disc_model(points_gr)
            fake_cord = cord_disc_model(points_pr)

            '''loss calculation'''
            '''     hm loss'''
            hm_loss_reg, hm_loss_disc, hm_reg_total_loss = self.hm_regressor_loss(hm_gr=heatmaps_gr, hm_pr=heatmaps_pr,
                                                                                  pnt_gr=points_gr, pnt_pr=points_pr,
                                                                                  epoch=epoch)
            hm_real_loss, hm_fake_loss, hm_disc_total_loss = self.hm_discriminator_loss(real_output=real_hm,
                                                                                        fake_output=fake_hm)
            '''     coordinate loss'''
            c_loss_reg, c_loss_disc, c_reg_total_loss = self.coord_regressor_loss(pnt_gr=points_gr, pnt_pr=points_pr,
                                                                                  hm_gr=heatmaps_gr, hm_pr=heatmaps_pr,
                                                                                  epoch=epoch)
            c_real_loss, c_fake_loss, c_disc_total_loss = self.coord_discriminator_loss(real_output=real_cord,
                                                                                        fake_output=fake_cord)
            '''create custom loss'''
            '''     hm: put more focus'''
            hm_weight = 5
            hm_reg_total_loss = hm_weight * hm_reg_total_loss
            hm_disc_total_loss = hm_weight / 2 * hm_disc_total_loss
            '''      Xor loss for hm & cord discriminator'''

        ''' Calculate: Gradients'''
        '''     hm: '''
        gradients_of_hm_reg = hm_reg_tape.gradient(hm_reg_total_loss, hm_reg_model.trainable_variables)
        gradients_of_hm_disc = hm_disc_tape.gradient(hm_disc_total_loss, hm_disc_model.trainable_variables)
        '''     coord'''
        gradients_of_c_reg = cord_reg_tape.gradient(c_reg_total_loss, cord_reg_model.trainable_variables)
        gradients_of_c_disc = cord_disc_tape.gradient(c_disc_total_loss, cord_disc_model.trainable_variables)

        '''apply Gradients:'''
        '''     hm: '''
        hm_reg_optimizer.apply_gradients(zip(gradients_of_hm_reg, hm_reg_model.trainable_variables))
        hm_disc_optimizer.apply_gradients(zip(gradients_of_hm_disc, hm_disc_model.trainable_variables))
        '''     coord'''
        cord_reg_optimizer.apply_gradients(zip(gradients_of_c_reg, cord_reg_model.trainable_variables))
        cord_disc_optimizer.apply_gradients(zip(gradients_of_c_disc, cord_disc_model.trainable_variables))

        '''printing loss Values: '''
        tf.print("->EPOCH: ", str(epoch), "->STEP: ", str(step),
                 "|||HEATMAP:->", 'hm_reg_total_loss:{', hm_reg_total_loss, '}', 'hm_loss_reg:{ ', hm_loss_reg, '} hm_loss_disc:{', hm_loss_disc,
                 '} hm_disc_total_loss:{', hm_disc_total_loss, '}', 'hm_real_loss:{ ', hm_real_loss, '} hm_fake_loss:{', hm_fake_loss,

                 "|||COORD:->", 'c_reg_total_loss:{', c_reg_total_loss, '}', 'c_loss_reg:{ ', c_loss_reg, '} c_loss_disc:{', c_loss_disc,
                 '} c_disc_total_loss:{', c_disc_total_loss, '}', 'c_real_loss:{ ', c_real_loss, '} c_fake_loss:{', c_fake_loss,
                 )

        with summary_writer.as_default():
            '''     hm'''
            tf.summary.scalar('hm_g_total_loss', hm_reg_total_loss, step=epoch)
            tf.summary.scalar('hm_d_total_loss', hm_disc_total_loss, step=epoch)
            tf.summary.scalar('hm_loss_reg', hm_loss_reg, step=epoch)
            tf.summary.scalar('hm_loss_disc', hm_loss_disc, step=epoch)
            '''     coord'''

    def train(self):
        summary_writer = tf.summary.create_file_writer(
            "./train_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        hm_reg_model = self.make_hm_generator_model()
        hm_disc_model = self.make_hm_discriminator_model()
        cord_reg_model = self.make_cord_generator_model()
        cord_disc_model = self.make_cord_discriminator_model()

        hm_reg_optimizer = self._get_optimizer(lr=2e-4)
        hm_disc_optimizer = self._get_optimizer(lr=1e-4)
        cord_reg_optimizer = self._get_optimizer(lr=2e-2)
        cord_disc_optimizer = self._get_optimizer(lr=1e-2)

        x_train_filenames, x_val_filenames, y_train_filenames, y_val_filenames = self._create_generators()

        step_per_epoch = len(x_train_filenames) // LearningConfig.batch_size

        for epoch in range(LearningConfig.epochs):
            for batch_index in range(step_per_epoch):
                images, heatmaps_gr, points_gr = self._get_batch_sample(batch_index, x_train_filenames,
                                                                        y_train_filenames)

                images = tf.cast(images, tf.float32)
                points_gr = tf.cast(points_gr, tf.float32)

                self.train_step(epoch=epoch, step=batch_index, images=images, heatmaps_gr=heatmaps_gr,
                                points_gr=points_gr, hm_reg_model=hm_reg_model,
                                hm_disc_model=hm_disc_model, cord_reg_model=cord_reg_model,
                                cord_disc_model=cord_disc_model,
                                hm_reg_optimizer=hm_reg_optimizer, hm_disc_optimizer=hm_disc_optimizer,
                                cord_reg_optimizer=cord_reg_optimizer, cord_disc_optimizer=cord_disc_optimizer,
                                summary_writer=summary_writer)
            # if (epoch + 1) % 2 == 0:
            #     self._create_ckpt(epoch=epoch, hm_generator_optimizer=hm_reg_optimizer,
            #                       cord_generator_optimizer=cord_reg_optimizer,
            #                       hm_discriminator_optimizer=hm_disc_optimizer,
            #                       cord_discriminator_optimizer=cord_disc_optimizer,
            #                       hm_generator=hm_reg_model, cord_generator=cord_reg_model,
            #                       hm_discriminator=hm_disc_model, cord_discriminator=cord_disc_model)

        # -----------------------------------------------------

    def _create_cord_regressor_net(self, input_tensor, input_shape, is_trainable=True):
        """
        This is the main network, we use for predicting points:
        input:
            X: img
            Y: [points]

        :param input_tensor:
        :param input_shape:
        :return: keras model created for the geo-hm regression task.
        """
        cnn = CNNModel()
        model = cnn.get_model(input_tensor=input_tensor, arch=self.cord_regressor_arch, num_landmark=self.num_landmark,
                              input_shape=input_shape, num_face_graph_elements=None)
        if self.cord_regressor_weight is not None:
            model.load_weights(self.cord_regressor_weight)
        model.trainable = is_trainable
        model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=self._get_optimizer(), metrics=['mse'])
        return model

    def _create_hm_regressor_net(self, input_tensor, input_shape, is_trainable=True):
        """
        This is the main network, we use for predicting hm:
        input:
            X: img
            Y: [hm]

        :param input_tensor:
        :param input_shape:
        :return: keras model created for the geo-hm regression task.
        """
        cnn = CNNModel()
        model = cnn.get_model(input_tensor=input_tensor, arch=self.hm_regressor_arch, num_landmark=self.num_landmark,
                              input_shape=input_shape, num_face_graph_elements=None)
        if self.hm_regressor_weight is not None:
            model.load_weights(self.hm_regressor_weight)

        model.trainable = is_trainable

        model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=self._get_optimizer(), metrics=['mse'])
        return model

    def _create_hm_discriminator_net(self, input_tensor, input_shape, is_trainable=True):
        """
        This is the discriminator network, being used at the second stage when we want to discriminate
        the real and fake data, generated by the RegressorNetwork
        :param input_tensor:
        :param input_shape:
        :return:
        """

        cnn = CNNModel()
        model = cnn.get_model(input_tensor=input_tensor, arch=self.hm_discriminator_arch, input_shape=input_shape,
                              num_landmark=self.num_landmark, num_face_graph_elements=None)
        if self.hm_discriminator_weight is not None:
            model.load_weights(self.hm_discriminator_weight)
        model.trainable = is_trainable
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=self._get_optimizer(lr=1e-4, decay=1e-6),
                      metrics=['accuracy'])
        return model

    def _create_cord_discriminator_net(self, input_tensor, input_shape, is_trainable=True):
        """
        This is the discriminator network, being used at the second stage when we want to discriminate
        the real and fake data, generated by the RegressorNetwork
        :param input_tensor:
        :param input_shape:
        :return:
        """
        cnn = CNNModel()
        model = cnn.get_model(input_tensor=input_tensor, arch=self.cord_discriminator_arch, input_shape=input_shape,
                              num_landmark=self.num_landmark, num_face_graph_elements=None)
        if self.cord_discriminator_weight is not None:
            model.load_weights(self.cord_discriminator_weight)
        model.trainable = is_trainable
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=self._get_optimizer(lr=1e-4, decay=1e-6),
                      metrics=['accuracy'])
        return model

    def flatten_hm(self, heatmaps, lbl):
        result = tf.expand_dims(tf.math.reduce_sum(heatmaps, axis=-1), axis=3)
        # for i in range(LearningConfig.batch_size):
        #     imgpr.print_image_arr_heat(lbl + '_' + str(i), result[i])
        return result

    def fuse_hm(self, heatmaps, points):
        # return tf.expand_dims(tf.math.reduce_sum(heatmaps, axis=-1), axis=3)
        heatmaps_r = tf.expand_dims(tf.math.reduce_sum(heatmaps, axis=-1), axis=3)

        # heatmaps = K.eval(heatmaps)
        # points = K.eval(points)
        fg_gt = np.expand_dims(self._points_to_2d(points), axis=3)  # the result should be [bs, 56,56,1]
        # heatmaps_gr = np.expand_dims(np.sum(heatmaps, axis=-1), axis=3)
        return np.subtract(heatmaps_r, fg_gt)

    def _prepare_hm_discriminator_model_input(self, heatmaps_gr, heatmaps_pr, points_gr, points_pr, img, index):
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
        img = resize(img, (LearningConfig.batch_size, InputDataSize.hm_size, InputDataSize.hm_size, 3))

        '''convert points to 2d facial graph:'''
        fg_gt = np.expand_dims(self._points_to_2d(points_gr), axis=3)  # the result should be [bs, 56,56,1]
        fg_pr = np.expand_dims(self._points_to_2d(points_pr), axis=3)
        '''sum heatmaps'''
        heatmaps_gr = np.expand_dims(np.sum(heatmaps_gr, axis=-1), axis=3)
        heatmaps_pr = np.expand_dims(np.sum(heatmaps_pr, axis=-1), axis=3)

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
        # real_X = np.concatenate([heatmaps_gr, fg_gt], axis=-1)
        # fake_X = np.concatenate([heatmaps_pr, fg_pr], axis=-1)

        real_X = np.subtract(heatmaps_gr, fg_gt)
        fake_X = np.subtract(heatmaps_pr, fg_pr)

        # imgpr.print_image_arr_heat('real_0' + str(index), real_X[0], print_single=False)
        # imgpr.print_image_arr_heat('real_1' + str(index), real_X[1], print_single=False)
        # imgpr.print_image_arr_heat('fake_0' + str(index), fake_X[0], print_single=False)
        # imgpr.print_image_arr_heat('fake_1' + str(index), fake_X[1], print_single=False)

        '''create real/fake labels and attach to the input data'''
        X = np.concatenate((fake_X, real_X), axis=0)
        Y = np.zeros(2 * LearningConfig.batch_size)
        Y[:LearningConfig.batch_size] = 0.9  # use smooth labels

        return X, Y

    def _points_to_2d(self, _points):
        """

        :param _points:
        :return:
        """
        tf_rec = TFRecordUtility(self.num_landmark)
        image_utility = ImageUtility()
        hm_arr = []
        for i in range(LearningConfig.batch_size):
            _x_y, _x, _y = image_utility.create_landmarks_from_normalized(_points[i], InputDataSize.image_input_size,
                                                                          InputDataSize.image_input_size,
                                                                          InputDataSize.img_center,
                                                                          InputDataSize.img_center)
            hm_multi_layer = tf_rec.generate_hm(InputDataSize.hm_size, InputDataSize.hm_size, np.array(_x_y),
                                                self.hm_stride / 2, False)
            hm = np.sum(hm_multi_layer, axis=2)
            hm_arr.append(hm)
        return np.array(hm_arr)

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
            points_2d.append(pnt.reshape([LearningConfig.batch_size, len(pnt[1]) // 2, 2]))
        '''create the spare img for each facial part:'''
        hm_img = np.zeros(
            [LearningConfig.batch_size, InputDataSize.hm_size, InputDataSize.hm_size, self.num_face_graph_elements])
        # bs, 12 * 2
        for i in range(LearningConfig.batch_size):
            for j in range(self.num_face_graph_elements):
                t_hm = np.zeros([InputDataSize.hm_size, InputDataSize.hm_size])
                for x_y in points_2d[j][i]:
                    if not (0 <= x_y[0] <= InputDataSize.hm_size - 1):
                        x_y[0] = 0
                    if not (0 <= x_y[1] <= InputDataSize.hm_size - 1):
                        x_y[1] = 0
                    t_hm[int(x_y[1]), int(x_y[0])] = 1

                hm_img[i, :, :, j] = t_hm

        return hm_img

    def _fuse_hm_with_points(self, hm_tensor, points_tensor, img_tensor):
        """
        convert hm_tensor{bs, 56,56, 68} ==> {bs, 56,56,1} and concat it with the img{224, 224, 3}
        :param hm_tensor:
        :param points_tensor:
        :param img_tensor:
        :return:
        """
        return Lambda(lambda x: tf.expand_dims(tf.math.reduce_sum(x, axis=3), axis=3))(hm_tensor)

        # hm_tensor_t = tf.expand_dims(tf.math.reduce_sum(hm_tensor, axis=3), axis=3))

        # hm_tensor_summed = tf.expand_dims(tf.math.reduce_sum(hm_tensor, axis=3), axis=3)
        # t_pn_img_0 = self._convert_to_geometric(hm_tensor_summed, tf.cast(points_tensor, 'int64'))

        hm_tensor_summed = Lambda(lambda x: tf.expand_dims(tf.math.reduce_sum(x, axis=3), axis=3))(hm_tensor)
        # t_pn_img_0 = self._convert_to_geometric(hm_tensor_summed, tf.cast(points_tensor, 'int64'))

        t_pn_img_0 = Lambda(lambda x: self._convert_to_geometric(hm_tensor_summed, tf.cast(x, 'int64')))(points_tensor)
        t_pn_img = Lambda(lambda x: tf.reshape(tensor=x, shape=[tf.shape(hm_tensor_summed)[0], InputDataSize.hm_size,
                                                                InputDataSize.hm_size, 1]))(t_pn_img_0)

        # concat = Lambda(lambda x: keras.layers.Concatenate()([hm_tensor_summed, x]))(t_pn_img_0)
        concat = keras.layers.Concatenate()([hm_tensor_summed, t_pn_img])
        return concat

        '''fuse img + hm '''
        img_tensor_c = tf.image.resize(img_tensor, [InputDataSize.hm_size, InputDataSize.hm_size])
        hm_tensor_t = tf.expand_dims(tf.math.reduce_sum(hm_tensor, axis=3), axis=3)
        hm_tensor_c = tf.reshape(tensor=hm_tensor_t, shape=[tf.shape(img_tensor_c)[0], InputDataSize.hm_size,
                                                            InputDataSize.hm_size, 1])

        return Lambda(lambda x: keras.layers.Concatenate()([hm_tensor_c, img_tensor_c]))

        # '''the expected one'''
        # t_hm = regressor_output[0]
        # t_hm_cp = regressor_output[0]
        # t_pn = regressor_output[1]
        #
        # t_inp_img = tf.image.resize(regressor_input, [InputDataSize.hm_size, InputDataSize.hm_size])
        # t_hm = keras.layers.Concatenate()([t_inp_img, t_hm])
        #
        # t_pn_img = Lambda(lambda x: self._convert_to_geometric(t_hm_cp, tf.cast(x, 'int64')))(t_pn)
        # t_fused = Lambda(lambda x: self._fuse_tensors(t_hm, x))(t_pn_img)
        # return t_fused

    def _fuse_points_with_hms(self, points_tensor, hm_tensor):
        """
        :param points_tensor:
        :param img_tensor:
        :return:
        """
        hm_tensor = Lambda(lambda x: tf.expand_dims(tf.math.reduce_sum(x, axis=3), axis=3))(hm_tensor)

        return Lambda(lambda x: self._convert_to_geometric(hm_tensor, tf.cast(x, 'int64')))(points_tensor)

        # return points_tensor

        '''the expected one'''
        t_hm = regressor_output[0]
        t_hm_cp = regressor_output[0]
        t_pn = regressor_output[1]

        t_inp_img = tf.image.resize(regressor_input, [InputDataSize.hm_size, InputDataSize.hm_size])
        t_hm = keras.layers.Concatenate()([t_inp_img, t_hm])

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
        :param hm_img: bs * 56 * 56 * 1
        :param coordinates: bs * 136
        :return:
        """

        '''create a clean copy of generated image '''
        img_indices = tf.constant([[x] for x in range(LearningConfig.batch_size)])
        img_updates = tf.zeros([LearningConfig.batch_size, InputDataSize.hm_size,
                                InputDataSize.hm_size, 1], dtype=tf.float32)
        hm_img = tf.tensor_scatter_nd_update(hm_img, img_indices, img_updates)

        '''convert two_d_coords to facial part:'''
        # sep_1_d_cord = self._slice_face_graph_tensor(coordinates)
        sep_1_d_cord = coordinates

        '''convert all points to 2d:'''
        counter = 0
        indices = []

        '''convert to hm_scale: from  { from (-0.5, -0.5): (0.5, 0.5) => (0, 0): (56, 56)}'''
        cord_item = tf.map_fn(fn=lambda landmark: InputDataSize.hm_center + landmark * InputDataSize.hm_size,
                              elems=sep_1_d_cord)
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
                item = [batch_indx, y, x, counter]  # the coordination should be y,x NOT x,y
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
        return tf.keras.optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)
        # return adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=decay)

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
