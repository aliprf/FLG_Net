from configuration import DatasetName, DatasetType, \
    AffectnetConf, IbugConf, W300Conf, InputDataSize, LearningConfig
from hg_Class import HourglassNet

import tensorflow as tf
from tensorflow.keras import backend as K

# import tf.keras
from skimage.transform import resize

from tensorflow.keras.regularizers import l2, l1

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.keras.models import Model
from tensorflow.keras.applications import mobilenet_v2, mobilenet, resnet50, densenet
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Conv2DTranspose, BatchNormalization,\
    Activation, GlobalAveragePooling2D, DepthwiseConv2D, Dropout, ReLU, Concatenate, Input, GlobalMaxPool2D

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras.callbacks import CSVLogger
from clr_callback import CyclicLR
from datetime import datetime

import cv2
import os.path
from scipy.spatial import distance
import scipy.io as sio
# from keras.engine import InputLayer
# import coremltools

import efficientnet.tfkeras as efn
# import efficientnet.keras as efn


class CNNModel:
    def get_model(self, input_tensor, arch, num_landmark, num_face_graph_elements, input_shape):

        if arch == 'effGlassNet':
            model = self.create_effGlassNet(input_shape=input_shape, input_tensor=input_tensor,
                                            num_landmark=num_landmark, num_face_graph_elements=num_face_graph_elements)
        elif arch == 'effDiscrimNet':
            model = self.create_effDiscrimNet(input_shape=input_shape, input_tensor=input_tensor)

        elif arch == 'hm_reg_model':
            model = self.create_hm_reg_model(input_shape=input_shape, input_tensor=input_tensor, num_landmark=num_landmark)
        elif arch == 'cord_reg_model':
            model = self.create_cord_reg_model(input_shape=input_shape, input_tensor=input_tensor, num_landmark=num_landmark)
        elif arch == 'hm_Disc_model':
            model = self.create_hm_disc_model(input_shape=input_shape, input_tensor=input_tensor)
        elif arch == 'cord_Disc_model':
            model = self.create_cord_disc_model(input_shape=input_shape, input_tensor=input_tensor)


        return model

    def create_hm_reg_model(self, input_shape, input_tensor, num_landmark):
        """
        This is EfficientNet-B7 combined with one stack of StackedHourGlassNetwork used as heatmap & geo regressor network.
        :param input_shape:
        :param input_tensor:
        :param num_landmark:
        :return: model
        """

        eff_net = efn.EfficientNetB0(include_top=True,
                                     weights=None,
                                     input_tensor=input_tensor,
                                     input_shape=input_shape,
                                     pooling=None,
                                     classes=num_landmark)  # or weights='noisy-student'

        eff_net.layers.pop()
        inp = eff_net.input

        top_activation = eff_net.get_layer('top_activation').output
        x = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            kernel_initializer='he_uniform')(top_activation)  # 14, 14, 256
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            kernel_initializer='he_uniform')(x)  # 56, 56, 256
        bn_0 = BatchNormalization(name='bn_0')(x)
        x = ReLU()(bn_0)

        '''reduce to  7'''
        x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(
            x)  # 28, 28, 256
        bn_1 = BatchNormalization(name='bn_1')(x)  # 28, 28, 256
        x = ReLU()(bn_1)

        x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(
            x)  # 14, 14, 256
        bn_2 = BatchNormalization(name='bn_2')(x)  # 14, 14, 256
        x = ReLU()(bn_2)

        x = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform')(
            x)  # 7, 7 , 256
        bn_3 = BatchNormalization(name='bn_3')(x)  # 7, 7 , 256
        x = ReLU()(bn_3)

        '''increase to  56'''
        x = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            name='deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization()(x)
        x = tf.keras.layers.add([x, bn_2])  # 14, 14, 256
        x = ReLU()(x)

        x = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            name='deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization()(x)
        x = tf.keras.layers.add([x, bn_1])  # 28, 28, 256
        x = ReLU()(x)

        x = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), padding='same',
                            name='deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization()(x)
        x = tf.keras.layers.add([x, bn_0])  # 56, 56, 256

        '''out heatmap regression'''
        out_heatmap = Conv2D(num_landmark//2, kernel_size=1, padding='same', name='O_hm')(x)
        eff_net = Model(inp, [out_heatmap])
        eff_net.summary()

        model_json = eff_net.to_json()
        with open("./model_arch/hm_reg_model.json", "w") as json_file:
            json_file.write(model_json)
        return eff_net

    def create_cord_reg_model(self, input_shape, input_tensor, num_landmark):
        """
        This is EfficientNet-B7 combined with one stack of StackedHourGlassNetwork used as heatmap & geo regressor network.
        :param input_shape:
        :param input_tensor:
        :param num_landmark:
        :return: model
        """

        eff_net = efn.EfficientNetB0(include_top=True,
                                     weights=None,
                                     input_tensor=input_tensor,
                                     input_shape=input_shape,
                                     pooling=None,
                                     classes=num_landmark)  # or weights='noisy-student'

        eff_net.layers.pop()
        inp = eff_net.input

        top_activation = eff_net.get_layer('top_activation').output

        '''out for geo regression'''
        x = GlobalAveragePooling2D()(top_activation)
        x = Dropout(0.5)(x)
        out_geo = Dense(num_landmark, name='O_geo')(x)
        #
        eff_net = Model(inp, out_geo)
        eff_net.summary()

        model_json = eff_net.to_json()
        with open("./model_arch/cord_reg_model.json", "w") as json_file:
            json_file.write(model_json)
        return eff_net

    def create_cord_disc_model(self, input_shape, input_tensor):

        inputs = inputs = tf.keras.Input(shape=(input_shape,))
        x = Dense(128, activation="relu")(inputs)
        x = Dense(128, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(.5)(x)
        outputs = Dense(1)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cord_disc_model")
        model.summary()

        # model_json = model.to_json()
        # with open("./model_arch/Disc_cord_model.json", "w") as json_file:
        #     json_file.write(model_json)
        return model


    def create_hm_disc_model(self, input_shape, input_tensor):
        """
        This is EfficientNet-B7 used as a binary classifier network.
        :param input_shape:
        :param input_tensor:
        :return: model
        """
        eff_net = efn.EfficientNetB0(include_top=True,
                                     weights=None,
                                     input_tensor=input_tensor,
                                     input_shape=input_shape,
                                     pooling=None,
                                     classes=1)
        eff_net.summary()
        model_json = eff_net.to_json()
        with open("./model_arch/Disc_hm_model.json", "w") as json_file:
            json_file.write(model_json)
        return eff_net

    def create_effNet(self, input_shape, input_tensor, num_landmark):
        """
        This is EfficientNet-B7 main network.
        :param input_shape:
        :param input_tensor:
        :param num_landmark:
        :return: model
        """
        eff_net = efn.EfficientNetB7(include_top=True,
                                     weights=None,
                                     input_tensor=input_tensor,
                                     input_shape=input_shape,
                                     pooling=None,
                                     classes=num_landmark)

        model_json = eff_net.to_json()
        with open("./model_arch/effNet-b7_main.json", "w") as json_file:
            json_file.write(model_json)
        return eff_net

    def create_resnetDiscrimNet(self, input_shape, input_tensor):
        eff_net = tf.keras.applications.resnet.ResNet50(include_top=True, weights=None, input_tensor=input_tensor,
                                                     input_shape=input_shape, pooling=None, classes=1)
        eff_net.summary()
        model_json = eff_net.to_json()
        with open("effDiscrimNet.json", "w") as json_file:
            json_file.write(model_json)
        return eff_net

    def hour_glass_network(self, num_classes=68, num_stacks=10, num_filters=256,
                           in_shape=(224, 224), out_shape=(56, 56)):
        hg_net = HourglassNet(num_classes=num_classes, num_stacks=num_stacks,
                              num_filters=num_filters,
                              in_shape=in_shape,
                              out_shape=out_shape)
        model = hg_net.build_model(mobile=True)
        return model

    def mobileNet_v2_main(self, tensor):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)
        # , classes=cnf.landmark_len)

        mobilenet_model.layers.pop()

        x = mobilenet_model.get_layer('global_average_pooling2d_1').output  # 1280
        x = Dense(LearningConfig.landmark_len, name='dense_layer_out_2', activation='relu',
                  kernel_initializer='he_uniform')(x)
        Logits = Dense(LearningConfig.landmark_len, name='out')(x)
        inp = mobilenet_model.input

        revised_model = Model(inp, Logits)

        revised_model.summary()
        # plot_model(revised_model, to_file='mobileNet_v2_main.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mobileNet_v2_main.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def calculate_flops(self, _arch, _output_len):

        if _arch == 'ASMNet':
            net = tf.keras.models.load_model('./final_weights/wflw_ds_.h5', compile=False)
        elif _arch == 'mobileNetV2':
            net = tf.keras.models.load_model('./final_weights/wflw_mn_.h5', compile=False)
        net._layers[0].batch_input_shape = (1, 224, 224, 3)

        with tf.Session(graph=tf.Graph()) as sess:
            K.set_session(sess)
            model_new = tf.keras.models.model_from_json(net.to_json())
            model_new.summary()

            #     x = net.get_layer('O_L').output  # 1280
            #     inp = net.input
            #     revised_model = Model(inp, [x])
            #     revised_model.build(tf.placeholder('float32', shape=(1, 448, 448, 3)))
            #     revised_model.summary()
            #     net = tf.tf.keras.models.load_model('./final_weights/ibug_ds_asm.h5')
            #     net = self.create_ASMNet(inp_tensor=None, inp_shape=(224, 224, 3), output_len=_output_len)
            #     net = self.create_ASMNet(inp_tensor=tf.placeholder('float32', shape=(1, 224, 224, 3)), inp_shape=None, output_len=_output_len)
            # elif _arch == 'mobileNetV2':
            #     # net = tf.tf.keras.models.load_model('./final_weights/ibug_mn_.h5')
            #
            #     # net = self.create_MobileNet(inp_tensor=None, inp_shape=(224, 224, 3), output_len=_output_len)
            #     # net = resnet50.ResNet50(input_shape=(224, 224, 3),  weights=None)
            #     # net = resnet50.ResNet50(input_tensor=tf.placeholder('float32', shape=(1, 224, 224, 3)),  weights=None)
            #     net = mobilenet_v2.MobileNetV2(alpha=1,  weights=None, input_tensor=tf.placeholder('float32', shape=(1, 224, 224, 3)))
            #     net.summary()

            run_meta = tf.RunMetadata()

            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

            opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
            params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

            print("FLOPS: {:,} --- Params: {:,}".format(flops.total_float_ops, params.total_parameters))
            return flops.total_float_ops, params.total_parameters
