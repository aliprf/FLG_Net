from configuration import DatasetName, DatasetType, \
    AffectnetConf, IbugConf, W300Conf, InputDataSize, LearningConfig
from hg_Class import HourglassNet

import tensorflow as tf
import keras
from skimage.transform import resize

from keras.regularizers import l2, l1

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.models import Model
from keras.applications import mobilenet_v2, mobilenet, resnet50, densenet
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, \
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, Dropout, ReLU, Concatenate, \
    Deconvolution2D, Input, GlobalMaxPool2D

from keras.callbacks import ModelCheckpoint
from keras import backend as K

from keras.optimizers import adam
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.callbacks import CSVLogger
from clr_callback import CyclicLR
from datetime import datetime

import cv2
import os.path
from keras.utils.vis_utils import plot_model
from scipy.spatial import distance
import scipy.io as sio
from keras.engine import InputLayer
# import coremltools

import efficientnet.keras as efn


class CNNModel:
    def get_model(self, train_images, arch, num_output_layers, output_len):

        if arch == 'ASMNet':
            # self.calculate_flops(arch, output_len)
            model = self.create_ASMNet(inp_tensor=train_images, inp_shape=None, output_len=output_len)

        elif arch == 'mobileNetV2':
            # self.calculate_flops(arch, output_len)
            model = self.create_MobileNet(inp_tensor=train_images, output_len=output_len, inp_shape=None)

        elif arch == 'mobileNetV2_nopose':
            model = self.create_MobileNet_nopose(inp_tensor=train_images, output_len=output_len)

        elif arch == 'efficientNet':
            model = self.create_efficientNet(inp_shape=[224,224,3], input_tensor=train_images, output_len=output_len)

        return model

    def calculate_flops(self, _arch, _output_len):

        if _arch == 'ASMNet':
            net = keras.models.load_model('./final_weights/wflw_ds_.h5', compile=False)
        elif _arch == 'mobileNetV2':
            net = keras.models.load_model('./final_weights/wflw_mn_.h5', compile=False)
        net._layers[0].batch_input_shape = (1, 224, 224, 3)

        with tf.Session(graph=tf.Graph()) as sess:
            K.set_session(sess)
            model_new = keras.models.model_from_json(net.to_json())
            model_new.summary()

            #     x = net.get_layer('O_L').output  # 1280
            #     inp = net.input
            #     revised_model = Model(inp, [x])
            #     revised_model.build(tf.placeholder('float32', shape=(1, 448, 448, 3)))
            #     revised_model.summary()
            #     net = tf.keras.models.load_model('./final_weights/ibug_ds_asm.h5')
            #     net = self.create_ASMNet(inp_tensor=None, inp_shape=(224, 224, 3), output_len=_output_len)
            #     net = self.create_ASMNet(inp_tensor=tf.placeholder('float32', shape=(1, 224, 224, 3)), inp_shape=None, output_len=_output_len)
            # elif _arch == 'mobileNetV2':
            #     # net = tf.keras.models.load_model('./final_weights/ibug_mn_.h5')
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

    def create_MobileNet_nopose(self, inp_tensor, output_len):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=inp_tensor,
                                                   pooling=None)
        # model_json = mobilenet_model.to_json()
        #
        # with open("mobileNet_v2_main.json", "w") as json_file:
        #     json_file.write(model_json)
        #
        # return mobilenet_model

        mobilenet_model.layers.pop()

        x = mobilenet_model.get_layer('global_average_pooling2d_1').output  # 1280
        out_landmarks = Dense(output_len, name='O_L')(x)
        out_poses = Dense(LearningConfig.pose_len, name='O_P')(x)

        inp = mobilenet_model.input

        revised_model = Model(inp, [out_landmarks])

        revised_model.summary()
        # plot_model(revised_model, to_file='mobileNet_v2_main.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mobileNet_v2_main_multi_out.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def create_efficientNet(self, inp_shape, input_tensor, output_len, is_teacher=True):
        if is_teacher:  # for teacher we use a heavier network
            eff_net = efn.EfficientNetB3(include_top=True,
                                         weights=None,
                                         input_tensor=input_tensor,
                                         input_shape=inp_shape,
                                         pooling=None,
                                         classes=output_len)
            # return self._create_efficientNet_3deconv(inp_shape, input_tensor, output_len)
        else:  # for student we use the small network
            eff_net = efn.EfficientNetB0(include_top=True,
                                         weights=None,
                                         input_tensor=input_tensor,
                                         input_shape=inp_shape,
                                         pooling=None,
                                         classes=output_len)  # or weights='noisy-student'

        eff_net.layers.pop()
        inp = eff_net.input

        x = eff_net.get_layer('top_activation').output
        x = GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(rate=0.3)(x)
        output = Dense(output_len, activation='linear', name='out')(x)

        eff_net = Model(inp, output)

        eff_net.summary()

        # plot_model(eff_net, to_file='eff_net.png', show_shapes=True, show_layer_names=True)

        # tf.keras.utils.plot_model(
        #     eff_net,
        #     to_file="eff_net.png",
        #     show_shapes=False,
        #     show_layer_names=True,
        #     rankdir="TB"
        # )

        # model_json = eff_net.to_json()
        # with open("eff_net.json", "w") as json_file:
        #     json_file.write(model_json)
        return eff_net

    def create_MobileNet(self, inp_tensor, output_len, inp_shape):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=inp_tensor,
                                                   pooling=None)
        # model_json = mobilenet_model.to_json()
        #
        # with open("mobileNet_v2_main.json", "w") as json_file:
        #     json_file.write(model_json)
        #
        # return mobilenet_model

        mobilenet_model.layers.pop()

        x = mobilenet_model.get_layer('global_average_pooling2d_1').output  # 1280
        out_landmarks = Dense(output_len, name='O_L')(x)
        out_poses = Dense(LearningConfig.pose_len, name='O_P')(x)

        inp = mobilenet_model.input

        revised_model = Model(inp, [out_landmarks, out_poses])

        revised_model.summary()
        # plot_model(revised_model, to_file='mobileNet_v2_main.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mobileNet_v2_main_multi_out.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model


    def create_ASMNet(self, output_len, inp_tensor=None, inp_shape=None):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=inp_tensor,
                                                   pooling=None)
        mobilenet_model.layers.pop()
        inp = mobilenet_model.input

        '''heatmap can not be generated from activation layers, so we use out_relu'''
        block_1_project_BN = mobilenet_model.get_layer('block_1_project_BN').output  # 56*56*24
        block_1_project_BN_mpool = GlobalAveragePooling2D()(block_1_project_BN)

        block_3_project_BN = mobilenet_model.get_layer('block_3_project_BN').output  # 28*28*32
        block_3_project_BN_mpool = GlobalAveragePooling2D()(block_3_project_BN)

        block_6_project_BN = mobilenet_model.get_layer('block_6_project_BN').output  # 14*14*64
        block_6_project_BN_mpool = GlobalAveragePooling2D()(block_6_project_BN)

        block_10_project_BN = mobilenet_model.get_layer('block_10_project_BN').output  # 14*14*96
        block_10_project_BN_mpool = GlobalAveragePooling2D()(block_10_project_BN)

        block_13_project_BN = mobilenet_model.get_layer('block_13_project_BN').output  # 7*7*160
        block_13_project_BN_mpool = GlobalAveragePooling2D()(block_13_project_BN)

        block_15_add = mobilenet_model.get_layer('block_15_add').output  # 7*7*160
        block_15_add_mpool = GlobalAveragePooling2D()(block_15_add)

        x = keras.layers.Concatenate()([block_1_project_BN_mpool, block_3_project_BN_mpool, block_6_project_BN_mpool,
                                        block_10_project_BN_mpool, block_13_project_BN_mpool, block_15_add_mpool])
        x = keras.layers.Dropout(rate=0.3)(x)
        ''''''
        out_landmarks = Dense(output_len,
                              kernel_regularizer=l2(0.01),
                              # activity_regularizer=l1(0.01),
                              bias_regularizer=l2(0.01),
                              name='O_L')(x)
        out_poses = Dense(LearningConfig.pose_len,
                          kernel_regularizer=l2(0.01),
                          # activity_regularizer=l1(0.01),
                          bias_regularizer=l2(0.01),
                          name='O_P')(x)

        revised_model = Model(inp, [out_landmarks, out_poses])

        revised_model.summary()
        model_json = revised_model.to_json()

        with open("ASMNet.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def hour_glass_network(self, num_classes=68, num_stacks=10, num_filters=256,
                           in_shape=(224, 224), out_shape=(56, 56)):
        hg_net = HourglassNet(num_classes=num_classes, num_stacks=num_stacks,
                              num_filters=num_filters,
                              in_shape=in_shape,
                              out_shape=out_shape)
        model = hg_net.build_model(mobile=True)
        return model

    def mn_asm_v0(self, tensor):
        """
            has only one output
            we use custom loss for this network and using ASM to correct points after that
        """

        # block_13_project_BN block_10_project_BN block_6_project_BN
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        '''heatmap can not be generated from activation layers, so we use out_relu'''
        x = mobilenet_model.get_layer('out_relu').output  # 7, 7, 1280
        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization(name='out_bn1')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization(name='out_bn2')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization(name='out_bn3')(x)

        out_heatmap = Conv2D(LearningConfig.point_len, kernel_size=1, padding='same', name='out_heatmap')(x)

        revised_model = Model(inp, out_heatmap)

        revised_model.summary()
        # plot_model(revised_model, to_file='mnv2_hm.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mn_asm_v0.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def mn_asm_v1(self, tensor):
        # block_13_project_BN block_10_project_BN block_6_project_BN
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input

        # '''block_1 {  block_6_project_BN 14, 14, 46 '''
        # x = mobilenet_model.get_layer('block_6_project_BN').output  # 14, 14, 46
        # x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
        #                     name='block_1_deconv_1', kernel_initializer='he_uniform')(x)  # 28, 28, 128
        # x = BatchNormalization(name='block_1_out_bn_1')(x)
        #
        # x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
        #                     name='block_1_deconv_2', kernel_initializer='he_uniform')(x)  # 56, 56, 128
        # x = BatchNormalization(name='block_1_out_bn_2')(x)
        #
        # block_1_out = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='block_1_out')(x)
        # '''block_1 }'''

        '''block_2 {  block_10_project_BN 14, 14, 96 '''
        x = mobilenet_model.get_layer('block_10_project_BN').output  # 14, 14, 96
        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_2_deconv_1', kernel_initializer='he_uniform')(x)  # 28, 28, 128
        x = BatchNormalization(name='block_2_out_bn_1')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_2_deconv_2', kernel_initializer='he_uniform')(x)  # 56, 56, 128
        x = BatchNormalization(name='block_2_out_bn_2')(x)

        block_2_out = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='block_2_out')(x)
        '''block_2 }'''

        '''block_3 {  block_13_project_BN 7, 7, 160 '''
        x = mobilenet_model.get_layer('block_13_project_BN').output  # 7, 7, 160
        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_3_deconv_1', kernel_initializer='he_uniform')(x)  # 14, 14, 128
        x = BatchNormalization(name='block_3_out_bn_1')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_3_deconv_2', kernel_initializer='he_uniform')(x)  # 28, 28, 128
        x = BatchNormalization(name='block_3_out_bn_2')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='block_3_deconv_3', kernel_initializer='he_uniform')(x)  # 56, 56, 128
        x = BatchNormalization(name='block_3_out_bn_3')(x)

        block_3_out = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='block_3_out')(x)

        '''block_3 }'''

        '''heatmap can not be generated from activation layers, so we use out_relu'''
        x = mobilenet_model.get_layer('out_relu').output  # 7, 7, 1280
        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization(name='out_bn1')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization(name='out_bn2')(x)

        x = Deconvolution2D(filters=128, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization(name='out_bn3')(x)

        out_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='out_heatmap')(x)

        revised_model = Model(inp, [
            # block_1_out,  # 85
            # block_2_out,  # 90
            # block_3_out,  # 97
            out_heatmap  # 100
        ])

        revised_model.summary()
        # plot_model(revised_model, to_file='mnv2_hm.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mn_asm_v1.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    # def create_multi_branch_mn(self, inp_shape, num_branches):
    #     branches = []
    #     inputs = []
    #     for i in range(num_branches):
    #         inp_i, br_i = self.create_branch_mn(prefix=str(i), inp_shape=inp_shape)
    #         inputs.append(inp_i)
    #         branches.append(br_i)
    #
    #     revised_model = Model(inputs[0], branches[0], name='multiBranchMN')
    #     # revised_model = Model(inputs, branches, name='multiBranchMN')
    #
    #     revised_model.layers.pop(0)
    #
    #     new_input = Input(shape=inp_shape)
    #
    #     revised_model = Model(new_input, revised_model.outputs)
    #
    #     revised_model.summary()
    #
    #     model_json = revised_model.to_json()
    #     with open("MultiBranchMN.json", "w") as json_file:
    #         json_file.write(model_json)
    #     return revised_model
    #

    def create_multi_branch_mn(self, inp_shape, num_branches):

        mobilenet_model = mobilenet_v2.MobileNetV2_mb(3, input_shape=inp_shape,
                                        alpha=1.0,
                                        include_top=True,
                                        weights=None,
                                        input_tensor=None,
                                        pooling=None)

        mobilenet_model.layers.pop()
        inp = mobilenet_model.input

        outputs = []
        for i in range(num_branches):
            prefix = str(i)
            '''heatmap can not be generated from activation layers, so we use out_relu'''
            x = mobilenet_model.get_layer('out_relu'+prefix).output  # 7, 7, 1280

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix + '_deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
            x = BatchNormalization(name=prefix + 'out_bn1')(x)

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix + '_deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
            x = BatchNormalization(name=prefix + 'out_bn2')(x)

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix + '_deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
            x = BatchNormalization(name=prefix + 'out_bn3')(x)

            out_heatmap = Conv2D(LearningConfig.point_len, kernel_size=1, padding='same', name=prefix + '_out_hm')(x)
            outputs.append(out_heatmap)

        revised_model = Model(inp, outputs)

        revised_model.summary()

        model_json = revised_model.to_json()
        with open("MultiBranchMN.json", "w") as json_file:
            json_file.write(model_json)
        return revised_model

    def create_branch_mn(self, prefix, inp_shape):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   pooling=None)
        mobilenet_model.layers.pop()
        inp = mobilenet_model.input

        '''heatmap can not be generated from activation layers, so we use out_relu'''
        x = mobilenet_model.get_layer('out_relu').output  # 7, 7, 1280

        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name=prefix + '_deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization(name=prefix + 'out_bn1')(x)

        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name=prefix + '_deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization(name=prefix + 'out_bn2')(x)

        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name=prefix + '_deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization(name=prefix + 'out_bn3')(x)

        out_heatmap = Conv2D(LearningConfig.point_len, kernel_size=1, padding='same', name=prefix + '_out_hm')(x)

        for layer in mobilenet_model.layers:
            layer.name = layer.name + '_' + prefix
        return inp, out_heatmap

    # def create_multi_branch_mn_one_input(self, inp_shape, num_branches):
    #     mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape,
    #                                                alpha=1.0,
    #                                                include_top=True,
    #                                                weights=None,
    #                                                input_tensor=None,
    #                                                pooling=None)
    #     mobilenet_model.layers.pop()
    #     inp = mobilenet_model.input
    #     outputs = []
    #     relu_name = 'out_relu'
    #     for i in range(num_branches):
    #         x = mobilenet_model.get_layer(relu_name).output  # 7, 7, 1280
    #         prefix = str(i)
    #         for layer in mobilenet_model.layers:
    #             layer.name = layer.name + prefix
    #
    #         relu_name = relu_name + prefix
    #
    #         '''heatmap can not be generated from activation layers, so we use out_relu'''
    #
    #         x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
    #                             name=prefix+'_deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
    #         x = BatchNormalization(name=prefix + 'out_bn1')(x)
    #
    #         x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
    #                             name=prefix+'_deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
    #         x = BatchNormalization(name=prefix +'out_bn2')(x)
    #
    #         x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
    #                             name=prefix+'_deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
    #         x = BatchNormalization(name=prefix+'out_bn3')(x)
    #
    #         out_heatmap = Conv2D(LearningConfig.point_len, kernel_size=1, padding='same', name=prefix+'_out_hm')(x)
    #         outputs.append(out_heatmap)
    #
    #     revised_model = Model(inp, outputs)
    #
    #     revised_model.summary()
    #
    #     model_json = revised_model.to_json()
    #     with open("MultiBranchMN.json", "w") as json_file:
    #         json_file.write(model_json)
    #     return revised_model

    def create_asmnet(self, inp_shape, num_branches):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=inp_shape,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=None,
                                                   pooling=None)

        mobilenet_model.layers.pop()
        inp = mobilenet_model.input
        outputs = []
        relu_name = 'out_relu'
        for i in range(num_branches):
            x = mobilenet_model.get_layer(relu_name).output  # 7, 7, 1280
            prefix = str(i)
            for layer in mobilenet_model.layers:
                layer.name = layer.name + prefix

            relu_name = relu_name + prefix

            '''heatmap can not be generated from activation layers, so we use out_relu'''

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix+'_deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
            x = BatchNormalization(name=prefix + 'out_bn1')(x)

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix+'_deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
            x = BatchNormalization(name=prefix +'out_bn2')(x)

            x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                                name=prefix+'_deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
            x = BatchNormalization(name=prefix+'out_bn3')(x)

            out_heatmap = Conv2D(LearningConfig.point_len, kernel_size=1, padding='same', name=prefix+'_out_hm')(x)
            outputs.append(out_heatmap)

        revised_model = Model(inp, outputs)

        revised_model.summary()

        model_json = revised_model.to_json()
        with open("asmnet.json", "w") as json_file:
            json_file.write(model_json)
        return revised_model

    def mnv2_hm(self, tensor):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=None,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)

        mobilenet_model.layers.pop()

        inp = mobilenet_model.input
        '''heatmap can not be generated from activation layers, so we use out_relu'''
        x = mobilenet_model.get_layer('out_relu').output  # 7, 7, 1280
        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv1', kernel_initializer='he_uniform')(x)  # 14, 14, 256
        x = BatchNormalization(name='out_bn1')(x)

        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv2', kernel_initializer='he_uniform')(x)  # 28, 28, 256
        x = BatchNormalization(name='out_bn2')(x)

        x = Deconvolution2D(filters=256, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu',
                            name='deconv3', kernel_initializer='he_uniform')(x)  # 56, 56, 256
        x = BatchNormalization(name='out_bn3')(x)

        out_heatmap = Conv2D(LearningConfig.landmark_len // 2, kernel_size=1, padding='same', name='out_heatmap')(x)

        revised_model = Model(inp, [
            out_heatmap,
        ])

        revised_model.summary()
        # plot_model(revised_model, to_file='mnv2_hm.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mnv2_hm.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

    def mobileNet_v2_main_discriminator(self, tensor, input_shape):
        mobilenet_model = mobilenet_v2.MobileNetV2(input_shape=input_shape,
                                                   alpha=1.0,
                                                   include_top=True,
                                                   weights=None,
                                                   input_tensor=tensor,
                                                   pooling=None)
        # , classes=cnf.landmark_len)

        mobilenet_model.layers.pop()

        x = mobilenet_model.get_layer('global_average_pooling2d_2').output  # 1280
        softmax = Dense(1, activation='sigmoid', name='out')(x)
        inp = mobilenet_model.input

        revised_model = Model(inp, softmax)

        revised_model.summary()
        # plot_model(revised_model, to_file='mobileNet_v2_main.png', show_shapes=True, show_layer_names=True)
        model_json = revised_model.to_json()

        with open("mobileNet_v2_main.json", "w") as json_file:
            json_file.write(model_json)

        return revised_model

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
