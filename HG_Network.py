from configuration import DatasetName, DatasetType, \
    AffectnetConf, IbugConf, W300Conf, InputDataSize, LearningConfig
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications import mobilenet_v2, mobilenet, resnet50, densenet
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Conv2DTranspose, BatchNormalization, Add, MaxPool2D,\
    Activation, GlobalAveragePooling2D, DepthwiseConv2D, Dropout, ReLU, Concatenate, Input, GlobalMaxPool2D, LeakyReLU, UpSampling2D

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.keras.callbacks import CSVLogger
from clr_callback import CyclicLR
from datetime import datetime
import os.path
from scipy.spatial import distance
import scipy.io as sio

class HGNet:
    def __init__(self, input_shape, num_landmark):
        self.initializer ='glorot_uniform'
        # self.initializer = tf.random_normal_initializer(0., 0.02)
        self.num_landmark = num_landmark
        self.input_shape = input_shape

    def _create_bottle_neck_blocks(self, input_layer, filters=256):
        x = Conv2D(filters, 1, strides=1, padding='same', kernel_initializer=self.initializer, use_bias=False)(input_layer)
        x = BatchNormalization(momentum=0.9)(x)
        o_1 = LeakyReLU()(x)

        x = Conv2D(filters//2, 1, strides=1, padding='same', kernel_initializer=self.initializer, use_bias=False)(o_1)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters//2, 3, strides=1, padding='same', kernel_initializer=self.initializer, use_bias=False)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU()(x)

        x = Conv2D(filters, 1, strides=1, padding='same', kernel_initializer=self.initializer, use_bias=False)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU()(x)

        x = Add()([o_1, x])

        return x

    def _create_conv_layer(self, input_layer, kernel_size, strides, filters):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer=self.initializer)(input_layer)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU()(x)
        return x

    def _create_block(self, inp):
        l1 = self._create_bottle_neck_blocks(input_layer=inp)
        l1_res = self._create_bottle_neck_blocks(input_layer=inp)
        x = MaxPool2D(pool_size=2, strides=2)(l1)  # 32

        l2 = self._create_bottle_neck_blocks(input_layer=x)
        l2_res = self._create_bottle_neck_blocks(input_layer=x)
        x = MaxPool2D(pool_size=2, strides=2)(l2)  # 16

        l3 = self._create_bottle_neck_blocks(input_layer=x)
        l3_res = self._create_bottle_neck_blocks(input_layer=x)
        x = MaxPool2D(pool_size=2, strides=2)(l3)  # 8

        l4 = self._create_bottle_neck_blocks(input_layer=x)
        l4_res = self._create_bottle_neck_blocks(input_layer=x)
        x = MaxPool2D(pool_size=2, strides=2)(l4)  # 4

        x = self._create_bottle_neck_blocks(input_layer=x)
        x = self._create_bottle_neck_blocks(input_layer=x)
        x = self._create_bottle_neck_blocks(input_layer=x)

        x = UpSampling2D(size=(2,2))(x)
        x = Add()([l4_res, x])
        x = self._create_bottle_neck_blocks(input_layer=x)

        x = UpSampling2D(size=(2,2))(x)
        x = Add()([l3_res, x])
        x = self._create_bottle_neck_blocks(input_layer=x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Add()([l2_res, x])
        x = self._create_bottle_neck_blocks(input_layer=x)

        x = UpSampling2D(size=(2, 2))(x)
        x = Add()([l1_res, x])
        x = self._create_bottle_neck_blocks(input_layer=x)

        '''final'''
        x = self._create_bottle_neck_blocks(input_layer=x)
        o_2_out = self._create_bottle_neck_blocks(input_layer=x)

        out_loss_1 = Conv2D(filters=self.num_landmark, kernel_size=1, strides=1,
                            padding='same', kernel_initializer=self.initializer)(x)
        x = BatchNormalization(momentum=0.9)(out_loss_1)
        x = LeakyReLU()(x)

        out_loss_next = self._create_bottle_neck_blocks(input_layer=x)

        finish = Add()([o_2_out, out_loss_next])
        return out_loss_1, finish

    def create_model(self):
        inp = Input(shape=self.input_shape)
        '''create header for 256 --> 64'''
        x = self._create_conv_layer(input_layer=inp, filters=64, kernel_size=7, strides=2)
        x = self._create_bottle_neck_blocks(input_layer=x)
        x = MaxPool2D(pool_size=2, strides=2)(x)  # 64

        '''main blocks'''
        out_loss_1, finish_1 = self._create_block(inp=x)
        out_loss_2, finish_2 = self._create_block(inp=finish_1)
        out_loss_3, finish_3 = self._create_block(inp=finish_2)
        out_loss_4, finish_4 = self._create_block(inp=finish_3)
        '''new'''

        revised_model = Model(inp, [out_loss_1, out_loss_2, out_loss_3, out_loss_4])

        revised_model.summary()
        tf.keras.utils.plot_model(revised_model, show_shapes=True, dpi=64)
        model_json = revised_model.to_json()
        with open("./model_arch/myHGN.json", "w") as json_file:
            json_file.write(model_json)
        return revised_model


# import tensorflow as tf
#
# from tensorflow.keras.layers import (
#     Add,
#     Concatenate,
#     Conv2D,
#     Input,
#     Lambda,
#     ReLU,
#     MaxPool2D,
#     UpSampling2D,
#     ZeroPadding2D,
#     BatchNormalization,
# )
#
#
# # [1] Stacked Hourglass Networks for Human Pose Estimation
#
#
# def BottleneckBlock(inputs, filters, strides=1, downsample=False, name=None):
#     """
#     "Our ﬁnal design makes extensive use of residual modules. Filters greater than 3x3 are never used,
#     and the bottlenecking restricts the total number of parameters at each layer curtailing total
#     memory usage. The module used in our network is shown in Figure 4." [1]
#     Ethan: can't tell what's the exact structure, so we need to refer to his source code here:
#     https://github.com/princeton-vl/pose-hg-train/blob/master/src/models/layers/Residual.lua
#     this follows ResNet V1 but also puts BN ahead which is used in ResNet V2
#     """
#     identity = inputs
#     if downsample:
#         identity = Conv2D(
#             filters=filters,  # lift channels first
#             kernel_size=1,
#             strides=strides,
#             padding='same',
#             kernel_initializer='he_normal')(inputs)
#
#     x = BatchNormalization(momentum=0.9)(inputs)
#     x = ReLU()(x)
#     x = Conv2D(
#         filters=filters // 2,
#         kernel_size=1,
#         strides=1,
#         padding='same',
#         kernel_initializer='he_normal')(x)
#
#     x = BatchNormalization(momentum=0.9)(x)
#     x = ReLU()(x)
#     x = Conv2D(
#         filters=filters // 2,
#         kernel_size=3,
#         strides=strides,
#         padding='same',
#         kernel_initializer='he_normal')(x)
#
#     x = BatchNormalization(momentum=0.9)(x)
#     x = ReLU()(x)
#     x = Conv2D(
#         filters=filters,
#         kernel_size=1,
#         strides=1,
#         padding='same',
#         kernel_initializer='he_normal')(x)
#
#     x = Add()([identity, x])
#     return x
#
#
# def HourglassModule(inputs, order, filters, num_residual):
#     """
#     https://github.com/princeton-vl/pose-hg-train/blob/master/src/models/hg.lua#L3
#     """
#     # Upper branch
#     up1 = BottleneckBlock(inputs, filters, downsample=False)
#
#     for i in range(num_residual):
#         up1 = BottleneckBlock(up1, filters, downsample=False)
#
#     # Lower branch
#     low1 = MaxPool2D(pool_size=2, strides=2)(inputs)
#     for i in range(num_residual):
#         low1 = BottleneckBlock(low1, filters, downsample=False)
#
#     low2 = low1
#     if order > 1:
#         low2 = HourglassModule(low1, order - 1, filters, num_residual)
#     else:
#         for i in range(num_residual):
#             low2 = BottleneckBlock(low2, filters, downsample=False)
#
#     low3 = low2
#     for i in range(num_residual):
#         low3 = BottleneckBlock(low3, filters, downsample=False)
#
#     up2 = UpSampling2D(size=2)(low3)
#
#     return up2 + up1
#
#
# def LinearLayer(inputs, filters):
#     x = Conv2D(
#         filters=filters,
#         kernel_size=1,
#         strides=1,
#         padding='same',
#         kernel_initializer='he_normal')(inputs)
#     x = BatchNormalization(momentum=0.9)(x)
#     x = ReLU()(x)
#     return x
#
#
# def StackedHourglassNetwork(
#         input_shape=(256, 256, 3), num_stack=4, num_residual=1,
#         num_heatmap=68):
#     """
#     https://github.com/princeton-vl/pose-hg-train/blob/master/src/models/hg.lua#L33
#     """
#     inputs = Input(shape=input_shape)
#
#     # initial processing of the image
#     x = Conv2D(
#         filters=64,
#         kernel_size=7,
#         strides=2,
#         padding='same',
#         kernel_initializer='he_normal')(inputs)
#     x = BatchNormalization(momentum=0.9)(x)
#     x = ReLU()(x)
#     x = BottleneckBlock(x, 128, downsample=True)
#     x = MaxPool2D(pool_size=2, strides=2)(x)
#     x = BottleneckBlock(x, 128, downsample=False)
#     x = BottleneckBlock(x, 256, downsample=True)
#
#     ys = []
#     for i in range(num_stack):
#         x = HourglassModule(x, order=4, filters=256, num_residual=num_residual)
#         for i in range(num_residual):
#             x = BottleneckBlock(x, 256, downsample=False)
#
#         # predict 256 channels like a fully connected layer.
#         x = LinearLayer(x, 256)
#
#         # predict final channels, which is also the number of predicted heatmap
#         y = Conv2D(
#             filters=num_heatmap,
#             kernel_size=1,
#             strides=1,
#             padding='same',
#             kernel_initializer='he_normal')(x)
#         ys.append(y)
#
#         # if it's not the last stack, we need to add predictions back
#         if i < num_stack - 1:
#             y_intermediate_1 = Conv2D(filters=256, kernel_size=1, strides=1)(x)
#             y_intermediate_2 = Conv2D(filters=256, kernel_size=1, strides=1)(y)
#             x = Add()([y_intermediate_1, y_intermediate_2])
#
#     revised_model = tf.keras.Model(inputs, ys, name='stacked_hourglass')
#     revised_model.summary()
#     # tf.keras.utils.plot_model(revised_model, show_shapes=True, dpi=64)
#     model_json = revised_model.to_json()
#     with open("./model_arch/NEW_HGN.json", "w") as json_file:
#         json_file.write(model_json)
#     return revised_model


# StackedHourglassNetwork()
