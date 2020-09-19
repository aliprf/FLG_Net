from tf_record_utility import TFRecordUtility
from configuration import DatasetName, DatasetType, AffectnetConf, IbugConf,\
    W300Conf, InputDataSize, CofwConf, WflwConf, LearningConfig
from cnn_model import CNNModel
from pca_utility import PCAUtility
from image_utility import ImageUtility
import numpy as np
from train import Train
from test import Test
from Facial_GAN import FacialGAN

import tensorflow as tf
import keras
import keras.backend as K

if __name__ == '__main__':

    pca_utility = PCAUtility()
    cnn_model = CNNModel()
    image_utility = ImageUtility()

    # tf_record_util = TFRecordUtility(IbugConf.num_of_landmarks*2)

    # tf_record_util.test_hm_accuracy()
    # tf_record_util.create_adv_att_img_hm()

    '''--> Preparing Test Data process:'''
    # tf_record_util.crop_and_save(dataset_name=DatasetName.ibug, dataset_type=DatasetType.wflw_full)
    # tf_record_util.normalize_points_and_save(dataset_name=DatasetName.ibug)
    # tf_record_util.detect_pose_and_save(dataset_name=DatasetName.ibug_test)
    # tf_record_util.create_tf_record(dataset_name=DatasetName.ibug_test, dataset_type=DatasetType.ibug_full,
    #                                 heatmap=False, accuracy=100, isTest=True)

    '''--> Preparing Train Data process:'''
    '''     augment, normalize, and save pts'''
    # tf_record_util = TFRecordUtility(IbugConf.num_of_landmarks*2)
    # tf_record_util.rotaate_and_save(dataset_name=DatasetName.ibug)
    # # we dont need to use this now# tf_record_util.random_augment_from_rotated(dataset_name=DatasetName.ibug)
    # '''     normalize the points and save'''
    # tf_record_util.normalize_points_and_save(dataset_name=DatasetName.ibug)
    # # tf_record_util.test_normalize_points(dataset_name=DatasetName.ibug)
    # tf_record_util.create_face_graph(dataset_name=DatasetName.ibug, dataset_type=None)


    '''--> retrive and test tfRecords'''
    # tf_record_util = TFRecordUtility(WflwConf.num_of_landmarks*2)
    # tf_record_util.test_tf_record()
    # tf_record_util.test_tf_record_hm()

    '''create point->imgName map'''
    # tf_record_util = TFRecordUtility(IbugConf.num_of_landmarks * 2)
    # tf_record_util.create_point_imgpath_map_tf_record(dataset_name=DatasetName.ibug)
    #
    # tf_record_util = TFRecordUtility(CofwConf.num_of_landmarks * 2)
    # tf_record_util.create_point_imgpath_map_tf_record(dataset_name=DatasetName.cofw)
    #
    # tf_record_util = TFRecordUtility(WflwConf.num_of_landmarks * 2)
    # tf_record_util.create_point_imgpath_map_tf_record(dataset_name=DatasetName.wflw)

    '''--> Evaluate Results'''
    '''for testing KT'''
    # test = Test(dataset_name=DatasetName.ibug_test, arch='efficientNet', num_output_layers=1,
    #             weight_fname='ds_ibug_ac_100_teacher.h5', has_pose=True, customLoss=False)

    # test = Test(dataset_name=DatasetName.ibug_test, arch='mobileNetV2_nopose', num_output_layers=1,
    #                 weight_fname='ds_ibug_ac_100_stu.h5', has_pose=True, customLoss=False)

    # test = Test(dataset_name=DatasetName.cofw_test, arch='mobileNetV2_nopose', num_output_layers=1,
    #             weight_fname='ds_cofw_ac_100_stu.h5', has_pose=True, customLoss=False)

    # test = Test(dataset_name=DatasetName.ibug_test, arch='mobileNetV2_nopose', num_output_layers=2,
    #             weight_fname='weights-94--0.01342.h5', has_pose=True, customLoss=False)


    '''--> Train Model'''
    fg = FacialGAN(dataset_name=DatasetName.ibug, geo_custom_loss=False, regressor_arch='effGlassNet',
                   discriminator_arch='effDiscrimNet', regressor_weight=None, discriminator_weight=None,
                   input_shape_reg=[InputDataSize.image_input_size, InputDataSize.image_input_size, 3],
                   input_shape_disc=[InputDataSize.hm_size, InputDataSize.hm_size,
                                     IbugConf.num_face_graph_elements * 2])
    fg.train_network()