from tf_record_utility import TFRecordUtility
from configuration import DatasetName, DatasetType, AffectnetConf, IbugConf, W300Conf, InputDataSize, CofwConf, WflwConf
from cnn_model import CNNModel
from pca_utility import PCAUtility
from image_utility import ImageUtility
import numpy as np
from train import Train
from test import Test

# from Train_Gan import TrainGan

import img_printer as imgp

if __name__ == '__main__':

    # x = np.random.normal(size=100)
    # imgp.print_histogram(x)
    # imgp.print_histogram2d(data[0], data[1])
    # imgp.print_histogram1()


    pca_utility = PCAUtility()
    cnn_model = CNNModel()
    image_utility = ImageUtility()

    # tf_record_util = TFRecordUtility(WflwConf.num_of_landmarks*2)
    # tf_record_util._create_face_graph(dataset_name=DatasetName.wflw, dataset_type=None)

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
    tf_record_util = TFRecordUtility(IbugConf.num_of_landmarks*2)
    # tf_record_util.rotaate_and_save(dataset_name=DatasetName.ibug)
    # we dont need to use this now# tf_record_util.random_augment_from_rotated(dataset_name=DatasetName.ibug)
    '''     normalize the points and save'''
    # tf_record_util.normalize_points_and_save(dataset_name=DatasetName.ibug)
    # tf_record_util.test_normalize_points(dataset_name=DatasetName.ibug)
    '''     generate pose using hopeNet'''
    # tf_record_util.detect_pose_and_save(dataset_name=DatasetName.cofw)

    '''     create and save PCA objects'''
    # pca_utility.create_pca_from_points(DatasetName.wflw, 80)
    # pca_utility.create_pca_from_npy(DatasetName.cofw, 90)
    # pca_utility.create_pca_from_npy(DatasetName.ibug, 90)
    # pca_utility.create_pca_from_npy(DatasetName.wflw, 90)
    # pca_utility.test_pca_validity(DatasetName.wflw, 80)

    '''     create tfRecord:'''
    # tf_record_util.create_tf_record(dataset_name=DatasetName.ibug_test, dataset_type=None, heatmap=False, accuracy=100)

    # tf_record_util = TFRecordUtility(IbugConf.num_of_landmarks * 2)
    # tf_record_util.create_tf_record(dataset_name=DatasetName.ibug, dataset_type=None, heatmap=True, accuracy=100, isTest=False)
    #
    # tf_record_util = TFRecordUtility(CofwConf.num_of_landmarks * 2)
    # tf_record_util.create_tf_record(dataset_name=DatasetName.cofw, dataset_type=None, heatmap=True, accuracy=100, isTest=False)
    #
    # tf_record_util = TFRecordUtility(WflwConf.num_of_landmarks * 2)
    # tf_record_util.create_tf_record(dataset_name=DatasetName.wflw, dataset_type=None, heatmap=True, accuracy=100, isTest=False)

    # tf_record_util.create_tf_record(dataset_name=DatasetName.cofw, dataset_type=None, heatmap=False, accuracy=100)
    # tf_record_util.create_tf_record(dataset_name=DatasetName.wflw, dataset_type=None, heatmap=False, accuracy=100, isTest=False)
    # tf_record_util.create_tf_record(dataset_name=DatasetName.wflw, dataset_type=None, heatmap=False, accuracy=95, isTest=False)

    '''--> retrive and test tfRecords'''
    # tf_record_util = TFRecordUtility(WflwConf.num_of_landmarks*2)
    tf_record_util = TFRecordUtility(IbugConf.num_of_landmarks*2)
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

    '''--> Train GAN:'''
    # trg = TrainGan()
    # trg.create_seq_model()

    '''--> Evaluate Results'''
    '''testing one-by-one'''
    # FLOPS: 2,058,952,561 --- Params: 2,879,014
    # test = Test(dataset_name=DatasetName.ibug_test, arch='ASMNet', num_output_layers=2, weight_fname='./final_weights/ibug_ds_.h5', has_pose=True, customLoss=False)
    # test = Test(dataset_name=DatasetName.ibug_test, arch='ASMNet', num_output_layers=2, weight_fname='./final_weights/ibug_ds_asm.h5', has_pose=True, customLoss=True)
    # test = Test(dataset_name=DatasetName.ibug_test, arch='mobileNetV2', num_output_layers=2, weight_fname='./final_weights/ibug_mn_.h5', has_pose=True, customLoss=False)
    # test = Test(dataset_name=DatasetName.ibug_test, arch='mobileNetV2', num_output_layers=2, weight_fname='./final_weights/ibug_mn_asm.h5', has_pose=True, customLoss=True)

    '''cofw'''
    # test = Test(dataset_name=DatasetName.cofw_test, arch='ASMNet', num_output_layers=2, weight_fname='./final_weights/cofw_ds_.h5', has_pose=True, customLoss=False)
    # test = Test(dataset_name=DatasetName.cofw_test, arch='ASMNet', num_output_layers=2, weight_fname='./final_weights/cofw_ds_asm.h5', has_pose=True, customLoss=True)
    # test = Test(dataset_name=DatasetName.cofw_test, arch='mobileNetV2', num_output_layers=2, weight_fname='./final_weights/cofw_mn_.h5', has_pose=True, customLoss=False)
    # test = Test(dataset_name=DatasetName.cofw_test, arch='mobileNetV2', num_output_layers=2, weight_fname='./final_weights/cofw_mn_asm.h5', has_pose=True, customLoss=True)

    # test = Test(dataset_name=DatasetName.cofw_test, arch='mobileNetV2', num_output_layers=2, weight_fname='./final_weights/0t_cofw_mn.h5', has_pose=False, customLoss=False)

    '''wflw'''
    # test = Test(dataset_name=DatasetName.wflw_test, arch='ASMNet', num_output_layers=2, weight_fname='./final_weights/wflw_ds_.h5', has_pose=False, customLoss=False)
    # test = Test(dataset_name=DatasetName.wflw_test, arch='ASMNet', num_output_layers=2, weight_fname='./final_weights/wflw_ds_asm.h5', has_pose=False, customLoss=True)
    # test = Test(dataset_name=DatasetName.wflw_test, arch='mobileNetV2', num_output_layers=2, weight_fname='./final_weights/wflw_mn_.h5', has_pose=False, customLoss=False)
    # test = Test(dataset_name=DatasetName.wflw_test, arch='mobileNetV2', num_output_layers=2, weight_fname='./final_weights/wflw_mn_asm.h5', has_pose=False, customLoss=True)

    # test = Test(dataset_name=DatasetName.ibug_test, arch='ASMNet', num_output_layers=2, weight_fname='asmnet_weights-200-0.00340.h5', has_pose=True)
    '''test all'''
    # test = Test(dataset_name=None, arch=None, num_output_layers=2, weight_fname=None, has_pose=True)
    # test.test_all_results('./final_weights', num_output_layers=2)

    '''for testing KT''' #nme_ch:  6.22    6.76
    # test = Test(dataset_name=DatasetName.ibug_test, arch='efficientNet', num_output_layers=1,
    #             weight_fname='ds_ibug_ac_100_teacher.h5', has_pose=True, customLoss=False)

    # test = Test(dataset_name=DatasetName.ibug_test, arch='mobileNetV2_nopose', num_output_layers=1,
    #                 weight_fname='ds_ibug_ac_100_stu.h5', has_pose=True, customLoss=False)

    # test = Test(dataset_name=DatasetName.cofw_test, arch='mobileNetV2_nopose', num_output_layers=1,
    #             weight_fname='ds_cofw_ac_100_stu.h5', has_pose=True, customLoss=False)

    # test = Test(dataset_name=DatasetName.ibug_test, arch='mobileNetV2_nopose', num_output_layers=2,
    #             weight_fname='weights-94--0.01342.h5', has_pose=True, customLoss=False)


    '''--> Train Model'''
    # trainer = Train(use_tf_record=True,
    #                 dataset_name=DatasetName.ibug,
    #                 custom_loss=False,
    #                 arch='ASMNet',
    #                 # arch='mobileNetV2',
    #                 inception_mode=False,
    #                 num_output_layers=2,
    #                 # weight='00-w-dasm.h5',
    #                 weight=None,
    #                 train_on_batch=False,
    #                 accuracy=95)


# FLOPS: 604,444,425 --- Params: 2,436,043
# FLOPS: 604,034,745 --- Params: 2,333,563
# FLOPS: 604,751,685 --- Params: 2,512,903

# FLOPS: 516,345,794 --- Params: 1,439,507
# FLOPS: 516,088,274 --- Params: 1,396,547
# FLOPS: 516,538,934 --- Params: 1,471,727