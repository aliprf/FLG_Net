from tf_record_utility import TFRecordUtility
from configuration import DatasetName, DatasetType, AffectnetConf, IbugConf, \
    W300Conf, InputDataSize, CofwConf, WflwConf, LearningConfig
from cnn_model import CNNModel
from pca_utility import PCAUtility
from image_utility import ImageUtility
import numpy as np
from train import Train
from test import Test
from Facial_GAN import FacialGAN
# from HM_regression_part import HmRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

if __name__ == '__main__':
    # width = height = 64
    # sigma = 3
    # x0 = y0 = 20
    # x = np.arange(0, width, 1, float)
    # y = np.arange(0, height, 1, float)[:, np.newaxis]
    # gaus = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    # gaus[gaus <= 0.01] = 0
    # ''''''
    # gaus_bg = np.invert(gaus == 0).astype(int)
    # # gaus_bg = (gaus == 0).astype(int)
    # gaus_fg_3 = ((gaus > 0) & (gaus <= 0.5)).astype(int)
    # gaus_fg_2 = ((gaus > 0.5) & (gaus <= 0.8)).astype(int)
    # gaus_fg_1 = (gaus > 0.8).astype(int)
    #
    # dpi = 80
    # width = 800 * 4
    # height = 800 * 4
    # figsize = width / float(dpi), height / float(dpi)
    # fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1]}, figsize=figsize)
    #
    # axs[0, 0].title.set_text("bg")
    # # axs[0, 0].imshow(gaus_bg, cmap=cm.coolwarm)
    # axs[0, 0].imshow(gaus_bg)  #, vmin=np.amin(gaus_bg), vmax=np.amax(gaus_bg), cmap=cm.coolwarm)
    #
    # axs[0, 1].title.set_text("fg 3 : (0.0, 0.5]")
    # axs[0, 1].imshow(gaus_fg_3)
    # # axs[0, 1].imshow(gaus_fg_3, vmin=np.amin(gaus_fg_3), vmax=np.amax(gaus_fg_3), cmap=cm.coolwarm)
    #
    # axs[1, 0].title.set_text("fg 2 : (0.5, 0.8]")
    # axs[1, 0].imshow(gaus_fg_2)
    #
    # axs[1, 1].title.set_text("fg 1 : (0.8, 1]")
    # axs[1, 1].imshow(gaus_fg_1)
    #
    # plt.tight_layout()
    # plt.savefig("z_guas_hm.png")
    #
    # ''''''
    #
    # fig_1 = plt.figure(figsize=figsize)
    # ax = fig_1.gca(projection='3d')
    # # Make data.
    # x = np.linspace(0, 64, 64)
    # y = np.linspace(0, 64, 64)
    # X, Y = np.meshgrid(x, y)
    # # Plot the surface.
    # surf = ax.plot_surface(X, Y, gaus_bg, cmap=cm.coolwarm, vmin=0, vmax=1,
    #                        linewidth=1, antialiased=True)
    # # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #
    # # Add a color bar which maps values to colors.
    # fig_1.colorbar(surf, shrink=1, aspect=10)
    #
    # plt.savefig('gaus_bg')
    ''''''

    pca_utility = PCAUtility()
    cnn_model = CNNModel()
    image_utility = ImageUtility()

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
    # tf_record_util = TFRecordUtility(IbugConf.num_of_landmarks * 2)
    # tf_record_util.rotaate_and_save(dataset_name=DatasetName.ibug)
    # tf_record_util.normalize_points_and_save(dataset_name=DatasetName.ibug)
    # # tf_record_util.test_normalize_points(dataset_name=DatasetName.ibug)
    # ## tf_record_util.create_face_graph(dataset_name=DatasetName.ibug, dataset_type=None)
    # tf_record_util.create_all_heatmap(dataset_name=DatasetName.ibug, dataset_type=None)

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

    # '''--> Train Model'''
    fg = FacialGAN(dataset_name=DatasetName.cofw, hm_regressor_arch='hm_reg_model',
                   cord_regressor_arch='cord_reg_model',
                   hm_discriminator_arch='hm_Disc_model', cord_discriminator_arch='cord_Disc_model',

                   hm_regressor_weight=None, cord_regressor_weight=None,
                   hm_discriminator_weight=None, cord_discriminator_weight=None,

                   input_shape_hm_reg=[InputDataSize.image_input_size, InputDataSize.image_input_size, 3],
                   input_shape_cord_reg=[InputDataSize.image_input_size, InputDataSize.image_input_size, 3],

                   input_shape_hm_disc=[InputDataSize.hm_size, InputDataSize.hm_size, 2],
                   # we concat flatten hm and img
                   input_shape_cord_disc=CofwConf.num_of_landmarks * 2)  # concat 2 generated and real array
    fg.train()

    '''Regression Train'''
    # hm_reg = HmRegression(dataset_name=DatasetName.ibug, hm_regressor_arch='hm_reg_model', hm_regressor_weight=None,
    #                       input_shape_hm_reg=[InputDataSize.image_input_size, InputDataSize.image_input_size, 3])
    # hm_reg.train()

    '''for test'''
    # test = Test(dataset_name= DatasetName.ibug_test, weight_fname='./training_checkpoints/cord_reg_11_.h5')