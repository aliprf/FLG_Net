class DatasetName:
    affectnet = 'affectnet'
    w300 = 'w300'
    ibug = 'ibug'
    aflw = 'aflw'
    aflw2000 = 'aflw2000'
    cofw = 'cofw'
    wflw = 'wflw'

    wflw_test = 'wflw_test'
    cofw_test = 'cofw_test'
    ibug_test = 'ibug_test'



class DatasetType:
    ibug_challenging = 10
    ibug_comomn = 11
    ibug_full = 12

    wflw_full = 20
    wflw_blur = 21
    wflw_expression = 22
    wflw_illumination = 23
    wflw_largepose = 24
    wflw_makeup = 25
    wflw_occlusion = 26


class LearningConfig:
    CLR_METHOD = "triangular"
    MIN_LR = 1e-7
    MAX_LR = 1e-2
    STEP_SIZE = 10
    batch_size = 2
    steps_per_validation_epochs = 5
    epochs = 200


class InputDataSize:
    image_input_size = 224
    img_center = image_input_size //2  # 112

    hm_size = image_input_size//4  # 56
    hm_center = hm_size//2 #  28

    # landmark_face_len = 54
    # landmark_nose_len = 18
    # landmark_eys_len = 24
    # landmark_mouth_len = 40
    # pose_len = 3


class AffectnetConf:
    csv_train_path = '/media/ali/extradata/facial_landmark_ds/affectNet/training.csv'
    csv_evaluate_path = '/media/ali/extradata/facial_landmark_ds/affectNet/validation.csv'
    csv_test_path = '/media/ali/extradata/facial_landmark_ds/affectNet/test.csv'

    tf_train_path = '/media/ali/extradata/facial_landmark_ds/affectNet/train.tfrecords'
    tf_test_path = '/media/ali/extradata/facial_landmark_ds/affectNet/eveluate.tfrecords'
    tf_evaluation_path = '/media/ali/extradata/facial_landmark_ds/affectNet/test.tfrecords'

    sum_of_train_samples = 200000  # 414800
    sum_of_test_samples = 30000
    sum_of_validation_samples = 5500

    img_path_prefix = '/media/ali/extradata/facial_landmark_ds/affectNet/Manually_Annotated_Images/'


class Multipie:
    lbl_path_prefix = '/media/ali/extradata/facial_landmark_ds/multi-pie/MPie_Labels/labels/all/'
    img_path_prefix = '/media/ali/extradata/facial_landmark_ds/multi-pie/'

    origin_number_of_all_sample = 2000
    origin_number_of_train_sample = 1950
    origin_number_of_evaluation_sample = 50
    augmentation_factor = 100


class W300Conf:
    tf_common = '/media/ali/data/test_common.tfrecords'
    tf_challenging = '/media/ali/data/test_challenging.tfrecords'
    tf_full = '/media/ali/data/test_full.tfrecords'

    img_path_prefix_common = '/media/ali/extradata/facial_landmark_ds/from_ibug/test_set/common/'
    img_path_prefix_challenging = '/media/ali/extradata/facial_landmark_ds/from_ibug/test_set/challenging/'
    img_path_prefix_full = '/media/ali/extradata/facial_landmark_ds/from_ibug/test_set/full/'

    number_of_all_sample_common = 554
    number_of_all_sample_challenging = 135
    number_of_all_sample_full = 689


class WflwConf:
    # Wflw_prefix_path = '/media/data3/ali/FL/wflw/'  # --> Zeus
    # Wflw_prefix_path = '/media/data2/alip/FL/wflw/'  # --> Atlas
    Wflw_prefix_path = '/media/ali/data/wflw/'  # --> local

    img_path_prefix = Wflw_prefix_path + 'all/'
    # rotated_img_path_prefix = Wflw_prefix_path + '0_rotated/'
    rotated_img_path_prefix = Wflw_prefix_path + '1_train_images_pts_dir/'
    train_images_dir = Wflw_prefix_path + '1_train_images_pts_dir/'
    normalized_points_npy_dir = Wflw_prefix_path + '2_normalized_npy_dir/'
    graph_face_dir = Wflw_prefix_path + '3_graph_face/'
    pose_npy_dir = Wflw_prefix_path + '4_pose_npy_dir/'

    test_img_path_prefix = Wflw_prefix_path + 'test_all/'
    test_images_dir = Wflw_prefix_path + 'test_images_pts_dir/'
    test_normalized_points_npy_dir = Wflw_prefix_path + 'test_normalized_npy_dir/'
    test_pose_npy_dir = Wflw_prefix_path + 'test_pose_npy_dir/'
    train_hm_dir = Wflw_prefix_path + 'train_hm_dir/'

    tf_train_path = Wflw_prefix_path + 'train.tfrecords'
    tf_evaluation_path = Wflw_prefix_path + 'evaluation.tfrecords'

    tf_test_path = Wflw_prefix_path + 'test_full.tfrecords'
    tf_test_path_blur = Wflw_prefix_path + 'test_blur.tfrecords'
    tf_test_path_expression = Wflw_prefix_path + 'test_expression.tfrecords'
    tf_test_path_illumination = Wflw_prefix_path + 'test_illumination.tfrecords'
    tf_test_path_largepose = Wflw_prefix_path + 'test_largepose.tfrecords'
    tf_test_path_makeup = Wflw_prefix_path + 'test_makeup.tfrecords'
    tf_test_path_occlusion = Wflw_prefix_path + 'test_occlusion.tfrecords'

    tf_train_path_95 = Wflw_prefix_path + 'train_90.tfrecords'
    tf_evaluation_path_95 = Wflw_prefix_path + 'evaluation_90.tfrecords'

    orig_number_of_training = 7500
    orig_number_of_test = 2500

    orig_of_all_test_blur = 773
    orig_of_all_test_expression = 314
    orig_of_all_test_illumination = 689
    orig_of_all_test_largepose = 326
    orig_of_all_test_makeup = 206
    orig_of_all_test_occlusion = 736

    number_of_all_sample = 0  # just images. dont count both img and lbls
    number_of_train_sample = int(number_of_all_sample * 0.95)  # 95 % for train
    number_of_evaluation_sample = int(number_of_all_sample * 0.05) # 5% for evaluation

    augmentation_factor = 4  # create . image from 4
    augmentation_factor_rotate = 15  # create . image from 15
    num_of_landmarks = 98
    num_face_graph_elements = 8

class CofwConf:
    # Cofw_prefix_path = '/media/data3/ali/FL/cofw/'  # --> Zeus
    # Cofw_prefix_path = '/media/data2/alip/FL/cofw/'  # --> Atlas
    Cofw_prefix_path = '/media/ali/data/cofw/'  # --> local

    img_path_prefix = Cofw_prefix_path + 'all/'
    # rotated_img_path_prefix = Cofw_prefix_path + '0_rotated/'
    rotated_img_path_prefix = Cofw_prefix_path + '1_train_images_pts_dir/'
    train_images_dir = Cofw_prefix_path + '1_train_images_pts_dir/'
    normalized_points_npy_dir = Cofw_prefix_path + '2_normalized_npy_dir/'
    graph_face_dir = Cofw_prefix_path + '3_graph_face/'
    pose_npy_dir = Cofw_prefix_path + '4_pose_npy_dir/'

    test_img_path_prefix = Cofw_prefix_path + 'test_all/'
    test_images_dir = Cofw_prefix_path + 'test_images_pts_dir/'
    test_normalized_points_npy_dir = Cofw_prefix_path + 'test_normalized_npy_dir/'
    test_pose_npy_dir = Cofw_prefix_path + 'test_pose_npy_dir/'

    tf_train_path = Cofw_prefix_path + 'train.tfrecords'
    tf_test_path = Cofw_prefix_path + 'test.tfrecords'
    tf_evaluation_path = Cofw_prefix_path + 'evaluation.tfrecords'
    train_hm_dir = Cofw_prefix_path + 'train_hm_dir/'

    tf_train_path_95 = Cofw_prefix_path + 'train_95.tfrecords'
    tf_evaluation_path_95 = Cofw_prefix_path + 'evaluation_95.tfrecords'

    orig_number_of_training = 1345
    orig_number_of_test = 507

    number_of_all_sample = 0  # afw, train_helen, train_lfpw
    number_of_train_sample = int(number_of_all_sample * 0.95)  # 95 % for train
    number_of_evaluation_sample = int(number_of_all_sample * 0.05)  # 5% for evaluation

    augmentation_factor = 5  # create . image from 1
    augmentation_factor_rotate = 30  # create . image from 1
    num_of_landmarks = 28
    num_face_graph_elements = 8


class IbugConf:
    '''server_config'''
    # _Ibug_prefix_path = '/media/data3/ali/FL/ibug/'  # --> Zeus
    # _Ibug_prefix_path = '/media/data2/alip/FL/ibug/'  # --> Atlas
    _Ibug_prefix_path = '/media/ali/data/ibug/'  # --> local_0
    # _Ibug_prefix_path = '/media/ali/extradata/data/ibug/'  # --> local_1

    img_path_prefix = _Ibug_prefix_path + 'all/'
    rotated_img_path_prefix = _Ibug_prefix_path + '1_train_images_pts_dir/'
    train_images_dir = _Ibug_prefix_path + '1_train_images_pts_dir/'
    normalized_points_npy_dir = _Ibug_prefix_path + '2_normalized_npy_dir/'
    graph_face_dir = _Ibug_prefix_path + '3_graph_face/'
    pose_npy_dir = _Ibug_prefix_path + '4_pose_npy_dir/'

    tf_test_path_full = _Ibug_prefix_path + 'test_full.tfrecords'
    tf_test_path_common = _Ibug_prefix_path + 'test_common.tfrecords'
    tf_test_path_challenging = _Ibug_prefix_path + 'test_challenging.tfrecords'

    test_img_path_prefix = _Ibug_prefix_path + 'test_all/'
    test_images_dir = _Ibug_prefix_path + 'test_images_pts_dir/'
    test_normalized_points_npy_dir = _Ibug_prefix_path + 'test_normalized_npy_dir/'
    test_pose_npy_dir = _Ibug_prefix_path + 'test_pose_npy_dir/'

    train_hm_dir = _Ibug_prefix_path + 'train_hm_dir/'

    orig_number_of_training = 3148
    orig_number_of_test_full = 689
    orig_number_of_test_common = 554
    orig_number_of_test_challenging = 135

    '''after augmentation'''
    number_of_all_sample = 0   # afw, train_helen, train_lfpw
    number_of_train_sample = int(number_of_all_sample * 0.95)  # 95 % for train
    number_of_evaluation_sample = int(number_of_all_sample * 0.05)  # 5% for evaluation

    augmentation_factor = 4  # create . image from 1
    augmentation_factor_rotate = 20  # create . image from 1
    num_of_landmarks = 68
    num_face_graph_elements = 9

