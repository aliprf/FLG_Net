from configuration import DatasetName, DatasetType, AffectnetConf, IbugConf, \
    W300Conf, InputDataSize, LearningConfig, CofwConf, WflwConf
from image_utility import ImageUtility
import tensorflow as tf
import numpy as np
import os
from skimage.transform import resize
import csv
import sys
from PIL import Image
from pathlib import Path
import sqlite3
import cv2
import os.path
from keras import backend as K

from scipy import misc
from scipy.ndimage import gaussian_filter, maximum_filter
from numpy import save, load, asarray
import img_printer as imgpr
from tqdm import tqdm
from pca_utility import PCAUtility
import pickle
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

class TFRecordUtility:
    def __init__(self, number_of_landmark):
        self.number_of_landmark = number_of_landmark

    def test_hm_accuracy(self):
        images_dir = IbugConf.images_dir

        counter = 0
        for file in os.listdir(images_dir):
            if file.endswith(".pts"):
                points_arr = []
                file_name = os.path.join(images_dir, file)
                img_file_name = str(file_name)[:-3] + "jpg"
                if os.path.exists(img_file_name):
                    with open(file_name) as fp:
                        line = fp.readline()
                        cnt = 1
                        while line:
                            if 3 < cnt < 72:
                                x_y_pnt = line.strip()
                                x = float(x_y_pnt.split(" ")[0])
                                y = float(x_y_pnt.split(" ")[1])
                                points_arr.append(x)
                                points_arr.append(y)
                            line = fp.readline()
                            cnt += 1
                    hm = self.generate_hm(56, 56, np.array(points_arr), 1.0, False)
                    img = Image.open(img_file_name)
                    self.calculate_hm_to_point_accuracy(img, hm, points_arr, counter)
                    counter += 1

    def calculate_hm_to_point_accuracy(self, img, heatmap, orig_lbls, counter):
        image_utility = ImageUtility()

        x_h_c, y_h_c, xy_h_c = self.from_heatmap_to_point(heatmap, number_of_points=5)
        xy_h_o, x_h_o, y_h_o = image_utility.create_landmarks(landmarks=orig_lbls, scale_factor_x=1, scale_factor_y=1)

        imgpr.print_image_arr('orig' + str(counter), img, x_h_o, y_h_o)
        imgpr.print_image_arr('conv' + str(counter), img, x_h_c, y_h_c)

    def get_predicted_kp_from_htmap(self, heatmap, center, scale, outres):
        # nms to get location
        kplst = self.post_process_heatmap(heatmap)
        kps = np.array(kplst)

        return kps

        # # use meta information to transform back to original image
        # mkps = copy.copy(kps)
        # for i in range(kps.shape[0]):
        #     mkps[i, 0:2] = data_process.transform(kps[i], center, scale, res=outres, invert=1, rot=0)
        #
        # return mkps

    def post_process_heatmap(self, heatMap, kpConfidenceTh=0.2):
        kplst = list()
        for i in range(heatMap.shape[-1]):
            # ignore last channel, background channel
            _map = heatMap[:, :, i]
            _map = gaussian_filter(_map, sigma=0.3)
            _nmsPeaks = self.non_max_supression(_map, windowSize=3, threshold=1e-6)

            y, x = np.where(_nmsPeaks == _nmsPeaks.max())
            if len(x) > 0 and len(y) > 0:
                kplst.append((int(x[0]) * 4, int(y[0]) * 4, _nmsPeaks[y[0], x[0]]))
            else:
                kplst.append((0, 0, 0))
        return kplst

    def non_max_supression(self, plain, windowSize=3, threshold=1e-6):
        # clear value less than threshold
        under_th_indices = plain < threshold
        plain[under_th_indices] = 0
        return plain * (plain == maximum_filter(plain, footprint=np.ones((windowSize, windowSize))))

    def create_tf_record(self, dataset_name, dataset_type, heatmap, isTest,accuracy=100):

        if not heatmap:
            self._create_tfrecord_from_npy(dataset_name, dataset_type, isTest, accuracy)
        elif heatmap:
            self._create_tfrecord_from_npy_hm(dataset_name, dataset_type, isTest, accuracy)

        elif dataset_name == DatasetName.affectnet:
            self.__create_tfrecord_affectnet(dataset_type, need_augmentation=True)
        elif dataset_name == DatasetName.w300:
            self.__create_tfrecord_w300(dataset_type)
        elif dataset_name == DatasetName.ibug:
            if heatmap:
                self._create_tfrecord_ibug_all_heatmap(dataset_name)
        elif dataset_name == DatasetName.aflw:
            self.__create_tfrecord_aflw()

    def test_tf_record_hm(self, ):
        image_utility = ImageUtility()
        lbl_arr, img_arr, pose_arr, hm_arr = self.retrieve_tf_record_hm(WflwConf.tf_train_path,
                                                            number_of_records=100,
                                                             # number_of_records=WflwConf.orig_number_of_test,
                                                             only_label=False)
        counter = 0
        for lbl in lbl_arr:
            landmark_arr_flat_n, landmark_arr_x_n, landmark_arr_y_n = \
                image_utility.create_landmarks_from_normalized(lbl_arr[counter], 224, 224, 112, 112)

            imgpr.print_image_arr(str(counter), img_arr[counter], landmark_arr_x_n, landmark_arr_y_n)
            imgpr.print_image_arr_heat(str(counter+1000), hm_arr[counter], print_single=False)

            counter += 1

    def test_tf_record(self, ):
        image_utility = ImageUtility()
        lbl_arr, img_arr, pose_arr = self.retrieve_tf_record(IbugConf.tf_test_path_challenging,
                                                             number_of_records=50,
                                                             # number_of_records=IbugConf.orig_number_of_training,
                                                             only_label=False)
        lbl_arr_norm = []
        lbl_arr_norm_asm = []

        counter = 0
        for lbl in lbl_arr:
            landmark_arr_flat_n, landmark_arr_x_n, landmark_arr_y_n = \
                image_utility.create_landmarks_from_normalized(lbl_arr[counter], 224, 224, 112, 112)

            # imgpr.print_image_arr(str(counter), np.zeros([224,224]) , landmark_arr_x_n, landmark_arr_y_n)
            # imgpr.print_image_arr(str(counter)+'_img_', img_arr[counter], [], [])
            # imgpr.print_image_arr(str(counter)+'_img_',img_arr[counter], landmark_arr_x_n, landmark_arr_y_n)

            # landmark_90 = self._get_asm(lbl_arr[counter], 'ibug', 85)

            '''test image '''
            # landmark_arr_xy, landmark_arr_x, landmark_arr_y = image_utility.\
            #     create_landmarks_from_normalized(landmark_90, 224, 224, 112, 112)
            # imgpr.print_image_arr(str(counter)+'_90_', np.zeros([224,224]), landmark_arr_x, landmark_arr_y)


            # imgpr.print_histogram1(counter, np.array(landmark_arr_flat_n).reshape([98, 2]))

            # lbl_arr_norm.append(landmark_arr_flat_n)
            # # lbl_arr_norm.append(lbl)
            # # lbl_arr_norm_asm.append(self._get_asm(lbl, 'wflw', 80))
            # lbl_arr_norm_asm.append(self._get_asm(landmark_arr_flat_n, 'wflw', 80))

            counter += 1

        # imgpr.print_histogram_plt('main', 'full', lbl_arr_norm)
        # imgpr.print_histogram_plt('asm', 'full', lbl_arr_norm_asm)
        #
        # imgpr.print_histogram_plt('main', 'face', lbl_arr_norm)
        # imgpr.print_histogram_plt('asm', 'face', lbl_arr_norm_asm)

        return

        imgpr.print_arr('main', 'full', lbl_arr_norm)
        imgpr.print_arr('asm', 'full', lbl_arr_norm_asm)

        imgpr.print_arr('main', 'face', lbl_arr_norm)
        imgpr.print_arr('asm', 'face', lbl_arr_norm_asm)

        imgpr.print_arr('main', 'eyes', lbl_arr_norm)
        imgpr.print_arr('asm', 'eyes', lbl_arr_norm_asm)

        imgpr.print_arr('main', 'nose', lbl_arr_norm)
        imgpr.print_arr('asm', 'nose', lbl_arr_norm_asm)

        imgpr.print_arr('main', 'mouth', lbl_arr_norm)
        imgpr.print_arr('asm', 'mouth', lbl_arr_norm_asm)

        imgpr.print_arr('main', 'eyebrow', lbl_arr_norm)
        imgpr.print_arr('asm', 'eyebrow', lbl_arr_norm_asm)

        # lbl_arr_norm = np.mean(np.array(lbl_arr_norm), axis=0)
        # lbl_arr_norm_asm = np.mean(np.array(lbl_arr_norm_asm), axis=0)
        # imgpr.print_histogram1('main_', lbl_arr_norm.reshape([self.number_of_landmark//2, 2]))
        # imgpr.print_histogram1('ASM_', lbl_arr_norm_asm.reshape([self.number_of_landmark//2, 2]))


    def retrieve_tf_record_train(self, tfrecord_filename, number_of_records, only_label=True):
        print('retrieve_tf_record_train')

        with tf.Session() as sess:
            filename_queue = tf.train.string_input_producer([tfrecord_filename])
            image_raw, landmarks, pose, img_name = self.__read_and_decode(filename_queue)

            init_op = tf.initialize_all_variables()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            img_arr = []
            lbl_arr = []
            pose_arr = []
            img_name_arr = []

            for i in tqdm(range(number_of_records)):
                _image_raw, _landmarks, _pose, _img_name = sess.run([image_raw, landmarks, pose, img_name])
                # print(np.array(_landmarks).shape)

                if not only_label:
                    img = np.array(_image_raw)
                    img = img.reshape(InputDataSize.image_input_size, InputDataSize.image_input_size, 3)
                    img_arr.append(img)

                img_name_arr.append(_img_name)
                lbl_arr.append(_landmarks)
                pose_arr.append(_pose)

            coord.request_stop()
            coord.join(threads)
            """ the output image is x y x y array"""
            return lbl_arr, img_arr, pose_arr, img_name_arr

    def retrieve_tf_record_hm(self, tfrecord_filename, number_of_records, only_label=True):
        with tf.Session() as sess:
            filename_queue = tf.train.string_input_producer([tfrecord_filename])
            image_raw, landmarks, pose, heatmap, img_name = self.__read_and_decode_hm(filename_queue)

            init_op = tf.initialize_all_variables()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            img_arr = []
            lbl_arr = []
            pose_arr = []
            img_name_arr = []
            hm_arr = []

            for i in tqdm(range(number_of_records)):
                # _image_raw, _landmarks, _pose = sess.run([image_raw, landmarks, pose])
                _image_raw, _landmarks, _pose, _heatmap, _img_name = sess.run([image_raw, landmarks, pose, heatmap, img_name])

                if not only_label:
                    img = np.array(_image_raw)
                    img = img.reshape(InputDataSize.image_input_size, InputDataSize.image_input_size, 3)
                    img_arr.append(img)

                img_name_arr.append(_img_name)
                lbl_arr.append(_landmarks)
                pose_arr.append(_pose)
                hm_arr.append(_heatmap)

            coord.request_stop()
            coord.join(threads)
            """ the output image is x y x y array"""
            return lbl_arr, img_arr, pose_arr, hm_arr

    def retrieve_tf_record(self, tfrecord_filename, number_of_records, only_label=True):
        with tf.Session() as sess:
            filename_queue = tf.train.string_input_producer([tfrecord_filename])
            image_raw, landmarks, pose = self.__read_and_decode_test_set(filename_queue)
            # image_raw, landmarks, pose, img_name = self.__read_and_decode(filename_queue)

            init_op = tf.initialize_all_variables()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            img_arr = []
            lbl_arr = []
            pose_arr = []
            img_name_arr = []

            for i in tqdm(range(number_of_records)):
                _image_raw, _landmarks, _pose = sess.run([image_raw, landmarks, pose])
                # _image_raw, _landmarks, _pose, _img_name = sess.run([image_raw, landmarks, pose, img_name])

                if not only_label:
                    img = np.array(_image_raw)
                    img = img.reshape(InputDataSize.image_input_size, InputDataSize.image_input_size, 3)
                    img_arr.append(img)

                # img_name_arr.append(_img_name)
                lbl_arr.append(_landmarks)
                pose_arr.append(_pose)

            coord.request_stop()
            coord.join(threads)
            """ the output image is x y x y array"""
            return lbl_arr, img_arr, pose_arr

    def create_adv_att_img_hm(self):
        png_file_arr = []
        for file in os.listdir(IbugConf.rotated_img_path_prefix):
            if file.endswith(".jpg") or file.endswith(".png"):
                png_file_arr.append(os.path.join(IbugConf.rotated_img_path_prefix, file))

        number_of_samples = IbugConf.origin_number_of_all_sample

        image_utility = ImageUtility()

        for i in range(number_of_samples):
            img_file = png_file_arr[i]
            pts_file = png_file_arr[i][:-3] + "pts"

            if not os.path.exists(pts_file):
                continue

            points_arr = []
            with open(pts_file) as fp:
                line = fp.readline()
                cnt = 1
                while line:
                    if 3 < cnt < 72:
                        x_y_pnt = line.strip()
                        x = float(x_y_pnt.split(" ")[0])
                        y = float(x_y_pnt.split(" ")[1])
                        points_arr.append(x)
                        points_arr.append(y)
                    line = fp.readline()
                    cnt += 1

            img = Image.open(img_file)

            '''normalize image'''
            resized_img = np.array(img) / 255.0

            '''crop data: we add a small margin to the images'''
            landmark_arr_xy, landmark_arr_x, landmark_arr_y = image_utility.create_landmarks(landmarks=points_arr,
                                                                                             scale_factor_x=1,
                                                                                             scale_factor_y=1)

            '''augment the images, then normalize the landmarks based on the hyperface method'''
            for k in range(IbugConf.augmentation_factor):
                '''save the origin image as well'''
                if k == 0:
                    landmark_arr_flat_aug = landmark_arr_xy
                    img_aug = resized_img

                else:
                    '''save the augmented images'''
                    if k % 2 == 0:
                        landmark_arr_flat_aug, img_aug = image_utility.random_augmentation(landmark_arr_xy, resized_img)
                    else:
                        landmark_arr_flat_aug, img_aug = image_utility.augment(resized_img, landmark_arr_xy)

                '''test '''
                # imgpr.print_image_arr(k, img_aug, [], [])

                '''again resize image to 224*224 after augmentation'''
                resized_img_new = resize(img_aug,
                                         (InputDataSize.image_input_size, InputDataSize.image_input_size, 3)
                                         , anti_aliasing=True)

                # imgpr.print_image_arr(k, resized_img_new, [], [])

                dims = img_aug.shape
                height = dims[0]
                width = dims[1]
                scale_factor_y = InputDataSize.image_input_size / height
                scale_factor_x = InputDataSize.image_input_size / width

                '''retrieve and rescale landmarks in after augmentation'''
                landmark_arr_flat, landmark_arr_x, landmark_arr_y = \
                    image_utility.create_landmarks(landmarks=landmark_arr_flat_aug,
                                                   scale_factor_x=scale_factor_x,
                                                   scale_factor_y=scale_factor_y)

                # imgpr.print_image_arr(k, resized_img_new, landmark_arr_x, landmark_arr_y)

                '''normalize landmarks based on hyperface method'''
                width = len(resized_img_new[0])
                height = len(resized_img_new[1])
                x_center = width / 2
                y_center = height / 2
                landmark_arr_flat_normalized = []
                for p in range(0, len(landmark_arr_flat), 2):
                    landmark_arr_flat_normalized.append((x_center - landmark_arr_flat[p]) / width)
                    landmark_arr_flat_normalized.append((y_center - landmark_arr_flat[p + 1]) / height)

                '''test print after augmentation'''
                # landmark_arr_flat_n, landmark_arr_x_n, landmark_arr_y_n = image_utility.\
                #     create_landmarks_from_normalized(landmark_arr_flat_normalized, 224, 224, 112, 112)
                # imgpr.print_image_arr((i*100)+(k+1), resized_img_new, landmark_arr_x_n, landmark_arr_y_n)

                '''save image'''
                im = Image.fromarray((resized_img_new * 255).astype(np.uint8))
                file_name = IbugConf.ready_img_path_prefix + str(10000 * (i + 1) + k)
                im.save(str(file_name) + '.jpg')
                '''save points'''
                pnt_file = open(str(file_name) + ".pts", "w")
                pre_txt = ["version: 1 \n", "n_points: 68 \n", "{ \n"]
                pnt_file.writelines(pre_txt)
                points_txt = ""
                for l in range(0, len(landmark_arr_xy), 2):
                    points_txt += str(landmark_arr_xy[l]) + " " + str(landmark_arr_xy[l + 1]) + "\n"

                pnt_file.writelines(points_txt)
                pnt_file.write("} \n")
                pnt_file.close()

        return number_of_samples

    def retrieve_tf_record_test_set(self, tfrecord_filename, number_of_records, only_label=True):
        with tf.Session() as sess:
            filename_queue = tf.train.string_input_producer([tfrecord_filename])
            image_raw, landmarks, pose = self.__read_and_decode_test_set(filename_queue)

            init_op = tf.initialize_all_variables()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            img_arr = []
            lbl_arr = []

            for i in range(number_of_records):
                _landmarks, _pose, _image_raw = sess.run([landmarks, pose, image_raw])
                # _image_raw, _landmarks = sess.run([image_raw, pose, landmarks])

                if not only_label:
                    img = np.array(_image_raw)
                    img = img.reshape(InputDataSize.image_input_size, InputDataSize.image_input_size, 3)
                    img_arr.append(img)

                lbl_arr.append(_landmarks)

            coord.request_stop()
            coord.join(threads)
            """ the output image is x y x y array"""
            return lbl_arr, img_arr

    def create_training_tensor(sefl, tfrecord_filename, batch_size):
        SHUFFLE_BUFFER = 100
        BATCH_SIZE = batch_size

        dataset = tf.data.TFRecordDataset(tfrecord_filename)

        # Maps the parser on every file path in the array. You can set the number of parallel loaders here

        dataset = dataset.map(sefl.__parse_function, num_parallel_calls=32)

        # This dataset will go on forever
        dataset = dataset.repeat()

        # Set the number of data points you want to load and shuffle
        dataset = dataset.shuffle(SHUFFLE_BUFFER)

        # Set the batch size
        dataset = dataset.batch(BATCH_SIZE)

        # Create an iterator
        iterator = dataset.make_one_shot_iterator()

        # Create your tf representation of the iterator

        images, landmarks, pose = iterator.get_next()
        return images, landmarks, pose

    def __top_n_indexes_tensor(self, arr, n):
        shape = tf.shape(arr)
        top_values, top_indices = tf.nn.top_k(tf.reshape(arr, (-1,)), n)
        top_indices = tf.stack(((top_indices // shape[1]), (top_indices % shape[1])), -1)
        return top_values, top_indices

    def __top_n_indexes(self, arr, n):
        import bottleneck as bn
        idx = bn.argpartition(arr, arr.size - n, axis=None)[-n:]
        width = arr.shape[1]
        return [divmod(i, width) for i in idx]

    def __find_nth_biggest_avg_tensor(self, heatmap, points, scalar):
        weights, indices = self.__top_n_indexes_tensor(heatmap, points)

        x_indices = tf.cast(indices[:, 0], tf.float32)
        y_indices = tf.cast(indices[:, 1], tf.float32)
        '''weighted average over x and y'''
        w_avg_x = tf.scalar_mul(1 / tf.reduce_sum(weights), tf.reduce_sum([tf.multiply(x_indices, weights)]))
        w_avg_x = tf.scalar_mul(1 / 56, w_avg_x)

        w_avg_y = tf.scalar_mul(1 / tf.reduce_sum(weights), tf.reduce_sum([tf.multiply(y_indices, weights)]))
        w_avg_y = tf.scalar_mul(1 / 56, w_avg_y)

        return tf.stack([w_avg_x, w_avg_y])

    def __find_nth_biggest_avg(self, heatmap, points, scalar):
        indices = self.__top_n_indexes(heatmap, points)

        x_arr = []
        y_arr = []
        w_s = 0
        x_s = 0
        y_s = 0

        for index in indices:
            x_arr.append(index[1])
            y_arr.append(index[0])
            w_i = heatmap[index[1], index[0]]

            if w_i < 0:
                w_i *= -1

            if w_i == 0:
                w_i = 0.00000000001

            w_s += w_i
            x_s += (w_i * index[1])
            y_s += (w_i * index[0])

        if w_s > 0:
            x_s = (x_s / w_s) * scalar
            y_s = (y_s / w_s) * scalar
            return x_s, y_s
        else:
            return 0, 0

        # indices = indices[0]
        # return indices[1]*4, indices[0]*4

    def from_heatmap_to_point_tensor(self, heatmaps, number_of_points, scalar=4):  # 56*56*68 => {(x,y), (), }
        # x_points = []
        # y_points = []
        # xy_points = []
        # print(heatmaps.shape)
        # indices =

        x = tf.stack([self.__find_nth_biggest_avg_tensor(heatmaps[:, :, i], number_of_points, scalar)
                      for i in range(LearningConfig.point_len)], -1)
        return x

        # return np.array(x_points), np.array(y_points), np.array(xy_points)

        # for i in range(heatmaps.shape[2]):
        #     x, y = self.__find_nth_biggest_avg_tensor(heatmaps[:, :, i], number_of_points, scalar)
        #     x_points.append(x)
        #     y_points.append(y)
        #     xy_points.append(x)
        #     xy_points.append(y)
        # return np.array(x_points), np.array(y_points), np.array(xy_points)

    def from_heatmap_to_point(self, heatmaps, number_of_points, scalar=4):  # 56*56*68 => {(x,y), (), }
        """"""
        x_points = []
        y_points = []
        xy_points = []
        # print(heatmaps.shape)
        for i in range(heatmaps.shape[2]):
            x, y = self.__find_nth_biggest_avg(heatmaps[:, :, i], number_of_points, scalar)
            x_points.append(x)
            y_points.append(y)
            xy_points.append(x)
            xy_points.append(y)
        return np.array(x_points), np.array(y_points), np.array(xy_points)

    def __gaussian_k(self, x0, y0, sigma, width, height):
        """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
        """
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # def retrive_hm_and_test(self):

    def create_image_and_labels_name(self, dataset_name):
        images_dir = IbugConf.train_images_dir
        lbls_dir = IbugConf.train_hm_dir

        img_filenames = []
        lbls_filenames = []

        for file in os.listdir(images_dir):
            if file.endswith(".jpg") or file.endswith(".png"):
                # lbl_file = lbls_dir + str(file)[:-3] + "npy"  # filename and address
                lbl_file = str(file)[:-3] + "npy"  # just name
                if os.path.exists(lbls_dir + lbl_file):
                    # img_filenames.append(images_dir + str(file)) # filename and address
                    img_filenames.append(str(file))
                    lbls_filenames.append(lbl_file)

        return np.array(img_filenames), np.array(lbls_filenames)

    def load_hm_and_test(self, dataset_name):

        npy_dir_85 = IbugConf.train_hm_dir_85
        npy_dir_90 = IbugConf.train_hm_dir_90
        npy_dir_97 = IbugConf.train_hm_dir_97
        arr_err = []
        counter = 1
        for file in tqdm(os.listdir(IbugConf.train_hm_dir_85)):
            try:
                hm_85 = load(npy_dir_85 + file)
                hm_90 = load(npy_dir_90 + file)
                hm_97 = load(npy_dir_97 + file)
                imgpr.print_image_arr_heat(counter, hm_85, print_single=False)
                counter += 1
            except:
                arr_err.append(file)
                print('err: -->' + str(counter) + ' - f: --> ' + file)

                counter += 1

        return arr_err

    def detect_pose_and_save(self, dataset_name):

        if dataset_name == DatasetName.ibug:
            images_dir = IbugConf.train_images_dir
            pose_npy_dir = IbugConf.pose_npy_dir
        elif dataset_name == DatasetName.cofw:
            images_dir = CofwConf.train_images_dir
            pose_npy_dir = CofwConf.pose_npy_dir
        elif dataset_name == DatasetName.wflw:
            images_dir = WflwConf.train_images_dir
            pose_npy_dir = WflwConf.pose_npy_dir
        elif dataset_name == DatasetName.cofw_test:
            images_dir = CofwConf.test_images_dir
            pose_npy_dir = CofwConf.test_pose_npy_dir
        elif dataset_name == DatasetName.wflw_test:
            images_dir = WflwConf.test_images_dir
            pose_npy_dir = WflwConf.test_pose_npy_dir
        elif dataset_name == DatasetName.ibug_test:
            images_dir = IbugConf.test_images_dir
            pose_npy_dir = IbugConf.test_pose_npy_dir
        else:
            images_dir = ''
            pose_npy_dir = ''

        counter = 1
        for file in tqdm(os.listdir(images_dir)):
            if file.endswith(".png") or file.endswith(".jpg"):
                file_name = os.path.join(images_dir, file)
                file_name_save = str(file)[:-3] + "npy"

                img = np.array(Image.open(file_name)) / 255.0
                yaw_predicted, pitch_predicted, roll_predicted = pose_detector.detect(img, isFile=False, show=False)
                '''normalize pose -1 -> +1 '''
                min_degree = -65
                max_degree = 65
                yaw_normalized = 2 * ((yaw_predicted - min_degree) / (max_degree - min_degree)) - 1
                pitch_normalized = 2 * ((pitch_predicted - min_degree) / (max_degree - min_degree)) - 1
                roll_normalized = 2 * ((roll_predicted - min_degree) / (max_degree - min_degree)) - 1

                pose_array = np.array([yaw_normalized, pitch_normalized, roll_normalized])

                np_path = pose_npy_dir + file_name_save
                save(np_path, pose_array)
                counter += 1
        print("detect_pose_and_save DONE.")

    def test_normalize_points(self, dataset_name):
        img_util = ImageUtility()

        if dataset_name == DatasetName.ibug:
            images_dir = IbugConf.train_images_dir
            normalized_points_npy_dir = IbugConf.normalized_points_npy_dir
            num_of_landmarks = IbugConf.num_of_landmarks
        elif dataset_name == DatasetName.cofw:
            images_dir = CofwConf.train_images_dir
            normalized_points_npy_dir = CofwConf.normalized_points_npy_dir
            num_of_landmarks = CofwConf.num_of_landmarks
        elif dataset_name == DatasetName.wflw:
            images_dir = WflwConf.train_images_dir
            normalized_points_npy_dir = WflwConf.normalized_points_npy_dir
            num_of_landmarks = WflwConf.num_of_landmarks
        elif dataset_name == DatasetName.cofw_test:
            images_dir = CofwConf.test_images_dir
            normalized_points_npy_dir = CofwConf.test_normalized_points_npy_dir
            num_of_landmarks = CofwConf.num_of_landmarks

        elif dataset_name == DatasetName.wflw_test:
            images_dir = WflwConf.test_images_dir
            normalized_points_npy_dir = WflwConf.test_normalized_points_npy_dir
            num_of_landmarks = WflwConf.num_of_landmarks
        elif dataset_name == DatasetName.ibug_test:
            images_dir = IbugConf.test_images_dir
            normalized_points_npy_dir = IbugConf.test_normalized_points_npy_dir
            num_of_landmarks = IbugConf.num_of_landmarks

        counter = 1
        for file in tqdm(os.listdir(normalized_points_npy_dir)):
            img_name = os.path.join(images_dir, str(file)[:-3] + "jpg")
            img = Image.open(img_name)
            if file.endswith(".npy"):
                # if file != '2050002.npy':
                #     continue
                points_arr = []
                file_name = os.path.join(normalized_points_npy_dir, file)
                points_arr = load(file_name)

                landmark_arr_xy, landmark_arr_x, landmark_arr_y = img_util.create_landmarks_from_normalized(points_arr, 224, 224, 112, 112)
                imgpr.print_image_arr(counter + 1, img, landmark_arr_x, landmark_arr_y)
                counter += 1

    def normalize_points_and_save(self, dataset_name):
        img_util = ImageUtility()

        if dataset_name == DatasetName.ibug:
            images_dir = IbugConf.train_images_dir
            normalized_points_npy_dir = IbugConf.normalized_points_npy_dir
            num_of_landmarks = IbugConf.num_of_landmarks
        elif dataset_name == DatasetName.cofw:
            images_dir = CofwConf.train_images_dir
            normalized_points_npy_dir = CofwConf.normalized_points_npy_dir
            num_of_landmarks = CofwConf.num_of_landmarks
        elif dataset_name == DatasetName.wflw:
            images_dir = WflwConf.train_images_dir
            normalized_points_npy_dir = WflwConf.normalized_points_npy_dir
            num_of_landmarks = WflwConf.num_of_landmarks
        elif dataset_name == DatasetName.cofw_test:
            images_dir = CofwConf.test_images_dir
            normalized_points_npy_dir = CofwConf.test_normalized_points_npy_dir
            num_of_landmarks = CofwConf.num_of_landmarks

        elif dataset_name == DatasetName.wflw_test:
            images_dir = WflwConf.test_images_dir
            normalized_points_npy_dir = WflwConf.test_normalized_points_npy_dir
            num_of_landmarks = WflwConf.num_of_landmarks
        elif dataset_name == DatasetName.ibug_test:
            images_dir = IbugConf.test_images_dir
            normalized_points_npy_dir = IbugConf.test_normalized_points_npy_dir
            num_of_landmarks = IbugConf.num_of_landmarks

        counter = 1
        for file in tqdm(os.listdir(images_dir)):
            if file.endswith(".pts"):
                points_arr = []
                file_name = os.path.join(images_dir, file)
                file_name_save = str(file)[:-3] + "npy"

                img_name_save = os.path.join(images_dir, str(file)[:-3] + "jpg")
                img = Image.open(img_name_save)
                with open(file_name) as fp:
                    line = fp.readline()
                    cnt = 1
                    while line:
                        if 3 < cnt <= num_of_landmarks + 3:
                            x_y_pnt = line.strip()
                            x = float(x_y_pnt.split(" ")[0])
                            y = float(x_y_pnt.split(" ")[1])
                            points_arr.append(x)
                            points_arr.append(y)
                        line = fp.readline()
                        cnt += 1

                normalized_points = self.generate_normalized_points(np.array(points_arr),
                                                                    InputDataSize.image_input_size,
                                                                    InputDataSize.image_input_size)
                np_path = normalized_points_npy_dir + file_name_save

                '''these are for test'''
                image_utility = ImageUtility()
                landmark_arr_flat_n, landmark_arr_x_n, landmark_arr_y_n = image_utility.\
                    create_landmarks_from_normalized(normalized_points,
                                                     InputDataSize.image_input_size,
                                                     InputDataSize.image_input_size,
                                                     InputDataSize.image_input_size/2,
                                                     InputDataSize.image_input_size/2
                                                     )
                # imgpr.print_image_arr(counter+1, img, landmark_arr_x_n, landmark_arr_y_n)
                ''''''
                save(np_path, normalized_points)
                counter += 1
        print('normalize_points_and_save COMPLETED!!!')

    def generate_normalized_points(self, points_arr, width, height):
        """normalize landmarks based on hyperface method"""

        x_center = width / 2
        y_center = height / 2
        landmark_arr_flat_normalized = []
        for p in range(0, len(points_arr), 2):
            landmark_arr_flat_normalized.append((points_arr[p] - x_center) / width)
            landmark_arr_flat_normalized.append((points_arr[p + 1] - y_center) / height)
        return landmark_arr_flat_normalized

    def generate_hm_and_save(self, dataset_name, pca_percentage=100):
        pca_util = PCAUtility()

        if pca_percentage != 100:
            eigenvalues = load('pca_obj/' + dataset_name + pca_util.eigenvalues_prefix + str(pca_percentage) + ".npy")
            eigenvectors = load('pca_obj/' + dataset_name + pca_util.eigenvectors_prefix + str(pca_percentage) + ".npy")
            meanvector = load('pca_obj/' + dataset_name + pca_util.meanvector_prefix + str(pca_percentage) + ".npy")

        images_dir = IbugConf.train_images_dir
        npy_dir = IbugConf.train_hm_dir

        if pca_percentage == 85:
            npy_dir = IbugConf.train_hm_dir_85
        if pca_percentage == 90:
            npy_dir = IbugConf.train_hm_dir_90
        if pca_percentage == 97:
            npy_dir = IbugConf.train_hm_dir_97

        counter = 1
        for file in tqdm(os.listdir(images_dir)):
            if file.endswith(".pts"):
                points_arr = []
                file_name = os.path.join(images_dir, file)
                file_name_save = str(file)[:-3] + "npy"
                with open(file_name) as fp:
                    line = fp.readline()
                    cnt = 1
                    while line:
                        if 3 < cnt < 72:
                            x_y_pnt = line.strip()
                            x = float(x_y_pnt.split(" ")[0])
                            y = float(x_y_pnt.split(" ")[1])
                            points_arr.append(x)
                            points_arr.append(y)
                        line = fp.readline()
                        cnt += 1
                if pca_percentage != 100:
                    b_vector_p = pca_util.calculate_b_vector(points_arr, True, eigenvalues, eigenvectors, meanvector)
                    points_arr_new = meanvector + np.dot(eigenvectors, b_vector_p)
                    points_arr = points_arr_new.tolist()

                hm = self.generate_hm(56, 56, np.array(points_arr), 2, False)
                hm_f = npy_dir + file_name_save

                # imgpr.print_image_arr_heat(counter, hm, print_single=True)

                save(hm_f, hm)
                counter += 1
        print('generate_hm_and_save COMPLETED!!!')

    def generate_partial_hm(self, height, width, landmarks_arr, s=3.0, upsample=True):
        hm_ar =[]
        for landmark in landmarks_arr:
            point_wise_hm = self.generate_hm(height, width, landmark, s, upsample)
            partial_hm = np.sum(point_wise_hm, axis=2)
            hm_ar.append(partial_hm)
        return hm_ar

    def generate_hm(self, height, width, landmarks, s=3.0, upsample=True):
        Nlandmarks = len(landmarks)
        hm = np.zeros((height, width, Nlandmarks // 2), dtype=np.float32)

        j = 0
        for i in range(0, Nlandmarks, 2):

            if upsample:
                x = float(landmarks[i]) * InputDataSize.image_input_size + InputDataSize.img_center
                y = float(landmarks[i + 1]) * InputDataSize.image_input_size + InputDataSize.img_center
            else:
                x = landmarks[i]
                y = landmarks[i + 1]

            x = int(x // 4)
            y = int(y // 4)

            hm[:, :, j] = self.__gaussian_k(x, y, s, height, width)
            j += 1
        return hm

    def __parse_function_reduced(self, proto):
        keys_to_features = {'landmarks': tf.FixedLenFeature([self.number_of_landmark], tf.float32),
                            'heatmap': tf.FixedLenFeature([56, 56, 68], tf.float32),
                            'image_raw': tf.FixedLenFeature([InputDataSize.image_input_size,
                                                             InputDataSize.image_input_size, 3], tf.float32)}

        parsed_features = tf.parse_single_example(proto, keys_to_features)
        _images = parsed_features['image_raw']
        _landmarks = parsed_features["landmarks"]
        _heatmap = parsed_features["heatmap"]
        return _images, _landmarks, _heatmap

    def __parse_function(self, proto):
        keys_to_features = {'landmarks': tf.FixedLenFeature([self.number_of_landmark], tf.float32),
                            'pose': tf.FixedLenFeature([InputDataSize.pose_len], tf.float32),
                            'image_raw': tf.FixedLenFeature([InputDataSize.image_input_size,
                                                             InputDataSize.image_input_size, 3], tf.float32)}

        parsed_features = tf.parse_single_example(proto, keys_to_features)
        _images = parsed_features['image_raw']
        _landmarks = parsed_features["landmarks"]
        _pose = parsed_features["pose"]

        return _images, _landmarks, _pose


    def __create_tfrecord_aflw(self):
        counter = 1

        # Open the sqlite database
        conn = sqlite3.connect('/media/ali/extradata/facial_landmark_ds/aflw/aflw-db/aflw/data/aflw.sqlite')
        c = conn.cursor()
        c2 = conn.cursor()

        # Creating the query string for retriving: roll, pitch, yaw and faces position
        # Change it according to what you want to retrieve
        select_string = "faceimages.filepath, faces.face_id, facepose.roll, facepose.pitch, facepose.yaw, facerect.x, facerect.y, facerect.w, facerect.h"
        from_string = "faceimages, faces, facepose, facerect"
        where_string = "faces.face_id = facepose.face_id and faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id"
        query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string

        # It iterates through the rows returned from the query
        for row in c.execute(query_string):

            query_string_facial = "SELECT * FROM  FeatureCoords WHERE FeatureCoords.face_id ="
            # Creating the full path names for input and output
            input_path = self.images_path + str(row[0])
            output_path = self.storing_path + str(row[0])

            # If the file exist then open it
            if (os.path.isfile(input_path) == True):

                query_string_facial += str(row[1])

                facial_array = np.zeros(shape=[21, 3])
                for facial_row in c2.execute(query_string_facial):
                    feature_id = facial_row[1]
                    x = facial_row[2]
                    y = facial_row[3]
                    visibility = 1
                    facial_array[feature_id - 1] = [visibility, x, y]
                    # print(facial_row)
                print("-------")

                # image = cv2.imread(input_path)  # load in grayscale
                img = Image.open(input_path)
                image = np.array(img) / 255.0

                # Image dimensions
                image_h, image_w, image_ch = image.shape

                # Roll, pitch and yaw
                roll = row[2]
                pitch = row[3]
                yaw = row[4]

                # Face rectangle coords
                face_x = row[5]
                face_y = row[6]
                face_w = row[7]
                face_h = row[8]

                # Error correction
                if (face_x < 0): face_x = 0
                if (face_y < 0): face_y = 0
                if (face_w > image_w):
                    face_w = image_w
                    face_h = image_w
                if (face_h > image_h):
                    face_h = image_h
                    face_w = image_h

                # Crop the face from the image
                image_cropped = np.copy(image[face_y:face_y + face_h, face_x:face_x + face_w])

                # relocate landmarks after resize
                for i in range(len(facial_array)):
                    if facial_array[i][0] == 1:
                        facial_array[i][1] = abs(face_x - facial_array[i][1])
                        facial_array[i][2] = abs(face_y - facial_array[i][2])

                # image_rescaled = resize(image_cropped,
                #                         (InputDataSize.image_input_size, InputDataSize.image_input_size, 3),
                #                         anti_aliasing=True)

                image_utility = ImageUtility()
                landmark_arr_flat, landmark_arr_x, landmark_arr_y = \
                    image_utility.create_landmarks_aflw(landmarks=facial_array, scale_factor_x=1, scale_factor_y=1)

                imgpr.print_image_arr(counter, image_cropped, landmark_arr_x, landmark_arr_y)

                # # Printing the information
                # print("Counter: " + str(counter))
                # print("iPath:    " + input_path)
                # print("oPath:    " + output_path)
                # print("Roll:    " + str(roll))
                # print("Pitch:   " + str(pitch))
                # print("Yaw:     " + str(yaw))
                # print("x:       " + str(face_x))
                # print("y:       " + str(face_y))
                # print("w:       " + str(face_w))
                # print("h:       " + str(face_h))
                # print("")

                # Increasing the counter
                counter = counter + 1

                # if the file does not exits it return an exception
            else:
                raise ValueError('Error: I cannot find the file specified: ' + str(input_path))

        # Once finished the iteration it closes the database
        c.close()

    def __create_tfrecord_affectnet(self, dataset_type, need_augmentation):
        fileDir = os.path.dirname(os.path.realpath('__file__'))

        csv_data_file = None
        tfrecord_filename = None
        number_of_samples = 0

        image_utility = ImageUtility()

        if dataset_type == DatasetType.data_type_train:
            tfrecord_filename = AffectnetConf.tf_train_path
            csv_data_file = AffectnetConf.csv_train_path
            print('creating train_tf.record...')
            number_of_samples = AffectnetConf.sum_of_train_samples

        elif dataset_type == DatasetType.data_type_test:
            tfrecord_filename = AffectnetConf.tf_test_path
            csv_data_file = AffectnetConf.csv_test_path
            print('creating test_tf.record...')
            number_of_samples = AffectnetConf.sum_of_test_samples

        elif dataset_type == DatasetType.data_type_validation:
            tfrecord_filename = AffectnetConf.tf_evaluation_path
            csv_data_file = AffectnetConf.csv_evaluate_path
            number_of_samples = AffectnetConf.sum_of_validation_samples
            print('creating validation_tf.record...')

        writer = tf.python_io.TFRecordWriter(tfrecord_filename)

        try:
            with open(csv_data_file) as csvfile:
                file_reader = csv.reader(csvfile, delimiter=',')
                k = 0
                for row in file_reader:
                    if k > number_of_samples:
                        break
                    img_path = row[0]
                    landmarks = row[5]

                    img_path = AffectnetConf.img_path_prefix + img_path
                    my_file = Path(img_path)

                    if my_file.is_file():
                        msg = '\033[92m' + " sample number " + str(k + 1) + \
                              " created." + '\033[94m' + "remains " + str(number_of_samples - k - 1)
                        sys.stdout.write('\r' + msg)
                        k += 1
                        # convert image to np array
                        img = Image.open(img_path)
                        # normalize data
                        img_arr = np.array(img) / 255.0

                        # resize image to 224*224
                        resized_img = resize(img_arr,
                                             (InputDataSize.image_input_size, InputDataSize.image_input_size, 3),
                                             anti_aliasing=True)
                        dims = img_arr.shape
                        height = dims[0]
                        width = dims[1]
                        scale_factor_y = InputDataSize.image_input_size / height
                        scale_factor_x = InputDataSize.image_input_size / width

                        # retrieve landmarks
                        landmark_arr_flat, landmark_arr_x, landmark_arr_y = \
                            image_utility.create_landmarks(landmarks=landmarks,
                                                           scale_factor_x=scale_factor_x,
                                                           scale_factor_y=scale_factor_y)

                        writable_img = np.reshape(resized_img,
                                                  [InputDataSize.image_input_size * InputDataSize.image_input_size * 3])
                        landmark_arr_flat = np.array(landmark_arr_flat)
                        feature = {'landmarks': self.__float_feature(landmark_arr_flat),
                                   'image_raw': self.__float_feature(writable_img)}
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        writer.write(example.SerializeToString())

                        # augmentation
                        if need_augmentation:
                            landmark_arr_flat_aug, img_aug = \
                                image_utility.random_augmentation(landmark_arr_flat, resized_img)
                            resized_img_aug = resize(img_aug,
                                                     (
                                                         InputDataSize.image_input_size, InputDataSize.image_input_size,
                                                         3),
                                                     anti_aliasing=True)
                            dims = img_aug.shape
                            height = dims[0]
                            width = dims[1]
                            scale_factor_y = InputDataSize.image_input_size / height
                            scale_factor_x = InputDataSize.image_input_size / width

                            landmark_arr_xy, landmark_arr_x, landmark_arr_y = \
                                image_utility.create_landmarks(landmarks=landmark_arr_flat_aug,
                                                               scale_factor_x=scale_factor_x,
                                                               scale_factor_y=scale_factor_y)

                            writable_img = np.reshape(resized_img_aug,
                                                      [
                                                          InputDataSize.image_input_size * InputDataSize.image_input_size * 3])
                            feature = {'landmarks': self.__float_feature(landmark_arr_xy),
                                       'image_raw': self.__float_feature(writable_img)}
                            example = tf.train.Example(features=tf.train.Features(feature=feature))
                            writer.write(example.SerializeToString())

                            # test print
                            # imgpr.print_image_arr(k, resized_img_aug, landmark_arr_x, landmark_arr_y)

        except Exception as e:
            print("\n TFRecordUtility -> Exception-->>")
            print(e)
        finally:
            writer.close()

        return number_of_samples

    '''w300 is just for test, so we use the basic bounding boxes, no augmentation and just test_tf_record'''

    def __create_tfrecord_w300(self, dataset_type):
        png_file_arr = []

        if dataset_type == 0:  # challenging
            for file in os.listdir(W300Conf.img_path_prefix_challenging):
                if file.endswith(".jpg") or file.endswith(".png"):
                    png_file_arr.append(os.path.join(W300Conf.img_path_prefix_challenging, file))

            writer_test = tf.python_io.TFRecordWriter(W300Conf.tf_challenging)
            number_of_samples = W300Conf.number_of_all_sample_challenging

        elif dataset_type == 1:  # common
            for file in os.listdir(W300Conf.img_path_prefix_common):
                if file.endswith(".jpg") or file.endswith(".png"):
                    png_file_arr.append(os.path.join(W300Conf.img_path_prefix_common, file))

            writer_test = tf.python_io.TFRecordWriter(W300Conf.tf_common)
            number_of_samples = W300Conf.number_of_all_sample_common
        else:  # full
            for file in os.listdir(W300Conf.img_path_prefix_full):
                if file.endswith(".jpg") or file.endswith(".png"):
                    png_file_arr.append(os.path.join(W300Conf.img_path_prefix_full, file))

            writer_test = tf.python_io.TFRecordWriter(W300Conf.tf_full)
            number_of_samples = W300Conf.number_of_all_sample_full

        image_utility = ImageUtility()

        for i in range(number_of_samples):
            img_file = png_file_arr[i]
            pts_file = png_file_arr[i][:-3] + "pts"

            points_arr = []
            with open(pts_file) as fp:
                line = fp.readline()
                cnt = 1
                while line:
                    if 3 < cnt < 72:
                        x_y_pnt = line.strip()
                        x = float(x_y_pnt.split(" ")[0])
                        y = float(x_y_pnt.split(" ")[1])
                        points_arr.append(x)
                        points_arr.append(y)
                    line = fp.readline()
                    cnt += 1
            #
            img = Image.open(img_file)

            # normalize img
            img_arr = np.array(img) / 255.0

            # crop data

            xy_points, x_points, y_points = image_utility.create_landmarks(landmarks=points_arr,
                                                                           scale_factor_x=1, scale_factor_y=1)
            img_arr, points_arr = image_utility.cropImg(img_arr, x_points, y_points, no_padding=True)

            # resize image to 224*224
            resized_img = resize(img_arr,
                                 (InputDataSize.image_input_size, InputDataSize.image_input_size, 3),
                                 anti_aliasing=True)
            dims = img_arr.shape
            height = dims[0]
            width = dims[1]
            scale_factor_y = InputDataSize.image_input_size / height
            scale_factor_x = InputDataSize.image_input_size / width

            # retrieve landmarks
            landmark_arr_xy, landmark_arr_x, landmark_arr_y = \
                image_utility.create_landmarks(landmarks=points_arr,
                                               scale_factor_x=scale_factor_x,
                                               scale_factor_y=scale_factor_y)

            '''normalize landmarks based on hyperface method'''
            width = len(resized_img[0])
            height = len(resized_img[1])
            x_center = width / 2
            y_center = height / 2
            landmark_arr_flat_normalized = []
            for p in range(0, len(landmark_arr_xy), 2):
                landmark_arr_flat_normalized.append((x_center - landmark_arr_xy[p]) / width)
                landmark_arr_flat_normalized.append((y_center - landmark_arr_xy[p + 1]) / height)

            '''creating landmarks for partial tasks'''
            landmark_face = landmark_arr_flat_normalized[0:54]  # landmark_face_len = 54
            landmark_nose = landmark_arr_flat_normalized[54:72]  # landmark_nose_len = 18
            landmark_eys = landmark_arr_flat_normalized[72:96]  # landmark_eys_len = 24
            landmark_mouth = landmark_arr_flat_normalized[96:136]  # landmark_mouth_len = 40

            '''test print after augmentation'''
            # landmark_arr_flat_n, landmark_arr_x_n, landmark_arr_y_n = image_utility.\
            #     create_landmarks_from_normalized(landmark_arr_flat_normalized, 224, 224, 112, 112)
            # imgpr.print_image_arr((i+1), resized_img, landmark_arr_x_n, landmark_arr_y_n)

            writable_img = np.reshape(resized_img,
                                      [InputDataSize.image_input_size * InputDataSize.image_input_size * 3])
            landmark_arr_flat_normalized = np.array(landmark_arr_flat_normalized)
            feature = {'landmarks': self.__float_feature(landmark_arr_flat_normalized),
                       'face': self.__float_feature(landmark_face),
                       'eyes': self.__float_feature(landmark_eys),
                       'nose': self.__float_feature(landmark_nose),
                       'mouth': self.__float_feature(landmark_mouth),
                       'image_raw': self.__float_feature(writable_img)}
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            writer_test.write(example.SerializeToString())
            msg = 'test --> \033[92m' + " sample number " + str(i + 1) + \
                  " created." + '\033[94m' + "remains " + str(number_of_samples - i - 1)
            sys.stdout.write('\r' + msg)

        writer_test.close()

        return number_of_samples

    def crop_and_save(self, dataset_name, dataset_type):
        number_of_samples = 0
        img_path_prefix = ''
        num_of_landmarks = 0

        if dataset_name == DatasetName.wflw:
            number_of_samples = WflwConf.orig_number_of_training
            img_path_prefix = WflwConf.img_path_prefix
            crop_img_path_prefix = WflwConf.train_images_dir
            pts_path_prefix = WflwConf.img_path_prefix
            num_of_landmarks = WflwConf.num_of_landmarks
            img_ext = "jpg"

        elif dataset_name == DatasetName.cofw_test:
            number_of_samples = CofwConf.orig_number_of_test
            img_path_prefix = CofwConf.test_img_path_prefix
            crop_img_path_prefix = CofwConf.test_images_dir
            pts_path_prefix = CofwConf.test_img_path_prefix
            num_of_landmarks = CofwConf.num_of_landmarks
            img_ext = "png"
        elif dataset_name == DatasetName.wflw_test:
            if dataset_type == DatasetType.wflw_full:
                number_of_samples = WflwConf.orig_number_of_test
            elif dataset_type == DatasetType.wflw_blur:
                number_of_samples = WflwConf.orig_of_all_test_blur
            elif dataset_type == DatasetType.wflw_largepose:
                number_of_samples = WflwConf.orig_of_all_test_largepose
            elif dataset_type == DatasetType.wflw_occlusion:
                number_of_samples = WflwConf.orig_of_all_test_occlusion
            elif dataset_type == DatasetType.wflw_makeup:
                number_of_samples = WflwConf.orig_of_all_test_makeup
            elif dataset_type == DatasetType.wflw_expression:
                number_of_samples = WflwConf.orig_of_all_test_expression
            elif dataset_type == DatasetType.wflw_illumination:
                number_of_samples = WflwConf.orig_of_all_test_illumination

            img_path_prefix = WflwConf.img_path_prefix  # both test and trains are in one Folder
            pts_path_prefix = WflwConf.test_img_path_prefix
            crop_img_path_prefix = WflwConf.test_images_dir
            num_of_landmarks = WflwConf.num_of_landmarks
            img_ext = "jpg"

        elif dataset_name == DatasetName.ibug_test:
            if dataset_type == DatasetType.ibug_challenging:
                number_of_samples = IbugConf.orig_number_of_test_challenging
            elif dataset_type == DatasetType.ibug_full:
                number_of_samples = IbugConf.orig_number_of_test_full
            elif dataset_type == DatasetType.ibug_comomn:
                number_of_samples = IbugConf.orig_number_of_test_common

            img_path_prefix = IbugConf.test_img_path_prefix
            pts_path_prefix = IbugConf.test_img_path_prefix

            crop_img_path_prefix = IbugConf.test_images_dir
            num_of_landmarks = IbugConf.num_of_landmarks
            img_ext = "jpg"

        pts_file_arr = []
        name_file_arr = []

        for file in os.listdir(pts_path_prefix):
            if file.endswith(".pts"):
                pts_file_arr.append(os.path.join(pts_path_prefix, file))
                name_file_arr.append(file)

        image_utility = ImageUtility()

        for i in tqdm(range(number_of_samples)):
            pts_file = pts_file_arr[i]

            if '#' in pts_file_arr[i]:
                img_file = img_path_prefix + name_file_arr[i].split('#')[0] + "." + img_ext
            else:
                if os.path.isfile(img_path_prefix + name_file_arr[i][:-3] + "jpg"):
                    img_file = img_path_prefix + name_file_arr[i][:-3] + "jpg"
                elif os.path.isfile(img_path_prefix + name_file_arr[i][:-3] + "png"):
                    img_file = img_path_prefix + name_file_arr[i][:-3] + "png"

            points_arr = []
            with open(pts_file) as fp:
                line = fp.readline()
                cnt = 1
                while line:
                    if 3 < cnt <= num_of_landmarks + 3:
                        x_y_pnt = line.strip()
                        x = float(x_y_pnt.split(" ")[0])
                        y = float(x_y_pnt.split(" ")[1])
                        points_arr.append(x)
                        points_arr.append(y)
                    line = fp.readline()
                    cnt += 1

            img = Image.open(img_file)
            img = np.array(img)

            resized_img = img
            landmark_arr_xy = points_arr

            image_utility.crop_and_save(resized_img, landmark_arr_xy,
                                            crop_img_path_prefix + str(10000 * (i + 1)),
                                            num_of_landmarks, dataset_name)
        print("crop_and_save")

    def rotaate_and_save(self, dataset_name):
        '''rotate image and save it in pts'''

        number_of_samples = 0
        img_path_prefix = ''
        rotated_img_path_prefix = ''
        augmentation_factor_rotate = 0
        num_of_landmarks = 0

        if dataset_name == DatasetName.ibug:
            number_of_samples = IbugConf.orig_number_of_training
            img_path_prefix = IbugConf.img_path_prefix
            rotated_img_path_prefix = IbugConf.rotated_img_path_prefix
            augmentation_factor_rotate = IbugConf.augmentation_factor_rotate
            num_of_landmarks = IbugConf.num_of_landmarks

        elif dataset_name == DatasetName.cofw:
            number_of_samples = CofwConf.orig_number_of_training
            img_path_prefix = CofwConf.img_path_prefix
            rotated_img_path_prefix = CofwConf.rotated_img_path_prefix
            augmentation_factor_rotate = CofwConf.augmentation_factor_rotate
            num_of_landmarks = CofwConf.num_of_landmarks

        elif dataset_name == DatasetName.wflw:
            number_of_samples = WflwConf.orig_number_of_training
            img_path_prefix = WflwConf.img_path_prefix
            rotated_img_path_prefix = WflwConf.rotated_img_path_prefix
            augmentation_factor_rotate = WflwConf.augmentation_factor_rotate
            num_of_landmarks = WflwConf.num_of_landmarks

        pts_file_arr = []

        for file in os.listdir(img_path_prefix):
            if file.endswith(".pts"):
                pts_file_arr.append(os.path.join(img_path_prefix, file))

        image_utility = ImageUtility()

        for i in tqdm(range(number_of_samples)):
            pts_file = pts_file_arr[i]

            if '#' in pts_file_arr[i]:
                img_file = pts_file_arr[i].split('#')[0] + ".jpg"
            else:
                img_file = pts_file_arr[i][:-3] + "jpg"
                if not os.path.exists(img_file):
                    img_file = pts_file_arr[i][:-3] + "png"
                    if not os.path.exists(img_file):
                        continue

            points_arr = []
            with open(pts_file) as fp:
                line = fp.readline()
                cnt = 1
                while line:
                    if 3 < cnt <= num_of_landmarks+3:
                        x_y_pnt = line.strip()
                        x = float(x_y_pnt.split(" ")[0])
                        y = float(x_y_pnt.split(" ")[1])
                        points_arr.append(x)
                        points_arr.append(y)
                    line = fp.readline()
                    cnt += 1

            img = Image.open(img_file)
            img = np.array(img)

            resized_img = img
            landmark_arr_xy = points_arr

            for j in range(augmentation_factor_rotate):
                image_utility.random_rotate(resized_img, landmark_arr_xy,
                                            rotated_img_path_prefix + str(10000 * (i + 1) + j),
                                            num_of_landmarks, dataset_name)
        print("rotaate_and_save")


    def random_augment_from_rotated(self, dataset_name):
        '''we use this function to augment images after rotation'''
        image_utility = ImageUtility()

        number_of_samples = 0
        rotated_img_path_prefix = ''
        num_of_landmarks = 0
        augmentation_factor = 0
        train_images_dir = ''

        if dataset_name == DatasetName.ibug:
            number_of_samples = IbugConf.orig_number_of_training
            rotated_img_path_prefix = IbugConf.rotated_img_path_prefix
            num_of_landmarks = IbugConf.num_of_landmarks
            augmentation_factor = IbugConf.augmentation_factor
            train_images_dir = IbugConf.train_images_dir

        elif dataset_name == DatasetName.cofw:
            number_of_samples = CofwConf.orig_number_of_training
            rotated_img_path_prefix = CofwConf.rotated_img_path_prefix
            num_of_landmarks = CofwConf.num_of_landmarks
            augmentation_factor = CofwConf.augmentation_factor
            train_images_dir = CofwConf.train_images_dir

        elif dataset_name == DatasetName.wflw:
            number_of_samples = WflwConf.orig_number_of_training
            rotated_img_path_prefix = WflwConf.rotated_img_path_prefix
            num_of_landmarks = WflwConf.num_of_landmarks
            augmentation_factor = WflwConf.augmentation_factor
            train_images_dir = WflwConf.train_images_dir

        elif dataset_name == DatasetName.ibug_test:
            number_of_samples = IbugConf.orig_number_of_test_challenging
            rotated_img_path_prefix = IbugConf.test_img_path_prefix
            num_of_landmarks = IbugConf.num_of_landmarks
            augmentation_factor = IbugConf.augmentation_factor
            train_images_dir = IbugConf.test_images_dir

        png_file_arr = []
        for file in os.listdir(rotated_img_path_prefix):
            if file.endswith(".jpg") or file.endswith(".png"):
                png_file_arr.append(os.path.join(rotated_img_path_prefix, file))

        for i in tqdm(range(len(png_file_arr))):
            img_file = png_file_arr[i]
            pts_file = png_file_arr[i][:-3] + "pts"

            points_arr = []
            with open(pts_file) as fp:
                line = fp.readline()
                cnt = 1
                while line:
                    if 3 < cnt <= num_of_landmarks + 3:
                        x_y_pnt = line.strip()
                        x = float(x_y_pnt.split(" ")[0])
                        y = float(x_y_pnt.split(" ")[1])
                        points_arr.append(x)
                        points_arr.append(y)
                    line = fp.readline()
                    cnt += 1

            img = Image.open(img_file)

            '''normalize image'''
            resized_img = np.array(img) / 255.0

            '''crop data: we add a small margin to the images'''
            landmark_arr_xy, landmark_arr_x, landmark_arr_y = image_utility.create_landmarks(landmarks=points_arr,
                                                                                             scale_factor_x=1,
                                                                                             scale_factor_y=1)
            '''augment the images, then normalize the landmarks based on the hyperface method'''
            for k in range(augmentation_factor):
                '''save the origin image as well'''
                if k == 0:
                    landmark_arr_flat_aug = landmark_arr_xy
                    img_aug = resized_img

                else:
                    '''save the augmented images'''
                    if k % 2 == 0:
                        landmark_arr_flat_aug, img_aug = image_utility.random_augmentation(landmark_arr_xy,
                                                                                           resized_img, num_of_landmarks)
                    else:
                        landmark_arr_flat_aug, img_aug = image_utility.augment(resized_img,
                                                                               landmark_arr_xy, num_of_landmarks)

                '''test '''
                # imgpr.print_image_arr(k, img_aug, [], [])

                '''again resize image to 224*224 after augmentation'''
                resized_img_new = resize(img_aug,
                                         (InputDataSize.image_input_size, InputDataSize.image_input_size, 3)
                                         , anti_aliasing=True)

                # imgpr.print_image_arr(k, resized_img_new, [], [])

                dims = img_aug.shape
                height = dims[0]
                width = dims[1]
                scale_factor_y = InputDataSize.image_input_size / height
                scale_factor_x = InputDataSize.image_input_size / width

                '''retrieve and rescale landmarks in after augmentation'''
                landmark_arr_flat, landmark_arr_x, landmark_arr_y = \
                    image_utility.create_landmarks(landmarks=landmark_arr_flat_aug,
                                                   scale_factor_x=scale_factor_x,
                                                   scale_factor_y=scale_factor_y)

                # imgpr.print_image_arr(k, resized_img_new, landmark_arr_x, landmark_arr_y)

                '''normalize landmarks based on hyperface method'''
                width = len(resized_img_new[0])
                height = len(resized_img_new[1])
                x_center = width / 2
                y_center = height / 2
                landmark_arr_flat_normalized = []
                for p in range(0, len(landmark_arr_flat), 2):
                    landmark_arr_flat_normalized.append((x_center - landmark_arr_flat[p]) / width)
                    landmark_arr_flat_normalized.append((y_center - landmark_arr_flat[p + 1]) / height)

                '''test print after augmentation'''
                # landmark_arr_flat_n, landmark_arr_x_n, landmark_arr_y_n = image_utility.\
                #     create_landmarks_from_normalized(landmark_arr_flat_normalized, 224, 224, 112, 112)
                # imgpr.print_image_arr((i*100)+(k+1), resized_img_new, landmark_arr_x_n, landmark_arr_y_n)

                # heatmap_landmark = self.generate_hm(56, 56, landmark_arr_flat_normalized, s=1.0)

                # imgpr.print_image_arr_heat((i + 1) * (k + 1), heatmap_landmark)
                # imgpr.print_image_arr((i * 100) + (k + 1), heatmap_landmark_all, [], [])

                '''save image'''
                im = Image.fromarray((resized_img_new * 255).astype(np.uint8))
                file_name = train_images_dir + str(10000 * (i + 1) + k)
                im.save(str(file_name) + '.jpg')

                pnt_file = open(str(file_name) + ".pts", "w")
                pre_txt = ["version: 1 \n", "n_points: 68 \n", "{ \n"]
                pnt_file.writelines(pre_txt)
                points_txt = ""
                for l in range(0, len(landmark_arr_xy), 2):
                    points_txt += str(landmark_arr_xy[l]) + " " + str(landmark_arr_xy[l + 1]) + "\n"

                pnt_file.writelines(points_txt)
                pnt_file.write("} \n")
                pnt_file.close()

        print("random_augment_from_rotated DONE.")
        return number_of_samples

    def _get_asm(self, input, dataset_name, accuracy):
        pca_utils = PCAUtility()

        eigenvalues = load('pca_obj/' + dataset_name + pca_utils.eigenvalues_prefix + str(accuracy) + ".npy")
        eigenvectors = load('pca_obj/' + dataset_name + pca_utils.eigenvectors_prefix + str(accuracy) + ".npy")
        meanvector = load('pca_obj/' + dataset_name + pca_utils.meanvector_prefix + str(accuracy) + ".npy")

        b_vector_p = pca_utils.calculate_b_vector(input, True, eigenvalues, eigenvectors, meanvector)
        out = meanvector + np.dot(eigenvectors, b_vector_p)
        return out

    def create_point_imgpath_map_tf_record(self, dataset_name):
        print('create_point_imgpath_map_tf_record')
        map = {}
        if dataset_name == DatasetName.ibug:
            tf_path = IbugConf.tf_train_path
            sample_counts = IbugConf.number_of_train_sample
            landmarks_dir = IbugConf.normalized_points_npy_dir

        elif dataset_name == DatasetName.cofw:
            tf_path = CofwConf.tf_train_path
            sample_counts = CofwConf.number_of_train_sample
            landmarks_dir = CofwConf.normalized_points_npy_dir

        elif dataset_name == DatasetName.wflw:
            tf_path = WflwConf.tf_train_path
            sample_counts = WflwConf.number_of_train_sample
            landmarks_dir = WflwConf.normalized_points_npy_dir

        # sample_counts = 1708
        lbl_arr, img_arr, pose_arr, img_name_arr = self.retrieve_tf_record_train(tf_path,
                                                                                 number_of_records=sample_counts,
                                                                                 only_label=True)
        counter = 0
        # f = open("key_"+dataset_name, "a")
        for lbl in tqdm(lbl_arr):
            img_name = self._decode_tf_file_name(img_name_arr[counter].decode("utf-8"))
            landmark_key = lbl.tostring()
            # img_name = os.path.join(landmarks_dir, img_name)
            map[landmark_key] = img_name

            # f.write(str(landmark_key))
            counter += 1
        # f.close()

        pkl_file = open("map_" + dataset_name, 'wb')
        pickle.dump(map, pkl_file)
        pkl_file.close()

        file = open("map_" + dataset_name, 'rb')
        load_map = pickle.load(file)
        print(load_map)
        file.close()

    def create_point_imgpath_map(self, dataset_name):
        """
        create a map between facialLandmarks and image_name
        :param dataset_name:
        :return:
        """
        map = {}

        if dataset_name == DatasetName.ibug:
            img_dir = IbugConf.train_images_dir
            landmarks_dir = IbugConf.normalized_points_npy_dir

        elif dataset_name == DatasetName.cofw:
            img_dir = CofwConf.train_images_dir
            landmarks_dir = CofwConf.normalized_points_npy_dir

        elif dataset_name == DatasetName.wflw:
            img_dir = WflwConf.train_images_dir
            landmarks_dir = WflwConf.normalized_points_npy_dir

        file1 = open(dataset_name+".txt", "a")

        for img_file_name in tqdm(os.listdir(img_dir)):  #
            if img_file_name.endswith(".jpg") or img_file_name.endswith(".png"):

                '''load landmark npy, (has been augmented already)'''
                landmark_file_name = os.path.join(landmarks_dir, img_file_name[:-3] + "npy")
                landmark = str(load(landmark_file_name))
                # print(landmark)
                landmark_key = hash(landmark)
                # landmark_key = self.get_hash_key(landmark)
                map[landmark_key] = img_file_name
                # file1.write(landmark_key)

        # file1.close()

        # print(map)

        # np.save("map_" + dataset_name, map)
        # load_map = np.load("map_" + dataset_name+".npy", allow_pickle=True)

        pkl_file = open("map_" + dataset_name, 'wb')
        pickle.dump(map, pkl_file)
        pkl_file.close()

        file = open("map_" + dataset_name, 'rb')
        load_map = pickle.load(file)
        file.close()

        # print(load_map.get(landmark_key))
        # print(load_map)

    def get_hash_key(self, input):
        return hash(str(input).replace("\n", "").replace(" ", ""))

    def _decode_tf_file_name(self, file_name):
        return str(file_name).replace("X", "")

    def _encode_tf_file_name(self, file_name):
        while len(file_name) < 15:
            file_name = "X" + file_name
        return file_name

    def _create_tfrecord_from_npy_hm(self, dataset_name, dataset_type, isTest, accuracy=100):
        img_utils =ImageUtility()

        if dataset_name == DatasetName.ibug:
            img_dir = IbugConf.train_images_dir
            landmarks_dir = IbugConf.normalized_points_npy_dir
            pose_dir = IbugConf.pose_npy_dir
            num_train_samples = IbugConf.number_of_train_sample
            if accuracy == 100:
                tf_train_path = IbugConf.tf_train_path
                tf_evaluation_path = IbugConf.tf_evaluation_path
            elif accuracy == 90:
                tf_train_path = IbugConf.tf_train_path_95
                tf_evaluation_path = IbugConf.tf_evaluation_path_95

        elif dataset_name == DatasetName.cofw:
            img_dir = CofwConf.train_images_dir
            landmarks_dir = CofwConf.normalized_points_npy_dir
            pose_dir = CofwConf.pose_npy_dir
            num_train_samples = CofwConf.number_of_train_sample  # 95%
            if accuracy == 100:
                tf_train_path = CofwConf.tf_train_path
                tf_evaluation_path = CofwConf.tf_evaluation_path
            elif accuracy == 90:
                tf_train_path = CofwConf.tf_train_path_95
                tf_evaluation_path = CofwConf.tf_evaluation_path_95

        elif dataset_name == DatasetName.wflw:
            img_dir = WflwConf.train_images_dir
            landmarks_dir = WflwConf.normalized_points_npy_dir
            pose_dir = WflwConf.pose_npy_dir
            num_train_samples = WflwConf.number_of_train_sample  # 95%
            if accuracy == 100:
                tf_train_path = WflwConf.tf_train_path
                tf_evaluation_path = WflwConf.tf_evaluation_path
            elif accuracy == 90:
                tf_train_path = WflwConf.tf_train_path_95
                tf_evaluation_path = WflwConf.tf_evaluation_path_95

        elif dataset_name == DatasetName.wflw_test:
            img_dir = WflwConf.test_images_dir
            landmarks_dir = WflwConf.test_normalized_points_npy_dir
            pose_dir = WflwConf.test_pose_npy_dir
            if dataset_type == DatasetType.wflw_full:
                num_train_samples = WflwConf.orig_number_of_test
                tf_train_path = WflwConf.tf_test_path
            elif dataset_type == DatasetType.wflw_blur:
                num_train_samples = WflwConf.orig_of_all_test_blur
            elif dataset_type == DatasetType.wflw_largepose:
                num_train_samples = WflwConf.orig_of_all_test_largepose
                tf_train_path = WflwConf.tf_test_path_largepose
            elif dataset_type == DatasetType.wflw_occlusion:
                num_train_samples = WflwConf.orig_of_all_test_occlusion
                tf_train_path = WflwConf.tf_test_path_occlusion
            elif dataset_type == DatasetType.wflw_makeup:
                num_train_samples = WflwConf.orig_of_all_test_makeup
                tf_train_path = WflwConf.tf_test_path_makeup
            elif dataset_type == DatasetType.wflw_expression:
                num_train_samples = WflwConf.orig_of_all_test_expression
                tf_train_path = WflwConf.tf_test_path_expression
            elif dataset_type == DatasetType.wflw_illumination:
                num_train_samples = WflwConf.orig_of_all_test_illumination
                tf_train_path = WflwConf.tf_test_path_illumination
            tf_evaluation_path = None

        elif dataset_name == DatasetName.cofw_test:
            img_dir = CofwConf.test_images_dir
            landmarks_dir = CofwConf.test_normalized_points_npy_dir
            pose_dir = CofwConf.test_pose_npy_dir
            num_train_samples = CofwConf.orig_number_of_test
            tf_train_path = CofwConf.tf_test_path
            tf_evaluation_path = None

        elif dataset_name == DatasetName.ibug_test:
            img_dir = IbugConf.test_images_dir
            landmarks_dir = IbugConf.test_normalized_points_npy_dir
            pose_dir = IbugConf.test_pose_npy_dir
            if dataset_type == DatasetType.ibug_challenging:
                num_train_samples = IbugConf.orig_number_of_test_challenging
                tf_train_path = IbugConf.tf_test_path_challenging
            elif dataset_type == DatasetType.ibug_full:
                num_train_samples = IbugConf.orig_number_of_test_full
                tf_train_path = IbugConf.tf_test_path_full
            elif dataset_type == DatasetType.ibug_comomn:
                num_train_samples = IbugConf.orig_number_of_test_common
                tf_train_path = IbugConf.tf_test_path_common

            tf_evaluation_path = None

        counter = 0

        writer_train = tf.python_io.TFRecordWriter(tf_train_path)
        if tf_evaluation_path is not None:
            writer_evaluate = tf.python_io.TFRecordWriter(tf_evaluation_path)

        for file in os.listdir(img_dir):
            if file.endswith(".jpg") or file.endswith(".png"):
                img_tf_name = self._encode_tf_file_name(file)
                img_file_name = os.path.join(img_dir, file)

                '''load img and normalize it'''
                img = Image.open(img_file_name)
                img = np.array(img) / 255.0

                '''load landmark npy, (has been augmented already)'''
                #
                landmark_file_name = os.path.join(landmarks_dir, file[:-3] + "npy")
                if not os.path.exists(landmark_file_name):
                    continue
                landmark = load(landmark_file_name)

                '''load pose npy'''
                pose_file_name = os.path.join(pose_dir, file[:-3] + "npy")
                if not os.path.exists(pose_file_name):
                    pose = [0, 0, 0]
                else:
                    pose = load(pose_file_name)
                '''create new landmark using accuracy'''
                if accuracy != 100:
                    landmark = self._get_asm(landmark, dataset_name, accuracy)

                '''test image '''
                landmark_arr_xy, landmark_arr_x, landmark_arr_y = img_utils.create_landmarks_from_normalized(landmark, 224, 224, 112, 112)
                imgpr.print_image_arr(counter + 1, img, landmark_arr_x, landmark_arr_y)

                '''prepare img'''
                writable_img = np.reshape(img, [InputDataSize.image_input_size * InputDataSize.image_input_size * 3])

                '''create hm'''
                heatmap_landmark = self.generate_hm(56, 56, landmark, s=2.0)
                writable_heatmap = np.reshape(heatmap_landmark, [56 * 56 * self.number_of_landmark//2])

                '''create tf_record:'''
                if isTest: # no need for hm in test
                    feature = {'landmarks': self.__float_feature(landmark),
                               'pose': self.__float_feature(pose),
                               'image_raw': self.__float_feature(writable_img),
                               }
                else:
                    feature = {'landmarks': self.__float_feature(landmark),
                               'pose': self.__float_feature(pose),
                               'image_raw': self.__float_feature(writable_img),
                               'heatmap': self.__float_feature(writable_heatmap),
                               'image_name': self.__bytes_feature(img_tf_name.encode('utf-8'))
                               }

                example = tf.train.Example(features=tf.train.Features(feature=feature))

                if counter <= num_train_samples:
                    writer_train.write(example.SerializeToString())
                    msg = 'train --> \033[92m' + " sample number " + str(counter + 1) + \
                          " created." + '\033[94m' + "remains " + str(num_train_samples - counter - 1)
                    sys.stdout.write('\r' + msg)

                elif tf_evaluation_path is not None:
                    writer_evaluate.write(example.SerializeToString())
                    msg = 'eval --> \033[92m' + " sample number " + str(counter + 1) + \
                          " created." + '\033[94m' + "remains " + str(num_train_samples - counter - 1)
                    sys.stdout.write('\r' + msg)
                counter += 1

        writer_train.close()
        if tf_evaluation_path is not None:
            writer_evaluate.close()

    def _create_face_graph(self, dataset_name, dataset_type):
        if dataset_name == DatasetName.ibug:
            img_dir = IbugConf.train_images_dir
            landmarks_dir = IbugConf.normalized_points_npy_dir
            num_train_samples = IbugConf.number_of_train_sample

        if dataset_name == DatasetName.wflw:
            img_dir = WflwConf.train_images_dir
            landmarks_dir = WflwConf.normalized_points_npy_dir
            num_train_samples = WflwConf.number_of_train_sample

        if dataset_name == DatasetName.cofw:
            img_dir = CofwConf.train_images_dir
            landmarks_dir = CofwConf.normalized_points_npy_dir
            num_train_samples = CofwConf.number_of_train_sample

        counter= 0
        for file in os.listdir(img_dir):
            if file.endswith(".jpg") or file.endswith(".png"):
                img_file_name = os.path.join(img_dir, file)

                '''load img and normalize it'''
                img = Image.open(img_file_name)

                '''load landmark npy, (has been augmented already)'''
                landmark_file_name = os.path.join(landmarks_dir, file[:-3] + "npy")
                if not os.path.exists(landmark_file_name):
                    continue
                landmark = load(landmark_file_name)

                if dataset_name == DatasetName.ibug:
                    self._create_ibug_graph(counter, img, landmark)

                if dataset_name == DatasetName.cofw:
                    self._create_cofw_graph(counter, img, landmark)

                if dataset_name == DatasetName.wflw:
                    self._create_wflw_graph(counter, img, landmark)

                counter += 1

    def _create_wflw_graph(self, counter, img, landmark):
        face = np.array(landmark[0:66])
        l_ebrow = np.append(landmark[66:84], landmark[66:68])
        r_ebrow = np.append(landmark[84:102], landmark[84:86])
        l_eye = np.append(landmark[120:136], landmark[120:122])
        r_eye = np.append(landmark[136:152], landmark[136:138])
        nose = np.append(landmark[102:120], landmark[108:110])
        u_mouth = np.append(landmark[152:166], np.array([landmark[184:186], landmark[182:184], landmark[180:182],
                                                         landmark[178:180], landmark[176:178], landmark[152:154]]))

        l_mouth = np.append(landmark[164:176], np.array([landmark[152:154], landmark[176:178], landmark[190:192],
                                                         landmark[188:190], landmark[186:188], landmark[184:186]]))

        landmark_arr = []

        landmark_arr.append(face)
        landmark_arr.append(l_ebrow)
        landmark_arr.append(r_ebrow)
        landmark_arr.append(l_eye)
        landmark_arr.append(r_eye)
        landmark_arr.append(nose)
        landmark_arr.append(u_mouth)
        landmark_arr.append(l_mouth)

        # imgpr.print_partial(counter, img, landmark_arr)

        hm_arr = self.generate_partial_hm(InputDataSize.image_input_size//4, InputDataSize.image_input_size//4,
                                          landmark_arr, s=1.2)
        # for i in range(len(hm_arr)):
        #     imgpr.print_image_arr_heat('zHM_' + str(counter)+'_'+str(i), hm_arr[i])
        imgpr.print_partial_heat('z_HM' + str(counter)+'_', np.array(hm_arr), True)

    def _create_cofw_graph(self, counter, img, landmark):

        l_ebrow = np.array([landmark[0:2], landmark[8:10], landmark[4:6], landmark[10:12], landmark[0:2]]).reshape([10])
        r_ebrow = np.array([landmark[2:4], landmark[12:14], landmark[6:8], landmark[14:16], landmark[2:4]]).reshape([10])

        r_eye = np.array([landmark[16:18], landmark[24:26], landmark[20:22], landmark[26:28], landmark[16:18]]).reshape([10])
        l_eye = np.array([landmark[18:20], landmark[28:30], landmark[22:24], landmark[30:32], landmark[18:20]]).reshape([10])
        nose = np.array([landmark[36:38], landmark[40:42], landmark[38:40], landmark[36:38]]).reshape([8])
        mouth = np.array([landmark[44:46], landmark[48:50], landmark[46:48],
                          landmark[54:56], landmark[44:46], landmark[52:54], landmark[46:48]]).reshape([14])

        landmark_arr = []

        landmark_arr.append(l_ebrow)
        landmark_arr.append(r_ebrow)
        landmark_arr.append(l_eye)
        landmark_arr.append(r_eye)
        landmark_arr.append(nose)
        landmark_arr.append(mouth)

        # imgpr.print_partial(counter, img, landmark_arr)
        hm_arr = self.generate_partial_hm(InputDataSize.image_input_size // 4, InputDataSize.image_input_size // 4,
                                          landmark_arr, s=2.5)
        # for i in range(len(hm_arr)):
        #     imgpr.print_image_arr_heat('zHM_' + str(counter)+'_'+str(i), hm_arr[i])
        imgpr.print_partial_heat('z_HM' + str(counter) + '_', np.array(hm_arr), True)
        imgpr.print_image_arr('z_img' + str(counter) + '_', img, [],[])

    def _create_ibug_graph(self, counter, img, landmark):

        face = landmark[0:34] # line
        l_ebrow = landmark[34:44] # line
        r_ebrow = landmark[44:54] # line

        nose_bridge = landmark[54:62] # line
        nose = np.append(landmark[60:72], landmark[60:62])
        nose = np.append(nose, np.array([landmark[54:56], landmark[62:64]]))
        nose = np.append(nose, np.array([landmark[54:56], landmark[70:72]]))

        l_eye = np.append(landmark[72:84], landmark[72:74])
        r_eye = np.append(landmark[84:96], landmark[84:86])
        u_lip = np.append(landmark[96:110], np.array([landmark[128:130], landmark[126:128],
                                                      landmark[124:126], landmark[122:124],landmark[120:122]]))
        l_lip = np.append(landmark[110:120], np.array([landmark[134:136],
                                                       landmark[132:134],landmark[130:132], landmark[110:112]]))

        landmark_arr = []
        landmark_arr.append(face)
        landmark_arr.append(l_ebrow)
        landmark_arr.append(r_ebrow)
        landmark_arr.append(nose_bridge)
        landmark_arr.append(nose)
        landmark_arr.append(l_eye)
        landmark_arr.append(r_eye)
        landmark_arr.append(u_lip)
        landmark_arr.append(l_lip)

        # imgpr.print_partial(counter, img, landmark_arr)
        hm_arr = self.generate_partial_hm(InputDataSize.image_input_size // 4, InputDataSize.image_input_size // 4,
                                          landmark_arr, s=1.8)
        # for i in range(len(hm_arr)):
        #     imgpr.print_image_arr_heat('zHM_' + str(counter)+'_'+str(i), hm_arr[i])
        imgpr.print_partial_heat('z_HM' + str(counter) + '_', np.array(hm_arr), True)
        # imgpr.print_image_arr('z_img' + str(counter) + '_', img, [], [])

    def _create_tfrecord_from_npy(self, dataset_name, dataset_type, isTest, accuracy=100):
        """we use this function when we have already created and nrmalzed both landmarks and poses"""

        if dataset_name == DatasetName.ibug:
            img_dir = IbugConf.train_images_dir
            landmarks_dir = IbugConf.normalized_points_npy_dir
            pose_dir = IbugConf.pose_npy_dir
            num_train_samples = IbugConf.number_of_train_sample
            if accuracy == 100:
                tf_train_path = IbugConf.tf_train_path
                tf_evaluation_path = IbugConf.tf_evaluation_path
            elif accuracy == 95:
                tf_train_path = IbugConf.tf_train_path_95
                tf_evaluation_path = IbugConf.tf_evaluation_path_95

        elif dataset_name == DatasetName.cofw:
            img_dir = CofwConf.train_images_dir
            landmarks_dir = CofwConf.normalized_points_npy_dir
            pose_dir = CofwConf.pose_npy_dir
            num_train_samples = CofwConf.number_of_train_sample  # 95%
            if accuracy == 100:
                tf_train_path = CofwConf.tf_train_path
                tf_evaluation_path = CofwConf.tf_evaluation_path
            elif accuracy == 95:
                tf_train_path = CofwConf.tf_train_path_95
                tf_evaluation_path = CofwConf.tf_evaluation_path_95

        elif dataset_name == DatasetName.wflw:
            img_dir = WflwConf.train_images_dir
            landmarks_dir = WflwConf.normalized_points_npy_dir
            pose_dir = WflwConf.pose_npy_dir
            num_train_samples = WflwConf.number_of_train_sample  # 95%
            if accuracy == 100:
                tf_train_path = WflwConf.tf_train_path
                tf_evaluation_path = WflwConf.tf_evaluation_path
            elif accuracy == 95:
                tf_train_path = WflwConf.tf_train_path_95
                tf_evaluation_path = WflwConf.tf_evaluation_path_95

        elif dataset_name == DatasetName.wflw_test:
            img_dir = WflwConf.test_images_dir
            landmarks_dir = WflwConf.test_normalized_points_npy_dir
            pose_dir = WflwConf.test_pose_npy_dir
            if dataset_type == DatasetType.wflw_full:
                num_train_samples = WflwConf.orig_number_of_test
                tf_train_path = WflwConf.tf_test_path
            elif dataset_type == DatasetType.wflw_blur:
                num_train_samples = WflwConf.orig_of_all_test_blur
            elif dataset_type == DatasetType.wflw_largepose:
                num_train_samples = WflwConf.orig_of_all_test_largepose
                tf_train_path = WflwConf.tf_test_path_largepose
            elif dataset_type == DatasetType.wflw_occlusion:
                num_train_samples = WflwConf.orig_of_all_test_occlusion
                tf_train_path = WflwConf.tf_test_path_occlusion
            elif dataset_type == DatasetType.wflw_makeup:
                num_train_samples = WflwConf.orig_of_all_test_makeup
                tf_train_path = WflwConf.tf_test_path_makeup
            elif dataset_type == DatasetType.wflw_expression:
                num_train_samples = WflwConf.orig_of_all_test_expression
                tf_train_path = WflwConf.tf_test_path_expression
            elif dataset_type == DatasetType.wflw_illumination:
                num_train_samples = WflwConf.orig_of_all_test_illumination
                tf_train_path = WflwConf.tf_test_path_illumination
            tf_evaluation_path = None

        elif dataset_name == DatasetName.cofw_test:
            img_dir = CofwConf.test_images_dir
            landmarks_dir = CofwConf.test_normalized_points_npy_dir
            pose_dir = CofwConf.test_pose_npy_dir
            num_train_samples = CofwConf.orig_number_of_test
            tf_train_path = CofwConf.tf_test_path
            tf_evaluation_path = None

        elif dataset_name == DatasetName.ibug_test:
            img_dir = IbugConf.test_images_dir
            landmarks_dir = IbugConf.test_normalized_points_npy_dir
            pose_dir = IbugConf.test_pose_npy_dir
            if dataset_type == DatasetType.ibug_challenging:
                num_train_samples = IbugConf.orig_number_of_test_challenging
                tf_train_path = IbugConf.tf_test_path_challenging
            elif dataset_type == DatasetType.ibug_full:
                num_train_samples = IbugConf.orig_number_of_test_full
                tf_train_path = IbugConf.tf_test_path_full
            elif dataset_type == DatasetType.ibug_comomn:
                num_train_samples = IbugConf.orig_number_of_test_common
                tf_train_path = IbugConf.tf_test_path_common


            tf_evaluation_path = None

        counter = 0

        writer_train = tf.python_io.TFRecordWriter(tf_train_path)
        if tf_evaluation_path is not None:
            writer_evaluate = tf.python_io.TFRecordWriter(tf_evaluation_path)

        for file in os.listdir(img_dir):
            if file.endswith(".jpg") or file.endswith(".png"):
                img_tf_name = self._encode_tf_file_name(file)
                # img_tf_name_ = self._decode_tf_file_name(img_tf_name)

                img_file_name = os.path.join(img_dir, file)

                '''load img and normalize it'''
                img = Image.open(img_file_name)
                img = np.array(img) / 255.0

                '''load landmark npy, (has been augmented already)'''
                landmark_file_name = os.path.join(landmarks_dir, file[:-3] + "npy")
                if not os.path.exists(landmark_file_name):
                    continue
                landmark = load(landmark_file_name)

                '''load pose npy'''
                pose_file_name = os.path.join(pose_dir, file[:-3] + "npy")
                if not os.path.exists(pose_file_name):
                    continue
                pose = load(pose_file_name)
                '''create new landmark using accuracy'''
                if accuracy != 100:
                    landmark = self._get_asm(landmark, dataset_name, accuracy)
                    '''test image after ASM: '''

                '''create tf_record:'''
                writable_img = np.reshape(img,
                                          [InputDataSize.image_input_size * InputDataSize.image_input_size * 3])

                if isTest:
                    feature = {'landmarks': self.__float_feature(landmark),
                               'pose': self.__float_feature(pose),
                               'image_raw': self.__float_feature(writable_img)
                               }
                else:
                    feature = {'landmarks': self.__float_feature(landmark),
                               'pose': self.__float_feature(pose),
                               'image_raw': self.__float_feature(writable_img),
                               'image_name': self.__bytes_feature(img_tf_name.encode('utf-8')),
                               }

                example = tf.train.Example(features=tf.train.Features(feature=feature))

                if counter <= num_train_samples:
                    writer_train.write(example.SerializeToString())
                    msg = 'train --> \033[92m' + " sample number " + str(counter + 1) + \
                          " created." + '\033[94m' + "remains " + str(num_train_samples - counter - 1)
                    sys.stdout.write('\r' + msg)

                elif tf_evaluation_path is not None:
                    writer_evaluate.write(example.SerializeToString())
                    msg = 'eval --> \033[92m' + " sample number " + str(counter + 1) + \
                          " created." + '\033[94m' + "remains " + str(num_train_samples - counter - 1)
                    sys.stdout.write('\r' + msg)
                counter += 1

        writer_train.close()
        if tf_evaluation_path is not None:
            writer_evaluate.close()

    def __create_tfrecord_ibug_all_main(self):
        # try:
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        pst_file_arr = []
        png_file_arr = []
        for file in os.listdir(IbugConf.train_images_dir):  # both pts file and images are in one directory
            if file.endswith(".jpg") or file.endswith(".png"):
                png_file_arr.append(os.path.join(IbugConf.train_images_dir, file))

        writer_train = tf.python_io.TFRecordWriter(IbugConf.tf_train_path)
        writer_evaluate = tf.python_io.TFRecordWriter(IbugConf.tf_evaluation_path)

        number_of_samples = 100  # 100% after augmentation
        number_of_train = 95  # 95 % after augmentation
        number_of_evaluation = 5  # 5 % after augmentation

        image_utility = ImageUtility()

        for i in range(number_of_samples):
            img_file = png_file_arr[i]
            pts_file = png_file_arr[i][:-3] + "pts"

            points_arr = []
            with open(pts_file) as fp:
                line = fp.readline()
                cnt = 1
                while line:
                    if 3 < cnt < 72:
                        x_y_pnt = line.strip()
                        x = float(x_y_pnt.split(" ")[0])
                        y = float(x_y_pnt.split(" ")[1])
                        points_arr.append(x)
                        points_arr.append(y)
                    line = fp.readline()
                    cnt += 1

            img = Image.open(img_file)

            '''normalize image'''
            img_arr = np.array(img) / 255.0

            '''crop data: we add a small margin to the images'''
            xy_points, x_points, y_points = image_utility.create_landmarks(landmarks=points_arr,
                                                                           scale_factor_x=1, scale_factor_y=1)
            img_arr, points_arr = image_utility.cropImg(img_arr, x_points, y_points)

            '''resize image to 224*224'''
            resized_img = resize(img_arr,
                                 (InputDataSize.image_input_size, InputDataSize.image_input_size, 3),
                                 anti_aliasing=True)
            dims = img_arr.shape
            height = dims[0]
            width = dims[1]
            scale_factor_y = InputDataSize.image_input_size / height
            scale_factor_x = InputDataSize.image_input_size / width

            '''rescale and retrieve landmarks'''
            landmark_arr_xy, landmark_arr_x, landmark_arr_y = \
                image_utility.create_landmarks(landmarks=points_arr,
                                               scale_factor_x=scale_factor_x,
                                               scale_factor_y=scale_factor_y)

            '''augment the images, then normalize the landmarks based on the hyperface method'''
            for k in range(IbugConf.augmentation_factor):
                '''save the origin image as well'''
                if k == 0:
                    landmark_arr_flat_aug = landmark_arr_xy
                    img_aug = resized_img

                else:
                    '''save the augmented images'''
                    if k % 2 == 0:
                        landmark_arr_flat_aug, img_aug = image_utility.augment(resized_img, landmark_arr_xy)
                    else:
                        landmark_arr_flat_aug, img_aug = image_utility.random_augmentation(landmark_arr_xy, resized_img)

                '''test '''
                # imgpr.print_image_arr(k, img_aug, [], [])

                '''again resize image to 224*224 after augmentation'''
                resized_img_new = resize(img_aug,
                                         (InputDataSize.image_input_size, InputDataSize.image_input_size, 3)
                                         , anti_aliasing=True)

                # imgpr.print_image_arr(k, resized_img_new, [], [])

                dims = img_aug.shape
                height = dims[0]
                width = dims[1]
                scale_factor_y = InputDataSize.image_input_size / height
                scale_factor_x = InputDataSize.image_input_size / width

                '''retrieve and rescale landmarks in after augmentation'''
                landmark_arr_flat, landmark_arr_x, landmark_arr_y = \
                    image_utility.create_landmarks(landmarks=landmark_arr_flat_aug,
                                                   scale_factor_x=scale_factor_x,
                                                   scale_factor_y=scale_factor_y)
                '''calculate pose'''
                # detect = PoseDetector()
                resized_img_new_cp = np.array(resized_img_new)
                yaw_predicted, pitch_predicted, roll_predicted = detect.detect(resized_img_new_cp, isFile=False,
                                                                               show=False)
                '''normalize pose -1 -> +1 '''
                min_degree = -65
                max_degree = 65
                yaw_normalized = 2 * ((yaw_predicted - min_degree) / (max_degree - min_degree)) - 1
                pitch_normalized = 2 * ((pitch_predicted - min_degree) / (max_degree - min_degree)) - 1
                roll_normalized = 2 * ((roll_predicted - min_degree) / (max_degree - min_degree)) - 1

                pose_array = np.array([yaw_normalized, pitch_normalized, roll_normalized])

                '''normalize landmarks based on hyperface method'''
                width = len(resized_img_new[0])
                height = len(resized_img_new[1])
                x_center = width / 2
                y_center = height / 2
                landmark_arr_flat_normalized = []
                for p in range(0, len(landmark_arr_flat), 2):
                    landmark_arr_flat_normalized.append((x_center - landmark_arr_flat[p]) / width)
                    landmark_arr_flat_normalized.append((y_center - landmark_arr_flat[p + 1]) / height)

                '''creating landmarks for partial tasks'''
                landmark_face = landmark_arr_flat_normalized[0:54]  # landmark_face_len = 54
                landmark_nose = landmark_arr_flat_normalized[54:72]  # landmark_nose_len = 18
                landmark_eys = landmark_arr_flat_normalized[72:96]  # landmark_eys_len = 24
                landmark_mouth = landmark_arr_flat_normalized[96:136]  # landmark_mouth_len = 40

                '''test print after augmentation'''
                # landmark_arr_flat_n, landmark_arr_x_n, landmark_arr_y_n = image_utility.\
                #     create_landmarks_from_normalized(landmark_arr_flat_normalized, 224, 224, 112, 112)
                # imgpr.print_image_arr((i+1)*(k+1), resized_img_new, landmark_arr_x_n, landmark_arr_y_n)

                '''create tf_record'''
                writable_img = np.reshape(resized_img_new,
                                          [InputDataSize.image_input_size * InputDataSize.image_input_size * 3])
                landmark_arr_flat_normalized = np.array(landmark_arr_flat_normalized)
                feature = {'landmarks': self.__float_feature(landmark_arr_flat_normalized),
                           'face': self.__float_feature(landmark_face),
                           'eyes': self.__float_feature(landmark_eys),
                           'nose': self.__float_feature(landmark_nose),
                           'mouth': self.__float_feature(landmark_mouth),
                           'pose': self.__float_feature(pose_array),
                           'image_raw': self.__float_feature(writable_img)}
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                if i <= number_of_train:
                    writer_train.write(example.SerializeToString())
                    msg = 'train --> \033[92m' + " sample number " + str(i + 1) + \
                          " created." + '\033[94m' + "remains " + str(number_of_samples - i - 1)
                    sys.stdout.write('\r' + msg)

                else:
                    writer_evaluate.write(example.SerializeToString())
                    msg = 'eval --> \033[92m' + " sample number " + str(i + 1) + \
                          " created." + '\033[94m' + "remains " + str(number_of_samples - i - 1)
                    sys.stdout.write('\r' + msg)

        writer_train.close()
        writer_evaluate.close()

        return number_of_samples

    def __int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def __float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def __float_feature_single(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def __bytes_feature(self, value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def __read_and_decode_hm(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'landmarks': tf.FixedLenFeature([self.number_of_landmark],
                                                                               tf.float32),
                                               'pose': tf.FixedLenFeature([InputDataSize.pose_len], tf.float32),
                                               'image_raw': tf.FixedLenFeature([InputDataSize.image_input_size *
                                                                                InputDataSize.image_input_size * 3]
                                                                                , tf.float32),
                                               'heatmap': tf.FixedLenFeature([56 ,56 , self.number_of_landmark // 2],tf.float32),
                                               'image_name': tf.FixedLenFeature([], tf.string)
                                           })
        landmarks = features['landmarks']
        pose = features['pose']
        image_raw = features['image_raw']
        heatmap = features['heatmap']
        image_name = features['image_name']

        return image_raw, landmarks, pose,  heatmap, image_name

    def __read_and_decode(self, filename_queue):
        print('__read_and_decode')

        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'landmarks': tf.FixedLenFeature([self.number_of_landmark],
                                                                               tf.float32),
                                               'pose': tf.FixedLenFeature([InputDataSize.pose_len], tf.float32),
                                               'image_raw': tf.FixedLenFeature(
                                                   [InputDataSize.image_input_size *
                                                    InputDataSize.image_input_size * 3]
                                                   , tf.float32),
                                               "image_name": tf.FixedLenFeature([], tf.string)
                                           })
        landmarks = features['landmarks']
        pose = features['pose']
        image_raw = features['image_raw']
        image_name = features['image_name']
        return image_raw, landmarks, pose, image_name

    def __read_and_decode_test_set(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'landmarks': tf.FixedLenFeature([self.number_of_landmark],
                                                                               tf.float32),
                                               'pose': tf.FixedLenFeature([InputDataSize.pose_len], tf.float32),
                                               'image_raw': tf.FixedLenFeature(
                                                   [InputDataSize.image_input_size *
                                                    InputDataSize.image_input_size * 3]
                                                   , tf.float32)
                                           })
        landmarks = features['landmarks']
        image_raw = features['image_raw']
        pose = features['pose']
        return image_raw, landmarks, pose

    def test_tf_records_validity(self):
        image_utility = ImageUtility()
        lbl_arr, img_arr = self.retrieve_tf_record(tfrecord_filename=IbugConf.tf_train_path,
                                                   number_of_records=30, only_label=False, only_pose=False)
        for i in range(30):
            landmark_arr_xy, landmark_arr_x, landmark_arr_y = image_utility.create_landmarks(landmarks=lbl_arr[i],
                                                                                             scale_factor_x=1,
                                                                                             scale_factor_y=1)
            image_utility.print_image_arr(i, img_arr[i], landmark_arr_x, landmark_arr_y)
