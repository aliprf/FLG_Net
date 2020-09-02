import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from pathlib import Path

# import pandas as pd
# import seaborn as sns
from scipy import stats
import numpy as np
import random

import PIL.ImageDraw as ImageDraw
import PIL.Image as Image
from configuration import InputDataSize
from image_utility import ImageUtility

def print_partial(counter, img, landmarks_arr):
    image_utility = ImageUtility()
    # plt.figure()
    # plt.imshow(img)
    # implot = plt.imshow(img)

    index =0
    for lndm in landmarks_arr:
        image = Image.new("L", (InputDataSize.image_input_size // 4, InputDataSize.image_input_size // 4))
        draw = ImageDraw.Draw(image)

        # color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        landmark_arr_xy, landmark_arr_x, landmark_arr_y = image_utility.create_landmarks_from_normalized(lndm, 224, 224, 112, 112)

        # plt.scatter(x=landmark_arr_x[:], y=landmark_arr_y[:], c=color, s=20)
        # plt.plot(landmark_arr_x, landmark_arr_y, '-ok', c=color)

        # if index == 4 or index == 5 or index == 6 or index == 7:
        #     draw.polygon((landmark_arr_xy), fill='#ffffff')

        landmark_arr_xy = (np.array(landmark_arr_xy) // 4).tolist()
        draw.line((landmark_arr_xy), fill='#ffffff', width=2)

        img_np = np.asarray(image)
        plt.imshow(img_np)
        image.save('0_name_' + str(counter) + '_' + str(index) + '.png')
        index += 1

    # plt.axis('off')
    # plt.savefig('name_' + str(counter) + '.png', bbox_inches='tight')
    # # plt.show()
    # plt.clf()

def print_histogram2(counter, landmarks_arr):
    for data in landmarks_arr:
        data = np.array(data).reshape([98, 2])
        df = pd.DataFrame(data, columns=["y", "x"])
        x1 = sns.jointplot(x="y", y="x", data=df)
        x2 = sns.jointplot(x="y", y="x", data=df, kind="kde")

    x1.savefig('out1_' + str(counter) + '.png')
    x2.savefig('out2_' + str(counter) + '.png')

def print_histogram1(counter, data):
    if data is None:
        mean, cov = [0, 1], [(1, .5), (.5, 1)]
        data = np.random.multivariate_normal(mean, cov, 200)

    df = pd.DataFrame(data, columns=["y", "x"])
    x1 = sns.jointplot(x="y", y="x", data=df)
    x2 = sns.jointplot(x="y", y="x", data=df, kind="kde")
    x1.savefig('out1_'+str(counter)+'.png')
    x2.savefig('out2_'+str(counter)+'.png')


def print_histogram2d(x,y, data):
    with sns.axes_style("white"):
        x = sns.jointplot(x=x, y=y, kind="hex", color="k")
        x = sns.jointplot(x="x", y="y", data=data, kind="kde")
        x.savefig('out_1.png')

def print_histogram(data):
    sns.set(color_codes=True)
    sns.distplot(data)
    sns_plot = sns.distplot(data, kde=False, rug=True)
    sns_plot1 = sns.distplot(data, kde=False, fit=stats.gamma)
    sns_plot1.get_figure().savefig("output.png")


def print_image(image_name, landmarks_x, landmarks_y):

    my_file = Path(image_name)
    if my_file.is_file():
        im = plt.imread(image_name)
        implot = plt.imshow(im)

        plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='r', s=10)
        plt.show()


def print_image_arr_heat(k, image, print_single=False):
    import numpy as np
    for i in range(image.shape[2]):
        img = np.sum(image, axis=2)
        if print_single:
            plt.figure()
            plt.imshow(image[:, :, i])
            # implot = plt.imshow(image[:, :, i])
            plt.axis('off')
            plt.savefig('single_heat_' + str(i+(k*100)) + '.png', bbox_inches='tight')
            plt.clf()

    plt.figure()
    plt.imshow(img, vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig('heat_' + str(k) + '.png', bbox_inches='tight')
    plt.clf()

def print_histogram_plt(k, type, landmarks_arr):
    import matplotlib.ticker as ticker
    import random
    image_utility = ImageUtility()

    # var = np.var(landmarks_arr, axis=0)
    # mean = np.mean(landmarks_arr, axis=0)
    #
    plt.figure()
    for lndm in landmarks_arr:
        data = lndm
        color = '#b52b65'

        if type == 'face':
            data = lndm[0:64]
            color = '#ed6663'

        # color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

        # plt.plot(data, '-ok', c='#799351')
        plt.hist(data, bins=20, color=color, alpha=0.3, histtype='bar')
        # plt.hist(data, bins=num_of_bins, color=color, edgecolor='green', alpha=0.2)

        # plt.axis.set_major_formatter(ticker.PercentFormatter(xmax=len(landmark_arr_x)))

    # plt.text(0,15, 'mean: '+str(mean)+', var:' + str(var))
    plt.savefig('histo_' + str(type) + '_' + str(k) + '.png', bbox_inches='tight')
    plt.clf()

def print_histogram_plt_x_y(k, type, landmarks_arr):
    import matplotlib.ticker as ticker
    import random
    image_utility = ImageUtility()
    num_of_bins = 20

    plt.figure()
    for lndm in landmarks_arr:
        landmark_arr_xy, landmark_arr_x, landmark_arr_y = image_utility.create_landmarks(landmarks=lndm,
                                                                                         scale_factor_x=1,
                                                                                         scale_factor_y=1)
        data = landmark_arr_x
        data = landmark_arr_y
        # plt.plot(landmark_arr_x[0:32], landmark_arr_y[0:32], '-ok', c=color)
        plt.hist(data, bins=num_of_bins, color='blue', edgecolor='blue', alpha=0.3)
        # plt.hist2d(landmark_arr_x, landmark_arr_y, bins=num_of_bins, color='blue', edgecolor='black', alpha=0.5)
        # plt.axis.set_major_formatter(ticker.PercentFormatter(xmax=len(landmark_arr_x)))

    plt.savefig('histo_X_' + str(type) + '_' + str(k) + '.png', bbox_inches='tight')
    plt.clf()

    '''for Y'''
    plt.figure()
    for lndm in landmarks_arr:
        landmark_arr_xy, landmark_arr_x, landmark_arr_y = image_utility.create_landmarks(landmarks=lndm,
                                                                                         scale_factor_x=1,
                                                                                         scale_factor_y=1)
        data = landmark_arr_y
        # plt.plot(landmark_arr_x[0:32], landmark_arr_y[0:32], '-ok', c=color)
        plt.hist(data, bins=num_of_bins, color='green', edgecolor='green', alpha=0.3)
        # plt.axis.set_major_formatter(ticker.PercentFormatter(xmax=len(data)))

    plt.savefig('histo_Y_' + str(type) + '_' + str(k) + '.png', bbox_inches='tight')
    plt.clf()

def print_arr(k, type, landmarks_arr):
    import random
    image_utility = ImageUtility()

    plt.figure()
    for lndm in landmarks_arr:
        color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

        landmark_arr_xy, landmark_arr_x, landmark_arr_y = image_utility.create_landmarks(landmarks=lndm,
                                                                                         scale_factor_x=1,
                                                                                         scale_factor_y=1)
        if type == 'full':
            plt.scatter(x=landmark_arr_x[:], y=landmark_arr_y[:], c=color, s=2)
        elif type == 'face':
            plt.scatter(x=landmark_arr_x[0:32], y=landmark_arr_y[0:32], c=color, s=5)
            plt.plot(landmark_arr_x[0:32], landmark_arr_y[0:32], '-ok', c=color)

        elif type == 'eyes':
            plt.scatter(x=landmark_arr_x[60:75], y=landmark_arr_y[60:75], c=color, s=5)
            plt.plot(landmark_arr_x[60:75], landmark_arr_y[60:75], '-ok', c=color)

        elif type == 'nose':
            plt.scatter(x=landmark_arr_x[51:59], y=landmark_arr_y[51:59], c=color, s=5)
            plt.plot(landmark_arr_x[51:59], landmark_arr_y[51:59], '-ok', c=color)

        elif type == 'mouth':
            plt.scatter(x=landmark_arr_x[76:95], y=landmark_arr_y[76:95], c=color, s=5)
            plt.plot(landmark_arr_x[76:95], landmark_arr_y[76:95], '-ok', c=color)

        elif type == 'eyebrow':
            plt.scatter(x=landmark_arr_x[33:50], y=landmark_arr_y[33:50], c=color, s=5)
            plt.plot(landmark_arr_x[33:50], landmark_arr_y[33:50], '-ok', c=color)

    plt.axis('off')
    plt.savefig('name_' + str(type) + '_' + str(k) + '.png', bbox_inches='tight')
    # plt.show()
    plt.clf()


def print_image_arr(k, image, landmarks_x, landmarks_y):
    plt.figure()
    plt.imshow(image)
    implot = plt.imshow(image)

    for i in range(len(landmarks_x)):
        plt.text(landmarks_x[i], landmarks_y[i], str(i), fontsize=12, c='red',
                 horizontalalignment='center', verticalalignment='center',
                 bbox={'facecolor': 'blue', 'alpha': 0.3, 'pad': 0.0})

    plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#440047', s=60)
    plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='#fbecec', s=10)
    plt.axis('off')
    plt.savefig('name_' + str(k) + '.png', bbox_inches='tight')
    # plt.show()
    plt.clf()


def print_image_arr_2(k, image, landmarks_x, landmarks_y, xs, ys):
    plt.figure()
    plt.imshow(image)
    implot = plt.imshow(image)

    # plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='b', s=5)
    # for i in range(68):
    #     plt.annotate(str(i), (landmarks_x[i], landmarks_y[i]), fontsize=6)

    plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='b', s=5)
    plt.scatter(x=xs, y=ys, c='r', s=5)
    plt.savefig('sss'+str(k)+'.png')
    # plt.show()
    plt.clf()


def print_two_landmarks(image, landmarks_1, landmarks_2):
    plt.figure()
    plt.imshow(image)
    implot = plt.imshow(image)

    plt.scatter(x=landmarks_1[:68], y=landmarks_1[68:], c='b', s=10)
    plt.scatter(x=landmarks_2[:68], y=landmarks_2[68:], c='r', s=10)
    # plt.savefig('a'+str(landmarks_x[0])+'.png')
    plt.show()
    plt.clf()