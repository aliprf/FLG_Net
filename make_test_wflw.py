import scipy.io
import numpy as np
import os

# WFLW
file_dataset = open('./WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt')
os.mkdir('./test_imagesAnnotaion/')

for line in file_dataset:

    # print(line)
    adata = line.split(" ")

    a1 = adata[0:196:2]
    a2 = adata[1:196:2]
    a11 = [float(i) for i in a1]
    a12 = [float(i) for i in a2]
    a = np.zeros([98, 2])
    a[:, 0] = np.array(a11).T
    a[:, 1] = np.array(a12).T
    np.savetxt("./data_temp.pts", a, fmt = "%s")
    file_name_path = adata[-1]
    #print(file_name)
    # file_name = file_name.split("--")
    file_name = file_name_path.split("/")[1]
    file_path = file_name_path.split("/")[0]
    print(file_name)
    #file_name = file_name.split("/")[-1]
    file_train = open('./data_temp.pts', "r+")
    if not os.path.isdir('./test_imagesAnnotaion/' ):
        os.mkdir('./test_imagesAnnotaion/' )
    cnt = 0
    while os.path.exists('./test_imagesAnnotaion/' + file_name + '.pts'):
        cnt += 1
        file_name = file_name.split("#")[0]
        file_name = file_name + "#" + str(cnt)

    final_train = open('./test_imagesAnnotaion/' + file_name + '.pts', "w+")
    bxymin = a1[196:198]
    bxymax = a1[198:200]
    final_train.write('version: 1' + '\n' +'n_points:' + str(np.shape(a)[0]) + '\n' + '{' + '\n' + file_train.read() + '}' + '\n' + str(adata[196]) + ' ' + str(adata[197]) + '\n' + str(adata[198]) + ' ' + str(adata[199]))

    final_train.close()
    os.remove('./data_temp.pts')


print(cnt)
