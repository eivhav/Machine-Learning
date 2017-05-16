import numpy as np
from skimage.filters import threshold_otsu, threshold_local
from skimage import restoration
from skimage.feature import hog
from sklearn.decomposition import PCA



def convert_labels_to_one_hot(labels):
    alphlist = ['0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    one_hots = []
    for label in labels:
        one_hot = [0] * 26
        if label != 0:
            one_hot[int(label) - 1] = 1
            one_hots.append(one_hot)
        else:
            one_hots.append(one_hot)
    return one_hots



def preProcess_svm(images, params, PCA_model):

    if params['preprocess_method'] == "denoise_Threshold":
        for i in range(len(images)):
            images[i] *= 255
            images[i] = restoration.denoise_tv_chambolle(images[i], weight=0.001)
            binary_global = images[i] > threshold_otsu(images[i])
            images[i] = binary_global.flatten().reshape(400)
        return np.array(images), None

    if params['preprocess_method'] == "ThresholdPCA":
        for i in range(len(images)):
            images[i] *= 255
            binary_global = images[i] > threshold_otsu(images[i])
            images[i] = binary_global.flatten().reshape(400)
        X = np.array(images)
        if PCA_model is None:
            PCA_model = PCA(n_components=params['n_PCA_comp'], svd_solver='randomized', whiten=False).fit(X)
        return PCA_model.transform(X), PCA_model

    if params['preprocess_method'] == "denoiseHOG":
        for i in range(len(images)):
            images[i] *= 255
            images[i] = restoration.denoise_tv_chambolle(images[i], weight=0.001)
            images[i], hogimg = hog(images[i], orientations=10, pixels_per_cell=(4,4), cells_per_block=(2,2), visualise=True)
            if i % 1000 == 0: print("processing images:", i, "/", len(images))
        return np.array(images), None


def preProcess_knn(images, params):

    if params['preprocess_method'] == "denoise_Threshold":
        for i in range(len(images)):
            images[i] *= 255
            images[i] = restoration.denoise_tv_chambolle(images[i], weight=0.001)
            binary_global = images[i] > threshold_otsu(images[i])
            images[i] = binary_global.flatten().reshape(400)
        return np.array(images)

    if params['preprocess_method'] == "denoiseHOG":
        for i in range(len(images)):
            images[i] *= 255
            images[i] = restoration.denoise_tv_chambolle(images[i], weight=0.001)
            images[i], hogimg = hog(images[i], pixels_per_cell=(4,4), cells_per_block=(4,4), block_norm='L2-Hys', visualise=True)
        return np.array(images)




def preProcess_ConvNet(images, labels):
    X = np.array(images)
    X = X.reshape(X.shape[0], 20, 20, 1)
    y = convert_labels_to_one_hot(labels)
    y = np.array(y)
    return X, y



