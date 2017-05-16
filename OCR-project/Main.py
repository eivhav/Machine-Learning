
import numpy as np
from skimage.io import imread
import glob

from DataAugmentation import data_augmentation
from Detection import detection
from PreProcessing import preProcess_svm, preProcess_knn, preProcess_ConvNet
from Models import SVM, Knn, DeepConvNet


parameters = {'model_type': 'kNN',                      # 'SVM', 'kNN', or 'ConvNet'
              'augmented_data': False,                  # Add more 'fake' data
              'preprocess_method': 'denoiseHOG', # 'denoise_Threshold', "denoiseHOG", or "ThresholdPCA"(SVM only)

              # For SVM
              'n_PCA_comp': 40,                         # 40 is optimal

              # For kNN
              'knn_method': "knn_class",                # knn_class, knn_rad or knn_cent
              'n_neighbors': 4,
              'knn_weights': 'distance',
              'knn_radius': 10,

              # For ConvNet
              'model_version': 'v1',                     # v1 or v2
              'optimizer': 'adadelta',
              'load_model': True,                       # True if train new model
              'model_folder': "/home/havikbot/PycharmProjects/",    # Replace this if running ConvNet
              'model_save_version': "v1_8_1",

              # For detection
              'run_detection': True,                      # requires ConvNetModel
              }



def main(params):

    # Load data
    train_raw = load_data("training", params['augmented_data'])
    test_raw = load_data("test", False)
    y_train = np.array(train_raw[1])
    y_test = np.array(test_raw[1])
    print("Data loaded")


    # Run SVM model
    if params['model_type'] == "SVM":
        X_train, pca_model = preProcess_svm(train_raw[0], params=params, PCA_model=None)
        X_test, pca_model = preProcess_svm(test_raw[0], params=params, PCA_model=pca_model)
        print('Preprocessing Complete')

        if params['preprocess_method'] == "denoiseHOG": model = SVM("linear")
        else: model = SVM("svc")

        model.train_model(train=(X_train, y_train))
        model.test_model(test=(X_test, y_test))


    # Run kNN model
    elif params['model_type'] == 'kNN':
        X_train = preProcess_knn(train_raw[0], params)
        X_test = preProcess_knn(test_raw[0], params)
        print('Preprocessing Complete')

        model = Knn(method=params['knn_method'], n_neighbors=params['n_neighbors'],
                    weights=params['knn_weights'], radius=params['knn_radius'])
        model.train_model(train=(X_train, y_train))
        print("Testing. Accuracy:", model.test_model(test=(X_test, y_test)))


    # Run ConvNet model
    elif params['model_type'] =='ConvNet':
        X_train, y_train = preProcess_ConvNet(train_raw[0], train_raw[1])
        X_test, y_test = preProcess_ConvNet(test_raw[0], test_raw[1])
        print('Preprocessing Complete')

        model = DeepConvNet(params['model_version'], params['optimizer'])
        model_data_path = params['model_folder'] +'ConvNetModel' + params['model_save_version'] + ".h5"
        if params['load_model']:
            model.load_model(model_data_path)
        else:
            print(y_train.shape)
            model.train_model(train=(X_train, y_train), test=(X_test, y_test),
                              batch_size=256, nb_epochs=50, file_path=model_data_path)

        model.test_model(X_test, y_test, save_images=False)

        # Run detection
        if params['run_detection']:
            det = detection("detection-images")
            det.detect_letters(model=model.model, save_images=False)

    else:
        print("Invalid model_type", params['model_type'])





def load_data(data_set, augment_data):
    alphlist = ['0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    picture_list = []
    label_list = []
    for img in glob.glob(data_set + '/*.jpg'):
        p_img = imread(img, as_grey=True)
        picture_list.append(np.array(p_img) / 255)
        if data_set == 'test':
            label_list.append(img[5])
        else:
            label_list.append(img[9])

        for letter in alphlist:
            if label_list[-1] == letter:
                label_list.pop(-1)
                label_list.append(alphlist.index(letter))

    if augment_data:
        da = data_augmentation()
        picture_list, label_list = da.get_augmented_data(picture_list, label_list)
        print(len(picture_list))

    return picture_list, label_list


main(params=parameters)















