
import sklearn
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from keras.models import Model, load_model
from keras.layers import Flatten, Dense, Input, Reshape, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint


import numpy as np

from skimage import io
from sklearn import neighbors

from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier

class SVM():
    def __init__(self, type):
        if type == 'linear':
            self.clf = sklearn.svm.LinearSVC()
        else:
            self.clf = sklearn.svm.SVC()

    def train_model(self, train):
        self.clf.fit(train[0], train[1])

    def predict(self, data):
        return self.clf.predict(data)

    def test_model(self, test):
        score = 0
        for i in range(len(test[0])):
            if self.clf.predict(test[0][i])[0] == test[1][i]:
                score += 1
        print("Correct on test set: ", float(score)/len(test[0]))


class Knn():
    def __init__(self, method, n_neighbors, weights, radius):
        if method == 'knn_class':
            self.clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
        elif method == 'knn_rad':
            self.clf = RadiusNeighborsClassifier(radius=radius)
        elif method == 'knn_cent':
            self.clf = NearestCentroid()

    def train_model(self, train):
        self.clf.fit(train[0], train[1])

    def predict(self, data):
        return self.clf.predict(data)

    def test_model(self, test):
        return self.clf.score(test[0], test[1])



class DeepConvNet():
    def __init__(self, version, optimizer):
        if version == 'v1':
            print("Building v1 model...")
            self.model = self.ConvModelv1((20, 20, 1), 26)
            print(self.model.summary())
        elif version == 'v2':
            print("Building v2 model...")
            self.model = self.ConvModelv2((20, 20, 1), 26)
            print(self.model.summary())

        if self.model is not None:
            self.model.compile(optimizer=optimizer, loss='binary_crossentropy')
            print("Model compiled sucessfully ")


    def ConvModelv1(self, input_shape, nb_classes):
        img_input = Input(shape=input_shape, name='input')
        x = Convolution2D(32, 5, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
        x = Convolution2D(32, 5, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='block1_pool')(x)

        # Block 3
        x = Dropout(0.30)(x)
        x = Convolution2D(64, 5, 4, activation='relu', border_mode='same', name='block3_conv1')(x)
        x = Convolution2D(64, 5, 4, activation='relu', border_mode='same', name='block3_conv2')(x)
        x = Convolution2D(64, 5, 4, activation='relu', border_mode='same', name='block3_conv3')(x)
        x = MaxPooling2D((4, 4), strides=(1, 1), name='block3_pool')(x)
        x = MaxPooling2D((3, 3), strides=(1, 1), name='block3_pool2')(x)

        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dropout(0.30)(x)
        x = Dense(nb_classes, activation='softmax', name='predictions')(x)

        return Model(img_input, x, name='ConvModel')


    def ConvModelv2(self, input_shape, nb_classes):
        img_input = Input(shape=input_shape, name='input')
        x = Convolution2D(64, 5, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
        x = Convolution2D(64, 5, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), name='block1_pool')(x)

        # Block 3
        x = Dropout(0.30)(x)
        x = Convolution2D(64, 5, 4, activation='relu', border_mode='same', name='block2_conv1')(x)
        x = Convolution2D(64, 5, 4, activation='relu', border_mode='same', name='block2_conv2')(x)
        x = Convolution2D(64, 5, 4, activation='relu', border_mode='same', name='block2_conv3')(x)
        x = MaxPooling2D((4, 4), strides=(1, 1), name='block2_pool')(x)

        x = Dropout(0.30)(x)
        x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
        x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
        x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
        x = MaxPooling2D((4, 4), strides=(1, 1), name='block3_pool')(x)

        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dropout(0.30)(x)
        #x = Dense(1024, activation='relu', name='hidden')(x)
        x = Dense(nb_classes, activation='softmax', name='predictions')(x)

        return Model(img_input, x, name='ConvModelv2')


    def train_model(self, train, test, batch_size, nb_epochs, file_path):
        best_model = ModelCheckpoint(file_path, monitor='val_loss', verbose=0, save_best_only=True,
                                     save_weights_only=False)

        self.model.fit(x=train[0], y=train[1], batch_size=batch_size, nb_epoch=nb_epochs, verbose=2,
                       callbacks=[best_model], validation_data=(test[0], test[1]), shuffle=True) #validation_data=(test[0], test[1])
        self.model = load_model(filepath=file_path)


    def load_model(self, file_path):
        print("Loading model...")
        self.model = load_model(filepath=file_path)
        print(self.model.summary())

    def test_model(self, X_test, y_test, save_images):
        results = self.model.predict(X_test)
        score = 0
        for i in range(len(results)):
            if np.argmax(results[i]) == np.argmax(y_test[i]):
                score += 1
                if save_images: save_image(X_test[i], i, np.argmax(results[i]), "correct")
            else:
                if save_images: save_image(X_test[i], i, np.argmax(results[i]), "incorrect")

        print("result", score / y_test.shape[0])








def save_image(array, i, label, folder):
    alphlist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']
    array *= 255
    rescaled_im = np.array(array, dtype='uint8').reshape(20, 20)
    io.imsave("/home/havikbot/PycharmProjects/data_out/"+folder+"/"+str(i) + "-" + alphlist[label] + ".png", rescaled_im)