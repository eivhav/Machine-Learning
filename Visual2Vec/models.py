
from keras.applications import ResNet50, InceptionV3, VGG16, VGG19, imagenet_utils, inception_v3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras import backend
from keras.layers import Input, merge, Reshape
from keras.layers.core import Dense, Lambda, Reshape, Dropout
from keras.layers.convolutional import Convolution1D
from keras.models import Model, load_model
from keras.preprocessing import image
import time

import tensorflow as tf
tf.python.control_flow_ops = tf

class modelsClass():



    word_embedding_size = 300
    pic_module_output_size = 2048
    hidden1_size = 2048
    hidden2_size = 1024 #300
    hidden3_size = 600
    final_embedding_size = 300    #300
    p = [0.45, 0.40, 0.30, 0.05]
    J = 3 #  Number fo negative examples

    hinge_margin = 0.05

    training_model = None
    prediction_model = None
    visual_model = None



    def R(self, vects):
        # Calculates the similarity of two vectors.
        (x, y) = vects
        return backend.dot(tf.nn.l2_normalize(x, dim=1), backend.transpose(tf.nn.l2_normalize(y, dim=1)))

    def GESD(self):
        gamma_value = 1
        c_value = 1
        dot = lambda a, b: backend.batch_dot(a, b, axes=1)
        l2_norm = lambda a, b: backend.sqrt(backend.sum(backend.square(a - b), axis=1, keepdims=True))
        euclidean = lambda x: 1 / (1 + l2_norm(x[0], x[1]))
        sigmoid = lambda x: 1 / (1 + backend.exp(-1 * gamma_value * (dot(x[0], x[1]) + c_value)))
        return lambda x: euclidean(x) * sigmoid(x)


    def cosine_lambda(self):
        return lambda x: backend.dot(tf.nn.l2_normalize(x[0], dim=1), backend.transpose(tf.nn.l2_normalize(x[1], dim=1)))


    def build_softmax_model(self, optimizer, sim_type, neg_samples):
        self.J = neg_samples

        pic_input = Input(shape = (self.pic_module_output_size,))
        pos_labels = Input(shape = (self.word_embedding_size, ))
        neg_labels = [Input(shape = (self.word_embedding_size, )) for j in range(self.J)]

        pic_drop_out1 = Dropout(self.p[0])(pic_input)
        pic_hidden1 = Dense(self.hidden1_size, activation="tanh", input_dim=self.pic_module_output_size)(pic_drop_out1)

        pic_drop_out2 = Dropout(self.p[1])(pic_hidden1)
        pic_hidden2 = Dense(self.hidden2_size, activation="tanh", input_dim=self.hidden1_size)(pic_drop_out2)

        pic_drop_out3 = Dropout(self.p[2])(pic_hidden2)
        pic_hidden3 = Dense(self.hidden3_size, activation="tanh", input_dim=self.hidden2_size)(pic_drop_out3)

        pic_drop_out4 = Dropout(self.p[3])(pic_hidden3)
        pic_output = Dense(self.final_embedding_size, activation="tanh", input_dim=self.hidden3_size)(pic_drop_out4)

        pos_output = pos_labels
        neg_outputs = neg_labels

        # This layer calculates the cosine similarity between the semantic representations of
        # a image and a embedding.
        R_layer = Lambda(self.R, output_shape = (1,))

        R_pos = R_layer([pic_output, pos_output])
        R_negs = [R_layer([pic_output, neg_output]) for neg_output in neg_outputs]

        concat_Rs = merge([R_pos] + R_negs, mode = "concat")
        concat_Rs = Reshape((self.J + 1, 1))(concat_Rs)

        # In this step, we multiply each R(I, E) value by gamma. We can learn gamma's value by pretending it's
        # a single, 1 x 1 kernel.
        Rs_with_gamma = Convolution1D(1, 1, border_mode = "same", input_shape = (self.J + 1, 1), activation = "linear", bias = False)(concat_Rs)

        # Next, we exponentiate each of the gamma x R(I, E+) values.
        exp = Lambda(lambda x: backend.exp(x), output_shape=(self.J + 1,))(Rs_with_gamma)
        exp = Reshape((self.J + 1,))(exp)

        # Finally, we use the softmax function to calculate the P(E+|I).
        prob = Lambda(lambda x: -backend.log(x[0][0] / backend.sum(x[0])))(exp)
        prob2 = backend.cast(prob, tf.float32)

        # We define our models
        self.training_model = Model(input=[pic_input, pos_labels] + neg_labels, output=prob2)
        def y_pred_loss(y_true, y_pred):
            return y_pred
        self.training_model.compile(optimizer=optimizer, loss=y_pred_loss)
        self.prediction_model = Model(input=pic_input, output=pic_output)

        print("Softmax model built")



    def build_hinge_model(self, optimizer, sim_type, neg_samples):
        self.J = neg_samples

        pic_input = Input(shape=(self.pic_module_output_size,))
        pos_labels = Input(shape=(self.word_embedding_size,))
        neg_labels = [Input(shape=(self.word_embedding_size,)) for j in range(self.J)]

        pic_drop_out1 = Dropout(self.p[0])(pic_input)
        pic_hidden1 = Dense(self.hidden1_size, activation="tanh", input_dim=self.pic_module_output_size)(pic_drop_out1)

        pic_drop_out2 = Dropout(self.p[1])(pic_hidden1)
        pic_hidden2 = Dense(self.hidden2_size, activation="tanh", input_dim=self.hidden1_size)(pic_drop_out2)

        pic_drop_out3 = Dropout(self.p[2])(pic_hidden2)
        pic_hidden3 = Dense(self.hidden3_size, activation="tanh", input_dim=self.hidden2_size)(pic_drop_out3)

        pic_drop_out4 = Dropout(self.p[3])(pic_hidden3)
        pic_output = Dense(self.final_embedding_size, activation="tanh", input_dim=self.hidden3_size)(pic_drop_out4)

        pos_output = pos_labels
        neg_outputs = neg_labels

        hinge_sim_pos = merge([pic_output, pos_output], mode=self.GESD(), output_shape=lambda _: (None, 1))
        hinge_sim_neg = merge([pic_output, neg_outputs[0]], mode=self.GESD(), output_shape=lambda _: (None, 1))

        hinge_loss = Lambda(lambda x: abs((self.hinge_margin - x[0] + x[1])))
        prob = hinge_loss([hinge_sim_pos, hinge_sim_neg])

        self.training_model = Model(input=[pic_input, pos_labels] + neg_labels, output=prob)

        def y_pred_loss(y_true, y_pred):
            return y_pred
        self.training_model.compile(optimizer=optimizer, loss=y_pred_loss)

        self.prediction_model = Model(input=pic_input, output=pic_output)

        print("Hinge-loss model built")



    def build_visual_model(self, vis_model, out_layer):
        if vis_model == 'InceptionV3':
            inception_model = InceptionV3(weights='imagenet', include_top=True)
            if out_layer == 'top': self.visual_model = inception_model
            else: self.visual_model = Model(input = inception_model.input, output=inception_model.get_layer(out_layer).output)

        elif vis_model == 'ResNet50':
            resnet_model = self.visual_model = ResNet50(weights='imagenet', include_top=True)
            if out_layer == 'top': self.visual_model = resnet_model
            else: self.visual_model = Model(input=resnet_model.input, output=resnet_model.get_layer(out_layer).output)

        elif vis_model == 'VGG16':
            self.visual_model = VGG16(weights='imagenet', include_top=True)

        elif vis_model == 'VGG19':
            self.visual_model = VGG19(weights='imagenet', include_top=True)



    def run_visual_model(self, input_path, picture_ids, sub_folders, vis_model, out_layer):
        if self.visual_model is None: self.build_visual_model(vis_model, out_layer)
        predictions = dict()
        print("Running predictions. Total:" + str(len(picture_ids)) , " Visual model", vis_model, out_layer)
        count = 0
        target_size = (224, 224)
        if vis_model == 'InceptionV3': target_size = (299, 299)

        for picture in picture_ids:
            count += 1
            pic_path = picture
            if picture in sub_folders: pic_path = sub_folders[picture] + picture
            img = image.load_img(input_path + pic_path + '.jpg', target_size=target_size)

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            if vis_model == 'InceptionV3': x = inception_v3.preprocess_input(x)
            else: x = imagenet_utils.preprocess_input(x)

            predictions[picture] = self.visual_model.predict(x)[0]
            if count % 1000 == 0: print(count, "of", len(picture_ids))

        return predictions


    def run_validation(self, nb_matches, val_ids_predictions, train_ids_predictions):

        validate_semantics = [[],[]]
        training_semantics = [[],[]]
        result_dict = dict()

        print("Extracting training data semantics...")
        for id_key in train_ids_predictions.keys():
            training_semantics[0].append(id_key)
            training_semantics[1].append(train_ids_predictions[id_key])

        preds = self.prediction_model.predict(np.array(np.array(training_semantics[1])))
        for i in range(len(training_semantics[0])): training_semantics[1][i] = preds[i]
        print("   Done.")

        print("Extracting validation data semantics...")
        for id_key in val_ids_predictions.keys():
            validate_semantics[0].append(id_key)
            validate_semantics[1].append(val_ids_predictions[id_key])

        preds = self.prediction_model.predict(np.array(np.array(validate_semantics[1])))
        for i in range(len(validate_semantics[0])): validate_semantics[1][i] = preds[i]
        print("   Done.")

        print("Calculating similarities...")
        sim_matrix = cosine_similarity(validate_semantics[1], training_semantics[1])
        print("   Done.")

        print("Finding best matches...")
        total_matches = 0
        # First we find the sim threshold which gives right amount of matches
        # This value is emperical set to be 0.5 % of training data

        sim_threshold = 0.60
        for s in range(100, 0, -1):
            nb_ind = 0
            for i in range(0, len(validate_semantics[1])):
                nb_ind += len(np.where(np.array(sim_matrix[i]) > (s/100))[0])
            avg_matches = (nb_ind) / len(validate_semantics[1])
            if avg_matches > (nb_matches * len(training_semantics[1])):
                print("     Found sim_threshold to be ", s/100, "@", avg_matches)
                sim_threshold = s/100
                break

        for i in range(len(validate_semantics[1])):
            similar_keys = []
            similar_sims = []
            np_sims = np.array(sim_matrix[i])
            indicies = np.where(np_sims > sim_threshold)[0]

            for ind in indicies:
                similar_keys.append(training_semantics[0][ind])
                similar_sims.append(sim_matrix[i][ind])
                total_matches += 1
            result_dict[validate_semantics[0][i]] = [similar_keys, similar_sims]

        print("   Done. On avg ", (total_matches / len(validate_semantics[1])), " matches per val sample")
        return result_dict


    def save_model(self, data_path, my_model, params):
        my_model.save(data_path + 'models/' + 'model' + time.strftime("H:%d_%m_%Y") + '__' + params +'.h5')
        print("Model saved")

    def load_model(self, data_path, path, only_weights):
        if only_weights:
            self.training_model.load_weights(data_path + 'models/' +path)
            print("Model weights loaded. ")
        else:
            self.training_model = load_model(data_path + 'models/' +path)
            print("Model loaded. ")














