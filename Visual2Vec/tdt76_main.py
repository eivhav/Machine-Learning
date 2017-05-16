
from tdt76.models import modelsClass
from tdt76.data import data_class
import pickle
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os


class main():

    main_filepath = '/home/havikbot/'   # Please provide the main path here
    nb_negs = 5

    # for inception: 'top' or 'flatten'
    # for ResNet50: 'top or 'flatten_1'

    vis_module = 'InceptionV3'
    vis_module_outlayer = 'flatten'

    current_id = 'Incep3flat-v1'

    models = modelsClass()
    data_module = data_class(main_filepath)
    data_module.load_images_and_tags('train', 1000)
    #data_module.load_images_and_tags('validate', 101)

    models.build_hinge_model('adam', 'cos', nb_negs)
    #models.build_softmax_model('adam', 'cos', nb_negs)
    training_model = models.training_model

    def __init__(self, path):
        self.main_filepath = path



    def get_train_visual_predictions(self):
        if os.path.isfile(self.main_filepath + 'visual_pred_'+self.current_id) is True:
            print("training predictions found ")
            return self.data_module.load_visual_predicted_data(self.current_id)
        else:
            print("Training predictions not found. Generating them from image module.  ")
            print(" This is a lengthy process (45 min). The predictions ca be downloaded from link in the readme file ")
            predictions_train = self.models.run_visual_model(self.main_filepath+'train/pics/',
                                                             self.data_module.training_images_and_tags.keys(),
                                                             self.data_module.sub_folders_train,
                                                             self.vis_module,
                                                             self.vis_module_outlayer)
            self.data_module.save_visual_predicted_data(predictions_train, self.current_id)
            return predictions_train


    def get_val_visual_predictions(self, load, picture_folder_path, image_ids, sub_folder_dict):
        if os.path.isfile(self.main_filepath + 'visual_pred_validate-'+self.current_id) is True and load is True:
            print("validation predictions found ")
            return self.data_module.load_visual_predicted_data(self.current_id)
        else:
            print("Validation predictions not found. Generating them from image module. Takes ~30 sec per 1k ")
            predictions_validate = self.models.run_visual_model(picture_folder_path,
                                                                image_ids, sub_folder_dict,
                                                                self.vis_module,
                                                                self.vis_module_outlayer)
            self.data_module.save_visual_predicted_data(predictions_validate, 'validate-'+self.current_id)
            print("   Done")
            return predictions_validate

    def train_model(self, saved):

        if saved is True:
            training_data = self.data_module.get_data_from_pickle('training', self.current_id, self.nb_negs, 0.7)
        else:
            training_data = self.data_module.get_data('training', self.current_id, 0 , True, self.nb_negs, 0.7)

        y = np.ones(training_data[0].shape[0])
        earlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        checkpoint = ModelCheckpoint(self.main_filepath+'last_save.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')

        hist = self.training_model.fit(training_data, y, nb_epoch=100, verbose=1, batch_size=100, validation_split=0.2, callbacks=[earlyStop, checkpoint])
        self.models.training_model.load_weights(self.main_filepath+'last_save.h5')


    def run_model_get_results(self, ):
        self.models.training_model.load_weights(self.main_filepath+'last_save.h5')
        training_preds = self.data_module.load_visual_predicted_data(self.current_id)
        validation_preds = self.data_module.load_visual_predicted_data('validate-'+self.current_id)
        return self.models.run_validation(0.005, validation_preds, training_preds)



    def score(self, label_dict, target='', selection=list(), n=50):

        """
        4 Calculate the score of a selected set compared to the target image.
        5
        6 :param label_dict: dictionary of labels, keys are image IDs
        7 :param target: image ID of the query image
        8 :param selection: the list of IDs retrieved
        9 :param n: the assumed number of relevant images. Kept fixed at 50
        10 :return: the calculated score
        """

        # Remove the queried element
        selection = list(set(selection) - set([target]))

        # k is the number of retrieved elements
        k = len(selection)
        if target in label_dict.keys():
            target_dict = dict(label_dict[target])
        else:
            print("Couldn’t find " + target + " in the dict keys.")
            target_dict = {}

        # Current score will accumulate the element-wise scores,
        # before rescaling by scaling by 2/(k*n)
        current_score = 0.0

        # Calculate best possible score of image
        best_score = sum(target_dict.values())

        # Avoid problems with div zero. If best_score is 0.0 we will
        # get 0.0 anyway, then best_score makes no difference
        if best_score == 0.0:
            best_score = 1.0

        # Loop through all the selected elements
        for selected_element in selection:
            # If we have added a non-existing image we will not get
            # anything, and create a dict with no elements
            # Otherwise select the current labels
            if selected_element in label_dict.keys():
                selected_dict = dict(label_dict[selected_element])
            else:
                print("Couldn’t find " + selected_element + " in the dict keys.")
                selected_dict = {}

            # Extract the shared elements
            common_elements = list(set(selected_dict.keys()) & set(target_dict.keys()))
            if len(common_elements) > 0:
                # for each shared element, the potential score is the
                # level of certainty in the element for each of the
                # images, multiplied together
                element_scores = [selected_dict[element] * target_dict[element] for element in common_elements]
                # We sum the contributions, and that’s it
                current_score += sum(element_scores) / best_score
            else:
                # If there are no shared elements,
                # we won’t add anything
                pass

            # We are done after scaling
        return current_score * 2 / (k + n)



    all_labels = data_module.training_images_and_tags.copy()
    all_labels.update(data_module.val_images_and_tags)

    #train_model(False)

    #results = run_model_get_results()



    def train(self, location='./train/'):
        """
        The training procedure is triggered here. OPTIONAL to run; everything that is required for testing the model
        must be saved to file (e.g., pickle) so that the test procedure can load, execute and report
        :param location: The location of the training data folder hierarchy
        :return: nothing
        """
        self.get_train_visual_predictions()
        self.train_model(saved=False)
        #pass






    def test(self, queries=list(), location='./test'):
        """
        Test your system with the input. For each input, generate a list of IDs that is returned
        :param queries: list of image-IDs. Each element is assumed to be an entry in the test set. Hence, the image
        with id <id> is located on my computer at './test/pics/<id>.jpg'. Make sure this is the file you work with...
        :param location: The location of the test data folder hierarchy
        :return: a dictionary with keys equal to the images in the queries - list, and values a list of image-IDs
        retrieved for that input
        """
        self.training_model.load_weights(self.main_filepath + 'last_save.h5')

        training_preds = self.get_train_visual_predictions()
        testing_preds = self.get_val_visual_predictions(False, location, queries, dict())

        result = self.models.run_validation(0.005, testing_preds, training_preds)

        my_return_dict = dict()
        for r in result:
            my_return_dict[r] = result[r][0]
        return my_return_dict









