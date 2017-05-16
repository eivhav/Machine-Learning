
import pickle, random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine as cos
import os.path


class data_class():

    data_path = '/home/havikbot/'  # Please provide the main path here

    training_images_and_tags = dict()
    val_images_and_tags = dict()
    sub_folders_train = dict()
    sub_folders_val = dict()

    vocabulary = set()
    word_embeddings = dict()
    used_word_embeddings = dict()


    embeddings_file = 'glove.6B.300d.txt'
    word_embeddings_used_file = 'word_embeddings_used'

    def __init__(self, data_path):
        self.data_path = data_path



    def load_images_and_tags(self, type, nb_folders):
        if type == 'train':
            target_dict = self.training_images_and_tags
            sub_folders = self.sub_folders_train

        else:
            target_dict = self.val_images_and_tags
            sub_folders = self.sub_folders_val


        if nb_folders == 0:
            pickle_file = pickle.load(open(self.data_path + type +'/pickle/combined.pickle', 'rb'))
            for pic_id in pickle_file.keys():
                new_key = pic_id
                if len(pickle_file[pic_id]) > 0:
                    target_dict[new_key] = pickle_file[pic_id]
                    for label in pickle_file[pic_id]:
                        self.vocabulary.add(label[0])

        else:
            for i in range(nb_folders):
                index = '00000000' + str(i) + '00'
                pickle_file = pickle.load(open(self.data_path + type+'/pickle/descriptions' + index[-9:] + '.pickle', 'rb'))
                for pic_id in pickle_file.keys():
                    new_key = pic_id
                    sub_folders[new_key] = index[-9:] + '/'
                    if len(pickle_file[pic_id]) > 0:
                        target_dict[new_key] = pickle_file[pic_id]
                        for label in pickle_file[pic_id]:
                            self.vocabulary.add(label[0])

        print("ImageID and labels loaded.  Vocabulary size:", len(self.vocabulary))




    def init_label_embeddings(self):
        if os.path.isfile(self.data_path + self.word_embeddings_used_file) is True:
            self.used_word_embeddings = pickle.load(open(self.data_path + self.word_embeddings_used_file, 'rb'))
            print("Word embeddings loaded from stored embedding dict")
        else:
            print("No embeddings_file found. Initlizing from Glove")
            lines = open(self.data_path + self.embeddings_file).readlines()
            c = 0
            for line in lines:
                word = line.split(' ')[0]
                embs = line.split(' ')[1:]
                self.word_embeddings[word] = embs
                c += 1
                if c % 10000 == 0:
                    print(c)

            print("Word embeddings loaded from Glove")
            labels = list(self.vocabulary)
            for label in labels:
                if label in self.word_embeddings:
                    self.used_word_embeddings[label] = [float(v.strip()) for v in self.word_embeddings[label]]
                else:
                    sub_emb = [0] * 300
                    sub_label_num = 0
                    for sub_label in label.replace(',', '').replace(')', '').replace('(', '').replace('-', ' ').split(
                            " "):
                        if sub_label.strip() in self.word_embeddings:
                            values = self.word_embeddings[sub_label.strip()]
                            for i in range(len(values)):
                                sub_emb[i] += float(values[i].strip())
                            sub_label_num += 1
                    if sub_label_num > 0:
                        for j in range(len(sub_emb)): sub_emb[j] /= sub_label_num
                        self.used_word_embeddings[label] = sub_emb

            pickle.dump(self.used_word_embeddings, open(self.data_path + 'word_embeddings_used', 'wb'))

            print("Word embeddings file created", len(self.used_word_embeddings))


    def get_label_embedding(self, labels_with_weights, smooth_factor):
        embeddings = []
        weights_total = 0
        for label in labels_with_weights:
            label = (label[0], (float(label[1]) + smooth_factor) / (1 + smooth_factor))
            weights_total += label[1]

        for label in labels_with_weights:
            if label[0] in self.used_word_embeddings:
                values = self.used_word_embeddings[label[0]]
                embeddings.append([float(label[1]) * v for v in values])

        if len(embeddings) == 0: return None
        return np.mean(np.array(embeddings), axis=0)



    def get_negative_data(self, numpy_data, check_sim, limit):
        neg_data = numpy_data.copy()
        np.random.shuffle(neg_data)
        if check_sim:
            c = 0
            for i in range(len(neg_data)):
                sim = 1 - cos(neg_data[i], numpy_data[i])
                while sim > limit:
                    c += 1
                    neg_data[i] = neg_data[int(random.random()*np.shape(neg_data)[0])]
                    sim = 1 - cos(neg_data[i], numpy_data[i])

            print("Negative sample build. #Changes: ", c)

        return neg_data



    def get_data(self, data_type, visual_version, smooth_factor, save, nb_negs, sim_limit):
        vis_data = []
        label_data = []
        visual_predictions = pickle.load(open(self.data_path + 'visual_pred_'+visual_version, 'rb'))
        self.init_label_embeddings()
        for picture_id in visual_predictions.keys():
            if data_type == 'validate':
                embedding = self.get_label_embedding(self.val_images_and_tags[picture_id], smooth_factor)
            else:
                embedding = self.get_label_embedding(self.training_images_and_tags[picture_id], smooth_factor)

            if embedding is not None:
                if len(visual_predictions[picture_id].shape) == 2: vis_data.append(visual_predictions[picture_id][0])
                else: vis_data.append(visual_predictions[picture_id])
                label_data.append(embedding)

            if len(vis_data) % 5000 == 0: print("processed: ", len(vis_data),"/", len(visual_predictions.keys()))

        print("Loaded ", len(vis_data), " of ", data_type, " data.")

        print("Embeddings created. Total", len(self.used_word_embeddings))
        pickle.dump(self.used_word_embeddings, open(self.data_path + 'word_embeddings_used', 'wb'))

        if save is True:
            pickle.dump([np.array(vis_data), np.array(label_data)], open(self.data_path + data_type +'data_complete_'+visual_version, 'wb'))

        return [np.array(vis_data), np.array(label_data)] + \
               [self.get_negative_data(np.array(label_data), True, sim_limit) for i in range(nb_negs)]



    def get_data_from_pickle(self, data_type, versionid, nb_negs, sim_limit):
        data = pickle.load(open(self.data_path + data_type +'data_complete_'+versionid, 'rb'))
        return data + [self.get_negative_data(data[1], True, sim_limit) for i in range(nb_negs)]



    def save_visual_predicted_data(self, data_dict, version):
        pickle.dump(data_dict, open(self.data_path + 'visual_pred_'+version, 'wb'))


    def load_visual_predicted_data(self, version):
        return pickle.load(open(self.data_path + 'visual_pred_'+version, 'rb'))




