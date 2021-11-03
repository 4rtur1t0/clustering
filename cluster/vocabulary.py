"""
A Vocabulary class.
This is a clean implementation of a Tree.
Should be understood as very similar to the Tree used in hierarchical_clustering.py

"""
import pickle
import numpy as np
from cluster.distance_functions import compute_distance_function
import collections

class VocabularyTree():
    def __init__(self, root_word, distance_function):
        self.root_word = root_word
        self.distance_function = distance_function
        # the list of words as descriptors for brute force search
        self.leaf_word_list = []
        # the ids of the words
        self.leaf_word_id_list = []
        self.leaf_n_words_in_vocabulary_list = []
        # total number of words in vocabulary (
        self.N_words = None
        self.idf = []

    def get_all_words(self):
        """
        gets all nodes in the tree without recursive functions.
        """
        all_words = []
        current_level = []
        # add root node
        current_level.append(self.root_word)
        while True:
            next_level_nodes = []
            for child in current_level:
                # add parent node
                all_words.append(child)
                # add children recursively
                next_level_nodes.extend(child.get_child_words())
            current_level = next_level_nodes
            if len(current_level) == 0:
                return all_words

    def get_leaf_words(self):
        """
        Return leaf words:
            - nodes marked as closed
            - with no children nodes
        """
        leaf_words = []
        words = self.get_all_words()
        for word in words:
            if not word.has_children():
                leaf_words.append(word)
        return leaf_words

    def print_tree_words(self, verbose=False):
        current_level_words = []
        current_level_words.append(self.root_word)
        print(30 * '___')
        while True:
            next_level_words = []
            for word in current_level_words:
                if verbose:
                    print('|| word_id: ' + str(word.word_id) + ', ndp: ' + str(word.n_words_in_vocabulary) + ', word: '
                          + str(word.word), end='||')
                else:
                    print('|| word_id: ' + str(word.word_id) + ', ndp: ' + str(word.n_words_in_vocabulary), end='||')
                next_level_words.extend(word.get_child_words())
            if len(next_level_words) == 0:
                print()
                print(30 * '___')
                return
            print()
            print(30*'___')
            current_level_words = next_level_words

    def print_leaf_words(self, verbose=False):
        words = self.get_leaf_words()
        for word in words:
            if verbose:
                print('|| word_id: ' + str(word.word_id) + ', ndp: ' + str(word.n_words_in_vocabulary) + ', word: '
                      + str(word.word), end='||')
                print()
            else:
                print('|| word_id: ' + str(word.word_id) + ', ndp: ' + str(word.n_words_in_vocabulary), end='||')
                print()

    def build_leaf_word_list(self):
        words = self.get_leaf_words()
        for word in words:
            self.leaf_word_list.append(word.word)
            self.leaf_word_id_list.append(word.word_id)
            self.leaf_n_words_in_vocabulary_list.append(word.n_words_in_vocabulary)
        self.leaf_word_list = np.array(self.leaf_word_list, dtype=np.uint8)

    def project_to_vocabulary(self, descriptors, method='brute-force'):
        """
        transform a list of descriptors to a list of words.
        either, by looking with brute force on the leafs
        or by hierarchically going down the tree
        """
        if method == 'brute-force':
            word_ids_k, words_id = self.get_closest_words_brute_force(descriptors)
            # TODO: must count the number of words repeated in words
            # as well we have the number of words in the vocabulary
            # TODO: NEED THE TOTAL NUMBER OF WORDS?
        elif method == 'tree':
            # TODO: use hierarchical tree to find the closest words
            print('FINDING CLOSEST WORDS USING VOCABULARY TREE')
        else:
            print('Not implemented')
        # TODO: now compute histogram.. etc
        return word_ids_k, words_id

    def compute_bow_vector(self, descriptors, method='tf-idf'):
        """
        transform a list of descriptors to a list of words.
        either, by looking with brute force on the leafs
        or by hierarchically going down the tree
        """
        if method == 'brute-force':
            word_ids_k, words_id = self.get_closest_words_brute_force(descriptors)
            # TODO: must count the number of words repeated in words
            # as well we have the number of words in the vocabulary
            # TODO: NEED THE TOTAL NUMBER OF WORDS?
        elif method == 'tree':
            # TODO: use hierarchical tree to find the closest words
            print('FINDING CLOSEST WORDS USING VOCABULARY TREE')
        else:
            print('Not implemented')
        # TODO: now compute histogram.. etc
        return bow_vector

    def get_closest_words_brute_force(self, descriptors):
        """
        Compute the distance of each descriptor to each word.
        In this case, in order to speed things up, you can:
        compute the distance of each descriptor to all the words (using this)

        compute the distance of each word to all descriptors.
        """
        word_ids = []
        word_ids_k = []
        for descriptor in descriptors:
            d = compute_distance_function(self.leaf_word_list, descriptor, distance_function=self.distance_function)
            # find the min value, which corresponds to the word_id in word_id_list
            k = np.argmin(d)
            word_ids.append(self.leaf_word_id_list[k])
            word_ids_k.append(k)
        return word_ids_k, word_ids

    def compute_idf(self, images_descriptors):
        """
        Project all images to vocabulary. Compute:
        N: total number of documents (images)
        n_i: number of occurrences of word i in whole database.
        idf_i=log(N/n_i)
        rare words should get a higher weight whereas common words get a low weight

        CAUTION, INSTRUCTIONS:
        idf: input the database of images. idf computes the number of times that word i appears in the database over the
        total number of documents N.
        tf: input a single image. tf computes the ratio n_i/n where n_i is the total number of words i and n is the
        total number of words
        """
        print('Computing IDF')
        # number of documents
        N = len(images_descriptors)
        n_words_in_vocab = len(self.leaf_word_id_list)
        word_id_counter = np.zeros(n_words_in_vocab)
        i = 0
        print('Computing IDF: ')
        # counting the number of documents where word i appears
        for desc_in_image in images_descriptors:
            print('Percent complete: ', 100*i/N, '%', end='\r')
            # project to words, get a list of word_ids
            word_ids_k, words_id = self.get_closest_words_brute_force(desc_in_image)
            # compute logN/ni
            # caution, idf is computed after, computing if word i appears in n documents of the database.
            c = collections.Counter(word_ids_k)
            for key, val in c.items():
                # if val > 0, the word has appeared in this image
                if val > 0:
                    word_id_counter[key] += 1
            i += 1
        idf = np.zeros(n_words_in_vocab)
        for i in range(0, n_words_in_vocab):
            idf[i] = np.log((N + 1) / (word_id_counter[i] + 1)) + 1
        return idf

    def compute_tf(self, X_im):
        print('COMPUTING TF')
        # number of images to process
        N = len(X_im)
        n_words_in_vocab = len(self.leaf_word_id_list)
        # tf is a list of tf vectors for the list of input images
        tf = []
        i = 0
        for descriptors_in_image in X_im:
            print('Percent complete: ', 100*i/N, '%', end='\r')
            tfi = np.zeros(n_words_in_vocab)
            word_ids_k, words_id = self.get_closest_words_brute_force(descriptors_in_image)
            # find the ratio nid/nd for each word
            nd = len(descriptors_in_image)
            c = collections.Counter(word_ids_k)
            for key, val in c.items():
                tfi[key] = val/nd
            tf.append(tfi)
            i += 1
        return tf

    def compute_tf_idf(self, tf, idf):
        # idf to unit
        idf = idf/np.sum(idf)
        tf_idf_return = []
        print('COMPUTING TF-IDF')
        for tfi in tf:
            #tfi to unit
            tfi = tfi/np.sum(tfi)
            # multiply element by element
            tf_idf = np.multiply(tfi, idf)
            tf_idf = tf_idf/np.sum(tf_idf)
            tf_idf_return.append(tf_idf)
        return tf_idf_return

    def save_vocabulary(self, filename):
        filehandler = open(filename, 'wb+')
        pickle.dump(self, filehandler)


class Word():
    def __init__(self, word_id, word, n_words_in_vocabulary):
        self.word_id = word_id
        self.word = word
        self.n_words_in_vocabulary = n_words_in_vocabulary
        self.children = []

    def add_child_word(self, obj):
        # the children is at
        self.children.append(obj)

    def get_child_words(self):
        return self.children

    def has_children(self):
        if len(self.children) > 0:
            return True
        else:
            return False

    def __str__(self):
        return "word_id: %s, n_words_in_vocabulary: %s" % (self.word_id, self.n_words_in_vocabulary)

#
# class HistogramDescriptor():
#     def __init__(self):
#
#     def compute_hi

    # def plot_data(self, data, color='r'):
    #     # if no data is assigned to this node --> skip
    #     if self.data_indexes is None:
    #         return
    #     X = data[self.data_indexes]
    #     n_clusters = len(self.kmeans.centroids)
    #     for k in range(n_clusters):
    #         cluster_data = (self.kmeans.labels == k)
    #         plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=color, marker='.', s=20)
    #         plt.scatter(self.kmeans.centroids[k][0], self.kmeans.centroids[k][1], c='k', marker='.', s=150)

    # def plot_data(self, data, color='r'):
    #     # if no data is assigned to this node --> skip
    #     if self.data_indexes is None:
    #         return
    #     X = data[self.data_indexes]
    #     plt.scatter(X[:, 0], X[:, 1], c=color, marker='.', s=20)
    #     plt.scatter(self.centroid[0], self.centroid[1], c='k', marker='.', s=150)









# class Vocabulary():
#     """
#
#     """
#     def __init__(self, words, n_words, distance_function='euclidean'):
#         # the words themselves in descriptor space
#         self.words_vocabulary = words
#         # number of each word as seen on training data
#         self.n_words_vocabulary = n_words
#         # the number of words divided by the total number of words
#         self.histogram_words_vocabulary = n_words/np.sum(n_words)
#         self.distance_function = distance_function
#
#     def compute_bows(self, data, option='standard'):
#         """
#         Transform all descriptors in data to a set of Bag of Words vectors.
#
#         Different options exist:
#         - binary histogram. Considers the case in which each word exists or not in the data. The histogram is normalized
#                             to unit length.
#         - standard histogram. Considers the number of occurrences of each word
#         - tf-idf scheme histogram. A weighted histogram
#
#         Returns:
#         - a bag of words vector in the form of a histogram.
#         - the word associated to each datapoint in data.
#         - the number of each of the words.
#
#         :param data:
#         :return:
#         """
#         # words and number of words in each image
#         words, n_words, _ = self.get_similar_words(data)
#         # compute binary
#         if option == 'binary':
#             bow_vector = self.compute_binary_histogram(words, n_words)
#         # compute standard
#         elif option == 'standard':
#             bow_vector = self.compute_standard_histogram(words, n_words)
#         # compute tf_idf
#         elif option == 'tfidf':
#             bow_vector = self.compute_tfidf_histogram(words, n_words)
#         else:
#             raise Exception('Unknown option!')
#
#         return bow_vector, words, n_words
#
#
#     def get_similar_words(self, data):
#         """
#         Return the closest words each of the datapoints in data.
#         Using a brute-force scheme.
#
#         """
#         distance_mat = []
#         N = len(data)
#         words = -1*np.ones(N, dtype=np.uint32)
#         n_datapoints_words = np.zeros(len(self.words_vocabulary), dtype=np.uint32)
#         # find the distances of all datapoints to each cluster k
#         # for each cluster, the operation computes the distance to all datapoints
#         for k in range(len(self.words_vocabulary)):
#             centroid_k = self.words_vocabulary[k]
#             # distances of all datapoints to centroid k
#             d = compute_distance_function(data, centroid_k, self.distance_function)
#             distance_mat.append(d)
#         distance_mat = np.array(distance_mat)
#         datapoint_distances = np.zeros(N)
#         # next, we select the min distance of each datapoint to each cluster
#         for i in range(N):
#             col = distance_mat[:, i]
#             min_dist = np.amin(col)
#             k_i = np.where(col == min_dist)
#             words[i] = k_i[0][0]
#             datapoint_distances[i] = min_dist
#         # count existing datapoints for each word
#         for i in range(len(self.words_vocabulary)):
#             n_datapoints_words[i] = np.sum(words == i)
#         return words, n_datapoints_words, datapoint_distances
#
#     def replace_centroids(self, data):
#         """
#         Replace centroids.
#         Recompute the number of datapoints belonging to each new centroid (WORD).
#         Recompute the histogram.
#         """
#         for k in range(len(self.words_vocabulary)):
#             data_k = find_closest(data, self.words_vocabulary[k], self.distance_function)
#             self.words_vocabulary[k] = data_k
#         # transform to the new words
#         labels, n_datapoints_labels, _ = self.get_similar_words(data)
#         self.n_words_vocabulary = n_datapoints_labels
#         self.histogram_words_vocabulary = n_datapoints_labels/len(data)
#
#     def save_json(self, filename='vocabulary.json'):
#         """
#         Save the vocabulary as a json file
#         """
#         data = {'words_vocabulary': self.words_vocabulary.tolist(),
#                 'n_words_vocabulary': self.n_words_vocabulary.tolist(),
#                 'distance_function': self.distance_function}
#         with open(filename, 'w') as outfile:
#             json.dump(data, outfile)
#
#     def load_json(self, filename='vocabulary.json'):
#         """
#         Load the vocabulary from a json file
#         """
#         with open(filename) as json_file:
#             data = json.load(json_file)
#             self.words_vocabulary = data.get('words_vocabulary')
#             self.n_words_vocabulary = data.get('n_words_vocabulary')
#             self.distance_function = data.get('distance_function')
#
#     def compute_binary_histogram(self, words, n_words):
#         """
#         Return a binary histogram. Returning 1 if the word exists in the image and else 0. The histogram is normalized to unit norm.
#         """
#         bow_vector_bin = n_words > 0
#         bow_vector_bin = bow_vector_bin/len(n_words)
#         return bow_vector_bin
#
#     def compute_standard_histogram(self, words, n_words):
#         """
#         Return the likelihood of occurrence of each word.
#         """
#         bow_vector = n_words/len(words)
#         return bow_vector
#
#     def compute_tfidf_histogram(self, words, n_words):
#         """
#         Compute a tf-idf weight as
#         ti = nid/nd)*log(N/ni)
#         where:
#             nid: número de veces que aparece la palabra i en la imagen d.
#             nd: número de palabras en la imagen d
#             N: número total de imágenes en el mapa.
#             ni: número de veces que aparece la palabra i en la base de datos.
#
#         Finally the vector is normalized to unit length.
#         """
#         # start computing nid/nd
#         bow_vector = n_words / len(words)
#         for i in range(len(self.n_words_vocabulary)):
#             bow_vector[i] = bow_vector[i]*np.log(1/self.histogram_words_vocabulary[i])
#         return bow_vector/np.sum(bow_vector)


