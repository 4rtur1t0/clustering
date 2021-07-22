"""

A


"""
import numpy as np
from cluster.distance_functions import compute_distance_function, find_closest
import json

class Vocabulary():
    """

    """
    def __init__(self, words, n_words, distance_function='euclidean'):
        # the words themselves in descriptor space
        self.words_vocabulary = words
        # number of each word as seen on training data
        self.n_words_vocabulary = n_words
        # the number of words divided by the total number of words
        self.histogram_words_vocabulary = n_words/np.sum(n_words)
        self.distance_function = distance_function

    def compute_bows(self, data, option='standard'):
        """
        Transform all descriptors in data to a set of Bag of Words vectors.

        Different options exist:
        - binary histogram. Considers the case in which each word exists or not in the data. The histogram is normalized
                            to unit length.
        - standard histogram. Considers the number of occurrences of each word
        - tf-idf scheme histogram. A weighted histogram

        Returns:
        - a bag of words vector in the form of a histogram.
        - the word associated to each datapoint in data.
        - the number of each of the words.

        :param data:
        :return:
        """
        # words and number of words in each image
        words, n_words, _ = self.get_similar_words(data)
        # compute binary
        if option == 'binary':
            bow_vector = self.compute_binary_histogram(words, n_words)
        # compute standard
        elif option == 'standard':
            bow_vector = self.compute_standard_histogram(words, n_words)
        # compute tf_idf
        elif option == 'tfidf':
            bow_vector = self.compute_tfidf_histogram(words, n_words)
        else:
            raise Exception('Unknown option!')

        return bow_vector, words, n_words


    def get_similar_words(self, data):
        """
        Return the closest words each of the datapoints in data.
        Using a brute-force scheme.

        """
        distance_mat = []
        N = len(data)
        words = -1*np.ones(N, dtype=np.uint32)
        n_datapoints_words = np.zeros(len(self.words_vocabulary), dtype=np.uint32)
        # find the distances of all datapoints to each cluster k
        # for each cluster, the operation computes the distance to all datapoints
        for k in range(len(self.words_vocabulary)):
            centroid_k = self.words_vocabulary[k]
            # distances of all datapoints to centroid k
            d = compute_distance_function(data, centroid_k, self.distance_function)
            distance_mat.append(d)
        distance_mat = np.array(distance_mat)
        datapoint_distances = np.zeros(N)
        # next, we select the min distance of each datapoint to each cluster
        for i in range(N):
            col = distance_mat[:, i]
            min_dist = np.amin(col)
            k_i = np.where(col == min_dist)
            words[i] = k_i[0][0]
            datapoint_distances[i] = min_dist
        # count existing datapoints for each word
        for i in range(len(self.words_vocabulary)):
            n_datapoints_words[i] = np.sum(words == i)
        return words, n_datapoints_words, datapoint_distances

    def replace_centroids(self, data):
        """
        Replace centroids.
        Recompute the number of datapoints belonging to each new centroid (WORD).
        Recompute the histogram.
        """
        for k in range(len(self.words_vocabulary)):
            data_k = find_closest(data, self.words_vocabulary[k], self.distance_function)
            self.words_vocabulary[k] = data_k
        # transform to the new words
        labels, n_datapoints_labels, _ = self.get_similar_words(data)
        self.n_words_vocabulary = n_datapoints_labels
        self.histogram_words_vocabulary = n_datapoints_labels/len(data)

    def save_json(self, filename='vocabulary.json'):
        """
        Save the vocabulary as a json file
        """
        data = {'words_vocabulary': self.words_vocabulary.tolist(),
                'n_words_vocabulary': self.n_words_vocabulary.tolist(),
                'distance_function': self.distance_function}
        with open(filename, 'w') as outfile:
            json.dump(data, outfile)

    def load_json(self, filename='vocabulary.json'):
        """
        Load the vocabulary from a json file
        """
        with open(filename) as json_file:
            data = json.load(json_file)
            self.words_vocabulary = data.get('words_vocabulary')
            self.n_words_vocabulary = data.get('n_words_vocabulary')
            self.distance_function = data.get('distance_function')

    def compute_binary_histogram(self, words, n_words):
        """
        Return a binary histogram. Returning 1 if the word exists in the image and else 0. The histogram is normalized to unit norm.
        """
        bow_vector_bin = n_words > 0
        bow_vector_bin = bow_vector_bin/len(n_words)
        return bow_vector_bin

    def compute_standard_histogram(self, words, n_words):
        """
        Return the likelihood of occurrence of each word.
        """
        bow_vector = n_words/len(words)
        return bow_vector

    def compute_tfidf_histogram(self, words, n_words):
        """
        Compute a tf-idf weight as
        ti = nid/nd)*log(N/ni)
        where:
            nid: número de veces que aparece la palabra i en la imagen d.
            nd: número de palabras en la imagen d
            N: número total de imágenes en el mapa.
            ni: número de veces que aparece la palabra i en la base de datos.

        Finally the vector is normalized to unit length.
        """
        # start computing nid/nd
        bow_vector = n_words / len(words)
        for i in range(len(self.n_words_vocabulary)):
            bow_vector[i] = bow_vector[i]*np.log(1/self.histogram_words_vocabulary[i])
        return bow_vector/np.sum(bow_vector)


