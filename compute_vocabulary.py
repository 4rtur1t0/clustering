"""
LAST VERSION TO COMPUTE BAG OF WORDS
"""
import json
from itertools import cycle
import numpy as np
import time
from sklearn.metrics import silhouette_score
from cluster.hkmeans import HKMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, SparsePCA


def read_descriptors(path):
    descriptors_input_path = path # + '/mav0/descriptors.json'
    with open(descriptors_input_path, 'r') as json_file:
        data = json.load(json_file)
        desc1 = data['descriptors_front']
        desc2 = data['descriptors_back']
        # extend list 1
        desc1.extend(desc2)
        return desc1


def flatten_list(all_descriptors):
    """
    Flattens list of descriptors: original list is of size
        n_landmarks list with [1500 x 32] in each
    """
    ret_descriptors = []
    for descriptors_image in all_descriptors:
        for desc in descriptors_image:
            ret_descriptors.append(desc)
    return ret_descriptors


def plot_result(X, y, centroids):
    # Plot init seeds along side sample data
    plt.figure(1)
    colors = cycle(['r', 'g', 'b', 'k', 'c', 'm', 'y'])
    for k in set(y):
        cluster_data = (y == k)
        plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=next(colors), marker='.', s=50)
        # plt.scatter(centroids[k][0], centroids[k][1], c='k', marker='.', s=200)
    plt.title("RESULT! Data clustered")
    plt.show()
    return X, y


def load_data(sampling=None, max_index=None):
    all_descriptors = read_descriptors('data/descriptors.json')
    all_descriptors = flatten_list(all_descriptors)
    if max_index is None:
        max_index = len(all_descriptors)
    if sampling is None:
        sampling = 1
    all_descriptors = all_descriptors[0:max_index:sampling]
    X = np.array(all_descriptors, np.uint8)
    return X


def save_bow_vectors(bow_vector, words, n_words, filename='bagofwords.json'):
    """
    Save the bag of words
    """
    data = {'bow_vector': bow_vector.tolist(),
            'words': words.tolist(),
            'n_words': n_words.tolist()}
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    n_samples = 5000
    kw = 3
    lw = 2
    # load ORB DESCRIPTORS
    X = load_data(sampling=1, max_index=n_samples)
    X = np.unpackbits(X, axis=1)

    kmeans_params = {'tol': 0.001,
                     'max_iter': 300,
                     'distance_function': 'hamming',
                     'centroid_replacement': True,
                     'averaging_function': 'mean-round', # this is majority voting for binary descriptors if unpacked
                     'init_method': 'kmeans++',
                     'plot_progress': True}
    start_time = time.time()
    hk = HKMeans(kw=kw, lw=lw, kmeans_params=kmeans_params)
    hk.print_tree()
    hk.fit(X)
    elapsed_time = time.time() - start_time
    hk.print_tree()
    hk.print_words()
    # hk.plot_tree_data(X)
    print('Clustering took: ', elapsed_time, '(s)')

    print('Finding predictions via brute-force:')
    start_time = time.time()
    new_labels, _ = hk.predict_brute_force(X)
    elapsed_time = time.time() - start_time
    print('Took: ', elapsed_time, '(s)')
    print('Done predicting brute Force. found: ', len(set(new_labels)), ' different labels')

    print('Finding predictions via top-down:')
    start_time = time.time()
    new_labels, _ = hk.predict_top_down(X)
    elapsed_time = time.time() - start_time
    print('Took: ', elapsed_time)
    print('Done predicting top down. found: ', len(set(new_labels)), ' different labels')

    print("Clustered ", len(X), " n descriptors")
    print('Trained a tree with (kw, lw): ', kw, ', ', lw)
    print('Total number of nodes: ', hk.get_number_of_nodes())
    print('Total number of leaves: ', hk.get_number_of_leaves())
    print('Total number of words expected: ', hk.get_expected_number_of_words())
    print('Total number of words found: ', hk.count_number_of_words())
    print('Total cost of fit (at last hierarchy level): ', hk.get_total_cost())
    print('Number of datapoints per leaf (WORD): ', hk.get_n_datapoints_per_word())
    print('Fit time: ', elapsed_time, '(s)')
    # print("Sum of errors in classification (a small number is due to the approximation of the top-down classification): ", np.sum(np.abs(hk.leaf_labels - new_labels)))
    # print("Silhouette Coefficient: %0.3f" % silhouette_score(X, hk.leaf_labels, sample_size=n_samples))
    print("Silhouette Coefficient of tree classification: %0.3f" % silhouette_score(X, new_labels, sample_size=n_samples))


    # Xred = SparsePCA(n_components=2).fit_transform(X)

    # plot_result(Xred, new_labels, hk.leaf_centroids)

    print('Creating vocabulary object!')
    # create the vocabulary!!!
    voc = hk.create_vocabulary()
    print('Replacing centroids by true prototypes')
    # optionally, replace each original word by the closest, real, existing descripotr
    # no need to replace centroids if centroid_replacement=True
    # voc.replace_centroids(X)
    print('Saving vocabulary')
    voc.save_json('voc.json')
    print('Computing BOW vectors')
    bow_vector_st, words, n_words = voc.compute_bows(X, option='standard')
    print(bow_vector_st)
    # bow_vector_bin, words, n_words = voc.compute_bows(X, option='binary')
    # bow_vector_tfidf, words, n_words = voc.compute_bows(X, option='tfidf')

    # save bag of word vectors
    save_bow_vectors(bow_vector_st, words, n_words, 'bagofwords.json')






