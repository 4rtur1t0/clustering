"""
functions to load descriptors from files

"""
from sklearn.datasets import make_blobs
from sklearn.manifold import Isomap, SpectralEmbedding, TSNE
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import json


def generate_2D_data(n_samples=250, centers=2):
    X, y = make_blobs(n_samples=n_samples,
                      centers=centers,
                      cluster_std=0.10,
                      random_state=0)
    plt.figure(0)
    colors = cycle(['r', 'g', 'b', 'k', 'c', 'm', 'y'])
    for k in range(centers):
        cluster_data = (y == k)
        plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=next(colors), marker='.', s=20)
    plt.title("Simulated data")
    plt.show(block=False)
    # plt.close(0)
    return X


def load_ORB_data(paths, mode='both', camera=None):
    all_descriptors = []
    # if paths is None:
    #     all_descriptors = read_descriptors('data/descriptors.json')
    # else:
    if mode == 'both':
        # read several paths
        for path in paths:
            # front and back are considered a single document and retrieved jointly
            descriptors = read_descriptors(path + '/mav0/cam0/descriptors.json')
            for desc in descriptors:
                X = np.array(desc, np.uint8)
                X = np.unpackbits(X, axis=1)
                all_descriptors.append(X)
            descriptors = read_descriptors(path + '/mav0/cam1/descriptors.json')
            for desc in descriptors:
                X = np.array(desc, np.uint8)
                X = np.unpackbits(X, axis=1)
                all_descriptors.append(X)
    else:
        # read several paths
        for path in paths:
            # front and back are considered a single document and retrieved jointly
            descriptors = read_descriptors(path + '/mav0/' + camera + '/descriptors.json')
            for desc in descriptors:
                X = np.array(desc, np.uint8)
                X = np.unpackbits(X, axis=1)
                all_descriptors.append(X)
    return all_descriptors


def read_descriptors(path):
    descriptors_input_path = path # + '/mav0/descriptors.json'
    with open(descriptors_input_path, 'r') as json_file:
        data = json.load(json_file)
        return data['descriptors']


def flatten_and_sampling(descriptors, sampling=None, max_index=None):
    # next, flatten list in order to obtain a flat list of ORB descriptors
    all_descriptors = flatten_list(descriptors)
    if max_index is None:
        max_index = len(all_descriptors)
    if sampling is None:
        sampling = 1
    all_descriptors = all_descriptors[0:max_index:sampling]
    all_descriptors = np.array(all_descriptors, np.uint8)
    # # finally, unpack descriptors to get binary descriptors
    # X = np.unpackbits(X, axis=1)
    return all_descriptors


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


# def plot_result(X, y, centroids):
#     # Plot init seeds along side sample data
#     plt.figure()
#     colors = cycle(['r', 'g', 'b', 'k', 'c', 'm', 'y'])
#     for k in set(y):
#         cluster_data = (y == k)
#         plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=next(colors), marker='.', s=50)
#         # plt.scatter(centroids[k][0], centroids[k][1], c='k', marker='.', s=200)
#     plt.title("RESULT! Data clustered")
#     plt.show()
#     return

# def plot_results_projecting(X, y, centroids):
#     from cluster.distance_functions import compute_hamming_distances
#     # Plot init seeds along side sample data
#     plt.figure()
#     colors = cycle(['r', 'g', 'b', 'k', 'c', 'm', 'y'])
#     on = np.unpackbits(np.array([255, 255, 255, 255, 255, 255, 255, 255,
#                                  255, 255, 255, 255, 255, 255, 255, 255,
#                                  255, 255, 255, 255, 255, 255, 255, 255,
#                                  255, 255, 255, 255, 255, 255, 255, 255], dtype=np.uint8))
#     off = np.unpackbits(np.array([0, 0, 0, 0, 0, 0, 0, 0,
#                                   0, 0, 0, 0, 0, 0, 0, 0,
#                                   0, 0, 0, 0, 0, 0, 0, 0,
#                                   0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8))
#     for k in set(y):
#         cluster_data = (y == k)
#         # project all data of that cluster to to two dimensions, using two hamming distances.
#         # d1 is the distance to [0  0 00 0 ]
#         # d2 is the distance to [11111]
#         d1 = compute_hamming_distances(X[cluster_data], on)
#         d2 = compute_hamming_distances(X[cluster_data], off)
#         # plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=next(colors), marker='.', s=50)
#         plt.scatter(d1, d2, c=next(colors), marker='.', s=50)
#         d1c = hamming(centroids[k], on)
#         d2c = hamming(centroids[k], off)
#         # plt.scatter(centroids[k][0], centroids[k][1], c='k', marker='.', s=200)
#         plt.scatter(d1c, d2c, c=next(colors), marker='*', s=50)
#         plt.annotate(str(k), (d1c, d2c))
#
#     plt.title("RESULT! Data clustered")
#     plt.show()
#     return
#
#
# def plot_results_isomap(X, y, centroids):
#     embedding = Isomap(n_components=2, metric='manhattan', p=1)
#     X_red = embedding.fit_transform(X)
#     plot_results(X_red, y, title='RESULT! Data clustered. Using ISOMAP CLUSTERING TO VISUALIZE')
#
#     # centroids_red = embedding.transform(centroids)
#     # colors = cycle(['r', 'g', 'b', 'k', 'c', 'm', 'y'])
#     #
#     # for k in set(y):
#     #     cluster_data = (y == k)
#     #     plt.scatter(X_red[cluster_data, 0], X_red[cluster_data, 1], c=next(colors), marker='.', s=50)
#     #     plt.scatter(centroids_red[k][0], centroids_red[k][1], c='k', marker='.', s=200)
#     # plt.title("RESULT! Data clustered. Using ISOMAP TO VISUALIZE")
#     # plt.show()


# def plot_results_spectral(X, y, centroids):
#     embedding = SpectralEmbedding(n_components=2)
#     X_red = embedding.fit_transform(X)
#     plot_results(X_red, y, title='RESULT! Data clustered. Using SPECTRAL CLUSTERING TO VISUALIZE')
#
#
#
# def plot_results_TSNE(X, y, centroids):
#     embedding = TSNE(n_components=2,
#                      init='pca',
#                      perplexity=500)
#     X_red = embedding.fit_transform(X)
#     plot_results(X_red, y, title='RESULT! Data clustered. Using tsne TO VISUALIZE')


# def plot_results(X, y, title):
#     colors = cycle(['r', 'g', 'b', 'k', 'c', 'm', 'y'])
#     markers = cycle(['.', '+', 'o', '*', '1', 'p', 's', 'x', 'X'])
#     strs = []
#     i = 0
#     for k in set(y):
#         cluster_data = (y == k)
#         color = next(colors)
#         marker = next(markers)
#         plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=color, marker=marker, s=50)
#         # plt.scatter(centroids_red[k][0], centroids_red[k][1], c='k', marker='.', s=200)
#         strs.append(str(i))
#         i += 1
#     plt.legend(strs)
#     plt.title(title)
#     plt.show()

#
# def precompute_distances(X, labels, output_size_ratio):
#     idx = np.random.choice(len(X), int(len(X)*output_size_ratio), replace=False)
#     X = X[idx]
#     labels = labels[idx]
#     # create a square distance matrix
#     D = np.zeros((len(X), len(X)))
#     for i in range(len(X)):
#         for j in range(len(X)):
#             a = X[i, :]
#             b = X[j, :]
#             d = hamming(a, b)
#             D[i, j] = d
#     return D, labels


# def save_bow_vectors(bow_vector, words, n_words, filename='bagofwords.json'):
#     """
#     Save the bag of words
#     """
#     data = {'bow_vector': bow_vector.tolist(),
#             'words': words.tolist(),
#             'n_words': n_words.tolist()}
#     with open(filename, 'w') as outfile:
#         json.dump(data, outfile)