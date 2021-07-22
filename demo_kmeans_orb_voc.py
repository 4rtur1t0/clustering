"""

"""
import json
import numpy as np
import time

from sklearn.metrics import silhouette_score

from cluster.hkmeans import HKMeans
from cluster.kmeans import KMeans


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


if __name__ == '__main__':
    kw = 2
    lw = 4
    n_samples = 20000
    # Loading 1000 ORB descriptors from file
    X = load_data(1, n_samples)
    # in this case, the averaging makes sense if the bits are unpacked.
    # the average and discretization are a form of k-majority voting
    print("Converting to long binary descriptors")
    X = np.unpackbits(X, axis=1)

    kmeans_params = {'tol': 0.001, 'max_iter': 500,
                     'distance_function': 'euclidean',
                     'centroid_replacement': False,
                     'init_method': 'kmeans++',
                     'plot_progress': True}

    hk = HKMeans(kw=kw, lw=lw, kmeans_params=kmeans_params)
    hk.print_tree()

    start_time = time.time()
    hk.fit(X)

    # for each centroid, replace with the closest one at each level
    # given the new datapoints, assign them to each of the previous clusters
    # hk.reassign_datapoints()

    # # predict the same data to obtain the same labels
    # # starts comparing from the top of the tree
    # # may find some
    # new_labels, _ = hk.predict_top_down(X)
    # # reassign each data point to its closest centroid
    # new_labels, _ = hk.predict_brute_force(X)

    elapsed_time = time.time() - start_time
    hk.print_tree()
    hk.print_words()
    # for each centroid, replace with the closest existing datapoint
    #
    hk.replace_centroids(X)
    hk.print_words()

    hk.plot_tree_data(X)
    print("Clustering data with size: ", len(X))
    print('Trained a tree with (kw, lw): ', kw, ', ', lw)
    print('Total number of nodes: ', hk.get_number_of_nodes())
    print('Total number of leaves: ', hk.get_number_of_leaves())
    print('Total number of words expected: ', hk.get_expected_number_of_words())
    print('Total number of words found: ', hk.count_number_of_words())
    print('Total cost of fit (at last hierarchy level): ', hk.get_total_cost())
    print('Mean and variance in the number of datapoints per leaf (WORD): ', hk.get_n_datapoints_per_word())
    print('Fit time: ', elapsed_time, '(s)')
    print("Silhouette Coefficient: %0.3f" % silhouette_score(X, hk.leaf_labels, sample_size=n_samples))

    voc = hk.create_vocabulary(X)

    voc.save_json('voc.json')

    # print(hk.leaf_labels)
    # print(new_labels)


    plot_result(X, hk.leaf_labels, hk.leaf_centroids)