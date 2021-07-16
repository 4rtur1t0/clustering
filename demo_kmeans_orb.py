"""

A demo of the k-means with ORB descriptors computed on a set of images.

In this case, a hierarchical clustering is performed.

Using:
    - Binary ORB descriptors numbers.
    - A Hamming distance function to compute the distance between two descriptors.
    - An average mean function to find each cluster center.

    Best results are obtained when using the k-means++ initialization method.
    Use the option plot_progress=True to view the movement and assignation of the centroids at each algorithm step.

    Canal de métodos de clustering:
https://www.youtube.com/watch?v=AmiuQVw0SoM

Revisar método de la silueta
https://link.springer.com/article/10.1186/s12862-018-1163-8

MIrar libros
https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470316801.ch2

https://books.google.es/books?id=bzMzUHyYEBQC&pg=PA179&lpg=PA179&dq=k+medoids+majority+voting&source=bl&ots=WpZVFnHeUC&sig=ACfU3U2yokPDxJeuIFVhbYbAR5xr1CnWug&hl=es&sa=X&ved=2ahUKEwij1e2H9MTxAhWQsRQKHYqIBQ4Q6AF6BAgfEAM#v=onepage&q=k%20medoids%20majority%20voting&f=false

"""
import json
import numpy as np
import time
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


# def convert_to_bin(data):
#     print("Converting to long binary descriptors")
#     # data_bin = []
#     data_bin = np.unpackbits(data, axis=1)
#     # for datapoint in data:
#     #     datapoint_bin = np.array([])
#     #
#     #     # converts every feature in the ORB descriptor from int to an 8-bits binary representation
#     #     for feature in datapoint:
#     #         d = np.binary_repr(feature, width=8)
#     #         d = d.replace("", " ")[1: -1]
#     #         d_bin = np.fromstring(d, dtype=np.uint8, sep=' ')
#     #         datapoint_bin = np.append(datapoint_bin, d_bin)
#     #     data_bin.append(datapoint_bin)
#     # data_bin = np.array(data_bin, dtype=np.uint8)
#     print("End conversion")
#     return data_bin


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
    n_clusters = 200

    # Loading 1000 ORB descriptors from file
    X = load_data(1, 20000)
    # in this case, the averaging makes sense if the bits are unpacked.
    # the average and discretization are a form of k-majority voting
    print("Converting to long binary descriptors")
    X = np.unpackbits(X, axis=1)


    clf = KMeans(k=n_clusters, tol=0.001, max_iter=500,
                 distance_function='hamming',
                 averaging_function='mean-round',
                 centroid_replacement=False,
                 init_method='kmeans++',
                 plot_progress=False)
    print("Using data size: ", len(X))
    start_time = time.time()
    clf.fit(X)
    # clf.encode_centroids()
    elapsed_time = time.time() - start_time
    print('Fit time: ', elapsed_time, '(s)')
    print("Current cost is: ", clf.cost)
    print("Centroids are: ")
    # for c in clf.centroids:
    #     print(clf.centroids[c])

    # pack data and also the resulting centroids
    X = np.packbits(X, axis=1)
    clf.pack_centroids()

    # classify and plot
    clf.plot(X, 'Final result!')



    # clf.plot(X, 'Final result!')

    # distance_mat = np.zeros((20, 20))
    # for i in range(n_clusters):
    #     for j in range(n_clusters):
    #         centroid_i = clf.centroids[i]
    #         centroid_j = clf.centroids[j]
    #         d = np.linalg.norm(centroid_i-centroid_j)
    #         distance_mat[i, j] = d

    # print(distance_mat)
