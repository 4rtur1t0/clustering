"""
LAST VERSION TO COMPUTE BAG OF WORDS


https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76

https://betterprogramming.pub/a-friendly-guide-to-nlp-tf-idf-with-python-example-5fcb26286a33

https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/

"""
# import json
import numpy as np
import time
from cluster.hierarchical_clustering import HierarchicalClustering
# import matplotlib.pyplot as plt
# from cluster.distance_functions import hamming
from cluster.load_data import load_ORB_data, flatten_and_sampling
#import pickle
from config import EXPERIMENT_CONFIG


paths = EXPERIMENT_CONFIG.configurations.get('vocabulary_paths')


if __name__ == '__main__':
    max_index = None
    sampling = 1000
    # branching factor
    Kw = 3
    # depth level of the tree
    Lw = 6
    min_number_of_points_per_cluster = 100
    experiment_name = 'kw' + str(Kw)+'lw' + str(Lw)
    print('COMPUTING HIERARCHICAL CLUSTERING TO COMPUTE VOCABULARY TREE')
    print('PROCESSING THE FOLLOWING PATHS: ', paths)
    # load ORB DESCRIPTORS
    X_im = load_ORB_data(paths=paths)
    X = flatten_and_sampling(X_im, sampling=sampling, max_index=max_index)
    print('VOCABULARY FORMED FROM: ', len(X_im), ' images')
    print('TRYING TO CLUSTER: ', len(X), ' ORB descriptors')

    kmeans_params = {'tol': 0.5,
                     'max_iter': 300,
                     'distance_function': 'hamming',
                     'centroid_replacement': False,
                     'averaging_function': 'mean-round', # this is majority voting for binary descriptors if unpacked
                     'init_method': 'kmeans++',
                     'plot_progress': True}

    # X = generate_2D_data(n_samples=n_samples, centers=150)
    # kmeans_params = {'tol': 0.001,
    #                   'max_iter': 300,
    #                   'distance_function': 'euclidean',
    #                   'centroid_replacement': False,
    #                   'averaging_function': 'mean',  # this is majority voting for binary descriptors if unpacked
    #                   'init_method': 'kmeans++',
    #                   'plot_progress': False}
    start_time = time.time()
    hk = HierarchicalClustering(data=X, Kw=Kw, Lw=Lw,
                                kmeans_params=kmeans_params,
                                min_number_of_points_per_cluster=min_number_of_points_per_cluster)
    hk.fit()

    # plot
    print('INFO ON THE HIERARCHICAL TREE')
    print('NODES OF THE TREE')
    hk.print_tree_nodes(verbose=False)
    # plot only the names of leaf nodes
    print('LEAF NODES OF THE TREE')
    hk.print_leaf_nodes(verbose=False)
    # plot the data clustered hierarchically at different levels (graphics)
    # hk.plot_hierarchical_data()
    # plot leaf data (graphic)
    # hk.plot_leaf_data()
    hk.check_leaf_data(total_number=len(X))

    # compute the total inertia of the leaf nodes only!!!
    total_inertia = hk.compute_total_inertia()
    print('Total Inertia: ', total_inertia)
    print('Total number of leaf nodes: ', len(hk.get_leaf_nodes()))

    # EXPORT to a Vocabulary class to be saved
    vocabulary = hk.convert_to_vocabulary()
    # given the vocabulary tree, get the words as a list
    vocabulary.build_leaf_word_list()

    # find n documents where word i appears in the database and compute
    idf = vocabulary.compute_idf(X_im)
    # save IDF
    vocabulary.idf = idf
    mean_idf = np.mean(idf)
    cov_idf = np.cov(idf).item(0)
    print('Mean IDF: ', mean_idf)
    print('Cov IDF: ', cov_idf)
    #tf = vocabulary.compute_tf(X_im)

    #bow_vector = vocabulary.compute_bow_vector(tf, idf)
    print(30*'*')
    print('VOCABULARY TREE')
    print(30 * '*')
    vocabulary.print_tree_words(verbose=False)
    print(30 * '*')
    print('VOCABULARY LEAFS')
    print(30 * '*')
    vocabulary.print_leaf_words(verbose=False)
    print('SAVING VOCABULARY WITH PICKLE')
    vocabulary.save_vocabulary('voc_' + experiment_name + '.pkl')






