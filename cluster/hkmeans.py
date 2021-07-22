"""
TODO:

"""
from itertools import cycle
import numpy as np
from cluster.kmeans import KMeans
import matplotlib.pyplot as plt
from cluster.distance_functions import compute_distance_function
from cluster.vocabulary import Vocabulary


class Node():
    def __init__(self, id, n_clusters, kmeans_params):
        self.children = []
        self.parent = None
        # self.data = data
        self.indexes = None
        self.labels = None
        self.centroids = {}
        self.id = id
        self.leaf_id = None
        self.node_type = 'node'
        self.n_clusters = n_clusters
        self.kmeans_params = kmeans_params
        self.cost = None
        self.kmeans = None

    def init_indexes(self, data):
        self.indexes = np.arange(0, len(data))

    def add_child(self, obj):
        """
        Each node has a list of children
        """
        self.children.append(obj)

    def add_parent(self, obj):
        """
        Each Node has a single parent.
        """
        self.parent = obj

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children

    def get_n_datapoints(self):
        return len(self.indexes)

    def plot_data(self, data, color):
        # if no data is assigned to this node --> skip
        if self.indexes is None:
            return
        # plt.figure(1)
        X = data[self.indexes]
        for k in range(self.n_clusters):
            # col = colors[k]
            cluster_data = (self.labels == k)
            plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=color, marker='.', s=20)
            plt.scatter(self.centroids[k][0], self.centroids[k][1], c='k', marker='.', s=150)
        # plt.title("Data clustered at level" + str(level))
        # plt.show()

    def fit(self, data):
        # in this case, data has not been propagated to this node and no clustering needs to be performed
        # the node is not deleted
        if self.indexes is None:
            return
        # obtain the data that corresponds to the current node
        data = data[self.indexes]
        # check whether there are more clusters than data. In this case the clustering makes no sense and all datapoints
        # are selected as centroid with n_clusters = len(data)
        print('Clustering len(data)', len(data))
        if self.n_clusters > len(data):
            print('WARNING: cannot find more clusters than number of data.')
            print('WARNING: setting all datapoints as clusters.')
            self.n_clusters = len(data)
        km = KMeans(k=self.n_clusters,
                    tol=self.kmeans_params.get('tol', 0.001),
                    max_iter=self.kmeans_params.get('max_iter', 300),
                    distance_function=self.kmeans_params.get('distance_function', 'euclidean'),
                    centroid_replacement=self.kmeans_params.get('centroid_replacement', False),
                    averaging_function=self.kmeans_params.get('averaging_function', None),
                    init_method=self.kmeans_params.get('init_method', 'kmeans++'),
                    plot_progress=self.kmeans_params.get('plot_progress'))
        # fit current node with the data given
        km.fit(data)
        self.kmeans = km
        self.centroids = km.centroids
        self.labels = km.labels
        self.cost = km.cost
        # once clustered, propagate the data_indexes to this node's children
        # there are kw children of this node
        # caution, in the case where n_clusters is less than kw, some of the children nodes will not have data assigned
        # no subsequent clustering is performed for these Nodes
        self.propagate_indexes()

    def propagate_indexes(self):
        # if this is a leaf node, then finish, do not propagate indexes below
        if self.node_type == 'leaf':
            return

        for k in range(self.n_clusters):
            tf_indexes = (self.labels == k)
            self.children[k].indexes = self.indexes[tf_indexes]


class HKMeans():
    """
    Create a tree with branching factor kw with levels lw
    The tree stores the data
    """
    def __init__(self, kmeans_params, name='root', kw=2, lw=3):
        self.kw = kw
        self.Lw = lw
        self.name = name
        self.parent = None
        self.tree = {}
        self.leaf_centroids = None
        # self.leaf_labels = None
        self.kmeans_params = kmeans_params
        self.create_tree()

    def create_tree(self):
        node_id = 1
        leaf_node_id = 0
        for level in range(0, self.Lw+1):
            nodes = []
            # create nodes and assign each node to the parent at the previous level
            for k in range(np.power(self.kw, level)):
                # create current node
                curr_node = Node(str(node_id), n_clusters=self.kw, kmeans_params=self.kmeans_params)
                # add parent and childrens to the nodes
                if level > 0:
                    parent_index = int(np.floor(k / self.kw))
                    # select parent from previous level
                    parent = self.tree[level - 1][parent_index]
                    # select parent
                    curr_node.add_parent(parent)
                    # add children
                    parent.add_child(curr_node)
                # the root node has no parent
                if level == 0:
                    curr_node.node_type = 'root'
                # tag as leaf the last nodes
                if level == self.Lw:
                    curr_node.node_type = 'leaf'
                    curr_node.leaf_id = leaf_node_id
                    leaf_node_id += 1
                # save to the list
                nodes.append(curr_node)
                node_id += 1
            self.tree[level] = nodes

    def fit(self, data):
        """
        cluster all levels
        """
        # init indexes at root. The root node possesses all indexes of the whole dataset
        root = self.tree[0][0]
        root.init_indexes(data)
        # cluster the top levels. The last level (the leafs) receive the clustered indexes from the previous level.
        for level in range(self.Lw):
            print('Processing tree level: ', level, ' out of', self.Lw)
            nodes = self.tree[level]
            # cluster each of the nodes at this tree level
            for node in nodes:
                node.fit(data)
        # self.save_leaf_labels(data)
        self.save_centroids()
        print('Finished!')

    def predict(self, data, option='topdown'):
        """
        Predicts labels: two options are available.
                top down: traverses the tree from top to bottom. At each level, the next node is selected as the one that
                          is closest to its centroid. The process is repeated until we reach the leaf nodes.
                brute force: compares each datapoint to each of the leaf nodes and select the closest one
        """
        if option == 'topdown':
            return self.predict_top_down(data)
        elif option == 'bruteforce':
            return self.predict_brute_force(data)
        else:
            raise Exception('Unknown prediction method!')

    def predict_top_down(self, data):
        """
        Returns the label of the closest centroid of the leaves.
        predict works in a top down fashion. Starts at the top node and works down.
        This reduces the number of comparisons that need to be performed, however, the results are approximate.
        """
        labels = -1*np.ones(len(data), dtype=int)
        distances = np.zeros(len(data))
        for i in range(len(data)):
            # for every data start at the tree top and go down
            node = self.tree[0][0]
            while True:
                # print('Predict')
                label, distance = node.kmeans.predict(data[i])
                # switch to node below, according to the prediction
                node = node.children[label[0]]
                if node.kmeans is None:
                    labels[i] = node.leaf_id
                    distances[i] = distance[0]
                    break
        return labels, distances

    def predict_top_down_bis(self, data):
        """
        Returns the label of the closest centroid of the leaves.
        predict works in a top down fashion. Starts at the top node and works down.
        This reduces the number of comparisons that need to be performed, however, the results are approximate.
        """
        labels = -1*np.ones(len(data), dtype=int)
        distances = np.zeros(len(data))

        for i in range(self.Lw):
            # for every data start at the tree top and go down
            nodes = self.tree[i]
            for node in nodes:
                node.kmeans.predict(data)
                # the key is to propagate indexes
                node.propagate_indexes()
        # TODO: MUST RETURN labels of data
        return labels, distances


    def predict_brute_force(self, data):
        """
        Compare each datapoint to one of the centroids.
        In order to speed up the process, we compare all data
        """
        distance_mat = []
        N = len(data)
        labels = -1 * np.ones(N, dtype=np.int64)
        # find the distances of all datapoints to each cluster k
        # for each cluster, the operation computes the distance to all datapoints
        for k in range(len(self.leaf_centroids)):
            centroid_k = self.leaf_centroids[k, :]
            # distances of all datapoints to centroid k
            d = compute_distance_function(data, centroid_k, self.kmeans_params.get('distance_function'))
            distance_mat.append(d)
        distance_mat = np.array(distance_mat)
        datapoint_distances = np.zeros(N)
        # next, we select the min distance of each datapoint to each cluster
        for i in range(N):
            col = distance_mat[:, i]
            min_dist = np.amin(col)
            k_i = np.where(col == min_dist)
            labels[i] = k_i[0][0]
            datapoint_distances[i] = min_dist
        return labels, datapoint_distances

    # def replace_centroids(self, data):
    #     """
    #     For each centroid in the data, look for the closest datapoint and replace it.
    #     Also reassign all datapoints to the new centroids!
    #     """
    #     nodes = self.tree[self.Lw-1]
    #     for node in nodes:
    #         node.kmeans.centroid_replacement = True
    #         node.kmeans.update_centroids(data)
    #         node.labels, _ = node.predict(data, option='bruteforce')
    #         node.propagate_indexes()
    #     self.save_leaf_labels(data)
    #     self.save_centroids()


    def print_tree(self):
        for level in range(self.Lw+1):
            nodes = self.tree[level]
            print(30*"_")
            for node in nodes:
                try:
                    print(node.id, '  (', node.node_type, ')  ', end='')
                    print('(', node.get_n_datapoints(), ') ', end='')
                except TypeError:
                    continue
            print()
            print(30 * "_")

    def print_centroids(self):
        """
        Print_words and print_centroids do the same.
        """
        self.print_words()

    def print_words(self):
        """
        Print words and centroids that have been saved as self.leaf_centroids
        """
        # print, now print the leaves (leaves are the centroids of the last kmeans nodes.
        print("WORDS: (centroids at the last depth level)")
        for k in range(len(self.leaf_centroids)):
            print('WORD: ', k)
            print(self.leaf_centroids[k, :])

    def save_leaf_labels(self, data):
        """
        Assign labels to every datapoint in data.
        We do not use the middle nodes for classification and stick to the leaf nodes.
        """
        self.leaf_labels = -1*np.ones(len(data), dtype=int)
        node_leaves = self.tree[self.Lw]
        for node_leaf in node_leaves:
            current_label = node_leaf.leaf_id
            self.leaf_labels[node_leaf.indexes] = current_label

    def save_centroids(self):
        # nodes at the upper level
        pre_node_leaves = self.tree[self.Lw-1]
        centroids = []
        for node_leaf in pre_node_leaves:
            n_centroids = len(node_leaf.centroids)
            for k in range(n_centroids):
                centroids.append(node_leaf.centroids[k])
        self.leaf_centroids = np.array(centroids)

    def plot_tree_data(self, X):
        colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y']
        iterator = cycle(colors)
        for level in range(self.Lw):
            plt.figure(level)
            nodes = self.tree[level]
            for node in nodes:
                node.plot_data(X, color=next(iterator))
            plt.title('Tree at level ' + str(level))
        # finally plot the words
        # for k in range(len(self.leaf_centroids)):
        #     plt.scatter(self.leaf_centroids[k, 0], self.leaf_centroids[k, 1], c='b', marker='o', s=80)
        plt.show()

    def get_number_of_nodes(self):
        n_nodes = 0
        for level in range(self.Lw+1):
            n_nodes += len(self.tree[level])
        return n_nodes

    def get_number_of_leaves(self):
        n_leaves = len(self.tree[self.Lw])
        return n_leaves

    def get_expected_number_of_words(self):
        n_words = np.power(self.kw, self.Lw)
        return n_words

    def count_number_of_words(self):
        return len(self.leaf_centroids)

    def get_n_datapoints_per_word(self):
        """
        Get the number of datapoints associated to each of the words.
        """
        words = self.tree[self.Lw]
        datapoints = []
        for word in words:
            try:
                datapoints.append(len(word.indexes))
            except:
                datapoints.append(0)

        datapoints = np.array(datapoints)
        return datapoints

    def get_total_cost(self):
        leaves = self.tree[self.Lw - 1]
        cost = 0
        for leaf in leaves:
            if leaf.cost is None:
                continue
            cost += leaf.cost
        return cost

    def create_vocabulary(self):
        """
            Returns the list of words and the number of datapoints
        """
        words, n_words = self.get_words()
        return Vocabulary(words=words, n_words=n_words, distance_function=self.kmeans_params.get('distance_function'))

    def get_words(self):
        """
        Returns the list of words and the number of datapoints
        """
        words = self.leaf_centroids
        n_words = self.get_n_datapoints_per_word()
        return words, n_words





