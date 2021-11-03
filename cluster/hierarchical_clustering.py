"""
TODO:

"""
from itertools import cycle
import numpy as np
from cluster.kmeans import KMeans
import matplotlib.pyplot as plt


class HierarchicalClustering():
    def __init__(self, data, Lw, Kw, kmeans_params, min_number_of_points_per_cluster=10):
        """
        Start by populating the tree with a single node.
        The data is stored in self.data. All other methods use the indexes to refer to this data
        """
        self.data = data
        self.node_id_counter = 0
        data_indexes = np.arange(0, len(self.data))
        # create the root node, at level 1
        root_node = Node(name='node1_0', L=1, k=0,
                         node_id=self.node_id_counter,
                         n_clusters=Kw,
                         data_indexes=data_indexes,
                         centroid=None,
                         kmeans_params=kmeans_params)
        # mark this root node as open, so that it can grow other child nodes beneath
        root_node.mark_open()
        # start by creating a tree with all data indexes
        self.tree = Tree(root_node=root_node, kmeans_params=kmeans_params, Lw=Lw, Kw=Kw)
        self.kmeans_params = kmeans_params
        self.min_number_of_points_per_cluster = min_number_of_points_per_cluster
        self.Kw = Kw
        self.Lw = Lw

    def fit(self):
        """
        Iteratively find the nodes that are open and cluster them.
        The process is described below:
        1) Find nodes that are Open nodes.
        2) For each open node, fit the data using the kmeans algorithm and expand the tree. Each node is expanded
            by creating kw nodes that are their children and assigning their data indexes.
            Each node is expanded if it meets the following requirements:
                - their number of datapoints is above the min_numper_of_datapoints_per_cluster threshold.
                - the node has not reached the desired depth level Lw.
        3) Repeat at 1) until no more open node exist.
        """
        while True:
            # get all leaves that need further processing
            open_nodes = self.tree.get_open_nodes()
            if len(open_nodes) == 0:
                break
            for onode in open_nodes:
                print(str(onode))
                # data is selected internally at each node based on the indexes
                onode.fit(self.data)
                # close current node
                onode.mark_closed()
                self.expand_tree(onode)
                # self.print_tree_nodes()
        print("Ended Hierarchical fit process!!")

    def expand_tree(self, clustered_node):
        """
        Expands the tree by adding children to the clustered node.
        Children are added always but:
            - marked as closed if the number of points is below min_number_points (e.g. 50)
            - marked as open.

            If marked as open, the nodes will be further processed down.
            clustered_node is the parent and we are adding children to this node.
        """
        centroids = clustered_node.kmeans.centroids
        # for centroid in centroids:
        for k in range(len(centroids)):
            current_centroid = centroids[k]
            # check data indexes that have been labeled as k
            tf_indexes = (clustered_node.kmeans.labels == k)
            data_indexes_cluster_k = clustered_node.data_indexes[tf_indexes]
            n_datapoints_cluster_k = len(data_indexes_cluster_k)
            # the child node is at the next level. get level L  and kw of each node
            L_name = clustered_node.L + 1
            # k_name = (clustered_node.k)*self.Kw + k
            self.node_id_counter += 1
            # name = 'node' + str(L_name) + '_' + str(k_name)
            name = 'node' + str(L_name) + '_id_' + str(self.node_id_counter)
            # print(name)
            # add centroids
            # add the data indexes that belong to the node
            node = Node(name=name,
                        L=L_name,
                        k=k,
                        node_id=self.node_id_counter,
                        n_clusters=self.Kw,
                        data_indexes=data_indexes_cluster_k,
                        centroid=current_centroid,
                        kmeans_params=self.kmeans_params)
            # if there are enough datapoints to further cluster down and we have not reach the
            # depth level Lw
            if (n_datapoints_cluster_k > self.min_number_of_points_per_cluster) and (clustered_node.L < self.Lw):
                # mark nodes as open if really needed
                node.mark_open()
            # in all the other cases, do not add the node
            else:
                node.mark_closed()
            # if we have reached the desired depth level, mark as closed
            if clustered_node.L >= self.Lw:
                node.mark_closed()
                # continue
                # add the created node to the clustered_node if conditions meet
            clustered_node.add_child_node(node)

    def print_tree_nodes(self, verbose=False):
        current_level_nodes = []
        current_level_nodes.append(self.tree.root_node) # = root_node.get_child_nodes()
        print(30 * '___')
        while True:
            next_level_nodes = []
            for node in current_level_nodes:
                if verbose:
                    print('||' + str(node.name) + ', ndp: ' + str(node.n_datapoints) + ', centroid: ' + str(node.centroid), end='||')
                else:
                    print('||' + str(node.name) + ', ndp: ' + str(node.n_datapoints) + ' ' + node.state, end='||')
                next_level_nodes.extend(node.get_child_nodes())
            if len(next_level_nodes) == 0:
                print()
                print(30 * '___')
                return
            print()
            print(30*'___')
            current_level_nodes = next_level_nodes

    def get_leaf_nodes(self):
        return self.tree.get_leaf_nodes()

    def print_leaf_nodes(self, verbose=False):
        nodes = self.tree.get_leaf_nodes()
        print(30*'___')
        for node in nodes:
            if verbose:
                print('||' + str(node.name) + ', ndp: ' + str(node.n_datapoints) + ', centroid: ' + str(node.centroid) + '||')
            else:
                print('||' + str(node.name)  + ', ndp: ' + str(node.n_datapoints) + ' ' + node.state + '||')
        print(30 * '___')

    def plot_hierarchical_data(self):
        current_level_nodes = self.tree.root_node.get_child_nodes()
        # current_level_nodes.append(self.tree.root_node)  # = root_node.get_child_nodes()
        # current_level_nodes.append(self.tree.root_node.get_child_nodes())
        print(30 * '___')
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        iterator = cycle(colors)
        current_level = 1
        while True:
            next_level_nodes = []
            plt.figure()
            plt.title('Tree at current level: ' + str(current_level))
            for node in current_level_nodes:
                print(str(node))
                node.plot_data(self.data, color=next(iterator))
                next_level_nodes.extend(node.get_child_nodes())
            current_level += 1
            plt.show(block=True)
            if len(next_level_nodes) == 0:
                return
            current_level_nodes = next_level_nodes
            plt.close()

    def plot_leaf_data(self):
        nodes = self.tree.get_leaf_nodes()
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        iterator = cycle(colors)
        plt.figure()
        plt.title('Datapoints for each leaf: ')
        for node in nodes:
            print(str(node))
            node.plot_data(self.data, color=next(iterator))
        plt.show(block=True)

    def check_leaf_data(self, total_number):
        """
        Checks that all datapoints have been assigned to leafs
        """
        nodes = self.tree.get_leaf_nodes()
        n_datapoints = 0
        for node in nodes:
            print(str(node))
            n_datapoints += node.n_datapoints
        if n_datapoints == total_number:
            print('CHECK CORRECT: ALL DATAPOINTS ASSIGNED TO LEAFS!!!')
        else:
            print('ERROR, SOME DATAPOINTS NOT FOUND IN LEAFS')

    def compute_total_inertia(self):
        """
        Checks that all datapoints have been assigned to leafs
        """
        from cluster.distance_functions import compute_distance_function
        nodes = self.tree.get_leaf_nodes()
        total_inertia = 0
        for node in nodes:
            print(str(node))
            d = compute_distance_function(self.data[node.data_indexes],
                                          node.centroid,
                                          distance_function=node.kmeans.distance_function)
            total_inertia += np.sum(d)
        return total_inertia

    def convert_to_vocabulary(self):
        from cluster.vocabulary import VocabularyTree, Word
        root_node = self.tree.root_node
        # create vocabulary tree and root word
        # translate from node to Word
        root_word = Word(word_id=root_node.node_id,
                         word=root_node.centroid,
                         n_words_in_vocabulary=len(root_node.data_indexes))
        voc = VocabularyTree(root_word=root_word, distance_function=root_node.kmeans.distance_function)
        current_level_nodes = []
        current_level_words = []
        current_level_nodes.append(root_node)
        current_level_words.append(root_word)
        while True:
            next_level_nodes = []
            next_level_words = []
            k = 0
            for node in current_level_nodes:
                # print('||' + str(node.name) + ', ndp: ' + str(node.n_datapoints) + ', centroid: ' + str(node.centroid), end='||')
                child_nodes = node.get_child_nodes()
                next_level_nodes.extend(child_nodes)
                for ch in child_nodes:
                    word = Word(word_id=ch.node_id,
                                word=ch.centroid,
                                n_words_in_vocabulary=len(ch.data_indexes))
                    current_level_words[k].add_child_word(word)
                next_level_words.extend(current_level_words[k].get_child_words())
                k += 1
            if len(next_level_nodes) == 0:
                print()
                print(30 * '___')
                return voc
            print()
            print(30*'___')
            current_level_nodes = next_level_nodes
            current_level_words = next_level_words




class Tree():
    def __init__(self, root_node,  kmeans_params, name='tree', Lw=5, Kw=3):
        self.root_node = root_node
        self.name = name
        # branching factor
        self.Kw = Kw
        # depth level of tree
        self.Lw = Lw
        self.kmeans_params = kmeans_params

    def get_all_nodes(self):
        """
        gets all nodes in the tree without recursive functions.
        """
        all_nodes = []
        current_level = []
        # add root node
        current_level.append(self.root_node)
        while True:
            next_level_nodes = []
            for child in current_level:
                # add parent node
                all_nodes.append(child)
                # add children recursively
                next_level_nodes.extend(child.get_child_nodes())
            current_level = next_level_nodes
            if len(current_level) == 0:
                return all_nodes
        # return nodes

    def get_open_nodes(self):
        """
        gets only the open nodes
        (marked as open)
        """
        open_nodes = []
        nodes = self.get_all_nodes()
        for node in nodes:
            if node.state == 'OPEN':
                open_nodes.append(node)
        return open_nodes

    def get_closed_nodes(self):
        """
        gets the closed nodes
        (marks as closed)
        closed nodes should not be further processed down in the hierarchical clustering
        """
        closed_nodes = []
        nodes = self.get_all_nodes()
        for node in nodes:
            if node.state == 'CLOSE':
                closed_nodes.append(node)
        return closed_nodes

    def get_leaf_nodes(self):
        """
        Return leaf nodes:
            - nodes marked as closed
            - with no children nodes
        """
        leaf_nodes = []
        nodes = self.get_all_nodes()
        for node in nodes:
            if node.state == 'CLOSE' and not node.has_children():
                leaf_nodes.append(node)
        return leaf_nodes


class Node():
    def __init__(self, name, node_id, data_indexes, centroid, kmeans_params, L=0, k=0, n_clusters=2):
        self.name = name
        self.children = []
        # valid states are
        self.state = 'OPEN'
        # level and children number
        self.L = L
        self.k = k
        self.node_id = node_id
        self.kmeans = KMeans(k=n_clusters, **kmeans_params)
        self.data_indexes = data_indexes
        self.n_datapoints = len(data_indexes)
        self.centroid = centroid

    def add_child_node(self, obj):
        obj.L = self.L + 1
        # the children is at
        self.children.append(obj)

    def get_child_nodes(self):
        return self.children

    def has_children(self):
        if len(self.children) > 0:
            return True
        else:
            return False

    def fit(self, data):
        """
        find clusters in current node.
        IMPORTANT: each node clusters its own data, stored in data_indexes
        As a result, we obtain centroids along with their own data indexes
        """
        print('Clustering Node:' + str(self))
        self.kmeans.fit(data=data[self.data_indexes])

    def mark_open(self):
        self.state = 'OPEN'

    def mark_closed(self):
        self.state = 'CLOSE'

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

    def plot_data(self, data, color='r'):
        # if no data is assigned to this node --> skip
        if self.data_indexes is None:
            return
        X = data[self.data_indexes]
        plt.scatter(X[:, 0], X[:, 1], c=color, marker='.', s=20)
        plt.scatter(self.centroid[0], self.centroid[1], c='k', marker='.', s=150)

    def __str__(self):
        return "%s, level: %s, id: %s, npd: %s, state: %s" % (self.name, self.L, self.node_id,
                                                              len(self.data_indexes),  self.state)




