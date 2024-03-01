import os
import pickle
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm
import itertools
import networkx as nx
import dgl
import torch
import torch.utils.data
from scipy.spatial.distance import cdist
from scipy import sparse as sp
import time
import hashlib
import csv
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

import pdb
from collections import Counter


def sigma(dists, kth=8):
    # Compute sigma and reshape
    try:
        # Get k-nearest neighbors for each node
        knns = np.partition(dists, kth, axis=-1)[:, kth::-1]
        sigma = knns.sum(axis=1).reshape((knns.shape[0], 1)) / kth
    except ValueError:  # handling for graphs with num_nodes less than kth
        num_nodes = dists.shape[0]
        # this sigma value is irrelevant since not used for final compute_edge_list
        sigma = np.array([1] * num_nodes).reshape(num_nodes, 1)

    return sigma + 1e-8  # adding epsilon to avoid zero value of sigma


def compute_adjacency_matrix_images(coord, feat, use_feat=True, kth=8):
    coord = coord.reshape(-1, 2)
    # Compute coordinate distance
    c_dist = cdist(coord, coord)
    # complex1 = torch.complex(feat, feat)
    if use_feat:
        # Compute feature distance
        # complex1 = complex1.reshape(-1, 1)
        f_dist = cdist(feat.unsqueeze(1), feat.unsqueeze(1))
        # Compute adjacency
        A = np.exp(- (c_dist / sigma(c_dist)) ** 2 - (f_dist / sigma(f_dist)) ** 2)
    else:
        A = np.exp(- (c_dist / sigma(c_dist)) ** 2)

    # Convert to symmetric matrix
    A = 0.5 * (A + A.T)
    A[np.diag_indices_from(A)] = 0
    return A


def compute_edges_list(A, kth=4 + 1):
    # Get k-similar neighbor indices for each node

    num_nodes = A.shape[0]
    new_kth = num_nodes - kth

    if num_nodes > 9:
        knns = np.argpartition(A, new_kth - 1, axis=-1)[:, new_kth:-1]
        knn_values = np.partition(A, new_kth - 1, axis=-1)[:, new_kth:-1]  # NEW
    else:
        # handling for graphs with less than kth nodes
        # in such cases, the resulting graph will be fully connected
        knns = np.tile(np.arange(num_nodes), num_nodes).reshape(num_nodes, num_nodes)
        knn_values = A  # NEW

        # removing self loop
        if num_nodes != 1:
            knn_values = A[knns != np.arange(num_nodes)[:, None]].reshape(num_nodes, -1)  # NEW
            knns = knns[knns != np.arange(num_nodes)[:, None]].reshape(num_nodes, -1)
    return knns, knn_values  # NEW


class ST_PixDGL(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir,
                 shuffle=False,
                 use_mean_px=True,
                 use_coord=True):
        self.graph_lists = []
        with open(data_dir, 'rb') as f:
        # with open('dataset_predict.pkl', 'rb') as f:  #############################0000000000000
            self.label, self.data_total, self.graph_data = pickle.load(f)
        self._prepare()
        # shuffle 数据集
        if shuffle:
            combined_data = list(zip(self.data_total, self.label, self.graph_data))
            random.shuffle(combined_data)
            shuffled_data, shuffled_labels, shuffled_graphs = zip(*combined_data)
            self.label, self.data_total, self.graph_data = shuffled_labels, shuffled_data, shuffled_graphs
         # self.data_total = pickle.load(f)

    def _prepare(self):
        self.Adj_matrices, self.node_features, self.edges_lists, self.edge_features = [], [], [], []

        for index, sample in enumerate(self.graph_data):
            mean_px1, coord = sample[:2]
            try:
                coord0 = coord / coord.max()
            except AttributeError:
                VOC_has_variable_image_sizes = True


            A = compute_adjacency_matrix_images(coord, mean_px1)  # using super-pixel locations + features

            edges_list, edge_values_list = compute_edges_list(A)  # NEW

            N_nodes = A.shape[0]

            mean_px1 = mean_px1.reshape(N_nodes, -1)
            # mean_px2 = mean_px2.reshape(N_nodes, -1)
            coord = coord.reshape(N_nodes, -1)
            x = np.concatenate((mean_px1, coord), axis=1)

            edge_values_list = edge_values_list.reshape(-1)  # NEW # TO DOUBLE-CHECK !

            self.node_features.append(x)
            self.edge_features.append(edge_values_list)  # NEW
            self.Adj_matrices.append(A)
            self.edges_lists.append(edges_list)
        for item in tqdm(range(len(self.node_features)), desc='Data_Preprocessing', unit='item'):
            # for index in range(len(self.node_features)):

            g = dgl.DGLGraph()
            g.add_nodes(self.node_features[item].shape[0])
            g.ndata['feat'] = torch.tensor(self.node_features[item])

            for src, dsts in enumerate(self.edges_lists[item]):

                if self.node_features[item].shape[0] == 1:
                    g.add_edges(src, dsts)
                else:
                    g.add_edges(src, dsts[dsts != src])


            g.edata['feat'] = torch.Tensor(self.edge_features[item]).unsqueeze(1)  # NEW
            self.graph_lists.append(g)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
            Get the idx^th sample.
            Parameters
            ---------
            idx : int
                The sample index.
            Returns
            -------
            (dgl.DGLGraph, int)
                DGLGraph with node feature stored in `feat` field
                And its label.
        """
        return self.label[idx], self.data_total[idx], self.graph_lists[idx], self.Adj_matrices[idx]

        # return  self.data_total[idx]
class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """

    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.data_lists = lists[0]
        self.labels = lists[1]
        self.graph_lists = lists[2]
        self.Adj_matrices = lists[3]
    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])


class ST_DatasetLoad(torch.utils.data.Dataset):
    def __init__(self, name, train_ratio=1, shuffle=False, num_val=200):
        """
            Takes input standard image dataset name (MNIST/CIFAR10)
            and returns the superpixels graph.

            This class uses results from the above SuperPix class.
            which contains the steps for the generation of the Superpixels
            graph from a superpixel .pkl file that has been given by
            https://github.com/bknyaz/graph_attention_pool

            Please refer the SuperPix class for details.
        """
        t_data = time.time()
        self.name = name

        use_mean_px = True  # using super-pixel locations + features
        # use_mean_px = False # using only super-pixel locations

        use_coord = True
        self.data = ST_PixDGL(data_dir=name,shuffle=shuffle,
                                use_mean_px=use_mean_px,
                                use_coord=use_coord)

        # self.train_ = SuperPixDGL(r"L:\code\gtcl", dataset=self.name, split='train3',
        #                           use_mean_px=use_mean_px,
        #                           use_coord=use_coord)
        # _test_graphs, _test_labels = self.test_[:2000]
        total_num = len(self.data.data_total)
        _val_data, _val_labels, _val_graphs, _val_adjs   = self.data[int(total_num * train_ratio): int(total_num * 1)]
        _train_data, _train_labels, _train_graphs, _train_adjs = self.data[0: int(total_num * train_ratio)]
        _test_data, _test_labels, _test_graphs, _test_adjs  = self.data[0:]
        self.val =DGLFormDataset (_val_data, _val_labels, _val_graphs, _val_adjs )
        self.train = DGLFormDataset(_train_data, _train_labels, _train_graphs, _train_adjs)
        self.test = DGLFormDataset(_test_data, _test_labels, _test_graphs, _test_adjs )


        print("[I] Data load time: {:.4f}s".format(time.time() - t_data))



    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        labels , data, graphs, adjs  = map(list, zip(*samples))
        # data = torch.tensor(np.array(data))
        labels = torch.tensor(np.array(labels), dtype=torch.long)
        tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_n]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        tab_sizes_e = [graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_e]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        for idx, graph in enumerate(graphs):
            graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
            graphs[idx].edata['feat'] = graph.edata['feat'].float()
        batched_graph = dgl.batch(graphs)

        data = torch.stack(data)
        return data, labels, batched_graph, snorm_n, snorm_e
        # return samples[1],  torch.tensor(samples[0])





import random

if __name__ == '__main__':
    # data_dir = 'data/data/superpixels/'
    # with open(data_dir + "MNIST" + '.pkl', "rb") as f:
    #     f = pickle.load(f)
    # data_dir1 = 'data/superpixels/'
    # with open(data_dir1 + "MNIST" + '.pkl', "rb") as f1:
    #     f1 = pickle.load(f1)
    DATASET_NAME = 'MNIST'
    dataset = ST_DatasetDGL(DATASET_NAME)

    # l = dataset.test.n_samples  # 代表数组的长度
    # zeros = int(l * 0.1)  # 代表数组中 0 的个数，为长度的 10%
    #
    # arr = [0] * zeros + [1] * (l - zeros)
    # random.shuffle(arr)

    # dataset.elements = [dataset.test[i] for i in range(l) if arr[i] == 0]
    with open(r"DATA1.pkl", 'wb') as f:
        # f1 = pickle.load(f1)
        pickle.dump([dataset.train, dataset.val, dataset.test], f)



