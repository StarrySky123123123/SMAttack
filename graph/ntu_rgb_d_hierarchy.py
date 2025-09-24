from audioop import reverse
import sys
import numpy as np

sys.path.extend(['../'])
from .hdgcn_tools import *


class Graph:
    def __init__(self, CoM=21, labeling_mode='spatial'):
        self.num_node = 25
        self.CoM = CoM
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_hierarchical_graph(self.num_node, get_edgeset(dataset='NTU', CoM=self.CoM))  # L, 3, 25, 25
        else:
            raise ValueError()
        return A, self.CoM


class Graph_Kinetics:
    def __init__(self, CoM=21, labeling_mode='spatial'):
        self.num_node = 18
        self.CoM = CoM
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = get_hierarchical_graph(self.num_node, get_edgeset(dataset='kinetics', CoM=self.CoM))  # L, 3, 25, 25
        else:
            raise ValueError()
        return A, self.CoM


if __name__ == '__main__':
    import tools

    g = Graph().A
    import matplotlib.pyplot as plt

    for i, g_ in enumerate(g[0]):
        plt.imshow(g_, cmap='gray')
        cb = plt.colorbar()
        plt.savefig('./graph_{}.png'.format(i))
        cb.remove()
        plt.show()
