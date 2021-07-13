import numpy as np

from collections import defaultdict


class Mfset:
    def __init__(self, s):
        """
        Merge-find set data structure representing a partition.

        :param s: set considered for the partition. Assumes distinct elements.
        """
        self.parent = dict()
        self.rank = dict()
        self.size = len(s)

        for x in s:
            self.parent[x] = x
            self.rank[x] = 0

    def union(self, x, y):
        """
        Merge two disjoint sets.
        :return: true if the union has been successful
        """
        if x not in self.parent or y not in self.parent:
            raise ValueError('Both elements must be in the Mfset')

        x = self.find(x)
        y = self.find(y)

        if x == y:
            return False

        if self.rank[x] == self.rank[y]:
            self.parent[y] = x
            self.rank[x] += 1
        elif self.rank[x] > self.rank[y]:
            self.parent[y] = x
        else:
            self.parent[x] = y

        self.size -= 1
        return True

    def find(self, x):
        """
        Find the representative member of the set that contains x.
        """
        if x not in self.parent:
            raise ValueError('Element must be in the Mfset')

        while self.parent[x] != x:
            x, self.parent[x] = self.parent[x], self.parent[self.parent[x]]

        return x

    def is_trivial(self):
        """
        Returns true if the partition P contains only the entire set S
        """
        return self.size == 1

    def __repr__(self):
        sets = defaultdict(set)
        for x in self.parent.keys():
            sets[self.find(x)].add(x)

        return str(list(sets.values()))


# ______________________________________________________________________________________________________________________


def mst_kruskal(graph: np.array):
    """
    Returns the Minimum Spanning Tree of the graph represented by a cost matrix with Kruskal's Algorithm, using the
    Mfset data-structure.

    :param graph: directed or undirected graph
    :return: MST (cost matrix), MST cost
    """
    if graph.shape[0] != graph.shape[1]:
        raise ValueError('Cost matrix must be square')

    if not np.alltrue(np.diag(graph) == 0):
        raise ValueError('Invalid cost matrix: cost of the elements in diagonal must be 0')

    mfs = Mfset(range(graph.shape[0]))
    sorted_edges = ((i, j) for i, j in zip(*np.unravel_index(np.argsort(graph, axis=None), graph.shape)) if j > i)

    mst = np.zeros(graph.shape)
    cost = 0

    for i, j in sorted_edges:
        if mfs.is_trivial():
            break

        if mfs.union(i, j):
            mst[i, j] = graph[i, j]
            cost += graph[i, j]

    return mst, cost


if __name__ == '__main__':
    g = np.array([
        [0, 25, 25, 13],
        [0, 0, 25, 13],
        [0, 0, 0, 13],
        [0, 0, 0, 0]
    ])

    mst, cost = mst_kruskal(g)

    print(mst)
    print(cost)
