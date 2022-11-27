import numpy as np
import networkx as nx
import time 

# Task1:

class BarabasiAlbertGraph:
    """
    :param n: number of nodes
    :param m: degree of a new node
    "param n0: starting number of nodes
    """
    def __init__(self, n, m, n0=None, compute_connections=False):
        self.n = n
        self.m = m

        if n0 is None:
            n0 = m 
        if m > n0:
            raise ValueError("m cannot be bigger than n0")
        
        self.next_node = n0
        self.node_list = np.arange(self.n)

        self.prob_num = np.zeros(self.n)
        self.prob_num[:n0] = 1
        self.prob_den = n0

        self.connections = None
        self.compute_connections = compute_connections
        if self.compute_connections:
            self.connections = {i: [] for i in range(n0)}

    def build_graph(self):
        for _ in range(self.next_node, self.n):
            self.single_turn()

    def single_turn(self):
        neighbors = self._generate_neighbors()
        self._add_node(neighbors)

    def _generate_neighbors(self):
        return np.random.choice(self.node_list[:self.next_node], 
            size=self.m, 
            replace=False,
            p=self.prob_num[:self.next_node] / self.prob_den)

    def _add_node(self, neighbors):
        self.prob_num[self.next_node] = self.m + 1
        self.prob_den += self.m * 2 + 1
        if self.compute_connections:
            self.connections[self.next_node] = list(neighbors)

        for n in neighbors:
            self.prob_num[n] += 1
            if self.compute_connections:
                self.connections[n].append(self.next_node)
        self.next_node += 1

    def get_degrees(self):
        return self.prob_num - 1

# Class for Task2:

class BarabasiAlbertNodeRemover(BarabasiAlbertGraph):
    def __init__(self, n, m, n0=None, removal_mode='random'):
        super().__init__(n, m, n0, compute_connections=True)

        # Removal mode "random" is 0
        # Removal mode "attack" is 1

        if removal_mode == 'random':
            self.remove_mode = 0
            self.remove_node = self._remove_random
        elif removal_mode == 'attack':
            self.remove_mode = 1
            self.remove_node = self._remove_attack
        else:
            raise ValueError('Wrong remove mode selected')

    def build(self):
        super().build_graph()
        if self.remove_mode == 0:
            self.node_list = set(self.node_list)
            self.prob_num = {u: self.prob_num[u] for u in range(self.n)}

    def _remove_random(self):
        u = np.random.choice(list(self.node_list))
        self._remove_node(u)

    def _remove_attack(self):
        u = np.random.choice(self.node_list, p=self.prob_num / self.prob_den)
        self._remove_node(u)

    def _remove_node(self, u):
        for v in self.connections[u]:
            self.connections[v].remove(u)
            self.prob_num[v] -= 1
            if not self.connections[v]:  # Remove v if it has no edges
                self._remove_node(v)

        self.prob_den -= self.prob_num[u] * 2 - 1
        del self.connections[u]

        if self.remove_mode == 0:
            self.node_list.remove(u)
            del self.prob_num[u]
        elif self.remove_mode == 1:
            self.prob_num[u] = 0


# run this file itself to test the script

if __name__ == '__main__':
    ba = BarabasiAlbertGraph(n=10, m=3, n0=5, compute_connections=True)
    print(ba.connections)

    ba.single_turn()
    print(ba.connections)

    ba.single_turn()
    print(ba.connections)

    ba.build_graph()
    print(ba.next_node)
    print(ba.connections)
    print(ba.prob_num)
    print(ba.get_degrees())

    print('\n\n\n Random mode:')

    bar = BarabasiAlbertNodeRemover(10, 2, removal_mode='random')
    bar.build()

    print(bar.compute_connections)
    bar.remove_node()
    print(bar.connections)
    bar.remove_node()
    bar.remove_node()
    print(bar.connections)

    print('Attack mode:')
    bar = BarabasiAlbertNodeRemover(10, 2, removal_mode='attack')
    bar.build()

    print(bar.connections)
    bar.remove_node()
    print(bar.connections)
    bar.remove_node()
    bar.remove_node()
    print(bar.connections)
