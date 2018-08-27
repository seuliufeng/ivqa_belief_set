import networkx as nx
import numpy as np


def find_connected_components(points):
    g = nx.Graph()
    g.add_edges_from(points)
    return [list(c) for c in nx.connected_components(g)]


def test_connected_components():
    def _test(connection):
        print('\nEdges:')
        print(connection)
        cliques = find_connected_components(connection)
        print('Connected components:')
        for c in cliques:
            print(c)

    connection = [[1, 2], [1, 4], [5, 6], [3, 4]]
    _test(connection)
    connection = [[1, 2], [1, 4], [3, 4], [5, 6]]
    _test(connection)
    connection = [[1, 2], [3, 4], [5, 6], [2, 3]]
    _test(connection)


if __name__ == '__main__':
    test_connected_components()
