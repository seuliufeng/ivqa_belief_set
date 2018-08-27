import networkx as nx
from networkx.algorithms.approximation import max_clique

def find_max_clique(points):
    g = nx.Graph()
    g.add_edges_from(points)
    import pdb
    pdb.set_trace()
    print(max_clique(g))


def test_max_clique():

    def _test(connection):
        cliques = find_max_clique(connection)

    connection = [[1, 2], [1, 4], [5, 6], [3, 4]]
    _test(connection)
    connection = [[1, 2], [1, 4], [3, 4], [5, 6]]
    _test(connection)
    connection = [[1, 2], [3, 4], [5, 6], [2, 3]]
    _test(connection)


if __name__ == '__main__':
    test_max_clique()

