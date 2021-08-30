import collections
import sys

import numpy as np
import pickle as pkl
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_data():
    names = ['dup.hua3.test_graphs']
    objects = []
    for i in range(len(names)):
        with open("../data/{}".format(names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                return pkl.load(f, encoding='latin1')
            else:
                return pkl.load(f)


GraphData = collections.namedtuple(
    "GraphData",
    ["from_idx", "to_idx", "node_features", "edge_features", "graph_idx", "n_graphs"],
)

def pair_graph(g1,g2,i):
    from_ida = g1[0][0]
    print("from_ida", from_ida.shape)
    from_idb = g2[0][0] + g1[0][4].shape[0]
    print("from_idb", from_idb.shape)
    from_ids = np.hstack((from_ida, from_idb))
    print("from_ids", from_ids)
    print(from_ids.shape)

    to_ida = g1[0][1]
    print("to_ida", to_ida.shape)
    to_idb = g2[0][1] + g1[0][4].shape[0]
    print("to_idb", to_idb.shape)
    to_ids = np.hstack((to_ida, to_idb))
    print("to_ids", to_ids)
    print(to_ids.shape)

    node_fa = g1[0][2]
    print("node_fa", node_fa.shape)
    node_fb = g2[0][2]
    print("node_fb", node_fb.shape)
    node_fs = np.append(node_fa, node_fb, axis=0)
    print("node_fs", node_fs)
    print("node_fs", node_fs.shape)

    edge_fa = g1[0][3]
    print("edge_fa", edge_fa.shape)
    edge_fb = g2[0][3]
    print("edge_fb", edge_fb.shape)

    edge_fs = np.append(edge_fa, edge_fb, axis=0)
    print("edge_fs", edge_fs)
    print("edge_fs", edge_fs.shape)

    graph_ida = g1[0][4]
    print(graph_ida)
    graph_idb = g2[0][4] + 2
    print(graph_idb)
    graph_ids = np.hstack((graph_ida, graph_idb))
    print(graph_ids)

    graph_na = g1[0][5]
    graph_ns = graph_na + 2
    print(graph_ns)
    labela = g1[1]
    labelb = g2[1]
    labels = np.hstack((labela, labelb))
    graph = GraphData(
            from_idx=np.array(from_ids),
            to_idx=np.array(to_ids),
            node_features=np.array(node_fs, dtype="float32"),
            edge_features=np.array(edge_fs, dtype="float32"),
            graph_idx=np.array(graph_ids), n_graphs=graph_ns)
    return graph,labels


def batch_graph(data, fro, batch):

    '''
    from_ids = data[for][0][0]
    from_ida = g1[0][0]
    print("from_ida", from_ida.shape)
    from_idb = g2[0][0] + g1[0][4].shape[0]
    print("from_idb", from_idb.shape)
    from_ids = np.hstack((from_ida, from_idb))
    print("from_ids", from_ids)
    print(from_ids.shape)
    '''
    # print(data[fro])
    from_ids = data[fro][0][0]
    to_ids = data[fro][0][1]
    num = data[fro][0][4].shape[0]
    # print(num)
    for i in range(fro + 1, fro + batch):
        from_idx = data[i][0][0] + num
        to_idx = data[i][0][1] + num
        num = num + data[i][0][4].shape[0]
        from_ids = np.hstack((from_ids, from_idx))
        to_ids = np.hstack((to_ids, to_idx))
    # print(from_ids)
    # print(to_ids)


    '''
    graph_ida = g1[0][4]
    print(graph_ida)
    graph_idb = g2[0][4] + 2
    print(graph_idb)
    graph_ids = np.hstack((graph_ida, graph_idb))
    print(graph_ids)
    '''
    graph_ids = data[fro][0][4]
    for i in range(fro+1, fro + batch):
        graph_idx = data[i][0][4] + 2*(i-fro)
        graph_ids = np.hstack((graph_ids, graph_idx))
    # print(graph_ids)


    '''
    graph_na = g1[0][5]
    graph_ns = graph_na + 2
    print(graph_ns)
    '''
    graph_ns = 2 * batch
    # print(graph_ns)

    '''
    labela = g1[1]
    labelb = g2[1]
    labels = np.hstack((labela, labelb))
    '''
    labels = data[fro][1]
    for i in range(fro+1, fro + batch):
        labels = np.hstack((labels, data[i][1]))
    # print(labels)

    # node_features
    node_fs = data[fro][0][2]
    for i in range(fro + 1, fro + batch):
        node_fx = data[i][0][2]
        node_fs = np.append(node_fs, node_fx, axis=0)
    # edge_features
    edge_fs = data[fro][0][3]
    empty = False
    if edge_fs.shape[0] == 0 or edge_fs.shape[1] == 0:
        empty = True
    for i in range(fro + 1, fro + batch):
        edge_fx = data[i][0][3]
        if edge_fx.shape[0] == 0 or edge_fx.shape[1] == 0:
            continue

        # print(edge_fx)
        # print("edge_fx.shape",edge_fx.shape)
        # print(len(edge_fx.shape))
        if empty:
            edge_fs = edge_fx
            empty = False
            continue
        edge_fs = np.append(edge_fs, edge_fx, axis=0)
    graph = GraphData(
            from_idx=np.array(from_ids),
            to_idx=np.array(to_ids),
            node_features=np.array(node_fs, dtype="float32"),
            edge_features=np.array(edge_fs, dtype="float32"),
            graph_idx=np.array(graph_ids), n_graphs=graph_ns)
    return graph, labels

#
# test_graphs = load_data()
# print(len(test_graphs))
# # print(test_graphs[0])
# # print(test_graphs[1])
# print(batch_graph(test_graphs, 0, 2))


