import os
import random
import numpy as np
import collections
import pickle as pkl
import sys
from util import localfile
from tqdm import tqdm
from repo_name import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0， 1"
"--------------------------------"
data_folder = 'data/clf'

dataset = [
    [data_folder + '/first_msr_pairs_pull_info_X.txt', data_folder + '/first_msr_pairs_pull_info_y.txt', "train"],
    # [data_folder + '/second_msr_pairs_pull_info_X.txt', data_folder + '/second_msr_pairs_pull_info_y.txt', "test"],
    [data_folder + '/first_nondup_pull_info_X.txt', data_folder + '/first_nondup_pull_info_y.txt', "train"],
    # [data_folder + '/test_pull_info_X.txt', data_folder + '/test_pull_info_y.txt', "test"]
    # [data_folder + '/test1_pull_info_X.txt', data_folder + '/test1_pull_info_y.txt', "test"]
]

prdata_path = "prdata"

GraphData = collections.namedtuple(
    "GraphData",
    ["from_idx", "to_idx", "node_features", "edge_features", "graph_idx", "n_graphs"],
)
"--------------------------------"
add_title = True
add_body = True
add_file_list_sim = True
add_overlap_files_len = True
add_code_sim1 = True
add_code_sim2 = True
add_location_sim1 = True
add_location_sim2 = True
add_pattern = True
add_time = True

weighted_graph = True
fileName = "remove"

"--------------------------------"
if not add_title:
    fileName = fileName + "_title"
if not add_body:
    fileName = fileName + "_body"
"--------------------------------"

try:
    window_size = int(sys.argv[1])
    fileName = fileName + "_" + sys.argv[1]
except:
    window_size = 9
    fileName = fileName + "_9"
    print('using default window size = %s' % window_size)


"--------------------------------"
print("FileName", fileName)
"--------------------------------"
# load pre-trained word embeddings
word_embeddings_dim = 300
word_embeddings = {}

with open('glove.6B.' + str(word_embeddings_dim) + 'd.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        data = line.split()
        word_embeddings[str(data[0])] = list(map(float, data[1:]))

data_list = []
train_data_list = []
test_data_list = []
label_list = []
train_label_list = []
test_label_list = []

for s in dataset:
    X = localfile.try_get_file(s[0])
    # print(len(X))
    y = localfile.try_get_file(s[1])
    data_list.extend(X)
    label_list.extend(y)
    if s[2] == "train":
        train_data_list.extend(X)
        train_label_list.extend(y)
    elif s[2] == "test":
        test_data_list.extend(X)
        test_label_list.extend(y)


train_ids = []
for train_name in tqdm(train_data_list):
    train_id = data_list.index(train_name)
    train_ids.append(train_id)
random.seed(1)
random.shuffle(train_ids)

test_ids = []
for test_name in tqdm(test_data_list):
    test_id = data_list.index(test_name)
    test_ids.append(test_id)
random.seed(1)
random.shuffle(test_ids)

ids = train_ids + test_ids
# all_data_list = data_list
# all_label_list = label_list
all_data_list = []
all_label_list = []
for i in ids:
    all_data_list.append(data_list[int(i)])
    all_label_list.append(label_list[int(i)])
print("all_data_list", len(all_data_list))

print("label", all_label_list)


word_set = set()

all_len = all_data_list.__len__()
for i in range(all_len):
    # print(i)
    if add_title:
        A_clean_title = all_data_list[i]["A_clean_title"]
        B_clean_title = all_data_list[i]["B_clean_title"]
        word_set.update(A_clean_title)
        word_set.update(B_clean_title)
    if add_body:
        A_clean_body = all_data_list[i]["A_clean_body"]
        B_clean_body = all_data_list[i]["B_clean_body"]
        word_set.update(B_clean_body)
        word_set.update(A_clean_body)


# print("word_set", word_set)
vocab = list(word_set)
vocab_size = len(vocab)
# print("vocab", vocab)
# print("vocab_size", vocab_size)


word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

# initialize out-of-vocabulary word embeddings
oov = {}
for v in vocab:
    oov[v] = np.random.uniform(-0.01, 0.01, word_embeddings_dim)


# label_set = set()
# for label in all_label_list:
#     label_set.add(label)
#
# label_list = list(label_set)
# print("label_list", label_list)

# select 90% training set
train_size = len(train_data_list)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size
test_size = len(test_data_list)

print("real_train_size", real_train_size)
print("val_size", val_size)
print("test_size", test_size)
print("len(dataset)", len(dataset))


def get_graph(doc_words, graph_id):
    # doc_words = all_data_list[i][]
    from_ids = []
    to_ids = []
    node_features = []
    edge_features = []
    graph_ids = []
    doc_len = len(doc_words)

    doc_vocab = list(set(doc_words))
    doc_nodes = len(doc_vocab)
    doc_word_id_map = {}
    for j in range(1, doc_nodes + 1):
        doc_word_id_map[doc_vocab[j - 1]] = j

    """
    for i in range(doc_len):
        if i != doc_len-1:
            from_ids.append(doc_word_id_map[doc_words[i]])
            to_ids.append(doc_word_id_map[doc_words[i+1]])
            edge_features.append([1.0])
    """
    # sliding windows
    windows = []
    if doc_len <= window_size:
        windows.append(doc_words)
    else:
        for j in range(doc_len - window_size + 1):
            window = doc_words[j: j + window_size]
            windows.append(window)

    word_pair_count = {}
    for window in windows:
        for p in range(1, len(window)):
            for q in range(0, p):
                word_p = window[p]
                word_p_id = word_id_map[word_p]
                word_q = window[q]
                word_q_id = word_id_map[word_q]
                if word_p_id == word_q_id:
                    continue
                word_pair_key = (word_p_id, word_q_id)
                # word co-occurrences as weights
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.
                # two orders
                word_pair_key = (word_q_id, word_p_id)
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.

    for key in word_pair_count:
        p = key[0]
        q = key[1]
        from_ids.append(doc_word_id_map[vocab[p]])
        to_ids.append(doc_word_id_map[vocab[q]])
        edge_features.append([word_pair_count[key]] if weighted_graph else 1.)

    for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
        node_features.append(list(word_embeddings[k]) if k in word_embeddings else oov[k])
        graph_ids.append(graph_id)

    graph = GraphData(
        from_idx=np.array(from_ids),
        to_idx=np.array(to_ids),
        node_features=np.array(node_features, dtype="float32"),
        edge_features=np.array(edge_features, dtype="float32"),
        graph_idx=np.array(graph_ids), n_graphs=2)
    return graph

def pair_graph(g1, g2, flag=False):
    from_ida = g1.from_idx
    # print("from_ida", from_ida.shape)
    from_idb = g2.from_idx + g1.node_features.shape[0]
    # print("from_idb", from_idb.shape)
    from_ids = np.hstack((from_ida, from_idb))
    # from_ids.append(0)
    # from_ids = np.append(from_ids, 0)
    # from_ids = np.append(from_ids, 0)
    # print("from_ids", from_ids)
    # print(from_ids.shape)

    to_ida = g1.to_idx
    # print("to_ida", to_ida.shape)
    to_idb = g2.to_idx + g1.node_features.shape[0]
    # print("to_idb", to_idb.shape)
    to_ids = np.hstack((to_ida, to_idb))
    # to_ids = np.append(to_ids, 1)
    # to_ids = np.append(to_ids, 1 + g1.node_features.shape[0])
    # print("to_ids", to_ids)
    # print(to_ids.shape)

    node_fa = g1.node_features
    # print("node_fa", node_fa.shape)
    node_fb = g2.node_features
    # print("node_fb", node_fb.shape)
    # node_fs = np.append(node_fa, node_fb, axis=0)
    # print("node_fs", node_fs)
    # print("node_fs", node_fs.shape)
    if node_fa.shape[0] == 0 and node_fb.shape[0] == 0:
        node_fs = [[]]
    elif node_fa.shape[0] == 0:
        node_fs = node_fb
    elif node_fb.shape[0] == 0:
        node_fs = node_fa
    else:
        node_fs = np.append(node_fa, node_fb, axis=0)
    edge_fa = g1.edge_features
    # print("edge_fa", edge_fa.shape)
    edge_fb = g2.edge_features
    # print("edge_fb", edge_fb.shape)
    if edge_fa.shape[0] == 0 and edge_fb.shape[0] == 0:
        edge_fs = [[]]
    elif edge_fa.shape[0] == 0 or edge_fa.shape[1] == 0:
        edge_fs = edge_fb
        if flag == False:
            from_ids = np.append(from_ids, 0)
            from_ids = np.append(from_ids, 1)
            to_ids = np.append(to_ids, 1)
            to_ids = np.append(to_ids, 0)
            edge_fs = np.append(edge_fs, [[100]], axis=0)
            edge_fs = np.append(edge_fs, [[100]], axis=0)
    elif edge_fb.shape[0] == 0 or edge_fb.shape[1] == 0:
        edge_fs = edge_fa
        if flag == False:
            from_ids = np.append(from_ids, 0)
            from_ids = np.append(from_ids, 1)
            to_ids = np.append(to_ids, 1)
            to_ids = np.append(to_ids, 0)
            edge_fs = np.append(edge_fs, [[100]], axis=0)
            edge_fs = np.append(edge_fs, [[100]], axis=0)
    else:
        edge_fs = np.append(edge_fa, edge_fb, axis=0)
        if flag == False:
            from_ids = np.append(from_ids, 0)
            from_ids = np.append(from_ids, 0)
            to_ids = np.append(to_ids, 1)
            to_ids = np.append(to_ids, 1 + g1.node_features.shape[0])
            edge_fs = np.append(edge_fs, [[100]], axis=0)
            edge_fs = np.append(edge_fs, [[100]], axis=0)
            from_ids = np.append(from_ids, 1)
            from_ids = np.append(from_ids, 1 + g1.node_features.shape[0])
            to_ids = np.append(to_ids, 0)
            to_ids = np.append(to_ids, 0)
            edge_fs = np.append(edge_fs, [[100]], axis=0)
            edge_fs = np.append(edge_fs, [[100]], axis=0)

    # print("edge_fs", edge_fs)
    # print("edge_fs", edge_fs.shape)
    # print("from_ids.shape", from_ids.shape)
    # print("to_ids.shape", to_ids.shape)
    graph_ida = g1.graph_idx
    # print(graph_ida)
    graph_idb = g2.graph_idx
    # print(graph_idb)
    graph_ids = np.hstack((graph_ida, graph_idb))

    # print(graph_ids)

    if flag == True:
        n_graphs = g1.n_graphs + g2.n_graphs
    else:
        n_graphs = 1

    graph = GraphData(
        from_idx=np.array(from_ids),
        to_idx=np.array(to_ids),
        node_features=np.array(node_fs, dtype="float32"),
        edge_features=np.array(edge_fs, dtype="float32"),
        graph_idx=np.array(graph_ids), n_graphs=n_graphs)
    return graph

def addFeatures(graph, data, graph_id, flag):
    title_sim = [[0.0 for i in range(300)]]
    title_sim[0][0] = data["title_sim"][0]
    # print(title_sim)
    body_sim = [[0.0 for i in range(300)]]
    body_sim[0][0] = data["body_sim"][0]

    # print(data["file_list_sim"]) # 1
    file_list_sim = [[0.0 for i in range(300)]]
    file_list_sim[0][0] = data["file_list_sim"]
    # print("file_list_sim", file_list_sim)

    # print(data["overlap_files_len"]) # 1
    overlap_files_len = [[0.0 for i in range(300)]]
    overlap_files_len[0][0] = data["overlap_files_len"]
    # print("overlap_files_len", overlap_files_len)

    # print(data["code_sim"]) # 2
    code_sim1 = [[0.0 for i in range(300)]]
    code_sim1[0][0] = data["code_sim"][0]
    code_sim2 = [[0.0 for i in range(300)]]
    code_sim2[0][0] = data["code_sim"][1]
    # print("code_sim", code_sim1)
    # print("code_sim", code_sim2)

    # print(data["location_sim"]) # 2
    location_sim1 = [[0.0 for i in range(300)]]
    location_sim1[0][0] = data["location_sim"][0]
    location_sim2 = [[0.0 for i in range(300)]]
    location_sim2[0][0] = data["location_sim"][1]
    # print("location_sim", location_sim1)
    # print("location_sim", location_sim2)

    # print(data["pattern"])  # 1
    pattern = [[0.0 for i in range(300)]]
    pattern[0][0] = data["pattern"]
    # print("pattern",pattern)

    # print(data["time"]) # 1
    time = [[0.0 for i in range(300)]]
    time[0][0] = data["time"]

    if flag:
        title_sim[0][0] = 1
        body_sim[0][0] = 1
        file_list_sim[0][0] = 1
        overlap_files_len[0][0] = 1
        code_sim1[0][0] = 1
        code_sim2[0][0] = 1
        location_sim1[0][0] = 1
        location_sim2[0][0] = 1
        pattern[0][0] = 1
        time[0][0] = 1
    # print("time", time)
    from_ids = graph.from_idx
    to_ids = graph.to_idx
    node_fs = graph.node_features
    edge_fs = graph.edge_features
    graph_ids = graph.graph_idx
    add_feature_num = 0
    graph_id = [graph_id]
    empty = False


    edge_num = from_ids.shape[0]
    if edge_num == 0:
        from_ids = [0]
        to_ids = [0]
        edge_fs = [[0]]

    node_num = node_fs.shape[0]
    # print("add_feature, node_fs.shape", node_fs.shape)
    if node_fs.shape[0] == 0:

        empty = True
    if not add_title:

        if empty:
            node_fs = title_sim
            # from_ids.append(1)
            empty = False
        else:
            node_fs = np.append(node_fs, title_sim, axis=0)
        graph_ids = np.append(graph_ids, graph_id, axis=0)
    # print(node_fs)

    if not add_body:
        if empty:
            node_fs = body_sim
            empty = False
        else:
            node_fs = np.append(node_fs, body_sim, axis=0)
        graph_ids = np.append(graph_ids, graph_id, axis=0)
    if add_file_list_sim:
        if empty:
            node_fs = file_list_sim
            empty = False
        else:
            node_fs = np.append(node_fs, file_list_sim, axis=0)

        from_ids = np.append(from_ids, [0], axis=0)
        to_ids = np.append(to_ids, [node_num + 1], axis=0)
        from_ids = np.append(from_ids, [node_num + 1], axis=0)
        to_ids = np.append(to_ids, [0], axis=0)
        node_num += 1
        edge_fs = np.append(edge_fs, [[100]], axis=0)
        edge_fs = np.append(edge_fs, [[100]], axis=0)

        add_feature_num = add_feature_num + 1
        graph_ids = np.append(graph_ids, graph_id, axis=0)
    if add_overlap_files_len:
        node_fs = np.append(node_fs, overlap_files_len, axis=0)

        from_ids = np.append(from_ids, [node_num], axis=0)
        to_ids = np.append(to_ids, [node_num + 1], axis=0)
        from_ids = np.append(from_ids, [node_num + 1], axis=0)
        to_ids = np.append(to_ids, [node_num], axis=0)
        node_num += 1
        edge_fs = np.append(edge_fs, [[100]], axis=0)
        edge_fs = np.append(edge_fs, [[100]], axis=0)

        add_feature_num = add_feature_num + 1
        graph_ids = np.append(graph_ids, graph_id, axis=0)
    if add_code_sim1:
        node_fs = np.append(node_fs, code_sim1, axis=0)

        from_ids = np.append(from_ids, [node_num], axis=0)
        to_ids = np.append(to_ids, [node_num + 1], axis=0)
        from_ids = np.append(from_ids, [node_num + 1], axis=0)
        to_ids = np.append(to_ids, [node_num], axis=0)
        node_num += 1
        edge_fs = np.append(edge_fs, [[100]], axis=0)
        edge_fs = np.append(edge_fs, [[100]], axis=0)

        add_feature_num = add_feature_num + 1
        graph_ids = np.append(graph_ids, graph_id, axis=0)
    if add_code_sim2:
        node_fs = np.append(node_fs, code_sim2, axis=0)

        from_ids = np.append(from_ids, [node_num], axis=0)
        to_ids = np.append(to_ids, [node_num + 1], axis=0)
        from_ids = np.append(from_ids, [node_num + 1], axis=0)
        to_ids = np.append(to_ids, [node_num], axis=0)
        node_num += 1
        edge_fs = np.append(edge_fs, [[100]], axis=0)
        edge_fs = np.append(edge_fs, [[100]], axis=0)

        add_feature_num = add_feature_num + 1
        graph_ids = np.append(graph_ids, graph_id, axis=0)
    if add_location_sim1:
        node_fs = np.append(node_fs, location_sim1, axis=0)

        from_ids = np.append(from_ids, [node_num], axis=0)
        to_ids = np.append(to_ids, [node_num + 1], axis=0)
        from_ids = np.append(from_ids, [node_num + 1], axis=0)
        to_ids = np.append(to_ids, [node_num], axis=0)
        node_num += 1
        edge_fs = np.append(edge_fs, [[100]], axis=0)
        edge_fs = np.append(edge_fs, [[100]], axis=0)

        add_feature_num = add_feature_num + 1
        graph_ids = np.append(graph_ids, graph_id, axis=0)
    if add_code_sim2:
        node_fs = np.append(node_fs, location_sim2, axis=0)

        from_ids = np.append(from_ids, [node_num], axis=0)
        to_ids = np.append(to_ids, [node_num + 1], axis=0)
        from_ids = np.append(from_ids, [node_num + 1], axis=0)
        to_ids = np.append(to_ids, [node_num], axis=0)
        node_num += 1
        edge_fs = np.append(edge_fs, [[100]], axis=0)
        edge_fs = np.append(edge_fs, [[100]], axis=0)

        add_feature_num = add_feature_num + 1
        graph_ids = np.append(graph_ids, graph_id, axis=0)
    if add_pattern:
        node_fs = np.append(node_fs, pattern, axis=0)

        from_ids = np.append(from_ids, [node_num], axis=0)
        to_ids = np.append(to_ids, [node_num + 1], axis=0)
        from_ids = np.append(from_ids, [node_num + 1], axis=0)
        to_ids = np.append(to_ids, [node_num], axis=0)
        node_num += 1
        edge_fs = np.append(edge_fs, [[100]], axis=0)
        edge_fs = np.append(edge_fs, [[100]], axis=0)

        add_feature_num = add_feature_num + 1
        graph_ids = np.append(graph_ids, graph_id, axis=0)
    if add_time:
        node_fs = np.append(node_fs, time, axis=0)

        from_ids = np.append(from_ids, [node_num], axis=0)
        to_ids = np.append(to_ids, [node_num + 1], axis=0)
        from_ids = np.append(from_ids, [node_num + 1], axis=0)
        to_ids = np.append(to_ids, [node_num], axis=0)
        node_num += 1
        edge_fs = np.append(edge_fs, [[100]], axis=0)
        edge_fs = np.append(edge_fs, [[100]], axis=0)

        add_feature_num = add_feature_num + 1
        graph_ids = np.append(graph_ids, graph_id, axis=0)

    root_node = [[0.0 for i in range(300)]]
    node_fs = np.append(root_node, node_fs, axis=0)
    graph_ids = np.append(graph_ids, graph_id, axis=0)

    # print("node_fs.shape", node_fs.shape)
    # print("node_fs", node_fs)
    # print(type(node_fs))
    # print(add_feature_num)
    graph = GraphData(
        from_idx=np.array(from_ids),
        to_idx=np.array(to_ids),
        node_features=np.array(node_fs, dtype="float32"),
        edge_features=np.array(edge_fs, dtype="float32"),
        graph_idx=np.array(graph_ids), n_graphs=1)
    return graph

def link2graph(A_graph, B_graph):
    return pair_graph(A_graph, B_graph, True)

def add_other_features(title_graph, body_graph, data, graph_id, flag):

    title_body = None
    if title_graph == None:
        title_body = body_graph
    elif body_graph == None:
        title_body = title_graph
    else:
        title_body = pair_graph(title_graph, body_graph)
    # print(title_body)
    # print("title_body", title_body)
    ret_graph = addFeatures(title_body, data, graph_id, flag)
    # print("ret_graph", ret_graph)
    return ret_graph

# build graph function
def build_graph(start, end):
    graphs = []
    for i in tqdm(range(start, end)):
        # print(i)

        A_clean_title = None
        A_title_graph = None
        B_clean_title = None
        B_title_graph = None
        if add_title:
           #  print("title")
            A_clean_title = all_data_list[i]["A_clean_title"]

            A_title_graph = get_graph(A_clean_title, 0)
            # print("A_title_graph", A_title_graph)
            B_clean_title = all_data_list[i]["B_clean_title"]

            B_title_graph = get_graph(B_clean_title, 1)
            # print("B_title_graph",B_title_graph)
        A_clean_body = None
        A_body_graph = None
        B_clean_body = None
        B_body_graph = None
        if add_body:
            # print("body")
            A_clean_body = all_data_list[i]["A_clean_body"]

            A_body_graph = get_graph(A_clean_body, 0)
            # print("A_body_graph", A_body_graph)
            B_clean_body = all_data_list[i]["B_clean_body"]

            B_body_graph = get_graph(B_clean_body, 1)
            # print("B_body_graph", B_body_graph)
        A_title_body_graph = add_other_features(A_title_graph, A_body_graph, all_data_list[i], 0, True)
        # print("A_title_body_graph", A_title_body_graph)
        B_title_body_graph = add_other_features(B_title_graph, B_body_graph, all_data_list[i], 1, False)
        # print("B_title_body_graph", B_title_body_graph)

        graph = link2graph(A_title_body_graph, B_title_body_graph)
        # print("graph", graph)

        labels = []
        label = all_label_list[i]
        labels.append(label)
        labels = np.array(labels)

        graphs.append([graph, labels])
    return graphs

if __name__ == "__main__":

    print('building graphs for training')
    train_graph = build_graph(start=0, end=real_train_size)
    print('building graphs for training + validation')
    train_val_graph = build_graph(start=real_train_size, end=train_size)
    print('building graphs for test')
    test_graph = build_graph(start=train_size, end=train_size + test_size)

    with open("data/%s.train_graphs" % fileName, 'wb') as f:
        pkl.dump(train_graph, f)
    with open("data/%s.train_val_graphs" % fileName, 'wb') as f:
         pkl.dump(train_val_graph, f)
    with open("data/%s.test_graphs" % fileName, 'wb') as f:
        pkl.dump(test_graph, f)
