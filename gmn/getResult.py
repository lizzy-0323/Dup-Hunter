import tensorflow as tf
# print(tf.__version__)
# print(tf.__path__)
#   package       version
#   tensorflow    1.13.1
#   dm-sonnet     1.18
import sys
import pickle as pkl
from utils import *
from models import *
# from train import *
from layers import *
import collections
import time
import random
import copy
from tqdm import tqdm
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

from repo_name import *
GraphData = collections.namedtuple(
    "GraphData",
    ["from_idx", "to_idx", "node_features", "edge_features", "graph_idx", "n_graphs"],
)

fileName = ""
windowSize = 3
globalStep = 1
"""Evaluation"""
def exact_hamming_similarity(x, y):
    """Compute the binary Hamming similarity."""
    match = tf.cast(tf.equal(x > 0, y > 0), dtype=tf.float32)
    return tf.reduce_mean(match, axis=1)

def compute_similarity(config, x, y):
    """Compute the distance between x and y vectors.

  The distance will be computed based on the training loss type.

  Args:
    config: a config dict.
    x: [n_examples, feature_dim] float tensor.
    y: [n_examples, feature_dim] float tensor.

  Returns:
    dist: [n_examples] float tensor.

  Raises:
    ValueError: if loss type is not supported.
  """
    if config["training"]["loss"] == "margin":
        # similarity is negative distance
        return -euclidean_distance(x, y)
    elif config["training"]["loss"] == "hamming":
        return exact_hamming_similarity(x, y)
    else:
        raise ValueError("Unknown loss type %s" % config["training"]["loss"])


def auc(scores, labels, **auc_args):
    """Compute the AUC for pair classification.

  See `tf.metrics.auc` for more details about this metric.

  Args:
    scores: [n_examples] float.  Higher scores mean higher preference of being
      assigned the label of +1.
    labels: [n_examples] int.  Labels are either +1 or -1.
    **auc_args: other arguments that can be used by `tf.metrics.auc`.

  Returns:
    auc: the area under the ROC curve.
  """
    scores_max = tf.reduce_max(scores)
    scores_min = tf.reduce_min(scores)
    # normalize scores to [0, 1] and add a small epislon for safety
    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)

    labels = (labels + 1) / 2
    # The following code should be used according to the tensorflow official
    # documentation:
    # value, _ = tf.metrics.auc(labels, scores, **auc_args)

    # However `tf.metrics.auc` is currently (as of July 23, 2019) buggy so we have
    # to use the following:
    _, value = tf.metrics.auc(labels, scores, **auc_args)
    return value

"""Build the model"""
def reshape_and_split_tensor(tensor, n_splits):
    """Reshape and split a 2D tensor along the last dimension.

  Args:
    tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
      multiple of `n_splits`.
    n_splits: int, number of splits to split the tensor into.

  Returns:
    splits: a list of `n_splits` tensors.  The first split is [tensor[0],
      tensor[n_splits], tensor[n_splits * 2], ...], the second split is
      [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
  """
    feature_dim = tensor.shape.as_list()[-1]
    # feature dim must be known, otherwise you can provide that as an input
    assert isinstance(feature_dim, int)
    tensor = tf.reshape(tensor, [-1, feature_dim * n_splits])
    return tf.split(tensor, n_splits, axis=-1)


def build_placeholders(node_feature_dim, edge_feature_dim):
    """Build the placeholders needed for the model.

  Args:
    node_feature_dim: int.
    edge_feature_dim: int.

  Returns:
    placeholders: a placeholder name -> placeholder tensor dict.
  """
    # `n_graphs` must be specified as an integer, as `tf.dynamic_partition`
    # requires so.
    return {
        "node_features": tf.placeholder(tf.float32, [None, node_feature_dim]),
        "edge_features": tf.placeholder(tf.float32, [None, edge_feature_dim]),
        "from_idx": tf.placeholder(tf.int32, [None]),
        "to_idx": tf.placeholder(tf.int32, [None]),
        "graph_idx": tf.placeholder(tf.int32, [None]),
        # only used for pairwise training and evaluation
        "labels": tf.placeholder(tf.int32, [None]),
    }

def build_model(config, node_feature_dim, edge_feature_dim):
    """Create model for training and evaluation.

  Args:
    config: a dictionary of configs, like the one created by the
      `get_default_config` function.
    node_feature_dim: int, dimensionality of node features.
    edge_feature_dim: int, dimensionality of edge features.

  Returns:
    tensors: a (potentially nested) name => tensor dict.
    placeholders: a (potentially nested) name => tensor dict.
    model: a GraphEmbeddingNet or GraphMatchingNet instance.

  Raises:
    ValueError: if the specified model or training settings are not supported.
  """
    encoder = GraphEncoder(**config["encoder"])
    aggregator = GraphAggregator(**config["aggregator"])
    if config["model_type"] == "embedding":
        model = GraphEmbeddingNet(encoder, aggregator, **config["graph_embedding_net"])
    elif config["model_type"] == "matching":
        model = GraphMatchingNet(encoder, aggregator, **config["graph_matching_net"])
    else:
        raise ValueError("Unknown model type: %s" % config["model_type"])

    training_n_graphs_in_batch = config["training"]["batch_size"]
    if config["training"]["mode"] == "pair":
        training_n_graphs_in_batch *= 2
    elif config["training"]["mode"] == "triplet":
        training_n_graphs_in_batch *= 4
    else:
        raise ValueError("Unknown training mode: %s" % config["training"]["mode"])

    placeholders = build_placeholders(node_feature_dim, edge_feature_dim)

    # training
    model_inputs = placeholders.copy()
    del model_inputs["labels"]
    model_inputs["n_graphs"] = training_n_graphs_in_batch
    graph_vectors = model(**model_inputs)

    if config["training"]["mode"] == "pair":
        x, y = reshape_and_split_tensor(graph_vectors, 2)
        labels = placeholders["labels"]
        # loss = pairwise_loss(
        #     x,
        #     y,
        #     placeholders["labels"],
        #     loss_type=config["training"]["loss"],
        #     margin=config["training"]["margin"],
        # )

        # optionally monitor the similarity between positive and negative pairs
        # is_pos = tf.cast(tf.equal(placeholders["labels"], 1), tf.float32)
        # is_neg = 1 - is_pos
        # n_pos = tf.reduce_sum(is_pos)
        # n_neg = tf.reduce_sum(is_neg)
        sim = compute_similarity(config, x, y)
        # sim_pos = tf.reduce_sum(sim * is_pos) / (n_pos + 1e-8)
        # sim_neg = tf.reduce_sum(sim * is_neg) / (n_neg + 1e-8)
    else:
        x_1, y, x_2, z = reshape_and_split_tensor(graph_vectors, 4)
        loss = triplet_loss(
            x_1,
            y,
            x_2,
            z,
            loss_type=config["training"]["loss"],
            margin=config["training"]["margin"],
        )

        # sim_pos = tf.reduce_mean(compute_similarity(config, x_1, y))
        # sim_neg = tf.reduce_mean(compute_similarity(config, x_2, z))

    # graph_vec_scale = tf.reduce_mean(graph_vectors ** 2)
    # if config["training"]["graph_vec_regularizer_weight"] > 0:
    #     loss += (
    #         config["training"]["graph_vec_regularizer_weight"] * 0.5 * graph_vec_scale
    #     )
        # 下面的无用
    # # monitor scale of the parameters and gradients, these are typically helpful
    # optimizer = tf.train.AdamOptimizer(
    #     learning_rate=config["training"]["learning_rate"]
    # )
    # grads_and_params = optimizer.compute_gradients(loss)
    # grads, params = zip(*grads_and_params)
    # grads, _ = tf.clip_by_global_norm(grads, config["training"]["clip_value"])
    # train_step = optimizer.apply_gradients(zip(grads, params))
    #
    # grad_scale = tf.global_norm(grads)
    # param_scale = tf.global_norm(params)
    #
    # # evaluation
    # model_inputs["n_graphs"] = config["evaluation"]["batch_size"] * 2
    # eval_pairs = model(**model_inputs)
    # x, y = reshape_and_split_tensor(eval_pairs, 2)
    # similarity = compute_similarity(config, x, y)
    # # pair_auc = auc(similarity, placeholders["labels"])
    #
    # model_inputs["n_graphs"] = config["evaluation"]["batch_size"] * 4
    # eval_triplets = model(**model_inputs)
    # x_1, y, x_2, z = reshape_and_split_tensor(eval_triplets, 4)
    # sim_1 = compute_similarity(config, x_1, y)
    # sim_2 = compute_similarity(config, x_2, z)
    # triplet_acc = tf.reduce_mean(tf.cast(sim_1 > sim_2, dtype=tf.float32))

    return (
        {
            # "train_step": train_step,
            "metrics": {
                "training": {
                    "x": x,
                    "y": y,
                    "sim": sim,
                    "label": labels,
                    # "loss": loss,
                    # "grad_scale": grad_scale,
                    # "param_scale": param_scale,
                    # "graph_vec_scale": graph_vec_scale,
                    # "sim_pos": sim_pos,
                    # "sim_neg": sim_neg,
                    # "sim_diff": sim_pos - sim_neg,

                },
                "validation": {
                    # "pair_auc": pair_auc,
                    # "triplet_acc": triplet_acc,
                },
            },
        },
        placeholders,
        model,
    )

def fill_feed_dict(placeholders, batch):
    """Create a feed dict for the given batch of data.

  Args:
    placeholders: a dict of placeholders.
    batch: a batch of data, should be either a single `GraphData` instance for
      triplet training, or a tuple of (graphs, labels) for pairwise training.

  Returns:
    feed_dict: a feed_dict that can be used in a session run call.
  """
    if isinstance(batch, GraphData):
        graphs = batch
        labels = None
    else:
        graphs, labels = batch

    feed_dict = {
        placeholders["node_features"]: graphs.node_features,
        placeholders["edge_features"]: graphs.edge_features,
        placeholders["from_idx"]: graphs.from_idx,
        placeholders["to_idx"]: graphs.to_idx,
        placeholders["graph_idx"]: graphs.graph_idx,
    }
    if labels is not None:
        feed_dict[placeholders["labels"]] = labels
        a = 1
    return feed_dict

test_graphs = []

def get_test_graphs(fn):
    for i in range(0, 14):
        # [0, 13]
        prefix = "../" + repo_br[i][0] + "-3000/"
        filename = prefix + fn +".test_graphs"
        test_graphs.append(filename)
    print(test_graphs)
    print(len(test_graphs))

dup_in_repo = [199, 152, 112, 104, 103, 74, 62, 61, 57, 55, 52, 46, 42, 30]

"""Main run process"""
if __name__ == "__main__":
    fileName = "remove_9"
    if len(sys.argv) == 4:
        fileName = sys.argv[1] + "_" + sys.argv[2]
        windowSize = int(sys.argv[2])
        globalStep = int(sys.argv[3])
    get_test_graphs(fileName)
    print(fileName)
    config = get_default_config()
    config["training"]["n_training_steps"] = 2
    tf.reset_default_graph()

    # Set random seeds
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.set_random_seed(seed + 2)

    # train_graphs,train_val_graphs, test_graphs = load_data()
    # print("len(train_graphs)", len(train_graphs))
    # print("len(train_val_graphs)", len(train_val_graphs))
    #
    # print("len(test_graphs)", len(test_graphs))
    # test_graphs = load_data()
    # print("len(test_graphs)", len(test_graphs))
    batch_size = config["training"]["batch_size"]
    print("batch_size", batch_size)
    tensors, placeholders, model = build_model(config, 300, 1)
    accumulated_metrics = collections.defaultdict(list)
    t_start = time.time()
    init_ops = (tf.global_variables_initializer(), tf.local_variables_initializer())

    # If we already have a session instance, close it and start a new one
    if "sess" in globals():
        sess.close()
    saver = tf.train.Saver()

    cfg = tf.ConfigProto()
    cfg.gpu_options.per_process_gpu_memory_fraction = 0.1
    with tf.Session(config=cfg) as sess:
        if os.path.exists('Model/25/checkpoint'):
            print("yes")
            saver.restore(sess, 'Model/25/lyl.GMN-9')
        else:
            print("no")
            init = tf.global_variables_initializer()
            sess.run(init)
            # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        test_graph = []

        # repo_result = open("result.txt", "w", encoding="UTF-8")
        aa = 0
        p1_all, p2_all, p3_all, p4_all, p5_all = 0.0, 0.0, 0.0, 0.0, 0.0
        r1_all, r2_all, r3_all, r4_all, r5_all = 0.0, 0.0, 0.0, 0.0, 0.0
        for graph in test_graphs:
            print(aa)
            aa += 1

            with open(graph, 'rb') as f:
                test_graph = pkl.load(f)
                # print(test_graph)

            size = len(test_graph)
            rate1, rate2, rate3, rate4, rate5 = int(size * 0.01), int(size * 0.02), int(size * 0.03), int(size * 0.04), int(size * 0.05)
            dup_all = dup_in_repo[aa - 1]
            print("size", size, "dup_all", dup_all,
                  "rate1", rate1, "rate2", rate2, "rate3", rate3, "rate4", rate4, "rate5", rate5)

            sim_list = {}
            label_list = {}
            for i in tqdm(range(len(test_graph))):
                batch = test_graph[i]
                # print(batch)
                sim, label = sess.run(
                    [tensors["metrics"]["training"]["sim"], tensors["metrics"]["training"]["label"]],
                    feed_dict=fill_feed_dict(placeholders, batch)
                )

                sim_list[i] = 1 / (1 + ((-sim[0]) ** 0.5))
                label_list[i] = label
                # print(sim_list[i], label_list[i])
            # print(sim_list)
            print(len(sim_list))

            sim_list_sorted_20 = [(x, y) for x, y in sorted(sim_list.items(), key=lambda x: x[1], reverse=True)][:rate5]
            sim_list_sorted_10 = sim_list_sorted_20[:rate4]
            sim_list_sorted_5 = sim_list_sorted_20[:rate3]
            sim_list_sorted_1 = sim_list_sorted_20[:rate2]
            sim_list_sorted_0 = sim_list_sorted_20[:rate1]
            # print(sim_list_sorted_20)
            # print(sim_list_sorted_10)
            # print(sim_list_sorted_5)
            # print(sim_list_sorted_1)
            # print(sim_list_sorted_0)
            label_list_sorted = []
            for x, y in sim_list_sorted_20:
                # print(label_list[x])
                label_list_sorted.extend(label_list[x])
            print(label_list_sorted)
            f_0, f_1, f_5, f_10, f_20 =0, 0, 0, 0, 0
            for i in range(0,len(label_list_sorted)):
                if label_list_sorted[i] == 1:
                    if i < rate1:
                        f_0 += 1
                    if i < rate2:
                        f_1 += 1
                    if i < rate3:
                        f_5 += 1
                    if i < rate4:
                        f_10 += 1
                    if i < rate5:
                        f_20 += 1
            p1, p2, p3, p4, p5 = f_0 / rate1, f_1 / rate2, f_5 / rate3, f_10 / rate4, f_20 / rate5
            r1, r2, r3, r4, r5 = f_0 / dup_in_repo[aa - 1], f_1 / dup_in_repo[aa - 1], f_5 / dup_in_repo[aa - 1], \
                                 f_10 / dup_in_repo[aa - 1], f_20 / dup_in_repo[aa - 1]
            p1_all, p2_all, p3_all, p4_all, p5_all = p1_all + p1, p2_all + p2, p3_all + p3, p4_all + p4, p5_all + p5
            r1_all, r2_all, r3_all, r4_all, r5_all = r1_all + r1, r2_all + r2, r3_all + r3, r4_all + r4, r5_all + r5
            print("%d\t%d\t%d\t%d\t%d"%(f_0, f_1, f_5,f_10,f_20))
            # print("%d\t%d\t%d\t%d\t%d" % (f_0, f_1, f_5, f_10, f_20), file = repo_result)
            print("%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n" % (p1, p2, p3, p4, p5, r1, r2, r3, r4, r5))
            # print("%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n" % (p1_all, p2_all, p3_all, p4_all, p5_all,
            #                                                     r1_all, r2_all, r3_all, r4_all, r5_all))
            # print("%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n" % (p1_all, p2_all, p3_all, p4_all, p5_all, r1_all, r2_all, r3_all, r4_all, r5_all), file=repo_result)
        p1_all, p2_all, p3_all, p4_all, p5_all = p1_all / 14, p2_all / 14, p3_all / 14, p4_all / 14, p5_all / 14
        r1_all, r2_all, r3_all, r4_all, r5_all = r1_all / 14, r2_all / 14, r3_all / 14, r4_all / 14, r5_all / 14
        print("%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n" % (p1_all, p2_all, p3_all, p4_all, p5_all,
                                                            r1_all, r2_all, r3_all, r4_all, r5_all))

