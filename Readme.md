
# Dup-Hunter

## Dup-Hunter: Detecting Duplicate Contributions in Fork-Based Development

Python library dependencies:
+ tensorflow -v : 1.13.1
+ numpy -v : 1.18.5
+ nltk -v : 3.4.5
+ flask -v : 1.1.1
+ GitHub-Flask -v : 3.2.0
+ gensim -v : 3.8.3
+ scipy -v : 1.4.1 
+ others: sklearn, bs4,

---

Dataset:

[dupPR]: Reference paper: Yu, Yue, et al. "A dataset of duplicate pull-requests in github." Proceedings of the 15th International Conference on Mining Software Repositories. ACM, 2018. (link: http://yuyue.github.io/res/paper/DupPR-msr2017.pdf)
<including: 2323 Duplicate PR pairs in 26 repos>

---
If you want to use our model quickly, five steps need to be done.

First, run `getData.py` to get the data information file;

Second, run `getGraph_train_data.py` to get the training data graph;

Third, run `getGraph_repo_test_data.py` to get the testing data graph;

Fourth, run `gmn/train.py` to train the model;

Fifth, run `gmn/getResult.py` to get the model result.

+ getData.py
  
    `python getData.py GMN N1`
    
    It will generate `xxx_pull_info_X.txt` and `xxx_pull_info_y.txt` files which include all data needed to build graph. The N1 indicates which model to use(GMN or Adaboost). The third parameter indicates the dataset size of the non-duplicate PR pairs used for training.
    
+ getGraph_train_data.py.py
    
    `python getGraph_train_data.py N1`
    
    It will take `xxx_pull_info_X.txt` and `xxx_pull_info_y.txt` files from `data/clf` directory to build the training data graph. The N1 indicates the sliding window size used to build the graph, then two graph files will be generated in the `data/clf` directory, as follows:
      
    `remove_XXX_N.train_graphs`, `remove_XXX_N.train_val_graphs`
    
    >Note: XXX can be` `, `title`, `body`. Where N represents the sliding window size.

+ getGraph_repo_test_data.py
   
    `python getGraph_repo_test_data.py N1 N2`
    
    It will take `xxx_pull_info_X.txt` and `xxx_pull_info_y.txt` files from each repository directory to build the testing data graph. The N1 indicates the sliding window size used to build the graph. The N2 indicates the graph building of which repository. Then `remove_XXX_N.test_graphs` will be generated in the each repository directory.

+ gmn/train.py  

    `python gmn/train.py`
    
    It will take `XXX.train_graphs` and `XXX.train_val_graphs` in the `data/clf` directory to train the model.
    
    >Note: XXX can be `remove'_N1`, `remove_title_N1`, `remove_body_N1`. The N1 means the slid window size.
    
+ gmn/getResult.py

     `python gmn/getResult.py`
    
    It will take `XXX.test_graphs` in each repository directory to test the model.
    

---

layers.py: Including the encoder layer, propagator layer, and Aggregator layer of the GMN model.

```
class GraphEncoder(snt.AbstractModule)
class GraphPropLayer(snt.AbstractModule)
class GraphAggregator(snt.AbstractModule)
``` 

getData.py: Get data from github using API.

``` python
# Set up the input dataset
# Get the text of title description and the similarity of features 
getData()
```

getGraph_train_data.py: Get graph built of text features and non-text features.

```
build_graph(start, end) # build graph for train, validate and test
train_graph = build_graph(start=0, end=real_train_size)
train_val_graph = build_graph(start=real_train_size, end=train_size)
test_graph = build_graph(start=train_size, end=train_size + test_size)
``` 

gmn/batch.py: Combine the data into a batch.

```
batch_graph(dataset, from, batch) # Start from the "from" position of the dataset, take batches and stack them up
```

train.py: Feed the graph built of the training dataset to the GMN.

```
# Load the data and the initial model
build_placeholders(node_feature_dim, edge_feature_dim)
def build_model(config, node_feature_dim, edge_feature_dim)
def fill_feed_dict(placeholders, batch)
```

getResult.py: Load the model saved during training, use test graph to test the model.
```
# The same as the train.py. The Precesion@k, recall@k, F1@k metrics are used to test model.
```

util.py: Contains the config of the model, but also the calculation formula for similarity and loss.
```
compute_cross_attention(x, y, sim)
euclidean_distance(x, y)
airwise_loss(x, y, labels, loss_type="margin", margin=0.0)
get_default_config()
```

nlp.py: Natural Language Processing model for calculating the text similarity.


```
m = Model(texts)
text_sim = query_sim_tfidf(tokens1, tokens2)
``` 


comp.py: Calculate the similarity for feature extraction.

``` 
# Set up the params of compare (different metrics).
# Check for init NLP model.
feature_vector = get_pr_sim_vector(pull1, pull2)
```


---

git.py: About GitHub API setting and fetching.

``` python
get_repo_info('repositories',
              'fork' / 'pull' / 'issue' / 'commit' / 'branch',
              renew_flag)

get_pull(repo, num, renew)
get_pull_commit(pull, renew)
fetch_file_list(pull, renew)
get_another_pull(pull, renew)
check_too_big(pull)
```


fetch_raw_diff.py: Get data from API, parse the raw diff.

```
parse_diff(file_name, diff) # parse raw diff
fetch_raw_diff(url) # parse raw diff from GitHub API
```


