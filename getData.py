import os
import re
import sys
import random
import copy
import numpy as np
from sklearn.ensemble import *
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn import svm
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from numpy import array
from multiprocessing import Pool

from util import localfile

from comp import *
from git import *

from tqdm import tqdm
from repo_name import *

# ----------------INPUT & CONFIG------------------------------------
# choose model
data_for = "GMN"
# data_for = "Adaboost"
trainSize = 50000
# --------------------------------------------------------------------
default_model = 'boost'  # default model: AdaBoost

data_folder = 'data/clf'
# path label type
dataset = [
    [data_folder + '/first_msr_pairs.txt', 1, 'train'],
    # [data_folder + '/second_msr_pairs.txt', 1, 'test'],
    [data_folder + '/first_nondup.txt', -1, 'train'],
    # [data_folder + '/second_nondup.txt', -1, 'test'],
]

# model save name
model_data_save_path_suffix = 'ok_text_%s_code_%s_%s' % (text_sim_type, code_sim_type, extract_sim_type)
# if add time or conf
if add_timedelta:
    model_data_save_path_suffix += '_add_time'

if add_conf:
    model_data_save_path_suffix += '_add_conf'

part_params = None

model_data_random_shuffle_flag = False
model_data_renew_flag = False

# ------------------------------------------------------------

# init NLP model
def init_model_with_pulls(pulls, save_id=None):
    t = [str(pull["title"]) for pull in pulls]
    b = []
    for pull in pulls:
        if pull["body"] and (len(pull["body"]) <= 2000):
            b.append(pull["body"])
    init_model_from_raw_docs(t + b, save_id)
    if code_sim_type == 'tfidf':
        c = []
        for pull in pulls:  # only added code
            try:
                if not check_large(pull):
                    p = copy.deepcopy(pull)
                    p["file_list"] = fetch_pr_info(p)
                    c.append(get_code_tokens(p)[0])
            except Exception as e:
                print('Error on get', pull['url'])

        init_code_model_from_tokens(c, save_id + '_code' if save_id is not None else None)


def init_model_with_repo(repo, save_id=None):
    print('init nlp model with %s data!' % repo)

    if save_id is None:
        save_id = repo.replace('/', '_') + '_allpr'
    else:
        save_id = repo.replace('/', '_') + '_' + save_id
    try:
        init_model_with_pulls([], save_id)
    except Exception as e:
        # print(e)
        # init_model_with_pulls(shuffle(get_repo_info(repo, 'pull'))[:10000], save_id)
        init_model_with_pulls(get_repo_info(repo, 'pull'), save_id)


# Calculate feature vector.
def get_sim(repo, num1, num2):
    if data_for == "GMN":
        return get_pull_info(repo, num1, num2)
    else:
        p1 = get_pull(repo, num1)
        p2 = get_pull(repo, num2)
        return get_pr_sim_vector(p1, p2)

def get_sim_wrap(args):
    return get_sim(*args)

def get_feature_vector(data, label, renew=False, out=None):
    print('Model Data Input=', data)
    if data_for == "GMN":
        print("GMN")
        default_path = data.replace('.txt', '') + '_pull_info'
        X_path, y_path = default_path + '_X.txt', default_path + '_y.txt'
        print("X_path", X_path, "y_path", y_path)
    else:
        default_path = data.replace('.txt', '') + '_feature_vector'
        out = default_path if out is None else default_path + '_' + out
        X_path, y_path = out + '_X.json', out + '_y.json'

    if os.path.exists(X_path) and os.path.exists(y_path) and (not renew):
        print('warning: feature vector already exists!')
        X = localfile.get_file(X_path)
        y = localfile.get_file(y_path)
        return X, y

    X, y = [], []

    # run with all PR's info model
    p = {}
    with open(data) as f:
        all_pr = f.readlines()

    for l in all_pr:
        r, n1, n2 = l.strip().split()
        #
        # if 'msr_pairs' not in data:
        #     if check_large(get_pull(r, n1)) or check_large(get_pull(r, n2)):
        #         continue

        if r not in p:
            p[r] = []
        p[r].append((n1, n2, label))

    print('all=', len(all_pr))

    # out_file = open(out + '_X_and_Y.txt', 'w+')
    #
    for r in p:
        init_model_with_repo(r)

    for r in p:
        print('Start running on', r)

        # init NLP model
        init_model_with_repo(r)

        print('pairs num=', len(p[r]))

        # run parallel
        for label in [-1, 1]:
            pairs = []
            result = []
            for z in tqdm(p[r]):
                if z[2] == label:

                    pairs.append((r, z[0], z[1]))
                    temp_result = get_sim_wrap((r, z[0], z[1]))
                    result.append(temp_result)
                    # print(result)
            # print("pairs",pairs)
            # with Pool(processes=20) as pool:
            #
            #     result = pool.map(get_sim_wrap, pairs)

            X.extend(result)
            y.extend([label for i in range(len(result))])
            # y.extend(pairs)

    # out_file.close()

    # save to local
    localfile.write_to_file(X_path, X)
    localfile.write_to_file(y_path, y)
    return (X, y)


# Build classification model
def getData():
    def model_data_prepare(dataset):
        X_train, y_train = [], []
        X_test, y_test = [], []

        for s in dataset:
            new_X, new_y = get_feature_vector(s[0], s[1], model_data_renew_flag, model_data_save_path_suffix)
            if s[2] == 'train':
                if len(new_X) == 50000:
                    X_train += new_X[0:trainSize]
                    y_train += new_y[0:trainSize]
                else:
                    X_train += new_X
                    y_train += new_y
            elif s[2] == 'test':
                X_test += new_X
                y_test += new_y

        # random sorts
        def get_ran_shuffle(X, y, train_percent=0.5):
            X, y = shuffle(X, y, random_state=12345)
            num = len(X)
            train_num = int(num * train_percent)
            X_train, X_test = X[:train_num], X[train_num:]
            y_train, y_test = y[:train_num], y[train_num:]
            return (X_train, y_train, X_test, y_test)

        # ran shuffle with train set and test set
        if model_data_random_shuffle_flag:
            X_train, y_train, X_test, y_test = get_ran_shuffle(X_train + X_test, y_train + y_test)

        return (X_train, y_train, X_test, y_test)

    print('--------------------------')
    print('Loading Data')
    X_train, y_train, X_test, y_test = model_data_prepare(dataset)
    if part_params:
        def extract_col(a, c):
            for i in range(len(a)):
                t = []
                for j in range(len(c)):
                    if c[j] == 1:
                        t.append(a[i][j])
                a[i] = t

        extract_col(X_train, part_params)
        extract_col(X_test, part_params)
        print('extract=', part_params)

    print('--------------------------')
    print('Size of Dataset: training_set', len(X_train), 'testing_set', len(X_test), 'feature_length=', len(X_train[0]))


def get_dataset():
    for i in range(0, 14):
        # [0, 13]
        prefix = repo_br[i][0] + "-3000/"
        filename = prefix + "dup_" + repo_br[i][0] + ".txt"
        filename1 = prefix + "non_dup_" + repo_br[i][0] + ".txt"
        dataset.append([filename, 1, "test"])
        dataset.append([filename1, -1, "test"])

if __name__ == "__main__":
    # input GMN/Adaboost trainingSize
    if len(sys.argv) == 3:
        data_for = sys.argv[1].strip()
        print("data_for", data_for)
        trainSize = int(sys.argv[2].strip())
        print("trainSize", trainSize)
    get_dataset()
    # print(dataset)
    getData()
