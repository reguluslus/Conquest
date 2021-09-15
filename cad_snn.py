# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import math
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

warnings.simplefilter(action='ignore', category=FutureWarning)


def read_data(file):
    df_feat = pd.read_csv(file, index_col=None)
    ground_truth = df_feat.iloc[:, -1].values

    return df_feat.iloc[:, :-1], ground_truth


def nearest_neighbours(X, k):
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    return neighbors


def find_nearest_neighbors_all(X, neighbors):
    return neighbors.kneighbors_graph(X).toarray()


def find_nearest_neighbors(k, neighbors):
    nn = neighbors.kneighbors(None, k, return_distance=False)
    return nn


def find_snn_distance(behavioral_data_train, behavioral_data_test, ref_groups, nn_b_train, nn_b_test, k,
                      dist="cosine"):
    nearest_neighbor_matrix = []
    for index, ref_group in enumerate(ref_groups):
        distances = []
        for x in ref_group:
            if dist == "cosine":
                distance = math.acos(len(np.intersect1d(nn_b_test[index], nn_b_train[x])) / k)
            else:
                distance = (1 - (len(np.intersect1d(nn_b_test[index], nn_b_train[x])) / len(nn_b_train[index])))

            dist_actual = euclidean_distances([behavioral_data_test[index]], [behavioral_data_train[x]])[0]
            distances.append(dist_actual / distance)
        nearest_neighbor_matrix.append(distances)
    return nearest_neighbor_matrix


def unify_scores(test_scores):
    from scipy.special import erf
    probs = np.zeros([len(test_scores), 2])
    _mu = np.mean(test_scores)
    _sigma = np.std(test_scores)

    pre_erf_score = (test_scores - _mu) / (_sigma * np.sqrt(2))
    erf_score = erf(pre_erf_score)
    probs[:, 1] = erf_score.clip(0, 1).ravel()
    probs[:, 0] = 1 - probs[:, 1]
    return probs


def contextual_anomaly_density(nearest_neighbor_matrix, a):
    return 1 / (np.mean(nearest_neighbor_matrix[a]) + 1e-10)


def contextual_anomaly_factor(nearest_neighbor_matrix_train, nearest_neighbor_matrix_test, ref_groups, a):
    ref_group = ref_groups[a]
    cad_a = contextual_anomaly_density(nearest_neighbor_matrix_test, a)
    sum_cads = 0
    for sample in ref_group:
        cad = contextual_anomaly_density(nearest_neighbor_matrix_train, sample)
        sum_cads = sum_cads + (cad / (cad_a + 1e-10))
    caf = sum_cads / len(ref_group)
    return caf


def score_combination_mean(score_list):
    score_list = np.asarray(score_list)
    return np.mean(score_list, axis=0)


def score_combination_max(score_list):
    score_list = np.asarray(score_list)
    return np.max(score_list, axis=0)


def rank_combination(scores_list):
    ranked_list = []
    for scores in scores_list:
        ranked_list.append(np.sort(scores))
