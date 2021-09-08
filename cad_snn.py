# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import math

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import jaccard
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)


# Method to compute Distance
# Define other distance functions here

def read_data_syn(num_of_gaussians, n_samples, file_name='features_syntatic_overlapping.csv'):
    df_feat = pd.read_csv(file_name, index_col=None)
    df_normal = df_feat[:n_samples * num_of_gaussians]
    df_anomaly = df_feat[n_samples * num_of_gaussians:]
    ground_truth = [0 for i in range(len(df_normal))]
    for a in range(len(df_anomaly)):
        ground_truth.append(1)
    return df_feat, ground_truth


def read_data_houses(file):
    df_feat = pd.read_csv(file, index_col=None)
    ground_truth = [0 for i in range(len(df_feat) - 1879)]
    for a in range(1879):
        ground_truth.append(1)
    return df_feat, ground_truth


def read_data(file):
    df_feat = pd.read_csv(file, index_col=None)
    ground_truth = df_feat.iloc[:, -1].values

    return df_feat.iloc[:, :-1], ground_truth


def read_data_abalone(file):
    df_feat = pd.read_csv(file, index_col=None)
    ground_truth = df_feat['9'].values
    # x1=ground_truth.count(0)
    # x2 = np.count_nonzero(ground_truth==1)

    # df_feat = df_feat.iloc[:, :]

    return df_feat.iloc[:, :-1], ground_truth


# df_normal, df_anomaly = read_data(num_of_clusters, 5000, 'synthetic2.csv')


def nearest_neighbours(X, k):
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    return neighbors


def find_reverse_nearest_neighbours(nn_list):
    ref_groups = []
    for i in range(len(nn_list)):
        reverse_neighbours = []
        for j, neighbours in enumerate(nn_list):
            if i in neighbours:
                reverse_neighbours.append(j)
        # revs = find_reverse_nearest_neighbours(nn_c, a)
        ref_group = [*nn_list[i], *reverse_neighbours]
        ref_group = list(set(ref_group))
        ref_groups.append(ref_group)
    return ref_groups


def find_nearest_neighbors_all(X, neighbors):
    return neighbors.kneighbors_graph(X).toarray()


def find_nearest_neighbors(k, neighbors):
    nn = neighbors.kneighbors(None, k, return_distance=False)
    return nn


def find_shared_nearest_neighbors(nn, k, a, b):
    a_neighbors = nn[a]
    b_neighbors = nn[b]
    shared_neighbors = np.intersect1d(a_neighbors, b_neighbors)
    return shared_neighbors, len(shared_neighbors)


def find_nearest_neighbors_shared(nn_c, nn_b):
    nearest_neighbor_matrix = []
    for index, ref_group in enumerate(nn_c):
        distances = []
        for x in ref_group:
            distances.append(1 - (len(np.intersect1d(nn_b[index], nn_b[x])) / len(nn_b[index])))
        nearest_neighbor_matrix.append(distances)
    return nearest_neighbor_matrix


def find_nearest_neighbors_shared_dis(behavioral_data_train, behavioral_data_test, ref_groups, nn_b_train, nn_b_test, k,
                                      dist="jaccard"):
    nearest_neighbor_matrix = []
    for index, ref_group in enumerate(ref_groups):
        distances = []
        for x in ref_group:
            if dist == "jaccard":
                distance = jaccard(nn_b_test[index].tolist(), nn_b_train[x].tolist())
            elif dist == "euc":
                distance = 1
            elif dist == "cosine":
                distance = math.acos(len(np.intersect1d(nn_b_test[index], nn_b_train[x])) / k)
            else:
                distance = (1 - (len(np.intersect1d(nn_b_test[index], nn_b_train[x])) / len(nn_b_train[index])))

            dist_actual = euclidean_distances([behavioral_data_test[index]], [behavioral_data_train[x]])[0]
            #distances.append(distance * math.sqrt(dist_actual))
            distances.append(dist_actual / distance)
        nearest_neighbor_matrix.append(distances)
    return nearest_neighbor_matrix


def find_nearest_neighbors_mutual(behavioral_data_train, behavioral_data_test, ref_groups, nn_b_train, nn_b_test, k):
    nearest_neighbor_matrix = []
    for index, ref_group in enumerate(ref_groups):
        distances = []
        for x in ref_group:
            if len(np.intersect1d(nn_b_test[index], nn_b_train[x])) > 0:
                distances.append(
                    1 - (1 / (euclidean_distances([behavioral_data_test[index]], [behavioral_data_train[x]])[0] + 1)))
            else:
                distances.append(1)
        nearest_neighbor_matrix.append(distances)
    return nearest_neighbor_matrix


def find_nearest_neighbors_mutual2(behavioral_data_train, behavioral_data_test, ref_groups, ref_groups_test, nn_b_test,
                                   k):
    nearest_neighbor_matrix = []
    for index, nn_b_test in enumerate(nn_b_test):
        distances = []
        for x in nn_b_test:
            if len(np.intersect1d(ref_groups_test[index], ref_groups[x])) > 0:
                distances.append(
                    1 / (euclidean_distances([behavioral_data_test[index]], [behavioral_data_train[x]])[0] + 1))
            else:
                distances.append(0)
        nearest_neighbor_matrix.append(distances)
    return nearest_neighbor_matrix


def find_nearest_neighbors_cosine_sim(behavioral_data, ref_groups, nn_b):
    nearest_neighbor_matrix = []
    for index, ref_group in enumerate(ref_groups):
        distances = []
        for x in ref_group:
            distances.append(cosine_similarity([np.sort(nn_b[index])], [np.sort(nn_b[index])])[0] / (
                    euclidean_distances([behavioral_data[index]], [behavioral_data[x]])[0] + 1e-10))
        nearest_neighbor_matrix.append(distances)
    return nearest_neighbor_matrix


def find_nearest_neighbors_cosine_dis(behavioral_data, ref_groups, nn_b):
    nearest_neighbor_matrix = []
    for index, ref_group in enumerate(ref_groups):
        distances = []
        for x in ref_group:
            distances.append(spatial.distance.cosine(np.sort(nn_b[index]), np.sort(nn_b[x])) *
                             euclidean_distances([behavioral_data[index]], [behavioral_data[x]])[0])
        nearest_neighbor_matrix.append(distances)
    return nearest_neighbor_matrix


def find_nearest_neighbors_euc(behavioral_data, nn_c, nn_b):
    nearest_neighbor_matrix = []
    for index, ref_group in enumerate(nn_c):
        index_to_neighbors = behavioral_data[ref_group]
        distances = []
        for x in ref_group:
            distances.append(euclidean_distances(index_to_neighbors, behavioral_data[nn_c[x]])[0])
        nearest_neighbor_matrix.append(distances)
    return nearest_neighbor_matrix


def find_nearest_neighbors_lof_simple(behavioral_data, nn_c):
    nearest_neighbor_matrix = []
    for index, ref_group in enumerate(nn_c):
        distances = []
        for x in ref_group:
            distances.append(euclidean_distances([behavioral_data[index]], [behavioral_data[x]])[0])
        nearest_neighbor_matrix.append(distances)
    return nearest_neighbor_matrix


def find_nearest_neighbors_lof_simple_extended(behavioral_data, ref_groups):
    nearest_neighbor_matrix = []
    for index, ref_group in enumerate(ref_groups):
        distances = []
        for x in ref_group:
            distances.append(euclidean_distances([behavioral_data[index]], [behavioral_data[x]])[0])
        nearest_neighbor_matrix.append(distances)
    return nearest_neighbor_matrix


def shared_nearest_neighbors_similarity_cosine(X, k, a, b):
    neighbors = nearest_neighbours(X, k)
    a_neighbors = find_nearest_neighbors(a, k, neighbors)
    b_neighbors = find_nearest_neighbors(b, k, neighbors)
    # shared_neighbors = np.intersect1d(a_neighbors, b_neighbors)
    return spatial.distance.cosine(a_neighbors, b_neighbors)


def contextual_anomaly_density(nearest_neighbor_matrix, a):
    shared_neighbors_list = []
    for i, sim in enumerate(nearest_neighbor_matrix[a]):
        if sim > 0:
            shared_neighbors_list.append(sim)
    if len(shared_neighbors_list) > 0:
        return 1 / np.mean(shared_neighbors_list)
    else:
        return 1 / 1e-10


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


def contextual_anomaly_density2(nearest_neighbor_matrix, a):
    return np.mean(nearest_neighbor_matrix[a]) + 1e-10
    # return sum_of_similarity / (len(shared_neighbors_list) + 1e-10)


def contextual_anomaly_density3(nearest_neighbor_matrix, a):
    return 1 / (np.mean(nearest_neighbor_matrix[a]) + 1e-10)


def shared_nearest_neighbors_similarity_2(X, k, a, b):
    dist = np.linalg.norm(a - b)
    return find_shared_nearest_neighbors(X, k, a, b) / dist


def contextual_anomaly_factor(nearest_neighbor_matrix, nn_c, a):
    ref_group = nn_c[a]
    cad_a = contextual_anomaly_density2(nearest_neighbor_matrix, a)
    sum_cads = 0
    for sample in ref_group:
        cad = contextual_anomaly_density2(nearest_neighbor_matrix, sample)
        sum_cads = sum_cads + (cad / (cad_a + 1e-10))
    caf = sum_cads / len(ref_group)
    return caf


def contextual_anomaly_factor2(nearest_neighbor_matrix_train, nearest_neighbor_matrix_test, ref_groups, a):
    ref_group = ref_groups[a]
    cad_a = contextual_anomaly_density3(nearest_neighbor_matrix_test, a)
    sum_cads = 0
    for sample in ref_group:
        cad = contextual_anomaly_density3(nearest_neighbor_matrix_train, sample)
        sum_cads = sum_cads + (cad / (cad_a + 1e-10))
    caf = sum_cads / len(ref_group)
    #caf=unify_scores(caf)
    return caf


# data, ground_truth = read_data_syn(5, 5000, 'synthetic.csv')
# test2('houses50.csv')


def precision_at(scores, ground_truth, top_index):
    sorted_indexes = np.argsort(np.asarray(scores))
    indexes_at = sorted_indexes[-top_index:]
    num_anomalies = 0
    for index in indexes_at:
        if ground_truth[index] == 1:
            num_anomalies = num_anomalies + 1

    return num_anomalies / top_index


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
