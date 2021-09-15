import numpy as np
from pymcdm.helpers import rrankdata
from pymcdm.methods import TOPSIS
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

import cad_snn
import context_values

data, ground_truth = cad_snn.read_data('pima.csv')


def find_combinations(num_of_features):
    import itertools as iter
    combination_list = [i for i in range(num_of_features)]
    combinations = [iter.combinations(combination_list, n) for n in range(1, len(combination_list))]
    flat_combinations = iter.chain.from_iterable(combinations)
    result = list(map(lambda x: [list(x), list(set(combination_list) - set(x))], flat_combinations))
    final_results = []
    for i, res in enumerate(result):
        final_results.append(res)

    return final_results


def test(true_context, X_train_original, X_test_original, y_test, k, dist=None):
    contextual_data_train = X_train_original.values[:, true_context[0]]
    behavioral_data_train = X_train_original.values[:, true_context[1]]
    contextual_data_test = X_test_original.values[:, true_context[0]]
    behavioral_data_test = X_test_original.values[:, true_context[1]]

    neighbors_con = cad_snn.nearest_neighbours(contextual_data_train, k)
    nn_con_train = neighbors_con.kneighbors(None, k, return_distance=False)
    nn_con_test = neighbors_con.kneighbors(contextual_data_test, k, return_distance=False)

    matrix_train = cad_snn.find_snn_distance(behavioral_data_train, behavioral_data_train, nn_con_train,
                                             nn_con_train, nn_con_train, k, dist=dist)
    matrix_test = cad_snn.find_snn_distance(behavioral_data_train, behavioral_data_test, nn_con_test,
                                            nn_con_train, nn_con_test, k, dist=dist)

    caf_scores = []

    for i in range(len(contextual_data_test)):
        caf = cad_snn.contextual_anomaly_factor(matrix_train, matrix_test, nn_con_test, i)
        caf_scores.append(caf)
    return caf_scores, average_precision_score(y_test, caf_scores)


def get_topsis_list(indexes, values):
    topsis = TOPSIS()
    weights = np.array([2, 1, 1])
    types = np.array([-1, -1, -1])
    pref = topsis(values, weights, types)
    ranking = rrankdata(pref)
    order = np.argsort(ranking)
    indexes = np.asarray(indexes)
    indexes = indexes[order]
    # ranking = np.array(ranking[:10])
    return indexes


combinations = find_combinations(8)
print('arrhythmia')
con_indexes = context_values.arr_context_4_indexes
con_vals = context_values.arr_context_4_vals
top_5 = get_topsis_list(con_indexes, con_vals)
top_5_contexts = context_values.index_to_context(top_5, combinations)

acc_array = []

for i in range(10):
    arr = []
    hash_table_scores = dict()
    X_train_original, X_test_original, y_train, y_test = train_test_split(data, ground_truth,
                                                                          test_size=0.3, stratify=ground_truth)
    for all_contexts in top_5_contexts:
        caf_list = []
        keys = []
        for j, c in enumerate(all_contexts):
            key = context_values.find_key(c, combinations)
            if hash_table_scores.get(key) is not None:
                scores = hash_table_scores[key]
            else:
                k = 40
                scores, acc = test(c, X_train_original, X_test_original, y_test, k, "cosine")
                hash_table_scores[key] = scores
            caf_list.append(scores)
            keys.append(key)

        scores_mean = cad_snn.score_combination_mean(caf_list)
        scores_max = cad_snn.score_combination_max(caf_list)

        print("index:", i)
        print(keys)
        print(average_precision_score(y_test, scores_mean))
        print(average_precision_score(y_test, scores_max))

        arr.append(average_precision_score(y_test, scores_mean))

    print("---------------------------------------------------")
    acc_array.append(arr)

acc_array = np.asarray(acc_array)

print(np.mean(acc_array, axis=0))
print(np.std(acc_array, axis=0))
