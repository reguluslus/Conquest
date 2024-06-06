import numpy as np
from pymoo.model.problem import Problem
from scipy import stats
from scipy.spatial.distance import pdist, squareform

import cad_snn


class ContextSearch(Problem):
    def __init__(self,
                 L,
                 n_max,
                 context_list,
                 data
                 ):
        super().__init__(n_var=len(L), n_obj=3, n_constr=1, elementwise_evaluation=True)
        self.L = L
        self.n_max = n_max
        self.contexts = context_list
        self.hash_table_kurtos = dict()
        self.hash_table_corr = dict()
        self.data=data

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = []
        f2 = []
        f3 = []
        a = self.L[x]
        for i in range(len(self.L[x])):
            # new_key=keys
            index1 = self.L[x][i]
            #print(index1)
            #print(self.contexts[index1])
            contextual_data = self.data[:, self.contexts[index1][0]]
            behavioral_data = self.data[:, self.contexts[index1][1]]
            key1 = str(self.contexts[index1][0])
            key2 = str(self.contexts[index1][1])
            kurtosis = self.get_kurtos(behavioral_data, contextual_data, key1, key2, k=20)
            dependency = self.get_correlation(behavioral_data, contextual_data, key1, key2)
            # Objective max kurtosis (min negative kurtosis)
            f1.append(0 - kurtosis)
            # Objective max dependency (min reverse dependency)
            f2.append(1 - dependency)
            # Objective 2
            for j in range(i + 1, len(self.L[x])):
                index2 = self.L[x][j]
                contextual_data2 = self.data[:, self.contexts[index2][0]]
                key3 = str(self.contexts[index2][0])
                redundancy = self.get_correlation(contextual_data2, contextual_data, key1, key3)
                # Objective min redundancy
                f3.append(redundancy)

        #print("kurtosis", np.mean(f1))
        #print("dependency", np.mean(f2))
        #print("redundancy", np.mean(f3))
        #print(self.L[x])

        out["F"] = np.array([np.mean(f1), np.mean(f2), np.mean(f3)])
        out["G"] = np.column_stack([0, 0, 0])

    def get_correlation(self, behavioral_data, contextual_data, key1, key2):
        if self.hash_table_corr.get(key1 + key2) is not None:
            dcor = self.hash_table_corr[key1 + key2]
        elif self.hash_table_corr.get(key2 + key1) is not None:
            dcor = self.hash_table_corr[key2 + key1]
        else:
            dcor = dist_corr(contextual_data, behavioral_data)
            self.hash_table_corr[key1 + key2] = dcor
        return dcor

    def get_kurtos(self, behavioral_data, contextual_data, key1, key2, k):
        if self.hash_table_kurtos.get(key1 + key2) is not None:
            dcor = self.hash_table_kurtos[key1 + key2]
        else:
            neighbors_con = cad_snn.nearest_neighbours(contextual_data, k)
            nn_con_train = neighbors_con.kneighbors(None, k, return_distance=False)

            matrix_train = cad_snn.find_snn_distance(behavioral_data, behavioral_data, nn_con_train,
                                                     nn_con_train, nn_con_train, k, dist='cosine')
            dcor = stats.kurtosis(np.mean(matrix_train, axis=1), fisher=True)
            #print(0 - dcor)
            self.hash_table_kurtos[key1 + key2] = dcor
        return dcor


def dist_corr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)

    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor


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


def find_combinations_digits(num_of_features):
    import itertools as iter
    import copy
    combination_list = [i for i in range(num_of_features)]
    combinations = [iter.combinations(combination_list, n) for n in range(1, len(combination_list))]
    flat_combinations = iter.chain.from_iterable(combinations)
    result = list(map(lambda x: [list(x), list(set(combination_list) - set(x))], flat_combinations))
    final_results = []
    for i, res in enumerate(result):
        new_res = []
        for i, x in enumerate(res):
            new_res_temp = []
            for y in x:
                new_res_temp.append(y + y)
                new_res_temp.append(y + y + 1)
            new_res.append(new_res_temp)
        final_results.append(new_res)

    return final_results




from pymoo.model.crossover import Crossover
from pymoo.model.mutation import Mutation
from pymoo.model.sampling import Sampling


class MySampling(Sampling):
    def __init__(self):
        self.offset = False

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=bool)
        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[:problem.n_max]
            X[k, I] = True

        return X


class BinaryCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.n_max - np.sum(both_are_true)

            I = np.where(np.logical_xor(p1, p2))[0]

            S = I[np.random.permutation(len(I))][:n_remaining]
            _X[0, k, S] = True

        return _X


class MyMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            X[i, np.random.choice(is_false)] = True
            X[i, np.random.choice(is_true)] = False

        return X