import os
import collections
import time
import copy
import traceback
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import euclidean_distances
import kmedoids
from caching import from_cache, to_cache
from infoclus_utils import *

from collections import defaultdict
from config import PROJECT_ROOT as ROOT

RUNTIME_OPTIONS = [0.01, 0.5, 1, 5, 10, 30, 60, 180, 300, 600, 1800, 3600, np.inf]
VAR_TPYE_THRESHOLD = 20
REPLACE_NAN = 0
EPSILON= 0.00001
Random_State = 42
KMEANS_COUNT = 30 # How many kmeans with different k we are going to consider, starting from the k passed in initailization
SPLITTING_STRATEGY = ['by_node','by_sibling']

DATA_FOLDER = os.path.join(ROOT, 'data')

class _Data:
    """
    Attributes:
        name
        dataset_folder
        cache_path
        data: pd.DataFrame
        data_raw: pd.DataFrame
        global_var_type
        var_type

        factorized_data: optional
        ls_mapping_chain_by_col: optional
    """
    def __init__(self, dataset_name, dataset_folder):
        self.name = dataset_name
        if dataset_folder is None:
            self.dataset_folder = os.path.join(DATA_FOLDER, dataset_name)
        else:
            self.dataset_folder = dataset_folder
        self.cache_path = os.path.join(self.dataset_folder, 'cache')

        df_data = pd.read_csv(os.path.join(self.dataset_folder, f'{dataset_name}.csv'))


        factorized_data, ls_mapping_chain_by_col, self.data, self.data_raw = get_scaled_data(df_data, REPLACE_NAN)
        self.size = self.data.shape[0]
        if factorized_data is not None and ls_mapping_chain_by_col is not None:
            self.factorized_data = factorized_data
            self.ls_mapping_chain_by_col = ls_mapping_chain_by_col

        df_var_type_complexity = get_var_type_complexity(self.data_raw, VAR_TPYE_THRESHOLD)
        self.var_type = df_var_type_complexity['var_type']
        if len(self.var_type.unique()) > 1:
            self.global_var_type = 'mixed'
        else:
            self.global_var_type = self.var_type.iloc[0]
        self._dls = df_var_type_complexity['var_complexity']

class _Embeddings:
    """
    Attributes:
        all_embeddings
        embedding
    """
    def __init__(self, data: _Data, embedding: np.ndarray=None):

        embeddings_path = os.path.join(data.dataset_folder, 'cache', 'embeddings.json')
        if os.path.exists(embeddings_path):
            with open("embeddings.json", "r", encoding="utf-8") as f:
                embeddings = json.load(f)
            self.all_embeddings = embeddings
        else:
            embeddings = get_embeddings(data.data_raw.values)
            self.all_embeddings = embeddings
            with open("embeddings.json", "w", encoding="utf-8") as f:
                json.dump(embeddings, f)

        if embedding is None:
            self.embedding = self.all_embeddings['tsne']
        else:
            self.all_embeddings['user_given'] = embedding
            self.embedding = embedding
            with open("embeddings.json", "w", encoding="utf-8") as f:
                json.dump(self.all_embeddings, f)


class _Model:
    """
    models
    Attributes:
        model

        meansForNodes
        varsForNodes
        nodesToPoints
        parents
    """
    def __init__(self, data_obj:_Data, linkage, embeddings_obj: _Embeddings):
        self.model = AgglomerativeClustering(linkage=linkage, distance_threshold=0, n_clusters=None)
        self.kmedoids_model = None
        self.model.fit(embeddings_obj.embedding)
        self._calc_statistics_numeric(data_obj.data.values)
        self._record_parents()

    def _calc_statistics_numeric(self, data: np.ndarray):

        n_samples = len(self.model.labels_)
        self.meansForNodes = {}
        self.varsForNodes = {}
        self.nodesToPoints = {}
        for i, merge in enumerate(self.model.children_):

            self.nodesToPoints[i + n_samples] = []

            for j, node in enumerate(merge):
                if node < n_samples:
                    self.meansForNodes[node] = data[node]
                    self.varsForNodes[node] = np.zeros_like(self.meansForNodes[node])
                    self.nodesToPoints[node] = [node]
                    self.nodesToPoints[i + n_samples].append(node)
                else:
                    self.nodesToPoints[i + n_samples].append(self.nodesToPoints[node])

            self.meansForNodes[i + n_samples] = recur_mean(self.meansForNodes[merge[0]], len(self.nodesToPoints[merge[0]]),
                                                     self.meansForNodes[merge[1]], len(self.nodesToPoints[merge[1]]))
            self.varsForNodes[i + n_samples] = recur_var(self.meansForNodes[merge[0]],
                                                               self.varsForNodes[merge[0]],
                                                               len(self.nodesToPoints[merge[0]]),
                                                               self.meansForNodes[merge[1]],
                                                               self.varsForNodes[merge[1]],
                                                               len(self.nodesToPoints[merge[1]])
                                                               )

        self.prior = [self.meansForNodes.get(len(data)*2-2), self.varsForNodes.get(len(data)*2-2)]

    def _record_parents(self):

        n_samples = len(self.model.labels_)

        self.parents = {}
        for index, children in enumerate(self.model.children_):
            left_child = children[0]
            right_child = children[1]
            self.parents[left_child] = index + n_samples
            self.parents[right_child] = index + n_samples

    def get_ancestors(self, node_idx):
        node_ancestors_idxes = []
        child = node_idx
        parent = self.parents[child]
        while self.parents.keys().__contains__(child):
            node_ancestors_idxes.append(self.parents[child])
            child = parent
            if self.parents.keys().__contains__(child):
                parent = self.parents[child]
            else:
                break
        return node_ancestors_idxes

    def find_closest_ancestor(self, node_ancestor_idxes, candidate_ancestors_with_labels):

        closest_ancestor = None
        closest_ancestor_cluster_label = None
        for index, ancestor_info in enumerate(candidate_ancestors_with_labels):
            ancestor_cluster_label = index
            ancestor_node_idx = ancestor_info[0]
            ancestor_ancestors = ancestor_info[1]
            if ancestor_node_idx in node_ancestor_idxes:
                if closest_ancestor is None:
                    closest_ancestor = ancestor_node_idx
                    closest_ancestor_cluster_label = ancestor_cluster_label
                elif closest_ancestor in ancestor_ancestors:
                    closest_ancestor = ancestor_node_idx
                    closest_ancestor_cluster_label = ancestor_cluster_label

        return closest_ancestor, closest_ancestor_cluster_label

    def get_samples_count_given_medoids_idxes(self, medoids_idxes):

        samples_count = 0
        for medoid_idx in medoids_idxes:
            cluster_label = medoid_idx
            set_of_samples = np.where(self.kmedoids_model.labels==cluster_label)[0]
            samples_count += len(set_of_samples)
        return samples_count

class _Paras:
    def __init__(self, data_obj: _Data, embeddings_obj: _Embeddings, model_obj: _Model,
                 alpha, emb_name, linkage, beta, min_att, max_att, run_id,
                 split_strategy_id,
                 modify_hierarchical, base_clusters):

        if modify_hierarchical:
            if base_clusters is None:
                self.base_clusters = int(data_obj.size / 5)
            else:
                self.base_clusters = base_clusters
            diss = euclidean_distances(embeddings_obj.embedding)
            model_obj.kmedoids_model = kmedoids.fasterpam(diss, self.base_clusters)
            data_obj.data = data_obj.data.iloc[model_obj.kmedoids_model.medoids]
            data_obj.size = data_obj.data.shape[0]
            embeddings_obj.embedding = embeddings_obj.embedding[model_obj.kmedoids_model.medoids]

        if alpha is None:
            self.alpha = int(data_obj.size / 10)
        else:
            self.alpha = alpha
        self.beta = beta
        self.min_att = min_att
        self.max_att = max_att
        self.runtime_id = run_id
        self.runtime = RUNTIME_OPTIONS[self.runtime_id]
        self.split_strategy = SPLITTING_STRATEGY[split_strategy_id]
        self._emb_name = emb_name
        self._linkage = linkage
        self._modify_hierarchical = modify_hierarchical

    @property
    def modify_hierarchical(self):
        return self._modify_hierarchical
    @property
    def linkage(self):
        return self._linkage
    @property
    def emb_name(self):
        return self._emb_name

    def update_paras(self, alpha, beta, min_att, max_att, run_id, split_strategy_id):
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if min_att is not None:
            self.min_att = min_att
        if max_att is not None:
            self.max_att = max_att
        if run_id is not None:
            self.runtime_id = run_id
            self.runtime = RUNTIME_OPTIONS[self.runtime_id]
        if split_strategy_id is not None:
            self.split_strategy = SPLITTING_STRATEGY[split_strategy_id]

    def get_paras(self):
        paras_val = {
            'emb_name': self.emb_name,
            'linkage': self.linkage,
            'alpha': self.alpha,
            'beta': self.beta,
            'min_att': self.min_att,
            'max_att': self.max_att,
            'runtime_id': self.runtime_id,
            'split_strategy': self.split_strategy,
            'modify_hierarchical': self.modify_hierarchical,
        }
        if self.modify_hierarchical:
            paras_val['base_clusters'] = self.base_clusters
        return paras_val

class _Result:
    def __init__(self, mean_prior: np.ndarray, var_prior: np.ndarray, sample_size: int):

        self.ic_opt = []  # ic of all attributes for each cluster
        self.si_opt = 0  # value of si for this clustering
        self.clusters_idxes_opt = [[0]*sample_size]  # all points belong to cluster 0
        self.attributes_opt = []  # chosen attributes for each cluster
        self.clusters_related_statistics_opt = [[mean_prior, var_prior, sample_size]]
        self._split_nodes_opt = []

        self.clustering = None
        self.count_clusters = None
        self.ic_val_per_cluster = None

    def update(self, ic_new, si_new, clusters_idxes_new, attributes_new, clusters_related_statistics_new, split_nodes_new):
        self.ic_opt = ic_new
        self.si_opt = si_new
        self.clusters_idxes_opt = clusters_idxes_new
        self.attributes_opt = attributes_new
        self.clusters_related_statistics_opt = clusters_related_statistics_new
        self._split_nodes_opt = split_nodes_new

    def extend_results(self, model_obj: _Model):

        clustering = np.full(len(model_obj.kmedoids_model.labels), -1)
        for cluster_label, medoids_idxes in enumerate(self.clusters_idxes_opt):
            for medoid in medoids_idxes:
                set_of_samples = np.where(model_obj.kmedoids_model.labels == medoid)[0]
                clustering[set_of_samples] = cluster_label
        self.clustering = clustering
        self.count_clusters = len(self.clusters_idxes_opt)
        self.ic_val_per_cluster = [0]*self.count_clusters
        for i in range(self.count_clusters):
            self.ic_val_per_cluster[i] = sum(self.ic_opt[i][j] for j in self.attributes_opt[i])

class InfoClus:

    def __init__(self, dataset_name: str,     # necessary

                 embedding: np.ndarray = None,  # optional: given a precomputed embedding
                 data_folder: str = None,     # optional

                 alpha=None, emb_name='tsne', linkage='single', beta=1.5, min_att=2, max_att=5, run_id=6,
                 split_strategy_id=0,
                 modify_hierarchical=True, base_clusters=None,

                 allow_cache = True
                 ):

        print('Initializing InfoClus ...')
        tic_initialization = time.time()

        self.epsilon = EPSILON
        self.allow_cache = allow_cache

        self.data_obj = _Data(dataset_name, data_folder)
        self.embeddings_obj = _Embeddings(self.data_obj, embedding)
        self.model_obj = _Model(self.data_obj, linkage, self.embeddings_obj)

        self.paras_obj = _Paras(self.data_obj, self.embeddings_obj, self.model_obj,
                 alpha, emb_name, linkage, beta, min_att, max_att, run_id,
                 split_strategy_id,
                 modify_hierarchical, base_clusters)
        self.result_obj = _Result(self.model_obj.prior[0], self.model_obj.prior[1], self.data_obj.size)

        if self.allow_cache:
            file_path = os.path.join(self.data_obj.cache_path, self.data_obj.name + '_modify_' + str(self.paras_obj.modify_hierarchical))
            to_cache(file_path, self)
            print(f'instance saved to {file_path}')

        toc_initialization = time.time()
        print(f'Initialization done, time: {toc_initialization - tic_initialization} s')

    ######################################## step 2: optimise: either run InfoClus or read from cache ########################################
    def optimise(self,
                 alpha=None, beta=None, min_att=None, max_att=None, run_id=None,split_strategy_id=None,
                 allow_cache=True, show_brief_result=False):
        '''
        optimise result with current hyperparameters, the process is as follows:
        1. update hyperparameters of self
        2. check cache
        3. start clustering when no cache
        4. print the clustering result
        '''
        # update hyperparameters of self
        self.allow_cache = allow_cache
        self.paras_obj.update_paras(alpha,beta,min_att,max_att,run_id,split_strategy_id)
        cache_name, previously_calculated = self.check_cache()
        # start clustering when no cache
        if previously_calculated is None:
            self.result_obj = _Result(self.model_obj.prior[0], self.model_obj.prior[1], self.data_obj.size)
            self._run_infoclus_agglomerative()
            if self.allow_cache:
                self.create_cache_version(cache_name)
        self.print_result_in_terminal()

    def print_result_in_terminal(self):
        print(
            f'\nInfoClus - Dataset: {self.data_obj.name} Emb: {self.paras_obj.emb_name} Alpha: {self.paras_obj.alpha} Beta: {self.paras_obj.beta} Ref. Runtime: {self.paras_obj.runtime}')
        if __debug__:
            print('checking if the sum of clusters idxes equals to the data size: ', end='')
        print(f'Count of Clusters: {len(self.result_obj.clusters_idxes_opt)}')
        for cluster_idx in range(len(self.result_obj.clusters_idxes_opt)):
            print(f"    cluster {cluster_idx}:")
            print(f'        count of points: {len(self.result_obj.clusters_idxes_opt[cluster_idx])}')
            print(f'        attributes: ', end='')
            for j in self.result_obj.attributes_opt[cluster_idx]:
                print(f'{self.data_obj.data.columns[j]} ', end='')
            print("")
        print("SI: ", self.result_obj.si_opt)

    ######################################## step 3: run InfoClus by agglomerative ########################################
    def _run_infoclus_agglomerative(self):
        '''
        Here is the core part of Infoclus algorithm, the process is as follows:
        1. initialization of all result-related variables as None
        2. iteration preparation and start iteration of splitting in limited time

        Note: one split means enumerating all possible splits(nodes) and choose the best one to split one cluster into two
        '''

        splitting_startegy = self.paras_obj.split_strategy

        #################################### step2: iteration #########################################

        if splitting_startegy == 'by_node':

            if self.paras_obj.modify_hierarchical:
                candidates_for_split = set(range(self.data_obj.size*2-2))
            else:
                candidates_for_split = set(range(self.data_obj.size, 2*self.data_obj.size-2))
            splitted_nodes_and_its_ancestors = [[self.data_obj.size * 2 - 2, []]]
            samples_size = self.data_obj.size
            if self.paras_obj.modify_hierarchical:
                samples_size = self.model_obj.get_samples_count_given_medoids_idxes(self._clusters_idxes_opt[0])
            self._ic_opt.append(ic_one_info(self._priors[:, 0], self._priors[:, 1], samples_size, self.model_obj.prior))

            count_iterations = 0
            start = time.time()
            print("\nsplitting by nodes start ... ")
            while len(candidates_for_split) > 0 and (time.time() - start < self.paras_obj.runtime):
                count_iterations += 1
                si, clusters_idxes, attributes, ic_matrix, statistics, split_nodes = self._choose_optimal_split_by_nodes(candidates_for_split)
                if si > self._si_opt:
                    self.result_obj.update(copy.deepcopy(ic_matrix), copy.deepcopy(si), copy.deepcopy(clusters_idxes),
                                           copy.deepcopy(attributes), copy.deepcopy(statistics), copy.deepcopy(split_nodes))
            print(f"{count_iterations} iterations done.")

    def _choose_optimal_split_by_nodes(self, candidates_for_split):

        largest_si = -1
        largest_clusters_idxes = []
        largest_attributes = []
        largest_ics = []
        largest_statistics_for_computing_ics = []
        largest_split_nodes = []
        largest_nodes_idx = None

        for index in candidates_for_split:
            node_idx = self.model_obj.parents[index][0]

            clusters_idxes = copy.deepcopy(self.result_obj.clusters_idxes_opt)
            ic_matrix = copy.deepcopy(self.result_obj.ic_opt)
            split_nodes = copy.deepcopy(self.result_obj._split_nodes_opt)
            statistics_for_computing_ics = copy.deepcopy(self.result_obj.clusters_related_statistics_opt)

            res = self._split_by_node(node_idx, clusters_idxes, ic_matrix, split_nodes, statistics_for_computing_ics)

            if res is None:
                candidates_for_split.remove(index)
                continue
            attributes, ic_attributes, dl, si = self._calc_optimal_attributes_dl(ic_matrix)

            if si > largest_si:
                largest_si = si
                largest_clusters_idxes = clusters_idxes
                largest_attributes = attributes
                largest_ics = ic_matrix
                largest_statistics_for_computing_ics = statistics_for_computing_ics
                largest_split_nodes = split_nodes

                largest_nodes_idx = node_idx
            candidates_for_split.remove(largest_nodes_idx)
        return largest_si, largest_clusters_idxes, largest_attributes, largest_ics, largest_statistics_for_computing_ics, largest_split_nodes

    def _split_by_node(self, node, clusters_idxes, ic_matrix, split_nodes, statistics):

        points_to_change = self.model_obj.nodesToPoints[node]
        statistics.append(
            [self.model_obj.meansForNodes[node],
             self.model_obj.varsForNodes[node],
             len(points_to_change)])

        for clus_idx, split_node in enumerate(split_nodes):

            previous_node_ancestors_indexes = split_node[1]

            if node in previous_node_ancestors_indexes:

                points_to_change = [point for point in points_to_change if point not in clusters_idxes[clus_idx]]
                if len(points_to_change) == 0:
                    return None
                elif len(points_to_change) < 0 :
                    print('error')
                statistics[-1] = recur_meanVar_remove(
                    statistics[-1][0], statistics[-1][1],statistics[-1][2],
                    statistics[clus_idx][0], statistics[clus_idx][1], statistics[clus_idx][2]
                )
            clusters_idxes.append(points_to_change)

        node_ancestors_idxes = self.model_obj.get_ancestors(node)
        closest_ancestor, previous_cluster_label = self.model_obj.find_closest_ancestor(node_ancestors_idxes, split_nodes)
        if closest_ancestor is not None:

            previous_cluster_idxes = [point for point in clusters_idxes[previous_cluster_label] if point not in clusters_idxes[-1]]
            clusters_idxes[previous_cluster_label] = previous_cluster_idxes
            if len(previous_cluster_idxes) == 0:
                return None
            elif len(previous_cluster_idxes) < 0:
                print('error')
            statistics[previous_cluster_label] = recur_meanVar_remove(
                statistics[previous_cluster_label][0],
                statistics[previous_cluster_label][1],
                statistics[previous_cluster_label][2],
                statistics[-1][0],
                statistics[-1][1],
                statistics[-1][2]
            )

        for clus_idx in [previous_cluster_label, -1]:
            samples_count = statistics[clus_idx][2]
            if self.paras_obj.modify_hierarchical:
                medoids_idxes = clusters_idxes[clus_idx]
                samples_count = self.model_obj.get_samples_count_given_medoids_idxes(medoids_idxes)
            ic_matrix[clus_idx] = ic_one_info(statistics[clus_idx][0], statistics[clus_idx][1],
                                                              samples_count, self.model_obj.prior)

        split_nodes.append([node, node_ancestors_idxes])
        return [clusters_idxes, ic_matrix, split_nodes, statistics]

    # get the best attribute for each cluster
    def _init_optimal_attributes_dl(self, ics):
        sortedic = np.dstack(np.unravel_index(np.argsort(-ics.ravel()), ics.shape))[0]
        find_index = sortedic[:, 0]
        attributes_total = []
        ic_attributes = 0
        dl = 0
        for i in range(len(ics)):
            index = np.where(find_index == i)[0][0:self.paras_obj.min_att]
            attributes = [sortedic[ind][1] for ind in index]
            attributes_total.append(attributes)
            ic_attributes += sum(ics[i, attributes])
            dl = dl + sum((self._dls.iloc[attribute]) for attribute in attributes)
            sortedic = np.delete(sortedic, index, axis=0)
            find_index = np.delete(find_index, index, axis=0)
        best_comb_val = ic_attributes / (self.paras_obj.alpha + dl ** self.paras_obj.beta)

        return attributes_total, ic_attributes, dl, best_comb_val, sortedic

    def _calc_optimal_attributes_dl(self, ics):
        '''
        return attributes set for each cluster
        '''
        ics = np.array(ics)
        attributes_total, ic_attributes, dl, best_comb_val, sortedic = self._init_optimal_attributes_dl(ics)
        out_max_att_limit = False
        while not out_max_att_limit and len(sortedic) > 0:
            extend_cluster_try = sortedic[0][0]
            extend_attr_try = sortedic[0][1]
            sortedic = np.delete(sortedic, 0, axis=0)
            if len(attributes_total[extend_cluster_try]) >= self.paras_obj.max_att:
                continue
            dl_try = dl + self._dls.iloc[extend_attr_try]
            ic_attributes_try = ic_attributes + ics[extend_cluster_try, extend_attr_try]
            si_try = ic_attributes_try / (self.paras_obj.alpha + (dl_try) ** self.paras_obj.beta)
            if si_try >= best_comb_val:
                best_comb_val = si_try
                attributes_total[extend_cluster_try].append(extend_attr_try)
                dl = dl_try
                ic_attributes = ic_attributes_try
                out_max_att_limit = all(len(attribute) >= self.paras_obj.max_att for attribute in attributes_total)
            else:
                break

        return attributes_total, ic_attributes, dl, best_comb_val

    def create_cache_version(self, cache_name):
        self.result_obj.extend_results(self.model_obj)
        pre_calc = {
            'paras': self.paras_obj,
            'results': self.result_obj
        }
        to_cache(os.path.join(self.data_obj.cache_path, cache_name), pre_calc)

    def check_cache(self):

        current_paras = self.paras_obj.get_paras()
        cache_name = self.data_obj.name + get_hashkey_from_dict(current_paras)

        pre_calc = from_cache(os.path.join(self.data_obj.cache_path, cache_name))
        if pre_calc is not None:
            print("From cache")
            self.result_obj = pre_calc['results']
        return cache_name, pre_calc


    #
    # def _create_linkage(self):
    #     # counts = np.zeros(self.model.children_.shape[0])
    #     n_samples = len(self.model.labels_)
    #     parents = np.full(self.model.children_.shape[0], -1)
    #
    #     # build empty distribution
    #     if self.global_var_type == 'categorical':
    #         count_of_uniques_per_attribute = [len(df) for df in self.datas.ls_mapping_chain_by_col]
    #         np_data = np.zeros((max(count_of_uniques_per_attribute), len(self.datas.data_raw.columns)))
    #         mask = np.arange(np_data.shape[0])[:, None] >= np.array(count_of_uniques_per_attribute)
    #         np_data[mask] = self.epsilon
    #         empty_distribution = pd.DataFrame(np_data, columns=self.datas.data_raw.columns)
    #
    #     # for each merge in agglomerative clustering, do computation
    #     for i, merge in enumerate(self.model.children_):
    #
    #         leafPoints = []
    #         self._nodesToPoints[i] = leafPoints
    #
    #         # left child in merge
    #         left_child = merge[0]
    #         current_count_left = 0  # points count of left child node
    #         # compute count, mean, vars, parent
    #         if left_child < n_samples:
    #             current_count_left += 1
    #             if self.global_var_type == 'mixed':
    #                 pass
    #             elif self.global_var_type == 'numeric':
    #                 m_left = self.datas.data.iloc[left_child].to_numpy()
    #                 var_left = np.zeros_like(m_left)
    #             elif self.global_var_type == 'categorical':
    #                 left_point = self.datas.data.iloc[left_child]
    #                 m_left = empty_distribution.copy()
    #                 for j_column in range(len(self.datas.data_raw.columns)):
    #                     att_value = left_point.values[j_column]
    #                     i_row = self.datas.ls_mapping_chain_by_col[j_column].loc[self.datas.ls_mapping_chain_by_col[j_column]['scaled'] == att_value].index[0]
    #                     m_left.iloc[i_row, j_column] = 1
    #             leafPoints.append(left_child)
    #         else:
    #             # current_count_left += counts[left_child - n_samples]
    #             parents[left_child - n_samples] = i  # correction by Fuyin Lai
    #             if self.global_var_type == 'mixed':
    #                 pass
    #             elif self.global_var_type == 'numeric':
    #                 m_left = self._meansForNodes.get(left_child - n_samples)
    #                 var_left = self._varsForNodes.get(left_child - n_samples)
    #             elif self.global_var_type == 'categorical':
    #                 m_left = self._distributionsForNodes.get(left_child - n_samples)
    #             leafPoints.extend(self._nodesToPoints[left_child - n_samples])
    #         # right child
    #         right_child = merge[1]
    #         current_count_right = 0
    #         # count, mean, vars, parent
    #         if right_child < n_samples:
    #             current_count_right += 1
    #             if self.global_var_type == 'mixed':
    #                 pass
    #             elif self.global_var_type == 'numeric':
    #                 m_right = self.datas.data.iloc[right_child].to_numpy()
    #                 var_right = np.zeros_like(m_right)
    #             elif self.global_var_type == 'categorical':
    #                 right_point = self.datas.data.iloc[right_child]
    #                 m_right = empty_distribution.copy()
    #                 for j_column in range(len(self.datas.data_raw.columns)):
    #                     att_value = right_point.values[j_column]
    #                     i_row = self.datas.ls_mapping_chain_by_col[j_column].loc[self.datas.ls_mapping_chain_by_col[j_column]['scaled'] == att_value].index[0]
    #                     m_right.iloc[i_row, j_column] = 1
    #             leafPoints.append(right_child)
    #         else:
    #             # current_count_right += counts[right_child - n_samples]
    #             parents[right_child - n_samples] = i  # correction by Fuyin Lai
    #             if self.global_var_type == 'mixed':
    #                 pass
    #             elif self.global_var_type == 'numeric':
    #                 m_right = self._meansForNodes.get(right_child - n_samples)
    #                 var_right = self._varsForNodes.get(right_child - n_samples)
    #             elif self.global_var_type == 'categorical':
    #                 m_right = self._distributionsForNodes.get(right_child - n_samples)
    #             leafPoints.extend(self._nodesToPoints[right_child - n_samples])
    #
    #         # new mean, var and count for node i
    #         if self.datas.global_var_type == 'mixed':
    #             pass
    #         elif self.global_var_type == 'numeric':
    #             meanForNode = self.recur_mean(m_left, current_count_left,
    #                                           m_right, current_count_right)
    #             self._meansForNodes[i] = meanForNode
    #             varForNode = self.recur_var(m_left, var_left, current_count_left,
    #                                         m_right, var_right, current_count_right)
    #             self._varsForNodes[i] = varForNode
    #         elif self.global_var_type == 'categorical':
    #             distForNode = self.recur_dist_categorical(m_left, current_count_left,
    #                                           m_right, current_count_right)
    #             self._distributionsForNodes[i] = distForNode
    #         # counts[i] = current_count_left + current_count_right
    #
    #     self._parents_of_all_nodes = {}
    #     index_start = len(self.datas.data)
    #     for index, children in enumerate(self.model.children_):
    #         left_child = children[0]
    #         right_child = children[1]
    #         self._parents_of_all_nodes[left_child] = index+index_start
    #         self._parents_of_all_nodes[right_child] = index+index_start
    #
    #     # update self
    #     # self._parents = parents  # without counting original points
    #     # self._linkage_matrix = np.column_stack([self.model.children_, self.model.distances_, counts])

    #
    # def _calc_priors_agglomerative(self):
    #     # TODO: rewrite, remove dl_indices for numeric
    #     if self.global_var_type == 'mixed':
    #         pass
    #     elif self.global_var_type == 'numeric':
    #         self._priors = np.array([self._meansForNodes[len(self.datas.data) - 2], self._varsForNodes[len(self.datas.data) - 2]]).T
    #         self._priorsGausM = self._meansForNodes[len(self.datas.data) - 2]
    #         self._priorsGausS = self._varsForNodes[len(self.datas.data) - 2]
    #         # Order attribute indices per dl to use later in dl optimisation
    #         unique_dls = sorted(set(self._dls))
    #         # Attributes indices split per dl, used to split IC into submatrix and later to find IC value of attribute
    #         self._dl_indices = collections.OrderedDict()
    #         for dl in unique_dls:
    #             # Fill dl_indices for one dl value
    #             indices = [i for i, value in enumerate(self._dls) if value == dl]
    #             self._dl_indices[dl] = indices
    #     elif self.global_var_type == 'categorical':
    #         self._priors = self._distributionsForNodes[len(self.datas.data) - 2]

    #
    # def _calc_priors_kmeans(self):
    #     if self.global_var_type == 'mixed':
    #         pass
    #     elif self.global_var_type == 'numeric':
    #         pass
    #         self._priors = np.array([np.mean(self.datas.data.values, axis=0), np.var(self.datas.data.values,axis=0)]).T
    #         self._priorsGausM = self._priors[:,0]
    #         self._priorsGausS = self._priors[:,1]
    #     elif self.global_var_type == 'categorical':
    #         count_of_uniques_per_attribute = [len(df) for df in self.datas.ls_mapping_chain_by_col]
    #         np_data = np.zeros((max(count_of_uniques_per_attribute), len(self.datas.data_raw.columns)))
    #         mask = np.arange(np_data.shape[0])[:, None] >= np.array(count_of_uniques_per_attribute)
    #         np_data[mask] = self.epsilon
    #         data_distribution = pd.DataFrame(np_data, columns=self.datas.data_raw.columns)
    #         data_size = len(self.datas.data)
    #         for att_label in range(len(data_distribution.columns)):
    #             for col_loc in range(len(self.datas.ls_mapping_chain_by_col[att_label])):
    #                 value = self.datas.ls_mapping_chain_by_col[att_label]['scaled'][col_loc]
    #                 value_count = np.sum(self.datas.data.values[:, att_label] == value)
    #                 value_proportion = value_count / data_size
    #                 data_distribution.iloc[col_loc, att_label] = value_proportion
    #         self._priors = data_distribution

    #
    # def _refresh_all_variables(self):
    #     self._si_opt = 0  # value of si for this clustering
    #     self._clustering_opt = None  # final clustering labels for each point
    #     self._attributes_opt = None  # chosen attributes for each cluster
    #     self._split_nodes_opt = []
    #
    #     self._ic_opt = []  # ic of all attributes for each cluster
    #
    #     self._clustersRelatedInfo = []  # means, vars, and counts for each cluster
    #     if self.global_var_type == 'mixed':
    #         pass
    #     elif self.global_var_type == 'numeric':
    #         self._clustersRelatedInfo.append([self._priors[:, 0], self._priors[:, 1], len(self.datas.data)])
    #     elif self.global_var_type == 'categorical':
    #         self._clustersRelatedInfo.append([self._priors, len(self.datas.data)])
    #
    #     self._clusters_idxes_opt = [list(range(len(self.datas.data)))] # all points belong to cluster 0
    #
    #     self._total_ic_opt = 0
    #     self._total_dl_opt = 0  # value for summing up length of attributes

    # def kl_categorical(self, distribution_cluster: np.ndarray, epsilon: float = 0.00001) -> np.ndarray:
    #     # kl(p||q) = kl(cluster||prior)
    #     # kl(p||q) = kl(cluster||prior)
    #
    #     p = copy.copy(distribution_cluster)
    #     q = copy.copy(self._priors.values)
    #     p_safe = np.where(p <= 0, epsilon, p)
    #
    #     kl_mid_value = p * np.log(p_safe/q)
    #     kl = np.sum(kl_mid_value, axis=0)
    #
    #     return kl

    # # todo: not revised yet, delete or remove kl_categorical to other places
    # def ic_categorical(self, distribution_cluster: pd.DataFrame, size_cluster: int) -> np.ndarray:
    #     ic = size_cluster * self.kl_categorical(distribution_cluster.values)
    #     return ic

    # def _update_clustering_from_idxes(self):
    #     if sum(len(cluster_idxes) for cluster_idxes in self._clusters_idxes_opt) == len(self.datas.data):
    #         pass
    #     else:
    #         print('Error, not matching all points.')
    #     cluster_labels = np.empty(len(self.datas.data), dtype=int)
    #     # Assign each index to its cluster label
    #     for cluster_id, cluster_idxes in enumerate(self._clusters_idxes_opt):
    #         cluster_labels[cluster_idxes] = cluster_id
    #     self._clustering_opt = cluster_labels

    #
    # # in principle, the code is done, but I need to run to check is everything ok
    # def _run_infoclus_kmeans(self):
    #     #################################### step1: initialization result-related variables #########################################
    #     # todo: most of here are not necessary for kmeans, but now is needed to guarantee the run of code
    #     self._clustering_opt = None  # final clustering labels for each point
    #     self._si_opt = 0  # value of si for this clustering
    #     self._clustersRelatedInfo = {}  # means, vars, and counts for each cluster
    #     self._attributes_opt = None  # chosen attributes for each cluster
    #     self._ic_opt = None  # ic of all attributes for each cluster
    #     self._total_ic_opt = 0
    #     self._total_dl_opt = 0  # value for summing up length of attributes
    #     self._nodes_opt = None  # the left nodes that could be used for further splitting
    #     self._split_nodes_opt = []  # splitted nodes and their classification label, tuple inside
    #     self._split_nodes_opt.append(("others", 0))
    #     clustering_new_info = {}
    #
    #     print("considering kmeans", end='')
    #     for i in range(KMEANS_COUNT):
    #         k = self.model.n_clusters + i
    #         clustering_info_k = {}
    #         model = KMeans(n_clusters=k, random_state=self.model.random_state)
    #         model.fit(self.embedding)
    #         clustering_new = model.labels_
    #         index_dict = defaultdict(list)
    #         for idx, label in enumerate(clustering_new):
    #             index_dict[label].append(idx)
    #         ics=[]
    #         for cluster_label in range(k):
    #             index_cluster = index_dict[cluster_label]
    #             cluster = self.datas.data.values[index_cluster]
    #             if self.global_var_type == 'mixed':
    #                 pass
    #             elif self.global_var_type == 'numeric':
    #                 mean_cluster = np.mean(cluster, axis=0)
    #                 var_cluster = np.var(cluster, axis=0)
    #                 count_cluster = len(cluster)
    #                 ic_cluster = self.ic_one_info(mean_cluster,var_cluster,count_cluster,self.model_obj.prior)
    #                 ics.append(ic_cluster)
    #             elif self.global_var_type == 'categorical':
    #                 count_of_uniques_per_attribute = [len(df) for df in self.datas.ls_mapping_chain_by_col]
    #                 np_data = np.zeros((max(count_of_uniques_per_attribute), len(self.datas.data_raw.columns)))
    #                 mask = np.arange(np_data.shape[0])[:, None] >= np.array(count_of_uniques_per_attribute)
    #                 np_data[mask] = self.epsilon
    #                 cluster_distribution = pd.DataFrame(np_data, columns=self.data_raw.columns)
    #                 cluster_size = len(cluster)
    #                 for att_label in range(len(cluster_distribution.columns)):
    #                     for col_loc in range(len(self.datas.ls_mapping_chain_by_col[att_label])):
    #                         value = self.datas.ls_mapping_chain_by_col[att_label]['scaled'][col_loc]
    #                         value_count = np.sum(cluster[:, att_label] == value)
    #                         value_proportion = value_count / cluster_size
    #                         cluster_distribution.iloc[col_loc, att_label] = value_proportion
    #                 ic_cluster = self.ic_categorical(cluster_distribution, cluster_size)
    #                 clustering_info_k[cluster_label] = [cluster_distribution, cluster_size]
    #                 ics.append(ic_cluster)
    #         attributes, ic_attributes, dl, si_val = self.calc_optimal_attributes_dl(ics)
    #         if si_val > self._si_opt:
    #             self._clustering_opt = clustering_new
    #             self._attributes_opt = attributes
    #             self._si_opt = si_val
    #             self._ic_opt = ic_attributes
    #             if self.global_var_type == 'categorical':
    #                 self._clustersRelatedInfo = clustering_info_k
    #     print(f"\n done K: {k}")
    #
    #
    # def _node_indices_split(self, clusters_idxes, node_idx, pre_index=None, max_label=0):
    #     '''
    #     change indices after splitting one node out into a new cluster
    #     '''
    #     # get pre index of clustering
    #     n_samples = len(self.model.labels_)
    #     indices = copy.deepcopy(pre_index)
    #     if pre_index is None:
    #         indices = np.zeros(n_samples, dtype=int)
    #     # get index that are going to change
    #     new_cluster = max_label + 1
    #     to_change = self._nodesToPoints[node_idx]
    #     old_cluster = indices[to_change[0]]
    #     # change indices
    #     indices[to_change] = new_cluster
    #     not_change = np.where(indices == old_cluster)
    #
    #     clusters_idxes.append(list(to_change))
    #     clusters_idxes[old_cluster] = list(not_change[0])
    #
    #     return indices, new_cluster, old_cluster, to_change, not_change

    #
    # def _choose_optimal_split_by_sibling(self):
    #
    #     # todo, check this function and make it run
    #     last_cluster_label = len(self._split_nodes_opt)-1
    #     new_left_cluster_label = last_cluster_label + 1
    #     new_right_cluster_label = last_cluster_label + 2
    #
    #     # for each node in nodes, split its two children as two clusters
    #     largest_si = 0
    #     for sibling in self.model.children_:
    #
    #         split_nodes = copy.deepcopy(self._split_nodes_opt)
    #         statistics_for_computing_ics_sibling = copy.deepcopy(self._clustersRelatedInfo)
    #
    #         left_node_ancestors_indexes = self._get_ancestors(sibling[0])
    #         right_node_ancestors_indexes = self._get_ancestors(sibling[1])
    #
    #         skipper_outer_loop = False
    #         for previous_split_node in self._split_nodes_opt:
    #             if sibling[0] == previous_split_node[0] or sibling[1] == previous_split_node[0]:
    #                 skipper_outer_loop = True
    #                 continue
    #         if skipper_outer_loop:
    #             continue
    #
    #         res = self._get_partition_given_sibling(
    #             sibling, split_nodes, statistics_for_computing_ics_sibling,
    #             new_left_cluster_label, new_right_cluster_label, left_node_ancestors_indexes, right_node_ancestors_indexes)
    #
    #         if res is not None:
    #
    #             clusters_idxes_sibling, statistics_for_computing_ics_sibling, ics_sibling, split_nodes_sibling = res[0], res[1], res[2], res[3]
    #             attributes, ic_attributes, dl, si_val = self.calc_optimal_attributes_dl(ics_sibling)
    #
    #             if __debug__:
    #                 if len(attributes) != len(clusters_idxes_sibling):
    #                     print('error')
    #                     traceback.print_stack()
    #                     sys.exit()
    #
    #             if si_val > largest_si:
    #                 largest_si = si_val
    #                 largest_clusters_idxes = clusters_idxes_sibling
    #                 largest_attributes = attributes
    #                 largest_ics = ics_sibling
    #                 largest_statistics_for_computing_ics = statistics_for_computing_ics_sibling
    #                 largest_split_nodes = split_nodes_sibling
    #
    #     return largest_si, largest_clusters_idxes, largest_attributes, largest_ics, largest_statistics_for_computing_ics, largest_split_nodes
    #
    # def _get_partition_given_sibling(self, sibling, split_nodes, statistics_for_computing_ics_sibling,
    #                                  new_left_cluster_label, new_right_cluster_label, left_node_ancestors_indexes, right_node_ancestors_indexes):
    #
    #     left_node_index = sibling[0]
    #     if left_node_index <= len(self.datas.data)-1:
    #         # left_points_to_change = [left_node_index]
    #         return None
    #
    #     previous_split_nodes = copy.deepcopy(split_nodes)
    #     clusters_idxes = copy.deepcopy(self._clusters_idxes_opt)
    #     ics_sibling = []
    #
    #     left_points_to_change = self._nodesToPoints[left_node_index-len(self.datas.data)]
    #     split_nodes.append([left_node_index, left_node_ancestors_indexes])
    #     statistics_for_computing_ics_sibling.append(
    #         [self._meansForNodes.get(left_node_index-len(self.datas.data)),
    #          self._varsForNodes.get(left_node_index-len(self.datas.data)),
    #          len(left_points_to_change)])
    #
    #     right_node_index = sibling[1]
    #     if right_node_index <= len(self.datas.data)-1:
    #         return None
    #
    #     right_points_to_change = self._nodesToPoints[right_node_index-len(self.datas.data)]
    #     split_nodes.append([right_node_index, right_node_ancestors_indexes])
    #     statistics_for_computing_ics_sibling.append(
    #         [self._meansForNodes.get(right_node_index-len(self.datas.data)), self._varsForNodes.get(right_node_index-len(self.datas.data)),
    #          len(right_points_to_change)])
    #
    #     # update new nodes
    #     for index, split_node in enumerate(previous_split_nodes):
    #
    #         previous_cluster_label = index
    #         previous_node_index = split_node[0]
    #         previous_node_ancestors_indexes = split_node[1]
    #
    #         if left_node_index in previous_node_ancestors_indexes:
    #             left_points_to_change = self._remove_points_by_nodes(left_points_to_change,
    #                                                                 clusters_idxes[previous_cluster_label])
    #             statistics_for_computing_ics_sibling[new_left_cluster_label] = self.recur_meanVar_remove(
    #                 statistics_for_computing_ics_sibling[new_left_cluster_label][0],
    #                 statistics_for_computing_ics_sibling[new_left_cluster_label][1],
    #                 statistics_for_computing_ics_sibling[new_left_cluster_label][2],
    #                 statistics_for_computing_ics_sibling[previous_cluster_label][0],
    #                 statistics_for_computing_ics_sibling[previous_cluster_label][1],
    #                 statistics_for_computing_ics_sibling[previous_cluster_label][2])
    #
    #         if right_node_index in previous_node_ancestors_indexes:
    #             right_points_to_change = self._remove_points_by_nodes(right_points_to_change,
    #                                                                  clusters_idxes[previous_cluster_label])
    #             statistics_for_computing_ics_sibling[new_right_cluster_label] = self.recur_meanVar_remove(
    #                 statistics_for_computing_ics_sibling[new_right_cluster_label][0],
    #                 statistics_for_computing_ics_sibling[new_right_cluster_label][1],
    #                 statistics_for_computing_ics_sibling[new_right_cluster_label][2],
    #                 statistics_for_computing_ics_sibling[previous_cluster_label][0],
    #                 statistics_for_computing_ics_sibling[previous_cluster_label][1],
    #                 statistics_for_computing_ics_sibling[previous_cluster_label][2])
    #
    #     # check new nodes' validity
    #     if type(statistics_for_computing_ics_sibling[new_left_cluster_label]) is type(None):
    #         del statistics_for_computing_ics_sibling[new_left_cluster_label]
    #         del split_nodes[new_left_cluster_label]
    #         del new_left_cluster_label
    #         new_right_cluster_label = new_right_cluster_label - 1
    #     else:
    #         clusters_idxes.append(left_points_to_change)
    #     if type(statistics_for_computing_ics_sibling[new_right_cluster_label]) is type(None):
    #         del statistics_for_computing_ics_sibling[new_right_cluster_label]
    #         del split_nodes[new_right_cluster_label]
    #         del new_right_cluster_label
    #     else:
    #         clusters_idxes.append(right_points_to_change)
    #
    #     # update previous split nodes
    #     closest_ancestor, previous_cluster_label = self._find_closest_ancestor(left_node_index, left_node_ancestors_indexes, previous_split_nodes)
    #
    #     if closest_ancestor is not None:
    #         try:
    #             statistics_for_computing_ics_sibling[previous_cluster_label] = self.recur_meanVar_remove(
    #                 statistics_for_computing_ics_sibling[previous_cluster_label][0],
    #                 statistics_for_computing_ics_sibling[previous_cluster_label][1],
    #                 statistics_for_computing_ics_sibling[previous_cluster_label][2],
    #                 statistics_for_computing_ics_sibling[new_left_cluster_label][0],
    #                 statistics_for_computing_ics_sibling[new_left_cluster_label][1],
    #                 statistics_for_computing_ics_sibling[new_left_cluster_label][2]
    #             )
    #             previous_cluster_idxes = self._remove_points_by_nodes(clusters_idxes[previous_cluster_label],
    #                                                                   clusters_idxes[new_left_cluster_label])
    #             if len(previous_cluster_idxes) > 0:
    #                 clusters_idxes[previous_cluster_label] = previous_cluster_idxes
    #             elif len(previous_cluster_idxes) == 0:
    #                 del clusters_idxes[previous_cluster_label]
    #             else:
    #                 print('error, count of cluster idxes could not be negative')
    #         except UnboundLocalError as e:
    #             pass
    #         if statistics_for_computing_ics_sibling[previous_cluster_label] is None:
    #             del statistics_for_computing_ics_sibling[previous_cluster_label]
    #             del split_nodes[previous_cluster_label]
    #             try:
    #                 new_left_cluster_label = new_left_cluster_label - 1
    #                 new_right_cluster_label = new_right_cluster_label - 1
    #             except UnboundLocalError as e:
    #                 pass
    #     closest_ancestor, previous_cluster_label = self._find_closest_ancestor(right_node_index, right_node_ancestors_indexes, previous_split_nodes)
    #     if closest_ancestor is not None:
    #         try:
    #             statistics_for_computing_ics_sibling[previous_cluster_label] = self.recur_meanVar_remove(
    #                 statistics_for_computing_ics_sibling[previous_cluster_label][0],
    #                 statistics_for_computing_ics_sibling[previous_cluster_label][1],
    #                 statistics_for_computing_ics_sibling[previous_cluster_label][2],
    #                 statistics_for_computing_ics_sibling[new_right_cluster_label][0],
    #                 statistics_for_computing_ics_sibling[new_right_cluster_label][1],
    #                 statistics_for_computing_ics_sibling[new_right_cluster_label][2]
    #             )
    #             previous_cluster_idxes = self._remove_points_by_nodes(clusters_idxes[previous_cluster_label],
    #                                                                   clusters_idxes[new_right_cluster_label])
    #
    #             if len(previous_cluster_idxes) > 0:
    #                 clusters_idxes[previous_cluster_label] = previous_cluster_idxes
    #             elif len(previous_cluster_idxes) == 0:
    #                 del clusters_idxes[previous_cluster_label]
    #             else:
    #                 print('error, count of cluster idxes could not be negative')
    #         except UnboundLocalError as e:
    #             pass
    #
    #         if statistics_for_computing_ics_sibling[previous_cluster_label] is None:
    #             del statistics_for_computing_ics_sibling[previous_cluster_label]
    #             del split_nodes[previous_cluster_label]
    #             try:
    #                 new_left_cluster_label = new_left_cluster_label - 1
    #                 new_right_cluster_label = new_right_cluster_label - 1
    #             except UnboundLocalError as e:
    #                 pass
    #
    #     if self.global_var_type == 'mixed':
    #         pass
    #     elif self.global_var_type == 'numeric':
    #         for index, statictics in enumerate(statistics_for_computing_ics_sibling):
    #             samples_count = statictics[2]
    #             cluster_label = index
    #             if self.modify_hierarchical:
    #                 medoids_idxes = clusters_idxes[cluster_label]
    #                 samples_count = self._get_samples_count_given_medoids_idxes(medoids_idxes)
    #             ics_sibling.append(ic_one_info(statictics[0], statictics[1], samples_count, self.model_obj.prior))
    #     elif self.global_var_type == 'categorical':
    #         pass
    #
    #     return (clusters_idxes, statistics_for_computing_ics_sibling, ics_sibling, split_nodes)
