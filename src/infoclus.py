import time
from typing import Optional

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import euclidean_distances
import kmedoids
from caching import from_cache, to_cache
from infoclus_utils import *

from config import PROJECT_ROOT, DATA_FOLDER

RUNTIME_OPTIONS = [0.01, 0.5, 1, 5, 10, 30, 60, 180, 300, 600, 1800, 3600, np.inf]
VAR_TYPE_THRESHOLD = 20
REPLACE_NAN = 0
EPSILON= 0.00001
Random_State = 42
KMEANS_COUNT = 30 # How many kmeans with different k we are going to consider, starting from the k passed in initailization
SPLITTING_STRATEGY = ['by_node','by_sibling']
COUNT_BASE_CLUSTERS = 100

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
        self.data: Optional[pd.DataFrame] = None
        self.data_raw: Optional[pd.DataFrame] = None

        if dataset_folder is None:
            self.dataset_folder = os.path.join(DATA_FOLDER, dataset_name)
        else:
            self.dataset_folder = dataset_folder
        self.cache_path = os.path.join(self.dataset_folder, 'cache')

        df_data = pd.read_csv(os.path.join(self.dataset_folder, f'{dataset_name}.csv'))

        factorized_data, ls_mapping_chain_by_col, self.data , self.data_raw = get_scaled_data(df_data, REPLACE_NAN)
        self.size = self.data.shape[0]
        if factorized_data is not None and ls_mapping_chain_by_col is not None:
            self.factorized_data = factorized_data
            self.ls_mapping_chain_by_col = ls_mapping_chain_by_col

        df_var_type_complexity = get_var_type_complexity(self.data_raw, VAR_TYPE_THRESHOLD)
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

        embeddings_path = os.path.join(data.dataset_folder, 'cache', 'embeddings.npz')
        if os.path.exists(embeddings_path):
            print('Loading embeddings...')
            embeddings_load = np.load(embeddings_path)
            embeddings = {k: embeddings_load[k] for k in embeddings_load.files}
            self.all_embeddings = embeddings
            print('Done')
        else:
            print('Creating embeddings...')
            tic = time.time()
            embeddings = get_embeddings(data.data_raw.values)
            self.all_embeddings = embeddings
            np.savez(embeddings_path, **embeddings)
            toc = time.time()
            print(f'Done, time: {toc - tic} s')

        if embedding is None:
            self.embedding = self.all_embeddings['tsne']
        else:
            self.all_embeddings['user_given'] = embedding
            self.embedding = embedding
            np.savez(embeddings_path, **self.all_embeddings)


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
    def __init__(self, data_obj:_Data, linkage):
        self.model = AgglomerativeClustering(linkage=linkage, distance_threshold=0, n_clusters=None)
        self.kmedoids_model = None

        self.prior = []
        self.kmedoids_mean = []
        self.kmedoids_var = []
        self.kmedoids_to_points = []

    def calc_statistics_numeric(self, data: np.ndarray, modify_hierarchy: bool):

        n_samples = len(self.model.labels_)
        self.meansForNodes = {}
        self.varsForNodes = {}
        self.nodesToPoints = {}
        for i, merge in enumerate(self.model.children_):

            self.nodesToPoints[i + n_samples] = []

            for j, node in enumerate(merge):
                if node < n_samples:
                    if modify_hierarchy:
                        self.meansForNodes[node] = self.kmedoids_mean[node]
                        self.varsForNodes[node] = self.kmedoids_var[node]
                        self.nodesToPoints[node] = self.kmedoids_to_points[node]
                    else:
                        self.meansForNodes[node] = data[node]
                        self.varsForNodes[node] = np.zeros_like(self.meansForNodes[node])
                        self.nodesToPoints[node] = [node]
                self.nodesToPoints[i + n_samples].extend(self.nodesToPoints[node])

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

    def record_parents(self):

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

    def compute_kmedoids_statistics(self, data: np.ndarray):
        for cluster_label in range(len(self.kmedoids_model.medoids)):
            set_of_samples = np.where(self.kmedoids_model.labels==cluster_label)[0]
            cluster = data[set_of_samples]
            self.kmedoids_mean.append(np.mean(cluster, axis=0))
            self.kmedoids_var.append(np.var(cluster, axis=0))
            self.kmedoids_to_points.append(set_of_samples)

    #
    # def get_samples_count_given_medoids_idxes(self, medoids_idxes):
    #
    #     samples_count = 0
    #     for medoid_idx in medoids_idxes:
    #         set_of_samples = self.kmedoids_count[medoid_idx]
    #         samples_count += len(set_of_samples)
    #     return samples_count

class _Paras:
    def __init__(self, data_obj: _Data, embeddings_obj: _Embeddings, model_obj: _Model,
                 alpha, emb_name, linkage, beta, min_att, max_att, run_id,
                 split_strategy_id,
                 modify_hierarchical, base_clusters):

        if alpha is None:
            self.alpha = int(data_obj.size / 10)
        else:
            self.alpha = alpha
        if modify_hierarchical:
            if base_clusters is None:
                self.base_clusters = min(int(data_obj.size / 5), COUNT_BASE_CLUSTERS)
            else:
                self.base_clusters = base_clusters
            diss = euclidean_distances(embeddings_obj.embedding)
            model_obj.kmedoids_model = kmedoids.fasterpam(diss, self.base_clusters)
            model_obj.compute_kmedoids_statistics(data_obj.data.values)
            data_obj.data = data_obj.data.iloc[model_obj.kmedoids_model.medoids]
            data_obj.size = data_obj.data.shape[0]
            embeddings_obj.embedding = embeddings_obj.embedding[model_obj.kmedoids_model.medoids]
        model_obj.model.fit(embeddings_obj.embedding)
        model_obj.calc_statistics_numeric(data_obj.data.values, modify_hierarchical)
        model_obj.record_parents()

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
    def __init__(self, mean_prior: np.ndarray, var_prior: np.ndarray, sample_size: int, data_raw_size: int):

        self.ic_opt = [[0]*len(mean_prior)]  # ic of all attributes for each cluster
        self.si_opt = 0  # value of si for this clustering
        self.clusters_idxes_opt = [[i for i in range(data_raw_size)]]  # all points belong to cluster 0
        self.attributes_opt = []  # chosen attributes for each cluster
        self.clusters_related_statistics_opt = [[mean_prior, var_prior, data_raw_size]]
        self._split_nodes_opt = [[sample_size * 2 - 2, []]]

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

        clustering = np.full(sum(len(points) for points in self.clusters_idxes_opt), -1)
        for i, points in enumerate(self.clusters_idxes_opt):
            clustering[points] = i
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
        self.model_obj = _Model(self.data_obj, linkage)

        self.paras_obj = _Paras(self.data_obj, self.embeddings_obj, self.model_obj,
                 alpha, emb_name, linkage, beta, min_att, max_att, run_id,
                 split_strategy_id,
                 modify_hierarchical, base_clusters)
        self.result_obj = _Result(self.model_obj.prior[0], self.model_obj.prior[1], self.data_obj.size, self.data_obj.data_raw.shape[0])

        if self.allow_cache:
            file_path = os.path.join(self.data_obj.cache_path, 'modify_' + str(self.paras_obj.modify_hierarchical))
            to_cache(file_path, self)
            print(f'instance saved to {file_path}')

        toc_initialization = time.time()
        print(f'Initialization done, time: {toc_initialization - tic_initialization} s')

    ######################################## step 2: optimise: either run InfoClus or read from cache ########################################
    def optimise(self,
                 alpha=None, beta=None, min_att=None, max_att=None, run_id=None,split_strategy_id=None,
                 allow_cache=True):
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
            self.result_obj = _Result(self.model_obj.prior[0], self.model_obj.prior[1], self.data_obj.size, self.data_obj.data_raw.shape[0])
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

        splitting_strategy = self.paras_obj.split_strategy

        #################################### step2: iteration #########################################

        if splitting_strategy == 'by_node':

            res_obj_local_opt = _Result(self.model_obj.prior[0], self.model_obj.prior[1], self.data_obj.size, self.data_obj.data_raw.shape[0])

            if self.paras_obj.modify_hierarchical:
                candidates_for_split = set(range(self.data_obj.size*2-2))
            else:
                candidates_for_split = set(range(self.data_obj.size, 2*self.data_obj.size-2))

            count_iterations = 0
            start = time.time()
            print("\nsplitting by nodes start ... ")
            while len(candidates_for_split) > 0 and (time.time() - start < self.paras_obj.runtime):
                count_iterations += 1
                self._choose_optimal_split_by_nodes(res_obj_local_opt, candidates_for_split)
                if res_obj_local_opt.si_opt > self.result_obj.si_opt:
                    self.result_obj.update(copy.deepcopy(res_obj_local_opt.ic_opt),
                                           copy.deepcopy(res_obj_local_opt.si_opt),
                                           copy.deepcopy(res_obj_local_opt.clusters_idxes_opt),
                                           copy.deepcopy(res_obj_local_opt.attributes_opt),
                                           copy.deepcopy(res_obj_local_opt.clusters_related_statistics_opt),
                                           copy.deepcopy(res_obj_local_opt._split_nodes_opt))
            print(f"{count_iterations} iterations done.")

    def _choose_optimal_split_by_nodes(self, res_obj_local_opt: _Result, candidates_for_split):

        largest_si = -1
        largest_clusters_idxes = []
        largest_ics = []
        largest_statistics = []
        largest_split_nodes = []
        largest_attributes = []
        largest_nodes_idx = None

        for node_idx in candidates_for_split.copy():

            clusters_idxes = copy.deepcopy(res_obj_local_opt.clusters_idxes_opt)
            ic_matrix = copy.deepcopy(res_obj_local_opt.ic_opt)
            split_nodes = copy.deepcopy(res_obj_local_opt._split_nodes_opt)
            statistics_for_computing_ics = copy.deepcopy(res_obj_local_opt.clusters_related_statistics_opt)
            res = self._split_by_node(node_idx, clusters_idxes, ic_matrix, split_nodes, statistics_for_computing_ics)

            if res is None:
                candidates_for_split.remove(node_idx)
                continue
            attributes, ic_attributes, dl, si = self._calc_optimal_attributes_dl(ic_matrix)

            if si > largest_si:
                largest_si = si
                largest_clusters_idxes = clusters_idxes
                largest_ics = ic_matrix
                largest_statistics = statistics_for_computing_ics
                largest_split_nodes = split_nodes
                largest_attributes = attributes
                largest_nodes_idx = node_idx

        if len(candidates_for_split) == 0:
            return
        candidates_for_split.remove(largest_nodes_idx)
        res_obj_local_opt.update(largest_ics, largest_si, largest_clusters_idxes,
                                 largest_attributes, largest_statistics, largest_split_nodes)

    def _split_by_node(self, node, clusters_idxes, ic_matrix, split_nodes, statistics):

        points_to_change = self.model_obj.nodesToPoints[node]
        statistics.append(
            [self.model_obj.meansForNodes[node],
             self.model_obj.varsForNodes[node],
             len(points_to_change)])
        ic_matrix.append([])

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
        statistics[-1][2] = len(points_to_change)
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
            ic_matrix[clus_idx] = ic_one_info(statistics[clus_idx][0], statistics[clus_idx][1],
                                                                  statistics[clus_idx][2], self.model_obj.prior)
        split_nodes.append([node, node_ancestors_idxes])

        return [clusters_idxes, ic_matrix, split_nodes, statistics]

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
            dl = dl + sum((self.data_obj._dls.iloc[attribute]) for attribute in attributes)
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
            dl_try = dl + self.data_obj._dls.iloc[extend_attr_try]
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
        cache_name = get_hashkey_from_dict(current_paras)

        pre_calc = from_cache(os.path.join(self.data_obj.cache_path, cache_name))
        if pre_calc is not None:
            print("From cache")
            self.result_obj = pre_calc['results']
        return cache_name, pre_calc