"""
Analyze activation-based clustering and compare to weight-based
"""

import sys
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.stats import entropy
from scipy.stats import spearmanr, kendalltau
from sklearn.cross_decomposition import CCA
from pathos.multiprocessing import ProcessPool
import copy
import pickle
import time
import tensorflow as tf
from sacred import Experiment
from sacred.observers import FileStorageObserver
from src.utils import load_weights, compute_pvalue
from src.cnn.extractor import extract_cnn_weights_filters_as_units
from src.cnn import CNN_MODEL_PARAMS, CNN_VGG_MODEL_PARAMS

from src.spectral_cluster_model import weights_array_to_cluster_quality, weights_to_graph, \
    cnn_tensors_to_flat_weights_and_graph, delete_isolated_ccs_refactored, compute_ncut, \
    get_inv_avg_commute_time

# set up some sacred stuff
activations_experiment = Experiment('activations_model')
activations_experiment.observers.append((FileStorageObserver.create('activations_runs')))

RANDOM_STATE = 42

@activations_experiment.config
def my_config():
    eigen_solver = 'arpack'
    assign_labels = 'kmeans'
    epsilon = 1e-8
    with_shuffle = True
    n_workers = 10
    n_samples = 100
    n_inputs = 784
    n_outputs = 10
    corr_type = 'spearman'  # must be in ['kendall', 'pearson', 'spearman']
    use_inv_avg_commute = False
    filter_norm = 1


def do_clustering_weights(network_type, weights_path, n_clusters, n_inputs, n_outputs,
                          exclude_inputs, eigen_solver, assign_labels, use_inv_avg_commute,
                          filter_norm, epsilon):

    weights_ = load_weights(weights_path)

    if any(len(wgts.shape) > 2 for wgts in weights_):
        weights_ = extract_cnn_weights_filters_as_units(weights_, filter_norm)
    if network_type == 'cnn':  # for the cnns, only look at conv layers
        cnn_params = CNN_VGG_MODEL_PARAMS if 'vgg' in str(weights_path).lower() else CNN_MODEL_PARAMS
        n_conv_layers = len(cnn_params['conv'])
        weights_ = weights_[1: n_conv_layers]  # n_conv_layers is in the config
    elif exclude_inputs:
        weights_ = weights_[1:-1]  # exclude inputs and outputs

    adj_mat_ = weights_to_graph(weights_)

    # delete unconnected components from the net
    _, adj_mat, weight_mask, _ = delete_isolated_ccs_refactored(weights_, adj_mat_,
                                                             is_testing=False)

    if use_inv_avg_commute:
        adj_mat = get_inv_avg_commute_time(adj_mat)

    # find cluster quality of this pruned net
    print("\nclustering unshuffled weights\n")
    unshuffled_ncut, clustering_labels = weights_array_to_cluster_quality(None, adj_mat,
                                                                          n_clusters,
                                                                          eigen_solver,
                                                                          assign_labels,
                                                                          epsilon,
                                                                          is_testing=False)
    ave_in_out = (1 - unshuffled_ncut / n_clusters) / (2 * unshuffled_ncut / n_clusters)
    ent = entropy(clustering_labels)
    label_proportions = np.bincount(clustering_labels) / len(clustering_labels)
    result = {'ncut': unshuffled_ncut,
              'ave_in_out': ave_in_out,
              'mask': weight_mask,  # node_mask is a 1d length n_unit boolean array
              'labels': clustering_labels,
              'label_proportions': label_proportions,
              'entropy': ent}
    return result


def get_corr_adj(activations_mat, corr_type):

    # kendall has less gross error sensitivity and slightly smaller empirical variance
    # https://www.tse-fr.eu/sites/default/files/medias/stories/SEMIN_09_10/STATISTIQUE/croux.pdf
    # but spearman is much faster to compute

    # get the pearson, kendall, and spearman r^2 values from the activations matrix where rows=units, cols=examples
    n_units = activations_mat.shape[0]
    if corr_type == 'pearson':
        corr_mat = np.corrcoef(activations_mat, rowvar=True)
    elif corr_type == 'spearman':
        corr_mat, _ = spearmanr(activations_mat, axis=1)  # pearson r of ranks
    elif corr_type == 'kendall':
        corr_mat = np.diag(np.ones(n_units))  # n_concordant_pair - n_discordant_pair / n_choose_2
        for i in range(n_units):
            for j in range(i):
                kendall_tau, _ = kendalltau(activations_mat[i], activations_mat[j])
                corr_mat[i, j] = kendall_tau
                corr_mat[j, i] = kendall_tau
    else:
        raise ValueError("corr_type must be in ['kendall', 'pearson', 'spearman']")
    assert corr_mat.shape == (n_units, n_units)

    corr_adj = corr_mat**2
    np.fill_diagonal(corr_adj, 0)

    corr_adj = np.nan_to_num(corr_adj)
    corr_adj[corr_adj < 0] = 0
    corr_adj[corr_adj > 1] = 1

    return corr_adj


def shuffle_and_cluster_activations(n_samples, corr_adj, n_clusters,
                                    eigen_solver, assign_labels, epsilon):

    n_units = corr_adj.shape[0]
    shuff_ncuts = []

    time_str = str(time.time())
    dcml_place = time_str.index('.')
    time_seed = int(time_str[dcml_place + 1:])
    np.random.seed(time_seed)

    for _ in range(n_samples):
        # shuffle all edges
        corr_adj_shuff = np.zeros((n_units, n_units))
        upper_tri = np.triu_indices(n_units, 1)
        edges = corr_adj[upper_tri]
        np.random.shuffle(edges)
        corr_adj_shuff[upper_tri] = edges
        corr_adj_shuff = np.maximum(corr_adj_shuff, corr_adj_shuff.T)

        # cluster
        shuffled_ncut, _ = weights_array_to_cluster_quality(None, corr_adj_shuff, n_clusters,
                                                            eigen_solver, assign_labels, epsilon,
                                                            is_testing=False)
        shuff_ncuts.append(shuffled_ncut)

    return np.array(shuff_ncuts)


def do_clustering_activations(network_type, activations_path, activations_mask_path, corr_type,
                              n_clusters, n_inputs, n_outputs, exclude_inputs, eigen_solver,
                              assign_labels, epsilon, n_samples, with_shuffle, n_workers):

    with open(activations_path, 'rb') as f:
        activations = pickle.load(f)
    with open(activations_mask_path, 'rb') as f:
        activations_mask = pickle.load(f)

    if network_type == 'cnn':  # for the cnns, only look at conv layers
        if 'stacked' in str(activations_path).lower():
            n_in = n_inputs * 2
        else:
            n_in = n_inputs
        cnn_params = CNN_VGG_MODEL_PARAMS if 'vgg' in str(activations_path).lower() else CNN_MODEL_PARAMS
        n_conv_filters = sum([cl['filters'] for cl in cnn_params['conv']])
        n_start = np.sum(activations_mask[:n_in])
        n_stop = n_start + np.sum(activations_mask[n_in: n_in+n_conv_filters])
        activations = activations[n_start:n_stop, :]
        activations_mask = activations_mask[n_in: n_in+n_conv_filters]
    elif exclude_inputs:
        n_in = n_inputs
        n_start = np.sum(activations_mask[:n_in])
        activations = activations[n_start: -n_outputs, :]
        activations_mask = activations_mask[n_in: -n_outputs]

    corr_adj = get_corr_adj(activations, corr_type)

    unshuffled_ncut, clustering_labels = weights_array_to_cluster_quality(None, corr_adj, n_clusters,
                                                                          eigen_solver, assign_labels, epsilon,
                                                                          is_testing=False)
    ave_in_out = (1 - unshuffled_ncut / n_clusters) / (2 * unshuffled_ncut / n_clusters)
    ent = entropy(clustering_labels)
    label_proportions = np.bincount(clustering_labels) / len(clustering_labels)
    result = {'activations': activations, 'corr_adj': corr_adj, 'mask': activations_mask,
              'ncut': unshuffled_ncut, 'ave_in_out': ave_in_out, 'labels': clustering_labels,
              'label_proportions': label_proportions, 'entropy': ent}

    if with_shuffle:
        n_samples_per_worker = n_samples // n_workers
        function_argument = (n_samples_per_worker, corr_adj,
                             n_clusters, eigen_solver,
                             assign_labels, epsilon)
        if n_workers == 1:
            print('No Pool! Single Worker!')
            shuff_ncuts = shuffle_and_cluster_activations(*function_argument)

        else:
            print(f'Using Pool! Multiple Workers! {n_workers}')

            workers_arguments = [[copy.deepcopy(arg) for _ in range(n_workers)]
                                 for arg in function_argument]

            with ProcessPool(nodes=n_workers) as p:
                shuff_ncuts_results = p.map(shuffle_and_cluster_activations,
                                            *workers_arguments)

            shuff_ncuts = np.concatenate(shuff_ncuts_results)

        shuffled_n_samples = len(shuff_ncuts)
        shuffled_mean = np.mean(shuff_ncuts, dtype=np.float64)
        shuffled_stdev = np.std(shuff_ncuts, dtype=np.float64)
        print('BEFORE', np.std(shuff_ncuts))
        percentile = compute_pvalue(unshuffled_ncut, shuff_ncuts)
        print('AFTER', np.std(shuff_ncuts))
        z_score = (unshuffled_ncut - shuffled_mean) / shuffled_stdev

        result.update({'n_samples': shuffled_n_samples,
                       'mean': shuffled_mean,
                       'stdev': shuffled_stdev,
                       'z_score': z_score,
                       'percentile': percentile})
    return result


def grid_cca(activations1, act_labels1, activations2, act_labels2, n_clusters):

    cca_grid = np.zeros((n_clusters, n_clusters))
    for clust_i in range(n_clusters):
        for clust_j in range(n_clusters):
            i_mask = act_labels1 == clust_i
            j_mask = act_labels2 == clust_j
            if sum(i_mask) == 0 or sum(j_mask) == 0:
                cca_grid[clust_i, clust_j] = 0
                cca_grid[clust_j, clust_i] = 0
            else:
                n_comps = min(sum(i_mask), sum(j_mask))
                cca = CCA(n_components=n_comps)
                cca.fit(activations1[i_mask].T, activations2[j_mask].T)
                cca_score = cca.score(activations1[i_mask].T, activations2[j_mask].T)
                cca_grid[clust_i, clust_j] = cca_score

    return cca_grid


def clustering_comparisons(activations, act_labels, act_mask, corr_adj, weight_labels, weight_mask,
                           n_clusters, n_samples, with_shuffle, epsilon):

    # only consider units that were connected in the weight and activation graphs
    weight_act_mask = weight_mask[act_mask]
    act_weight_mask = act_mask[weight_mask]
    activations = activations[weight_act_mask]
    act_labels = act_labels[weight_act_mask]
    weight_labels = weight_labels[act_weight_mask]

    n_units = len(act_labels)
    assert len(act_labels) == len(weight_labels)
    assert n_units == np.sum(act_mask * weight_mask)

    # get normalized mutual info between two clusterings
    nmi = normalized_mutual_info_score(act_labels, weight_labels)

    # get the ncut that results from using the activation adj mat with the weight-based clustering labels
    mask_corr_adj = corr_adj[weight_act_mask, :][:, weight_act_mask]
    transfer_ncut = compute_ncut(mask_corr_adj, weight_labels, epsilon)

    # next, calculate the average intra and inter cluster corr_adj based on the weight labels
    intra_adj = np.array([])
    inter_adj = np.array([])
    for label in range(n_clusters):
        weight_label_mask = weight_labels == label
        intra_adj = np.append(intra_adj, mask_corr_adj[weight_label_mask, :][:, weight_label_mask].flatten())
        inter_adj = np.append(inter_adj, mask_corr_adj[weight_label_mask, :][:, 1-weight_label_mask].flatten())

    intra_mean = np.sum(intra_adj) / (len(intra_adj) - n_units)  # correct denom to ignore 0 self edges
    inter_mean = np.mean(inter_adj)

    # cca_grid = grid_cca(activations, weight_labels, n_clusters)

    results = {'normalized_mutual_information': nmi, 'transfer_ncut': transfer_ncut,
               'intra_mean': intra_mean, 'inter_mean': inter_mean}  # , 'cca_grid': cca_grid}

    if with_shuffle:
        shuffled_nmis = []
        for _ in range(n_samples):
            np.random.shuffle(weight_labels)
            shuffled_nmis.append(normalized_mutual_info_score(act_labels, weight_labels))
        shuffled_nmis = np.array(shuffled_nmis)

        shuffled_mean = np.mean(shuffled_nmis)
        shuffled_stdev = np.std(shuffled_nmis)
        results.update({'n_samples': n_samples,
                        'mean': shuffled_mean,
                        'stdev': shuffled_stdev,
                        'z_score': (nmi - shuffled_mean) / shuffled_stdev,
                        'percentile': compute_pvalue(nmi, shuffled_nmis)})

    return results


@activations_experiment.automain
def cluster_and_compare(activations_path, activations_mask_path, corr_type, weights_path, n_samples,
                        n_clusters, n_inputs, n_outputs, exclude_inputs, eigen_solver, assign_labels,
                        filter_norm, epsilon, with_shuffle, use_inv_avg_commute, n_workers):

    if 'cnn' in str(activations_path):
        network_type = 'cnn'
    else:
        network_type = 'mlp'

    act_cluster_results = do_clustering_activations(network_type, activations_path, activations_mask_path,
                                                    corr_type, n_clusters, n_inputs, n_outputs,
                                                    exclude_inputs, eigen_solver, assign_labels, epsilon,
                                                    n_samples, with_shuffle, n_workers)
    weight_cluster_results = do_clustering_weights(network_type, weights_path, n_clusters, n_inputs,
                                                   n_outputs, exclude_inputs, eigen_solver, assign_labels,
                                                   use_inv_avg_commute, filter_norm, epsilon)

    all_results = {'weight_cluster_results': weight_cluster_results,
                   'act_cluster_results': act_cluster_results}

    activations = act_cluster_results['activations']
    act_labels = act_cluster_results['labels']
    act_mask = act_cluster_results['mask']
    weight_labels = weight_cluster_results['labels']
    weight_mask = weight_cluster_results['mask']
    corr_adj = act_cluster_results['corr_adj']

    cluster_comparison_results = clustering_comparisons(activations, act_labels, act_mask, corr_adj,
                                                        weight_labels, weight_mask, n_clusters,
                                                        n_samples, with_shuffle, epsilon)

    all_results.update({'cluster_comparison_results': cluster_comparison_results})

    return all_results
