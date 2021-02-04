"""General utilities function."""


from contextlib import (redirect_stdout, redirect_stderr,
                       contextmanager, ExitStack)
import os
import logging
import pickle
import json
import math
import itertools as it
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity
from src.cnn.extractor import extract_cnn_weights
from scipy.stats import chisquare, norm, kstest, entropy, chi2, combine_pvalues
from scipy.special import binom


# https://stackoverflow.com/questions/50691545/how-to-use-a-with-statement-to-suppress-sys-stdout-or-sys-stderr
@contextmanager
def suppress(out=True, err=True):
    with ExitStack() as stack:
        with open(os.devnull, 'w') as null:
            if out:
                stack.enter_context(redirect_stdout(null))
            if err:
                stack.enter_context(redirect_stderr(null))
            yield


# https://gist.github.com/simon-weber/7853144
@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


def load_model2(model_path):
    custom_objects = {}
    if 'unpruned' not in str(model_path):
        custom_objects['PruneLowMagnitude'] = sparsity.pruning_wrapper.PruneLowMagnitude
    
    return tf.keras.models.load_model(model_path,
                                      custom_objects=custom_objects)


def load_weights_from_checkpoint(cpkt_path):
    model = load_model2(cpkt_path)
    return extract_weights(model)


def load_weights(weights_path):
    if 'ckpt' in str(weights_path):
        with all_logging_disabled():
            weights = load_weights_from_checkpoint(weights_path)

    else:
        with open(weights_path, 'rb') as f:
            weights = pickle.load(f)
            
    # if 'cnn' in str(weights_path).lower() and expand_cnn:
    #     print('CNN!')
    #     weights, _ = extract_cnn_weights(weights, verbose=True)
        
    return weights


def extract_weights(model, with_bias=False, with_batch_norm=False):

    if not with_batch_norm:
        weights, biases = zip(*(layer.get_weights() for layer in model.layers
                                if any(type_ in layer._name for type_ in ('dense', 'conv2d'))))
        if with_bias:
            return weights, biases
        else:
            return weights

    else:
        weights = []
        biases = []
        for layer in model.layers:
            if any(type_ in layer._name for type_ in ('dense', 'conv2d')):
                weights.append(layer.get_weights()[0])
                biases.append(layer.get_weights()[1])
            elif 'batch_nornm' in layer._name:
                weights.append(layer.get_weights())
                biases.append(None)
        if with_bias:
            return weights, biases
        else:
            return weights


def picklify(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def unpicklify(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_sparsity(network, is_model=True):
    if is_model:
        weights = extract_weights(network)
    else:
        weights = network

    return [(w == 0).mean() for w in weights]


def splitter(iterable, sizes):
    """Split an iterable into successive slice by sizes.

    >>> list(splitter(range(6), [1, 2, 3]))
    [[0], [1, 2], [3, 4, 5]]
    """

    iterator = iter(iterable)
    for size in sizes:
        yield list(it.islice(iterator, size))


def get_weights_paths(weight_directory, norm=1):
    weight_directory_path = Path(weight_directory)
    if 'cnn' in str(weight_directory):
        if norm == 1:
            weight_paths = {'unpruned' in str(path): path
                            for path in weight_directory_path.glob('*-weights-filter-units_l1.pckl')}
        else:
            weight_paths = {'unpruned' in str(path): path
                            for path in weight_directory_path.glob('*-weights-filter-units_l2.pckl')}
    else:
        weight_paths = {'unpruned' in str(path): path
                        for path in weight_directory_path.glob('*-weights.pckl')}
    return weight_paths


def get_model_paths(model_directory):
    model_directory_path = Path(model_directory)
    model_paths = {'unpruned' in str(path): path
                   for path in model_directory_path.glob('*.h5')}
    return model_paths


def get_activations_paths(activations_directory):
    activations_directory_path = Path(activations_directory)
    activations_paths = {'unpruned' in str(path): path
                         for path in activations_directory_path.glob('*-activations.pckl')}
    return activations_paths


def get_activation_masks_paths(masks_directory):
    masks_directory_path = Path(masks_directory)
    activations_paths = {'unpruned' in str(path): path
                         for path in masks_directory_path.glob('*-activations_mask.pckl')}
    return activations_paths


def enumerate2(iterable, start=0, step=1):
    count = start
    for value in iterable:
        yield (count, value)
        count += step


def preprocess_dataset(dataset_path, hot_one=False):
    with open(dataset_path, 'rb') as f:
        ds = pickle.load(f)

    size = 784
    if 'stacked' in dataset_path:
        size *= 2
    elif 'cifar10_full' in dataset_path:
        size = 32**2 * 3

    if 'poly' not in dataset_path:

        ds['X_train'] = (ds['X_train'] / 255).reshape(-1, size)
        ds['X_test'] = (ds['X_test'] / 255).reshape(-1, size)

    assert not hot_one, NotImplementedError
    
    return ds


def build_clustering_results(clustering_results):
    """Build a DataFrame of the results given nested dictionary.
    
    The first dictionary level is the model name,
    the second level is shuffle method,
    the third is is_pruned (boolean)
    and the fourth is the return value of `run_spectral_clustering`.
    """
    
    results = []

    for model_name in clustering_results:
        for shuffle_method in clustering_results[model_name]:
            for is_unpruned in clustering_results[model_name][shuffle_method]:
            
                result = {'model': model_name.replace('CNN-', '').replace('+DROPOUT', ''),
                          'network': 'CNN' if 'CNN' in model_name else 'MLP',
                          'dropout': 'DROPOUT' in model_name,
                          'is_unpruned': is_unpruned,
                          'shuffle_method': shuffle_method,
                          }

                labels, metrics = clustering_results[model_name][shuffle_method][is_unpruned]

                result.update(metrics)
                results.append(pd.Series(result))

    return pd.DataFrame(results)


def compute_pvalue(x, arr, side='left'):
    """
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC379178/
    """

    assert side in ('right', 'left')
    
    if side == 'right':
        comparisons = (x < arr)
    elif side == 'left':
        comparisons = (arr < x)
    
    r = np.sum(comparisons)
    n = len(comparisons)
    
    return (r + 1) / (n + 1)


def extract_classification_metrics(path, flatten=True):
    metrics_file = (Path(path) / 'metrics.json')
    raw_metrics = json.loads(metrics_file.read_text())

    try:
        metrics = {type_:
                   {'train': {'acc': results['acc'][-1], 'loss':  results['loss'][-1]},
                   'test': {'acc': results['val_acc'][-1], 'loss':  results['val_loss'][-1]}}
                   for type_, results in raw_metrics.items()}
    except KeyError:
        metrics = {type_:
                       {'train': {'acc': results['accuracy'][-1], 'loss': results['loss'][-1]},
                        'test': {'acc': results['val_accuracy'][-1], 'loss': results['val_loss'][-1]}}
                   for type_, results in raw_metrics.items()}
    
    # todo: refactor
    if flatten:
        metrics = {type_:
                   {'train_acc': results['train']['acc'],
                    'train_loss':  results['train']['loss'],
                    'test_acc': results['test']['acc'],
                    'test_loss':  results['test']['loss']}
                   for type_, results in metrics.items()}

    return metrics


def extract_regression_metrics(path, flatten=True):
    metrics_file = (Path(path) / 'metrics.json')
    raw_metrics = json.loads(metrics_file.read_text())

    metrics = {type_:
                   {'train': {'loss': results['loss'][-1]},
                    'test': {'loss': results['val_loss'][-1]}}
               for type_, results in raw_metrics.items()}

    # todo: refactor
    if flatten:
        metrics = {type_:
                       {'train_loss': results['train']['loss'],
                        'test_loss': results['test']['loss']}
                   for type_, results in metrics.items()}

    return metrics


# https://github.com/python/cpython/blob/ecb035cd14c11521276343397151929a94018a22/Modules/itertoolsmodule.c#L2772
def multi_combinations_with_replacement(*iterables):
    
    pools = [tuple(iterable) for iterable in iterables]
    
    assert len({len(pool) for pool in pools}) == 1,\
           'All iterables should have the same length.'
    
    n = len(pools[0])
    r = len(pools)
    
    for indices in it.product(range(n), repeat=r):
        if sorted(indices) == list(indices):
            yield tuple(pool[i] for i, pool in zip(indices, pools))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float, np.float32, np.float64)):
            return float(obj)

        return json.JSONEncoder.default(self, obj)


def heatmap_fixed(mat, **kwargs):
    ax = sns.heatmap(mat, **kwargs) 

    ax.set_xlim(0, len(mat))
    ax.set_ylim(0, len(mat))

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    return ax


def cohen_d(d1, d2):
    """Calculate Cohen's d for independent samples."""
    # Taken from
    # https://machinelearningmastery.com/effect-size-measures-in-python/

    n1, n2 = len(d1), len(d2)

    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)

    # calculate the pooled standard deviation

    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

    u1, u2 = np.mean(d1), np.mean(d2)

    return (u1 - u2) / s


def cohen_d_stats(u1, u2, sd1, sd2, n1, n2=None):
    """Calculate Cohen's d for independent samples.
    
    NOTE: sd1 and sd2 assumed to be the population standard deviation.
    """

    if n2 is None:
        n2 = n1
 
    s1, s2 = sd1**2, sd2**2

    # calculate the pooled standard deviation

    s = np.sqrt((n1 * s1 + n2 * s2) / (n1 + n2 - 2))

    return (u1 - u2) / s


def chi2_categorical_test(percentiles, n_samples):
    # chi sq test based on percentile counts -- note that this
    # treats ranks as categorical which is a bad assumption

    num_ranks = n_samples + 1
    min_percentile = 1 / num_ranks
    ranks = np.round((percentiles / min_percentile) - 1).astype(int)
    true_counts = np.concatenate((np.bincount(ranks), np.zeros(num_ranks - np.max(ranks) - 1).astype(int)))
    expected_counts = np.ones(num_ranks) * np.mean(true_counts)
    _, chi_categorical_p = chisquare(f_obs=true_counts, f_exp=expected_counts)

    return chi_categorical_p


def chi2_test(z_scores):
    # test based on a chi square distributed test statistic based on the sum of squared
    # z scores note that in order to be more conservative, z_scores should be calculated with
    # mean and std estimators that take into account the value of the true lesion data

    test_stat = np.sum(z_scores ** 2)
    df = len(z_scores)
    chi_p = 1 - chi2.cdf(test_stat, df=df)

    return chi_p


def bates_test(percentiles, n_samples):
    # one-sided bates test based on shifted percentiles and a uniform approximation
    # the percentiles are shifted to have mean 0.5, so under the null, they will have
    # the same mean as the uniform but smaller central moments higher than the first
    # so this will be a conservative test
    # also note that this is a one-sided test

    min_percentile = 1 / (n_samples + 1)
    n_percentiles = len(percentiles)
    shifted_percentiles = percentiles - (min_percentile / 2)  # center around 0.5

    if n_percentiles <= 30:  # use the bates cdf (technically the irwin hall one)
        percentile_sum = np.sum(shifted_percentiles)  # the test statistic
        p = (1 / math.factorial(n_percentiles)) * \
            sum([(-1)**k * binom(n_percentiles, k) * (percentile_sum-k)**n_percentiles
                 for k in range(int(percentile_sum))])
    else:  # use a normal approximation
        percentile_mean = np.mean(shifted_percentiles)  # the test statistic
        sigma = math.sqrt(1 / (12 * n_percentiles))
        p = norm.cdf(percentile_mean, loc=0.5, scale=sigma)

    return p


def combine_ps(percentiles, n_samples, method='fisher'):

    assert method in ['fisher', 'stouffer']

    min_percentile = 1 / (n_samples + 1)
    shifted_percentiles = percentiles - (min_percentile / 2)  # center around 0.5

    p = combine_pvalues(shifted_percentiles, method=method)[1]

    return p
