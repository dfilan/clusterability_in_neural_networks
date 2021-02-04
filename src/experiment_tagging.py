"""Tags and paths for trained models."""

from pathlib import Path
from glob import glob
import os
from datetime import datetime
import numpy as np

BASE_PATH = '../models/'

MODEL_TAG_LOOKUP = {
'MNIST': 'mnist_mlp_20epochs',
'MNIST+DROPOUT': 'mnist_mlp_20epochs_dropout',
'MNIST+L1REG': 'mnist_mlp_20epochs_l1reg',
'MNIST+L2REG': 'mnist_mlp_20epochs_l2reg',
'HALVES-SAME-MNIST': 'halves_same_mnist_mlp_20epochs',
'HALVES-SAME-MNIST+DROPOUT': 'halves_same_mnist_mlp_20epochs_dropout',
'HALVES-SAME-MNIST+L1REG': 'halves_same_mnist_mlp_20epochs_l1reg',
'HALVES-SAME-MNIST+L2REG': 'halves_same_mnist_mlp_20epochs_l2reg',
'HALVES-MNIST': 'halves_mnist_mlp_20epochs',
'HALVES-MNIST+DROPOUT': 'halves_mnist_mlp_20epochs_dropout',
'HALVES-MNIST+L1REG': 'halves_mnist_mlp_20epochs_l1reg',
'HALVES-MNIST+L2REG': 'halves_mnist_mlp_20epochs_l2reg',
'MNIST+MOD-INIT': 'mnist_mlp_20epochs_mod_init',
'MNIST+MOD-INIT+DROPOUT': 'mnist_mlp_20epochs_mod_init_dropout',
'MNIST+MOD-INIT+L1REG': 'mnist_mlp_20epochs_mod_init_l1reg',
'MNIST+MOD-INIT+L2REG': 'mnist_mlp_20epochs_mod_init_l2reg',
'HALVES-SAME-MNIST+MOD-INIT': 'halves_same_mnist_mlp_20epochs_mod_init',
'HALVES-SAME-MNIST+MOD-INIT+DROPOUT': 'halves_same_mnist_mlp_20epochs_mod_init_dropout',
'HALVES-SAME-MNIST+MOD-INIT+L1REG': 'halves_same_mnist_mlp_20epochs_mod_init_l1reg',
'HALVES-SAME-MNIST+MOD-INIT+L2REG': 'halves_same_mnist_mlp_20epochs_mod_init_l2reg',
'HALVES-MNIST+MOD-INIT+L1REG': 'halves_mnist_mlp_20epochs_mod_init_l1reg',
'HALVES-MNIST+MOD-INIT+L2REG': 'halves_mnist_mlp_20epochs_mod_init_l2reg',

'MNIST+LUCID': 'mnist_mlp_30epochs_lucid',
'MNIST+DROPOUT+LUCID': 'mnist_mlp_30epochs_dropout_lucid',
'MNIST+MOD-INIT+LUCID': 'mnist_mlp_30epochs_mod_init_lucid',
'MNIST+MOD-INIT+DROPOUT+LUCID': 'mnist_mlp_30epochs_mod_init_dropout_lucid',

'SMALL-MNIST': 'small_mnist_mlp_10epochs',
'SMALL-MNIST+CLUSTERABILITY-GRADIENT': 'small_mnist_mlp_10epochs_cluster_gradient',

'CIFAR10': 'cifar10_mlp_20epochs',
'CIFAR10+DROPOUT': 'cifar10_mlp_100epochs_dropout',
'CIFAR10+L1REG': 'cifar10_mlp_20epochs_l1reg',
'CIFAR10+L2REG': 'cifar10_mlp_20epochs_l2reg',
'HALVES-CIFAR10': 'halves_cifar10_mlp_20epochs',
'HALVES-CIFAR10+DROPOUT': 'halves_cifar10_mlp_20epochs_dropout',
'HALVES-CIFAR10+L1REG': 'halves_cifar10_mlp_20epochs_l1reg',
'HALVES-CIFAR10+L2REG': 'halves_cifar10_mlp_20epochs_l2reg',
'HALVES-SAME-CIFAR10': 'halves_same_cifar10_mlp_20epochs',
'HALVES-SAME-CIFAR10+DROPOUT': 'halves_same_cifar10_mlp_20epochs_dropout',
'HALVES-SAME-CIFAR10+L1REG': 'halves_same_cifar10_mlp_20epochs_l1reg',
'HALVES-SAME-CIFAR10+L2REG': 'halves_same_cifar10_mlp_20epochs_l2reg',
'CIFAR10+MOD-INIT': 'cifar10_mlp_20epochs_mod_init',
'CIFAR10+MOD-INIT+DROPOUT': 'cifar10_mlp_20epochs_mod_init_dropout',
'CIFAR10+MOD-INIT+L1REG': 'cifar10_mlp_20epochs_mod_init_l1reg',
'CIFAR10+MOD-INIT+L2REG': 'cifar10_mlp_20epochs_mod_init_l2reg',
'HALVES-SAME-CIFAR10+MOD-INIT': 'halves_same_cifar10_mlp_20epochs_mod_init',
'HALVES-SAME-CIFAR10+MOD-INIT+DROPOUT': 'halves_same_cifar10_mlp_20epochs_mod_init_dropout',
'HALVES-SAME-CIFAR10+MOD-INIT+L1REG': 'halves_same_cifar10_mlp_20epochs_mod_init_l1reg',
'HALVES-SAME-CIFAR10+MOD-INIT+L2REG': 'halves_same_cifar10_mlp_20epochs_mod_init_l2reg',
'HALVES-CIFAR10+MOD-INIT+L1REG': 'halves_cifar10_mlp_40epochs_mod_init_l1reg',
'HALVES-CIFAR10+MOD-INIT+L2REG': 'halves_cifar10_mlp_40epochs_mod_init_l2reg',

'FASHION': 'fashion_mlp_20epochs',
'FASHION+DROPOUT': 'fashion_mlp_20epochs_dropout',
'FASHION+L1REG': 'fashion_mlp_20epochs_l1reg',
'FASHION+L2REG': 'fashion_mlp_20epochs_l2reg',
'HALVES-FASHION': 'halves_fashion_mlp_20epochs',
'HALVES-FASHION+DROPOUT': 'halves_fashion_mlp_20epochs_dropout',
'HALVES-FASHION+L1REG': 'halves_fashion_mlp_20epochs_l1reg',
'HALVES-FASHION+L2REG': 'halves_fashion_mlp_20epochs_l2reg',
'HALVES-SAME-FASHION': 'halves_same_fashion_mlp_20epochs',
'HALVES-SAME-FASHION+DROPOUT': 'halves_same_fashion_mlp_20epochs_dropout',
'HALVES-SAME-FASHION+L1REG': 'halves_same_fashion_mlp_20epochs_l1reg',
'HALVES-SAME-FASHION+L2REG': 'halves_same_fashion_mlp_20epochs_l2reg',
'FASHION+MOD-INIT': 'fashion_mlp_20epochs_mod_init',
'FASHION+MOD-INIT+DROPOUT': 'fashion_mlp_20epochs_mod_init_dropout',
'FASHION+MOD-INIT+L1REG': 'fashion_mlp_20epochs_mod_init_l1reg',
'FASHION+MOD-INIT+L2REG': 'fashion_mlp_20epochs_mod_init_l2reg',
'HALVES-SAME-FASHION+MOD-INIT': 'halves_same_fashion_mlp_20epochs_mod_init',
'HALVES-SAME-FASHION+MOD-INIT+DROPOUT': 'halves_same_fashion_mlp_20epochs_mod_init_dropout',
'HALVES-SAME-FASHION+MOD-INIT+L1REG': 'halves_same_fashion_mlp_20epochs_mod_init_l1reg',
'HALVES-SAME-FASHION+MOD-INIT+L2REG': 'halves_same_fashion_mlp_20epochs_mod_init_l2reg',
'HALVES-FASHION+MOD-INIT+L1REG': 'halves_fashion_mlp_20epochs_mod_init_l1reg',
'HALVES-FASHION+MOD-INIT+L2REG': 'halves_fashion_mlp_20epochs_mod_init_l2reg',

'POLY': 'poly_mlp_regression_20epochs',
'POLY+L1REG': 'poly_mlp_regression_20epochs_l1reg',
'POLY+L2REG': 'poly_mlp_regression_20epochs_l2reg',

'MNIST-CIFAR10': 'mnist-cifar10_mlp_30epochs',
'MNIST-FASHION': 'mnist-fashion_mlp_20epochs',
'FASHION-CIFAR10': 'fashion-cifar10_mlp_30epochs',
'MNIST-CIFAR10-SEPARATED': 'mnist-cifar10-separated_mlp_30epochs',
'MNIST-FASHION-SEPARATED': 'mnist-fashion-separated_mlp_20epochs',
'FASHION-CIFAR10-SEPARATED': 'fashion-cifar10-separated_mlp_30epochs',
'MNIST-CIFAR10+DROPOUT': 'mnist-cifar10_mlp_30epochs_dropout',
'MNIST-FASHION+DROPOUT': 'mnist-fashion_mlp_20epochs_dropout',
'FASHION-CIFAR10+DROPOUT': 'fashion-cifar10_mlp_30epochs_dropout',
'MNIST-CIFAR10-SEPARATED+DROPOUT': 'mnist-cifar10-separated_mlp_30epochs_dropout',
'MNIST-FASHION-SEPARATED+DROPOUT': 'mnist-fashion-separated_mlp_20epochs_dropout',
'FASHION-CIFAR10-SEPARATED+DROPOUT': 'fashion-cifar10-separated_mlp_30epochs_dropout',
'MNIST-CIFAR10+L1REG': 'mnist-cifar10_mlp_30epochs_l1reg',
'MNIST-CIFAR10+L2REG': 'mnist-cifar10_mlp_30epochs_l2reg',
'MNIST-FASHION+L1REG': 'mnist-fashion_mlp_20epochs_l1reg',
'MNIST-FASHION+L2REG': 'mnist-fashion_mlp_20epochs_l2reg',
'FASHION-CIFAR10+L1REG': 'fashion-cifar10_mlp_30epochs_l1reg',
'FASHION-CIFAR10+L2REG': 'fashion-cifar10_mlp_30epochs_l2reg',
'MNIST-CIFAR10-SEPARATED+L1REG': 'mnist-cifar10-separated_mlp_30epochs_l1reg',
'MNIST-CIFAR10-SEPARATED+L2REG': 'mnist-cifar10-separated_mlp_30epochs_l2reg',
'MNIST-FASHION-SEPARATED+L1REG': 'mnist-fashion-separated_mlp_20epochs_l1reg',
'MNIST-FASHION-SEPARATED+L2REG': 'mnist-fashion-separated_mlp_20epochs_l2reg',
'FASHION-CIFAR10-SEPARATED+L1REG': 'fashion-cifar10-separated_mlp_30epochs_l1reg',
'FASHION-CIFAR10-SEPARATED+L2REG': 'fashion-cifar10-separated_mlp_30epochs_l2reg',

'CNN-MNIST': 'mnist_cnn_10epochs',
'CNN-MNIST+LUCID': 'mnist_cnn_10epochs_lucid',
'CNN-MOD-INIT-MNIST+LUCID': 'mnist_cnn_10epochs_mod_init_lucid',
'CNN-MNIST+DROPOUT': 'mnist_cnn_10epochs_dropout',
'CNN-MNIST+DROPOUT+LUCID': 'mnist_cnn_10epochs_dropout_lucid',
'CNN-MNIST+L1REG': 'mnist_cnn_10epochs_l1reg',
'CNN-MNIST+L2REG': 'mnist_cnn_10epochs_l2reg',
'CNN-MNIST+MOD-INIT': 'mnist_cnn_10epochs_mod_init',
'CNN-MNIST+MOD-INIT+DROPOUT': 'mnist_cnn_10epochs_mod_init_dropout',
'CNN-MNIST+MOD-INIT+DROPOUT+LUCID': 'mnist_cnn_10epochs_mod_init_dropout_lucid',
'CNN-MNIST+MOD-INIT+L1REG': 'mnist_cnn_10epochs_mod_init_l1reg',
'CNN-MNIST+MOD-INIT+L2REG': 'mnist_cnn_10epochs_mod_init_l2reg',
'CNN-STACKED-MNIST': 'stacked_mnist_cnn_10epochs',
'CNN-STACKED-MNIST+DROPOUT': 'stacked_mnist_cnn_10epochs_dropout',
'CNN-STACKED-MNIST+L1REG': 'stacked_mnist_cnn_10epochs_l1reg',
'CNN-STACKED-MNIST+L2REG': 'stacked_mnist_cnn_10epochs_l2reg',
'CNN-STACKED-SAME-MNIST': 'stacked_same_mnist_cnn_10epochs',
'CNN-STACKED-SAME-MNIST+DROPOUT': 'stacked_same_mnist_cnn_10epochs_dropout',
'CNN-STACKED-SAME-MNIST+L1REG': 'stacked_same_mnist_cnn_10epochs_l1reg',
'CNN-STACKED-SAME-MNIST+L2REG': 'stacked_same_mnist_cnn_10epochs_l2reg',
'CNN-STACKED-SAME-MNIST+MOD-INIT': 'stacked_same_mnist_cnn_10epochs_mod_init',
'CNN-STACKED-SAME-MNIST+MOD-INIT+DROPOUT': 'stacked_same_mnist_cnn_10epochs_mod_init_dropout',
'CNN-STACKED-SAME-MNIST+MOD-INIT+L1REG': 'stacked_same_mnist_cnn_10epochs_mod_init_l1reg',
'CNN-STACKED-SAME-MNIST+MOD-INIT+L2REG': 'stacked_same_mnist_cnn_10epochs_mod_init_l2reg',
'CNN-STACKED-MNIST+MOD-INIT+L1REG': 'stacked_mnist_cnn_10epochs_mod_init_l1reg',
'CNN-STACKED-MNIST+MOD-INIT+L2REG': 'stacked_mnist_cnn_10epochs_mod_init_l2reg',

'CNN-CIFAR10': 'cifar10_cnn_10epochs',
'CNN-CIFAR10+DROPOUT': 'cifar10_cnn_10epochs_dropout',
'CNN-CIFAR10+L1REG': 'cifar10_cnn_10epochs_l1reg',
'CNN-CIFAR10+L2REG': 'cifar10_cnn_10epochs_l2reg',
'CNN-STACKED-CIFAR10': 'stacked_cifar10_cnn_10epochs',
'CNN-STACKED-CIFAR10+DROPOUT': 'stacked_cifar10_cnn_10epochs_dropout',
'CNN-STACKED-CIFAR10+L1REG': 'stacked_cifar10_cnn_10epochs_l1reg',
'CNN-STACKED-CIFAR10+L2REG': 'stacked_cifar10_cnn_10epochs_l2reg',
'CNN-STACKED-SAME-CIFAR10': 'stacked_same_cifar10_cnn_10epochs',
'CNN-STACKED-SAME-CIFAR10+DROPOUT': 'stacked_same_cifar10_cnn_10epochs_dropout',
'CNN-STACKED-SAME-CIFAR10+L1REG': 'stacked_same_cifar10_cnn_10epochs_l1reg',
'CNN-STACKED-SAME-CIFAR10+L2REG': 'stacked_same_cifar10_cnn_10epochs_l2reg',
'CNN-CIFAR10+MOD-INIT': 'cifar10_cnn_10epochs_mod_init',
'CNN-CIFAR10+MOD-INIT+DROPOUT': 'cifar10_cnn_10epochs_mod_init_dropout',
'CNN-CIFAR10+MOD-INIT+L1REG': 'cifar10_cnn_10epochs_mod_init_l1reg',
'CNN-CIFAR10+MOD-INIT+L2REG': 'cifar10_cnn_10epochs_mod_init_l2reg',
'CNN-CIFAR10-FULL': 'cifar10_full_cnn_75epochs',
'CNN-CIFAR10-FULL+DROPOUT': 'cifar10_full_cnn_75epochs_dropout',
'CNN-CIFAR10-FULL+L1REG': 'cifar10_full_cnn_75epochs_l1reg',
'CNN-CIFAR10-FULL+L2REG': 'cifar10_full_cnn_75epochs_l2reg',
'CNN-STACKED-SAME-CIFAR10+MOD-INIT': 'stacked_same_cifar10_cnn_10epochs_mod_init',
'CNN-STACKED-SAME-CIFAR10+MOD-INIT+DROPOUT': 'stacked_same_cifar10_cnn_10epochs_mod_init_dropout',
'CNN-STACKED-SAME-CIFAR10+MOD-INIT+L1REG': 'stacked_same_cifar10_cnn_10epochs_mod_init_l1reg',
'CNN-STACKED-SAME-CIFAR10+MOD-INIT+L2REG': 'stacked_same_cifar10_cnn_10epochs_mod_init_l2reg',
'CNN-STACKED-CIFAR10+MOD-INIT+L1REG': 'stacked_cifar10_cnn_10epochs_mod_init_l1reg',
'CNN-STACKED-CIFAR10+MOD-INIT+L2REG': 'stacked_cifar10_cnn_10epochs_mod_init_l2reg',

'CNN-FASHION': 'fashion_cnn_10epochs',
'CNN-FASHION+DROPOUT': 'fashion_cnn_10epochs_dropout',
'CNN-FASHION+L1REG': 'fashion_cnn_10epochs_l1reg',
'CNN-FASHION+L2REG': 'fashion_cnn_10epochs_l2reg',
'CNN-STACKED-FASHION': 'stacked_fashion_cnn_10epochs',
'CNN-STACKED-FASHION+DROPOUT': 'stacked_fashion_cnn_10epochs_dropout',
'CNN-STACKED-FASHION+L1REG': 'stacked_fashion_cnn_10epochs_l1reg',
'CNN-STACKED-FASHION+L2REG': 'stacked_fashion_cnn_10epochs_l2reg',
'CNN-STACKED-SAME-FASHION': 'stacked_same_fashion_cnn_10epochs',
'CNN-STACKED-SAME-FASHION+DROPOUT': 'stacked_same_fashion_cnn_10epochs_dropout',
'CNN-STACKED-SAME-FASHION+L1REG': 'stacked_same_fashion_cnn_10epochs_l1reg',
'CNN-STACKED-SAME-FASHION+L2REG': 'stacked_same_fashion_cnn_10epochs_l2reg',
'CNN-FASHION+MOD-INIT': 'fashion_cnn_10epochs_mod_init',
'CNN-FASHION+MOD-INIT+DROPOUT': 'fashion_cnn_10epochs_mod_init_dropout',
'CNN-FASHION+MOD-INIT+L1REG': 'fashion_cnn_10epochs_mod_init_l1reg',
'CNN-FASHION+MOD-INIT+L2REG': 'fashion_cnn_10epochs_mod_init_l2reg',
'CNN-STACKED-SAME-FASHION+MOD-INIT': 'stacked_same_fashion_cnn_10epochs_mod_init',
'CNN-STACKED-SAME-FASHION+MOD-INIT+DROPOUT': 'stacked_same_fashion_cnn_10epochs_mod_init_dropout',
'CNN-STACKED-SAME-FASHION+MOD-INIT+L1REG': 'stacked_same_fashion_cnn_10epochs_mod_init_l1reg',
'CNN-STACKED-SAME-FASHION+MOD-INIT+L2REG': 'stacked_same_fashion_cnn_10epochs_mod_init_l2reg',
'CNN-STACKED-FASHION+MOD-INIT+L1REG': 'stacked_fashion_cnn_10epochs_mod_init_l1reg',
'CNN-STACKED-FASHION+MOD-INIT+L2REG': 'stacked_fashion_cnn_10epochs_mod_init_l2reg',

'CNN-VGG-CIFAR10': 'cifar10_full_cnn_vgg_200epochs',
'CNN-VGG-CIFAR10+L1REG': 'cifar10_full_cnn_vgg_200epochs_l1reg',
'CNN-VGG-CIFAR10+L2REG': 'cifar10_full_cnn_vgg_200epochs_l2reg',
'CNN-VGG-CIFAR10+DROPOUT': 'cifar10_full_cnn_vgg_200epochs_dropout',
'CNN-VGG-CIFAR10+DROPOUT+L2REG': 'cifar10_full_cnn_vgg_200epochs_dropout_l2reg',
'CNN-VGG-CIFAR10+MOD-INIT': 'cifar10_full_cnn_vgg_200epochs_mod_init',


'LINE': 'line_mlp_20epochs',
'LINE+DROPOUT': 'line_mlp_20epochs_dropout',
'LINE-MNIST': 'line-mnist_mlp_20epochs',
'LINE-CIFAR10': 'line-cifar10_mlp_30epochs',
'LINE-MNIST-SEPARATED': 'line-mnist-separated_mlp_20epochs',
'LINE-CIFAR10-SEPARATED': 'line-cifar10-separated_mlp_30epochs',
'LINE-MNIST+DROPOUT': 'line-mnist_mlp_20epochs_dropout',
'LINE-CIFAR10+DROPOUT': 'line-cifar10_mlp_30epochs_dropout',
'LINE-MNIST-SEPARATED+DROPOUT': 'line-mnist-separated_mlp_20epochs_dropout',
'LINE-CIFAR10-SEPARATED+DROPOUT': 'line-cifar10-separated_mlp_30epochs_dropout',
'RANDOM': 'random_mlp_20epochs',
'RANDOM+DROPOUT': 'random_mlp_20epochs_dropout',
'MNIST-x1.5-EPOCHS': 'mnist_mlp_30epochs',
'MNIST-x1.5-EPOCHS+DROPOUT': 'mnist_mlp_30epochs_dropout',
'MNIST-x2-EPOCHS': 'mnist_mlp_40epochs',
'MNIST-x2-EPOCHS+DROPOUT': 'mnist_mlp_40epochs_dropout',
'MNIST-x10-EPOCHS': 'mnist_mlp_200epochs',
'MNIST-x10-EPOCHS+DROPOUT': 'mnist_mlp_200epochs_dropout',
'RANDOM-x50-EPOCHS': 'random_mlp_1000epochs',
'RANDOM-x50-EPOCHS+DROPOUT': 'random_mlp_1000epochs_dropout',
'RANDOM-OVERFITTING': 'random_mlp_100epochs',
'RANDOM-OVERFITTING+DROPOUT': 'random_mlp_100epochs_dropout',
'CNN-LINE': 'line_cnn_10epochs',
'CNN-LINE+DROPOUT': 'line_cnn_10epochs_dropout'
}


def get_model_path(model_tag, date_tag='*', time_tag='*', filter_='last',
                   model_base_path=BASE_PATH):
    
    assert model_tag in MODEL_TAG_LOOKUP, (f"The tag `{model_tag}` doesn't exist.")
    assert filter_ in ('last', 'all')
    
    base_path = Path(model_base_path)
    
    paths = base_path.glob(f'{date_tag}/{MODEL_TAG_LOOKUP[model_tag]}/{time_tag}')
    
    # We cannot check whether a generator is empty without iterating over it,
    # so we make it into a list
    paths = list(paths)
    assert paths, (f'No model path, which correspond to the tag \'{model_tag}\' and base path '
                   f'\'{model_base_path}\', was found!')
    
    # The "maximual" path is the latest, because we use the date and time tags format as string
    # <year>-<month>-<date> and <hour>-<minutes>-<seconds>, respectively.
    # So the lexical order corresponds to time order (early to recent).
    if filter_ == 'last':
        run_results = max(paths)
    # Same argument goes for sorting the paths, to get them by timestamp order
    elif filter_ == 'all':
        run_results = sorted(paths)
    
    return run_results
