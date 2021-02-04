"""Script for training neural network for binary classification of 28x28 grayscale images."""

import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path
import os
import numpy as np
import sacred
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from tensorflow_model_optimization.sparsity import keras as sparsity

from src.cluster_gradient import make_eigenval_op
from src.cnn import CNN_MODEL_PARAMS, CNN_VGG_MODEL_PARAMS
from src.cnn.extractor import extract_cnn_weights, extract_cnn_weights_filters_as_units
from src.pointers import DATA_PATHS
from src.utils import picklify, extract_weights, get_sparsity, NumpyEncoder

ex = sacred.Experiment('training')
ex.observers.append(sacred
                    .observers
                    .FileStorageObserver
                    .create('training_runs_dir'))


def generate_training_tag(network_type,
                          epochs,
                          dataset_name,
                          init_modules,
                          dropout,
                          l1reg,
                          l2reg,
                          lucid,
                          cluster_gradient):
    base_tag = f"{dataset_name}_{network_type}_{epochs}epochs"
    if init_modules > 0:
        base_tag += "_mod_init"
    if dropout:
        base_tag += "_dropout"
    if l1reg:
        base_tag += "_l1reg"
    if l2reg:
        base_tag += "_l2reg"
    if lucid:
        base_tag += "_lucid"
    if cluster_gradient:
        base_tag += "_cluster_gradient"
    return base_tag


@ex.config
def general_config():
    num_classes = 10
    network_type = None
    epochs = 0
    dataset_name = ""
    with_dropout = False
    with_l1reg = False
    with_l2reg = False
    l1reg_rate = 0.00005
    l2reg_rate = 0.00005
    lucid = False
    init_modules = 0  # 0 means no modular initialization
    if 'stacked' in dataset_name:
        depth = 2
        width, height = 28, 28
    elif lucid:
        depth = 3
        width, height = 28, 28
    else:
        depth = 1
        width, height = 28, 28
    size = width * height
    shuffle = True
    n_train = None
    cluster_gradient = False
    num_cluster_eigs = 0
    cluster_lambda = 0
    extract_activations = False
    act_fn = 'relu' # 'tanh'
    unroll_cnns = False  # Whether to unroll CNNs or treat each filter as a unit when extracting weights
    augmentation = False
    write_checkpoints = False
    num_cluster_grad_workers = 1 # number of parallel cores used to compute the
                                 # clusterability gradient, should be at most
                                 # the number of weight matrices
    training_tag = generate_training_tag(
        network_type,
        epochs,
        dataset_name,
        init_modules,
        with_dropout,
        with_l1reg,
        with_l2reg,
        lucid,
        cluster_gradient
    )
    model_dir_path = Path('./models/{}/{}/{}'.format(
        datetime.now().strftime('%Y%m%d'),
        training_tag,
        datetime.now().strftime('%H%M%S')
    ))
    tensorboard_log_dir = './logs'


@ex.config
def pruning_config():
    initial_sparsity = 0.50
    final_sparsity = 0.90
    begin_step = 0
    frequency = 10


@ex.named_config
def mlp_config():
    network_type = 'mlp'
    model_params = {'widths': [256, 256, 256, 256]}
    dataset_name = 'line'
    dropout_rate = 0.5
    epochs = 20
    pruning_epochs = 20
    batch_size = 128


@ex.named_config
def mlp_regression_config():
    network_type = 'mlp_regression'
    model_params = {'widths': [256, 256, 256, 256]}
    dataset_name = 'poly'
    dropout_rate = 0.
    epochs = 20
    pruning_epochs = 20
    batch_size = 128
    num_classes = 512
    size = 2


@ex.named_config
def cnn_config():
    # Reference: https://keras.io/examples/cifar10_cnn/
    network_type = 'cnn'
    dataset_name = ""
    model_params = CNN_MODEL_PARAMS
    conv_dropout_rate = 0.25
    dense_dropout_rate = 0.5
    epochs = 10
    pruning_epochs = 10
    batch_size = 64


@ex.named_config
def cnn_vgg_config():
    # Reference: https://github.com/geifmany/cifar-vgg
    network_type = 'cnn_vgg'
    dataset_name = ""
    model_params = CNN_VGG_MODEL_PARAMS
    epochs = 200
    pruning_epochs = 50
    batch_size = 128
    l2reg_rate = 0.0005
    depth = 3
    width, height = 32, 32
    size = width * height
    augmentation = True
    conv_dropout_rate = 0  # gets overridden
    dense_dropout_rate = 0  # gets overridden


@ex.named_config
def small_mlp_config():
    network_type = 'mlp'
    model_params = {'widths': [64, 64, 64]}
    dataset_name = 'small_mnist'
    width, height = 7,7
    epochs = 10
    pruning_epochs = 10
    batch_size = 128
    dropout_rate = 0.2 # dropout not actually used

    
@ex.named_config
def small_mlp_cluster_config():
    network_type = 'mlp'
    model_params = {'widths': [64, 64, 64]}
    dataset_name = 'small_mnist'
    width, height = 7,7
    epochs = 10
    pruning_epochs = 10
    cluster_gradient = True
    cluster_lambda = 0.1
    num_cluster_eigs = 3
    batch_size = 128
    num_cluster_grad_workers = 4
    dropout_rate = 0.2 # dropout not actually used


@ex.capture
def get_pruning_params(num_train_samples,
                       initial_sparsity, final_sparsity,
                       begin_step, frequency,
                       batch_size, pruning_epochs):

    end_step = (np.ceil(num_train_samples / batch_size).astype(np.int32)
                * pruning_epochs)

    return {'pruning_schedule': sparsity.PolynomialDecay(
        initial_sparsity=initial_sparsity,
        final_sparsity=final_sparsity,
        begin_step=begin_step,
        end_step=end_step,
        frequency=frequency
    )
    }


@ex.capture
def save_weights(model, model_path, network_type, unroll_cnns, _log):
        
    weight_path = str(model_path) + '-weights.pckl'

    weights = extract_weights(model)

    picklify(weight_path, weights)
    ex.add_artifact(weight_path)
    
    if network_type == 'cnn':

        _log.info('Expanding CNN layers...')

        if unroll_cnns:  # if extracting cnn weights via unrolling into mlp
        
            expanded_weights, constraints = extract_cnn_weights(
                weights, as_sparse=True, verbose=True
            )

            expanded_weight_path = str(model_path) + '-weights-expanded.pckl'
            constraintst_path = str(model_path) + '-constraints-expanded.pckl'

            picklify(expanded_weight_path,
                     expanded_weights)
            ex.add_artifact(expanded_weight_path)

            picklify(constraintst_path,
                     constraints)
            ex.add_artifact(constraintst_path)

        else:  # if extracting cn weights via treating each filter as a unit

            _log.info(f'Raw CNN model weight shapes: {[wgt.shape for wgt in weights]}')

            extracted_weights_l1 = extract_cnn_weights_filters_as_units(weights, norm=1)

            extracted_weights_l2 = extract_cnn_weights_filters_as_units(weights, norm=2)

            _log.info(f'Extracted CNN model weight shapes: {[ew.shape for ew in extracted_weights_l1]}')

            extracted_weight_path_l1 = str(model_path) + '-weights-filter-units_l1.pckl'
            extracted_weight_path_l2 = str(model_path) + '-weights-filter-units_l2.pckl'

            picklify(extracted_weight_path_l1, extracted_weights_l1)
            ex.add_artifact(extracted_weight_path_l1)
            picklify(extracted_weight_path_l2, extracted_weights_l2)
            ex.add_artifact(extracted_weight_path_l2)

    if network_type == 'cnn_vgg':

        _log.info('Expanding CNN_VGG layers...')

        weights = extract_weights(model, with_batch_norm=True)
        extracted_weights_l1 = extract_cnn_weights_filters_as_units(weights, norm=1, with_batch_norm=True)

        _log.info(f'Extracted CNN model weight shapes: {[ew.shape for ew in extracted_weights_l1]}')

        extracted_weight_path_l1 = str(model_path) + '-weights-filter-units_l1.pckl'

        picklify(extracted_weight_path_l1, extracted_weights_l1)
        ex.add_artifact(extracted_weight_path_l1)


@ex.capture
def save_activations(model, model_path, dset_X, batch_size, _log, width, height):
    # get activations across datasets and save to model dir
    # for conv layers, take the mean of each channel
    # https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer

    in_dims = width * height
    if len(dset_X.shape) == 4:
        in_dims *= dset_X.shape[-1]

    _log.info('Dataset dimensions: ' + str(dset_X.shape))

    n_test = dset_X.shape[0]
    n_test -= n_test % batch_size
    _log.info(f'Extracting activations on testing set with {n_test} examples...')

    inp = model.input  # input placeholder
    _log.info('Layer types: ' + str([type(layer) for layer in model.layers]))
    outputs = [layer.output for layer in model.layers if any(type_ in layer._name for type_ in ('dense', 'conv2d'))]
    # outputs = []
    # for layer in model.layers:
    #     if not isinstance(layer, tf.keras.layers.Dropout):
    #         if isinstance(layer, sparsity.pruning_wrapper.PruneLowMagnitude):
    #             if not isinstance(layer.layer, tf.keras.layers.Dropout):
    #                 outputs.append(layer.output)
    #         else:
    #             outputs.append(layer.output)

    functor = tf.keras.backend.function([inp, tf.keras.backend.learning_phase()],
                                        outputs)  # evaluation function

    activations_single_batch = functor([dset_X[:batch_size], 0])
    n_layers = len(activations_single_batch)
    activations_dims = [(in_dims,)]
    for lyr_single in activations_single_batch:
        shp = np.squeeze(lyr_single).shape
        activations_dims.append((shp[-1],))  # each filter is a unit if a cnn

    _log.info('Model layer unit dims: ' + str(activations_dims))

    activations = [np.zeros(((n_test,) + lyr_dims)) for lyr_dims in activations_dims]

    for test_i in range(0, n_test, batch_size):  # iterate through test set
        batch_in = dset_X[test_i: test_i + batch_size]
        if len(batch_in.shape) == 2:  # mlp case
            activations[0][test_i: test_i + batch_size] = batch_in
        else:  # cnn case
            batch_in_channels_first = np.transpose(batch_in, (0, 3, 1, 2))
            activations[0][test_i: test_i + batch_size] = np.reshape(batch_in_channels_first, [batch_size, -1])
        acts_batch = functor([dset_X[test_i: test_i + batch_size], 0])  # False for eval
        for lyr in range(n_layers):
            if len(acts_batch[lyr].shape) == 2:  # fc layer case
                activations[lyr+1][test_i: test_i + batch_size] = acts_batch[lyr]
            else:  # conv case
                activations[lyr+1][test_i: test_i + batch_size] = np.linalg.norm(acts_batch[lyr],
                                                                                 ord=2, axis=(1, 2))

    _log.info(f'Activations extracted for {n_test} datapoints with shapes: {[lyr.shape for lyr in activations]}')
    all_act_mat = np.hstack(activations).T  # after taking .T, each row is a unit and each col an example
    row_stds = np.std(all_act_mat, axis=1)
    activations_mask = row_stds != 0
    all_act_mat = all_act_mat[activations_mask]

    with open(str(model_path) + '-activations.pckl', 'wb') as f:  # dump in same dir as model weights
        pickle.dump(all_act_mat, f)
    with open(str(model_path) + '-activations_mask.pckl', 'wb') as f:  # dump in same dir as model weights
        pickle.dump(activations_mask, f)
    _log.info('Done extracting activations.')


@ex.capture
def load_data(dataset_name, num_classes, width, height, size,
              network_type, n_train, lucid, depth, _log):
    
    assert dataset_name in DATA_PATHS

    data_path = DATA_PATHS[dataset_name]

    with open(data_path, 'rb') as f:
        dataset = pickle.load(f)
    X_train = dataset['X_train']
    X_test = dataset['X_test']

    if network_type != 'mlp_regression':
    
        if (X_train.min() == 0
            and X_train.max() <= 255
            and X_train.max() >= 250
            and X_test.min() == 0
            and X_test.max() <= 255
            and X_test.max() >= 250):
            X_train = X_train / 255
            X_test = X_test / 255
        # elif (X_train.min() == 0 and X_train.max() == 1 and X_test.min() == 0 and X_test.max() == 1):
        #     pass
        else:
            raise ValueError('X_train and X_test should be either in the range [0, 255] or [0, 1].')

        assert X_train.min() == 0
        assert X_test.min() == 0
        assert X_train.max() <= 1
        assert X_test.max() <= 1
        assert X_train.max() >= 0.95
        assert X_test.max() >= 0.95

        y_train = tf.keras.utils.to_categorical(dataset['y_train'])
        y_test = tf.keras.utils.to_categorical(dataset['y_test'])
        assert y_train.shape[-1] == 10
        assert y_test.shape[-1] == 10

    else:

        y_train = dataset['y_train']
        y_test = dataset['y_test']

    if network_type == 'cnn' or network_type == 'cnn_vgg' or lucid:

        if 'stacked' in dataset_name:

            X_train = np.transpose(X_train, (0, 2, 3, 1))
            X_test = np.transpose(X_test, (0, 2, 3, 1))

        else:

            if lucid:
                X_train = np.tile(X_train, depth)
                X_test = np.tile(X_test, depth)

            X_train = X_train.reshape([-1, height, width, depth])
            X_test = X_test.reshape([-1, height, width, depth])

        assert X_train.shape[-3:] == (height, width, depth)
        assert X_test.shape[-3:] == (height, width, depth)

    elif network_type == 'mlp':
        X_train = X_train.reshape([-1, size])
        X_test = X_test.reshape([-1, size])

    if n_train is not None:
        X_train = X_train[:n_train]
        y_train = y_train[:n_train]

    _log.info(f'X_train/test shapes of {X_train.shape} and {X_test.shape}')

    return (X_train, y_train), (X_test, y_test)


@ex.capture
def create_mlp_layers(network_type, size, width, height, num_classes, model_params, act_fn,
                      with_dropout, dropout_rate, with_l1reg, with_l2reg, l1reg_rate, l2reg_rate, lucid):

    assert model_params['widths']

    if with_l2reg:
        lreg = True
        regularizer = tf.keras.regularizers.l2
        reg_rate = l2reg_rate
    elif with_l1reg:
        lreg = True
        regularizer = tf.keras.regularizers.l1
        reg_rate = l1reg_rate
    else:
        lreg = False
        regularizer = None
        reg_rate = 0

    if lucid:
        layers = [tf.keras.layers.Flatten(input_shape=(width, height, 3)),
                  tf.keras.layers.Dense(model_params['widths'][0],
                                        activation='relu')]

    else:
        if lreg:
            layers = [tf.keras.layers.Dense(model_params['widths'][0],
                                            kernel_regularizer=tf.keras.regularizers.l1(reg_rate),
                                            activation=act_fn, input_shape=(size,))]
        else:
            layers = [tf.keras.layers.Dense(model_params['widths'][0],
                                            activation=act_fn, input_shape=(size,))]

    if lreg:
        hidden_layers = [tf.keras.layers.Dense(layer_width, activation=act_fn,
                                               kernel_regularizer=regularizer(reg_rate))
                         for layer_width in model_params['widths'][1:]]
    else:
        hidden_layers = [tf.keras.layers.Dense(layer_width, activation=act_fn)
                         for layer_width in model_params['widths'][1:]]

    if with_dropout:
        new_hidden_layers = [tf.keras.layers.Dropout(dropout_rate)]

        for hidden in hidden_layers:
            new_hidden_layers.append(hidden)
            new_hidden_layers.append(tf.keras.layers.Dropout(dropout_rate))

        hidden_layers = new_hidden_layers
    
    layers.extend(hidden_layers)

    if network_type == 'mlp':
        if lreg:
            layers.append(tf.keras.layers.Dense(num_classes, activation='softmax',
                                                kernel_regularizer=regularizer(reg_rate)))
        else:
            layers.append(tf.keras.layers.Dense(num_classes, activation='softmax'))
    else:  # the network type is 'mlp_regression'
        if lreg:
            layers.append(tf.keras.layers.Dense(num_classes, activation=None,
                                                kernel_regularizer=regularizer(reg_rate)))
        else:
            layers.append(tf.keras.layers.Dense(num_classes, activation=None))
    
    return layers


@ex.capture
def create_cnn_layers(dataset_name, width, height, num_classes, model_params, act_fn,
                      with_dropout, conv_dropout_rate, dense_dropout_rate,
                      with_l1reg, l1reg_rate, with_l2reg, l2reg_rate, depth):

    assert model_params['conv']
    assert model_params['dense']

    if with_l2reg:
        lreg = True
        regularizer = tf.keras.regularizers.l2
        reg_rate = l2reg_rate
    elif with_l1reg:
        lreg = True
        regularizer = tf.keras.regularizers.l1
        reg_rate = l1reg_rate
    else:
        lreg = False
        regularizer = None
        reg_rate = 0

    layers = []
    
    conv_layers = []
    
    is_first = True

    for conv_params in model_params['conv']:
        conv_kwargs = {'input_shape': (width, height, depth)} if is_first else {}
        is_first = False

        if lreg:
            conv_layers.append(tf.keras.layers.Conv2D(conv_params['filters'],
                                                      conv_params['kernel_size'],
                                                      padding=conv_params['padding'],
                                                      activation=act_fn,
                                                      kernel_regularizer=regularizer(reg_rate),
                                                      **conv_kwargs))
        else:
            conv_layers.append(tf.keras.layers.Conv2D(conv_params['filters'],
                                                      conv_params['kernel_size'],
                                                      padding=conv_params['padding'],
                                                      activation=act_fn,
                                                      **conv_kwargs))
        if conv_params['max_pool_after']:
            conv_layers.append(tf.keras.layers.MaxPooling2D(pool_size=conv_params['max_pool_size'],
                                                            padding=conv_params['max_pool_padding']))
        if conv_params['batch_norm_after']:
            conv_layers.append(tf.keras.layers.BatchNormalization())
        if with_dropout:
            if 'dropout_after' in colv_params.keys():
                if conv_params['dropout_after']:
                    conv_layers.append(tf.keras.layers.Dropout(conv_params['dropout_rate']))
            else:
                conv_layers.append(tf.keras.layers.Dropout(conv_dropout_rate))
    
    layers.extend(conv_layers)

    dense_layers = [tf.keras.layers.Flatten()]

    for dense_params in model_params['dense']:
        if lreg:
            dense_layers.append(tf.keras.layers.Dense(dense_params['units'], activation=act_fn,
                                                      kernel_regularizer=regularizer(reg_rate)))
        else:
            dense_layers.append(tf.keras.layers.Dense(dense_params['units'], activation=act_fn))
        if dense_params['batch_norm_after']:
            dense_layers.append(tf.keras.layers.BatchNormalization())
        if with_dropout:
            if 'dropout_after' in dense_params.keys():
                dense_layers.append(tf.keras.layers.Dropout(dense_params['dropout_rate']))
            else:
                dense_layers.append(tf.keras.layers.Dropout(dense_dropout_rate))

    layers.extend(dense_layers)

    if lreg:
        layers.append(tf.keras.layers.Dense(num_classes, activation='softmax',
                                            kernel_regularizer=regularizer(reg_rate)))
    else:
        layers.append(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return layers


@ex.capture
def create_model(network_type, dataset_name, init_modules, model_params,
                 size, num_classes, lucid, _log):

    assert network_type in ('mlp', 'mlp_regression', 'cnn', 'cnn_vgg')
    
    if network_type == 'mlp' or network_type == 'mlp_regression':
        layers = create_mlp_layers()

    else:
        layers = create_cnn_layers()
    
    model = tf.keras.Sequential(layers)

    if (network_type == 'mlp' or network_type == 'mlp_regression') and init_modules > 0:

        down_weight = 0.6
        up_weight = 1 + (1 - down_weight) * (init_modules - 1)

        layer_widths = [size] + model_params['widths']
        if num_classes <= init_modules:
            assignments = [np.random.randint(0, init_modules, size=layer_widths[i])
                           for i in range(len(layer_widths))] + \
                          [np.array(range(num_classes))]
        else:
            assignments = [np.random.randint(0, init_modules, size=layer_widths[i])
                           for i in range(len(layer_widths))] + \
                          [np.random.randint(0, init_modules, size=num_classes)]

        dense_i = 0
        for lyr_i, lyr in enumerate(model.layers):
            if 'dense' not in lyr.name.lower():  # skip dropout layers
                continue
            if dense_i == 0:  # skip the first layer because the pixels aren't always the same
                dense_i += 1
                continue
            weights = K.eval(lyr.weights[0][:])
            in_assign = assignments[dense_i]
            out_assign = assignments[dense_i + 1]
            for in_i in range(weights.shape[0]):
                for out_i in range(weights.shape[1]):
                    if in_assign[in_i] == out_assign[out_i]:
                        weights[in_i, out_i] *= up_weight
                    else:
                        weights[in_i, out_i] *= down_weight
            if len(lyr.weights) > 1:
                model.layers[lyr_i].set_weights((weights, K.eval(lyr.weights[1][:])))
            else:
                model.layers[lyr_i].set_weights(weights)
            dense_i += 1

    elif init_modules > 0:

        down_weight = 0.8
        up_weight = 1 + (1 - down_weight) * (init_modules - 1)

        filter_counts = [cl['filters'] for cl in model_params['conv']]
        assignments = [np.random.randint(0, init_modules, size=filter_counts[i]) for i in range(len(filter_counts))]

        conv_i = 0
        for lyr_i, lyr in enumerate(model.layers):
            if 'conv' not in lyr.name.lower():  # skip dropout and pooling layers
                continue
            if conv_i == 0:  # skip the first layer because the pixels aren't always the same
                conv_i += 1
                continue
            # conv layer weights have shape (conv_height, conv_width, in_channels, out_channels)
            weights = K.eval(lyr.weights[0][:])
            in_assign = assignments[conv_i - 1]
            out_assign = assignments[conv_i]
            for in_i in range(weights.shape[2]):
                for out_i in range(weights.shape[3]):
                    if in_assign[in_i] == out_assign[out_i]:
                        weights[:, :, in_i, out_i] *= up_weight
                    else:
                        weights[:, :, in_i, out_i] *= down_weight
            if len(lyr.weights) > 1:
                model.layers[lyr_i].set_weights((weights, K.eval(lyr.weights[1][:])))
            else:
                model.layers[lyr_i].set_weights(weights)
            conv_i += 1

    return model


@ex.capture
def get_two_model_paths(model_dir_path, dataset_name, network_type):
    directory = f'{dataset_name}-{network_type}-'
    return (model_dir_path / (directory + 'unpruned'),
            model_dir_path / (directory + 'pruned'))


@ex.capture
def train_model(model, X_train, y_train, X_test, y_test,
                model_path, batch_size, epochs, shuffle,
                network_type, cluster_gradient, num_cluster_eigs,
                cluster_lambda, extract_activations,
                tensorboard_log_dir, model_dir_path, _log,
                model_params, augmentation, write_checkpoints,
                num_cluster_grad_workers, is_pruning=False,
                callbacks=None):

    if callbacks == None:
        callbacks = []

    if write_checkpoints:

        ckpt_path = f'{model_path}-{{epoch:04d}}.ckpt'
        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                                           save_weights_only=False,
                                                           verbose=1)

        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir,
                                                     update_freq='batch',
                                                     profile_batch=0)

        callbacks.extend([ckpt_callback, tb_callback])

    # useful for debugging:
    # class WeightsInfo(tf.keras.callbacks.Callback):
    #     """
    #     Prints out the result of self.model.get_weights()
    #     """
    #     def __init__(self):
    #         super(WeightsInfo, self).__init__()

    #     def on_train_batch_begin(self, batch, logs=None):
    #         self.weights = self.model.get_weights()
    #         for i, thing in enumerate(self.weights):
    #             print(f"element {i} of self.weights is {thing} and has shape {thing.shape}")
    
    class NormalizeWeights(tf.keras.callbacks.Callback):
        """
        Callback which applies the ReLU scaling symmetry throughout the network
        """

        def __init__(self):
            super(NormalizeWeights, self).__init__()
            self.model_widths = model_params['widths']

        def on_train_batch_begin(self, batch, logs=None):
            weights = self.model.get_weights()
            if is_pruning:
                # when pruning, format of weights is:
                # layer matrix, bias vector, mask array, cutoff, num updates
                # interestingly, mask array has same shape as layer index, 
                # not masking the bias vector
                entries_per_layer = 5
            else:
                # weights array contains weight matrices and bias arrays
                # first weights matrix has shape
                # (input_pixels, first_hidden_width)
                # second weights matrix has shape (first_hidden_width,)
                # assumption in code: number of elements of weights is twice
                # the number of weight matrices
                entries_per_layer = 2

            assert len(weights) == (entries_per_layer
                                    * (len(self.model_widths) + 1))

            # now: want to have a for loop over hidden layers and their neurons
            for hidden_layer_num in range(len(self.model_widths)):
                incoming_weight_mat = weights[entries_per_layer
                                              * hidden_layer_num]
                incoming_biases     = weights[(entries_per_layer
                                               * hidden_layer_num) + 1]
                outgoing_weight_mat = weights[entries_per_layer
                                              * (hidden_layer_num + 1)]
                scale_factors = []
                if is_pruning:
                    mask = weights[(entries_per_layer * hidden_layer_num) + 3]
                    pruned_incoming_weights = incoming_weight_mat * mask
                for neuron in range(self.model_widths[hidden_layer_num]):
                    # take the norm of the incoming weights
                    # that's the 'neuron' column of the relevant weight matrix,
                    # and the 'neuron' element of the relevant bias
                    if is_pruning:
                        incoming_weights = pruned_incoming_weights[:,neuron]
                    else:
                        incoming_weights = incoming_weight_mat[:,neuron]
                    all_inc_weights  = np.append(incoming_weights,
                                                 incoming_biases[neuron])
                    scale_factor  = np.sqrt(np.sum(all_inc_weights**2))
                    # modify scale factor to ensure that gradients don't blow
                    # up (this is basically he initialisation)
                    scale_factor /= np.sqrt(2)
                    scale_factors.append(scale_factor)
                    # now divide incoming weights by the scale factor
                    if scale_factor != 0:
                        incoming_weight_mat[:,neuron] /= scale_factor
                        incoming_biases[neuron]       /= scale_factor
                    # and multiply outgoing weights by the scale factor
                    outgoing_weight_mat[neuron,:] *= scale_factor
                assert (weights[entries_per_layer]
                        != self.model.get_weights()[entries_per_layer]).any()

            self.model.set_weights(weights)


    # callbacks.append(WeightsInfo())

    if network_type == 'mlp' and cluster_gradient:
        callbacks.append(NormalizeWeights())
        def custom_loss(model_arg):
            def loss(y_true, y_pred):
                layer_tensors = [l.weights[0] for l in model_arg.layers]
                eigenvals = make_eigenval_op(
                    num_cluster_eigs, num_cluster_grad_workers
                )(*layer_tensors)
                cluster_score = tf.reduce_sum(eigenvals)
                return (
                    categorical_crossentropy(y_true, y_pred)
                    + (cluster_lambda / num_cluster_eigs) * cluster_score
                )
            return loss
        my_loss = custom_loss(model)
        my_metrics = ['accuracy']
    elif network_type == 'mlp_regression':
        my_loss = tf.keras.losses.MeanSquaredError()
        my_metrics = ['mse']
    else:
        my_loss = 'categorical_crossentropy'
        my_metrics = ['accuracy']
    model.compile(loss=my_loss,
                  optimizer='adam',
                  metrics=my_metrics)

    _log.info(model.summary())

    # model.save_weights(ckpt_path.format(epoch=0))

    if augmentation:
        data_gen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False)
        data_gen.fit(X_train)
        hist = model.fit(data_gen.flow(X_train, y_train, batch_size=batch_size),
                         steps_per_epoch=X_train.shape[0] // batch_size,
                         epochs=epochs,
                         verbose=1,
                         validation_data=(X_test, y_test),
                         callbacks=callbacks,
                         shuffle=shuffle)
    else:
        hist = model.fit(X_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         validation_data=(X_test, y_test),
                         callbacks=callbacks,
                         shuffle=shuffle)

    loss, acc = model.evaluate(X_test, y_test)
    _log.info('Trained model - Test dataset, accuracy: {:5.2f}%, loss: {:5.4f}'
              .format(100*acc, loss))

    if extract_activations:  # for now, only do for mlps
        assert network_type != 'cnn_vgg'
        save_activations(model, model_path, X_test, batch_size, _log)

    model_hdf_path = str(model_path) + '.h5'
    model.save(model_hdf_path)
    ex.add_artifact(model_hdf_path)

    for cpkt_filename in model_dir_path.glob('*.ckpt'):
        ex.add_artifact(cpkt_filename)

    return hist.history


@ex.automain
def run(network_type, epochs, pruning_epochs,
        tensorboard_log_dir, model_dir_path,
        _log, _run):

    if network_type == 'mlp' or network_type == 'mlp_regression':  # don't use gpu for mlps
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif network_type == 'cnn':
        os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    else:  # use gpu for cnn_vggs
        os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    assert network_type in ('mlp', 'mlp_regression', 'cnn', 'cnn_vgg')

    _log.info('Emptying model directory...')
    if model_dir_path.exists():
        shutil.rmtree(model_dir_path)
    Path(model_dir_path).mkdir(parents=True)

    _log.info('Loading data...')
    (X_train, y_train), (X_test, y_test) = load_data()
    
    metrics = {}
    
    unpruned_model_path, pruned_model_path = get_two_model_paths()

    unpruned_model = create_model()

    _log.info('Training unpruned model...')
    metrics['unpruned'] = train_model(unpruned_model, X_train, y_train, X_test,
                                      y_test, unpruned_model_path,
                                      epochs=epochs, is_pruning=False)

    _log.info('Unpruned model sparsity: {}'.format(
        get_sparsity(unpruned_model)
    ))
    save_weights(unpruned_model, unpruned_model_path)

    pruning_params = get_pruning_params(X_train.shape[0])    
    pruned_model = sparsity.prune_low_magnitude(
        unpruned_model, **pruning_params
    )
    
    pruning_callbacks = [
        sparsity.UpdatePruningStep(),
        sparsity.PruningSummaries(log_dir=tensorboard_log_dir,
                                  profile_batch=0)
    ]

    _log.info('Training pruned model...')
    metrics['pruned'] = train_model(pruned_model, X_train, y_train, X_test,
                                    y_test, pruned_model_path,
                                    epochs=pruning_epochs,
                                    is_pruning=True,
                                    callbacks=pruning_callbacks)

    _log.info('Pruned model sparsity: {}'.format(get_sparsity(pruned_model)))

    save_weights(pruned_model, pruned_model_path)
    
    ex.add_source_file(__file__)

    with open(model_dir_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, cls=NumpyEncoder)

    return metrics
