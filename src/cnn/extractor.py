import itertools as it

import numpy as np
from scipy.sparse import coo_matrix
from src.cnn import CNN_MODEL_PARAMS


def sparsify(mat):
    """
    Make a matrix as sparse.
    
    Try configuration were tested on pruned CNN-MNIST on the preceptron machine
    
    1. Dense matrix
       peak memory: 7125.71 MiB, increment: 6818.84 MiB
       27.475 seconds       
    
    2. Only COO    <--- Choosen
       peak memory: 1759.01 MiB, increment: 1374.97 MiB
       29.456 seconds
       
    3. COO and then tocsc (as recomended)
       peak memory: 1708.50 MiB, increment: 1400.43 MiB
       30.339 seconds
    """
    return coo_matrix(mat)  #.tocsc()


# TODO: refactor with NumPy broadcasting for speed, maybe using Numba?
def expand_conv_layer(kernel, input_side, padding='valid', verbose=False, as_sparse=False):

    assert padding in ('valid', 'same')

    assert kernel.ndim == 4
    assert kernel.shape[0] == kernel.shape[1]

    kernel_side, _, input_n_channels, output_n_channels, = kernel.shape
    
    assert kernel_side % 2 == 1

    kernel_edge = kernel_side // 2

    output_side = input_side
    
    expanded_layer = np.zeros((input_side, input_side, input_n_channels,
                               output_side, output_side, output_n_channels)) 
    constrains = np.zeros((input_side, input_side, input_n_channels,
                               output_side, output_side, output_n_channels)) 

    for out_row, out_col, out_chan in it.product(range(output_side),
                                                 range(output_side),
                                                 range(output_n_channels)):
        if verbose:
            print('OUT', out_row, out_col, out_chan)
            
        for index, (kernel_row, kernel_col, in_chan) in enumerate(
                                                          it.product(
                                                              range(-kernel_edge, kernel_edge+1, 1),
                                                              range(-kernel_edge, kernel_edge+1, 1),
                                                              range(input_n_channels)
                                                          ), start=1
                                                        ):

            in_row = out_row + kernel_row
            in_col = out_col + kernel_col

            if in_row < 0 or in_col < 0 or in_row >= input_side or in_col >= input_side:
                continue

            kernel_value = kernel[kernel_row + kernel_edge, kernel_col+ kernel_edge, in_chan, out_chan]

            if verbose:
                print('... K', kernel_row, kernel_col, 'IN', in_row, in_col, in_chan, 'VAL', kernel_value)


            expanded_layer[in_row, in_col, in_chan, out_row, out_col, out_chan] = kernel_value

            constrains[in_row, in_col, in_chan, out_row, out_col, out_chan] = index

    # valid is reduced same
    if padding == 'valid':

        expanded_layer = expanded_layer[:, :, :,
                                        kernel_edge:-kernel_edge if kernel_edge else output_side,
                                        kernel_edge:-kernel_edge if kernel_edge else output_side,
                                        :]

        constrains = constrains[:, :, :,
                                kernel_edge:-kernel_edge if kernel_edge else output_side,
                                kernel_edge:-kernel_edge if kernel_edge else output_side,
                                :]

        output_side = input_side - kernel_side + 1

    expanded_layer = expanded_layer.reshape(input_side**2 * input_n_channels, output_side**2 * output_n_channels)

    constrains = constrains.reshape(input_side**2 * input_n_channels,
                                    output_side**2 * output_n_channels)

    if as_sparse:
        expanded_layer = sparsify(expanded_layer)
        constrains = sparsify(constrains)
        
    return expanded_layer, output_side, constrains


def expand_pool_layer(pool_size, input_side, n_channels, with_avg=False, verbose=False, as_sparse=False):
    
    assert pool_size[0] == pool_size[0]
    
    pool_side = pool_size[0]
    
    # assuming valid only
    # https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
    output_side = input_side // pool_side

    # expanded_pool = sparse.DOK(shape=(input_side, input_side, n_channels, output_side, output_side, n_channels))
    expanded_pool = np.zeros((input_side, input_side, n_channels, output_side, output_side, n_channels))
    
    value = 1 / pool_side**2 if with_avg else 1

    for out_row, out_col, chan, in it.product(range(output_side),
                                              range(output_side),
                                              range(n_channels)):
        
        in_start_row = out_row * pool_side
        in_start_col = out_col * pool_side
        
        in_end_row = in_start_row + pool_side
        in_end_col = in_start_col + pool_side

        if verbose:
            print('OUT', out_row, out_col,
                  'IN', (in_start_row, in_end_row), (in_start_col, in_end_col),
                  'CHAN', chan)

        expanded_pool[in_start_row:in_end_row,
                      in_start_col:in_end_col,
                      chan,
                      out_row, out_col, chan] = value
        
    expanded_pool = expanded_pool.reshape(input_side**2 * n_channels, output_side**2 * n_channels)
    constrains = np.zeros_like(expanded_pool)

    if as_sparse:
        expanded_pool = sparsify(expanded_pool)
        constrains = sparsify(constrains)
        
    return expanded_pool, output_side, constrains


def expand_bias_conv_layer(bias, output_side, as_sparse=False):
    bias = np.tile(bias, output_side**2)
    if as_sparse:
        bias = sparsify(bias)
    return bias


def build_bias_pool_layer(pool_layer, as_sparse=False):
    bias = np.zeros(pool_layer.shape[1])
    if as_sparse:
        bias = sparsify(bias)
    return bias


def extract_cnn_weights(weights, cnn_model_params=CNN_MODEL_PARAMS, input_side=28,
                        with_avg=False, as_sparse=False, verbose=False, biases=None):

    expanded_weights = []
    
    # This is buggy and wrong
    # Do not use `all_constraints` before fixing it!
    all_constraints = []

    weights_iter = iter(weights)
    
    if biases is not None:
        expanded_biases = []
        biases_iter = iter(biases)

    for conv_layer_param in cnn_model_params['conv']:

        kernel = next(weights_iter)
        layer_weights, input_side, constrains = expand_conv_layer(kernel, input_side,
                                                                  padding=conv_layer_param['padding'],
                                                                  as_sparse=as_sparse)
        expanded_weights.append(layer_weights)
        all_constraints.append(constrains)

        if biases is not None:
            bias = next(biases_iter)
            # the input_side is already updated
            # after the call to expand_conv_layer
            # so it is equal to the output_side of the CURRENT layer
            conv_layer_bias = expand_bias_conv_layer(bias, input_side, as_sparse)
            expanded_biases.append(conv_layer_bias)
            
        if verbose:
            print('Conv!', layer_weights.shape)

        if conv_layer_param['max_pool_after']:
            
            #assert conv_layer_param['max_pool_padding'] == 'valid'
            
            layer_weights, input_side, constrains = expand_pool_layer(conv_layer_param['max_pool_size'],
                                                                      input_side,
                                                                      kernel.shape[-1],
                                                                      with_avg=with_avg,
                                                                      as_sparse=as_sparse)
            
            expanded_weights.append(layer_weights)
            all_constraints.append(constrains)

            if biases is not None:
                # see comment above why the input_side
                # is in fact the output side
                pool_layer_bias = build_bias_pool_layer(layer_weights, as_sparse)
                expanded_biases.append(pool_layer_bias)


            if verbose:
                print('Pooling!', layer_weights.shape)

    for dense_layer_params in cnn_model_params['dense']:
        layer_weights = next(weights_iter)

        assert layer_weights.ndim == 2
        assert layer_weights.shape[1] == dense_layer_params

        constrains = np.ones_like(layer_weights)
        
        if as_sparse:
            layer_weights = sparsify(layer_weights)
            constrains = sparsify(constrains)

        expanded_weights.append(layer_weights)
        all_constraints.append(constrains)

        if biases is not None:
            # for a dense layer, the bias stays the same
            bias = next(biases_iter)
            if as_sparse:
                bias = sparsify(bias)
            expanded_biases.append(bias)

        if verbose:
            print('Dense!', layer_weights.shape)

    # softmax layer

    layer_weights = next(weights_iter)

    assert layer_weights.ndim == 2
    assert layer_weights.shape[0] == dense_layer_params

    constrains = np.ones_like(layer_weights)

    if as_sparse:
        layer_weights = sparsify(layer_weights)
        constrains = sparsify(constrains)

    expanded_weights.append(layer_weights)
    all_constraints.append(constrains)

    if biases is not None:
        # for a dense layer, the bias stays the same
        bias = next(biases_iter)
        if as_sparse:
            bias = sparsify(bias)
        expanded_biases.append(bias)

    if verbose:
        print('Dense! Softmax!', layer_weights.shape)
        
    if biases is not None:
        return expanded_weights, all_constraints, expanded_biases
    else:
        return expanded_weights, all_constraints


###################################################################################
# ImageNet
###################################################################################


def conv_tensor_to_adj(conv_weights, norm=1, scale=False):

    conv_shape = conv_weights.shape
    assert len(conv_shape) == 4
    conv_weights = np.abs(conv_weights)
    filters_in = conv_shape[2]
    filters_out = conv_shape[3]

    conv_weights = np.array([[np.linalg.norm(conv_weights[:, :, in_i, out_i], norm)
                              for out_i in range(filters_out)]
                             for in_i in range(filters_in)])

    if scale:
        conv_weights /= np.mean(conv_weights)

    return np.array(conv_weights)


def depthwise_conv_to_conv(dwc_weights):

    # takes in weights from tf.keras.depthwiseconv2d and gives its weights as a regular conv weight

    dwc_shape = dwc_weights.shape  # shape is [rows, cols, in_dim, out_mult]
    conv_weights = np.zeros(dwc_shape[:3] + (dwc_shape[2]*dwc_shape[3],))  # [rows, cols, in_dim, in_dim*out_mult]

    for in_i in range(dwc_shape[2]):  # for each input level
        for m in range(dwc_shape[3]):  # for each mult level
            conv_weights[:, :, in_i, in_i+m] = dwc_weights[:, :, in_i, m]

    return conv_weights


def extract_cnn_weights_filters_as_units(weights, norm=1, with_batch_norm=False,
                                         epsilon=0.001):

    # conv layer weights have shape (conv_height, conv_width, in_channels, out_channels)
    # fc layer weights have shape (in, out)
    # for our cnns, weight shapes are [(3, 3, 1, 32), (3, 3, 32, 32), (6272, 128), (128, 10)]
    # the returned weight shapes are [(784, 32), (32, 32), (32, 128),(128, 10)]
    # this function assumes that all conv layers have 'same' padding
    # this function associates weights between filters with the L2 norm of the kernel

    assert norm in (1, 2)

    if with_batch_norm:

        # If so, weights is a list full of weights that are either for conv layers or bn layers
        # Conv layers are arrays with shape [h, w, in_channels, out_channels]
        # Bn layers are lists of arrays [gamma, beta, mov_mean, mov_var] each with shape
        # [out_channels] for the previous conv layer

        collapsed_weights = []
        for w in weights:
            if isinstance(w, np.ndarray):  # if conv3d weights
                collapsed_weights.append(w)
            else:  # if bn weights
                assert isinstance(w, list)
                assert len(w) == 4, f'batch_norm weights list is of length {len(w)}'
                gamma, _, _, sigma_sq = w
                collapsed_weights[-1] *= gamma / np.sqrt(sigma_sq + epsilon)
        weights = collapsed_weights

    weights = [np.abs(w) for w in weights]  # absolute value
    weights_shapes = [w.shape for w in weights]  # extract shapes
    n_conv = sum(len(ws) == 4 for ws in weights_shapes)  # get num conv layers

    if n_conv == 0:
        return weights

    extracted_weights = []

    for conv_i in range(n_conv):

        extracted_weights.append(conv_tensor_to_adj(weights[conv_i], norm))

    if n_conv < len(weights_shapes):
        # get how many units each final kernel leads to after pooling
        n_reps = weights_shapes[n_conv][0] / weights_shapes[n_conv-1][3]
        # get the weights that connect the final conv kernels to the flattened units
        kernels_to_flat = np.hstack([np.repeat(np.expand_dims(extracted_weights[-1][:, k], 1), n_reps, axis=1)
                                     for k in range(weights_shapes[n_conv-1][3])])
        # get the composite connections from the final conv layer to the first fc layer's units
        extracted_weights.append(np.matmul(kernels_to_flat, weights[n_conv]))

        extracted_weights.extend(weights[n_conv+1:])  # add all fc layers

    return extracted_weights
