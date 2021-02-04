from functools import reduce

import numpy as np

from src.utils import extract_weights, load_model2
from src.cnn import CNN_MODEL_PARAMS
from src.cnn.extractor import extract_cnn_weights


# "For locally connected layers it is
# natural to give each unit its own bias, and for tiled convolution, it is natural to
# share the biases with the same tiling pattern as the kernels. For convolutional
# layers, it is typical to have one bias per channel of the output and share it across
# all locations within each convolution map." - Deep Learning Book, pp. 358

class SimpleMLP:
    def __init__(self, weights, biases, model_params):
        self._weights = [w.astype('float32').tocsc() for w in weights]
        try:
            self._biases = [b.toarray().astype('float32').flatten() for b in biases]
        except AttributeError:
            self._biases = biases.astype('float32')

        self._max_pools = self._extract_max_pools(model_params)
        
    def _extract_max_pools(self, model_params):

        max_pools = []

        for layer in model_params['conv']:
            if layer['max_pool_after']:
                max_pools.append({})
                max_pools.append({'max_pool_size': layer['max_pool_size'],
                                  'max_pool_padding': layer['max_pool_padding']})
            else:
                max_pools.append({})

        for layer in model_params['dense']:
            max_pools.append({})

        max_pools.append({})

        return max_pools


    def _appy_max_pool_valid_2x2(self, x, input_side, n_channels):

        n_data, n_elements = x.shape

        assert n_elements == input_side**2 * n_channels
        assert input_side % 2 == 0

        x = x.reshape(n_data, input_side, input_side, n_channels)

        cell_top_left     = x[:,  ::2,  ::2, :]
        cell_top_right    = x[:,  ::2, 1::2, :]
        cell_bottom_left  = x[:, 1::2,  ::2, :]
        cell_bottom_right = x[:, 1::2, 1::2, :]

        x = reduce(np.maximum, (cell_top_left,
                                cell_top_right,
                                cell_bottom_left,
                                cell_bottom_right))

        output_side = input_side // 2

        x = x.reshape(n_data, output_side**2 * n_channels)
    
        return x

    def _forward(self, x, masks=None):
        
        last_sentinel = [False] * (len(self._weights) - 1) + [True]
        
        if masks is None:
            masks = [None] * len(self._weights)

        for W, b, max_pool_data, is_last, mask in zip(self._weights,
                                                self._biases,
                                                self._max_pools,
                                                last_sentinel,
                                                masks):
            if not max_pool_data:
                x = x @ W + b
                
                if mask is not None:
                    x[:, mask] = 0
                    
                # No activation 
                if not is_last:
                    x = np.maximum(x, 0)  # ReLU
            
            else:
                # TODO: generalize this part
                # and use the data from model_params
                assert max_pool_data['max_pool_size'] == (2, 2)
                assert max_pool_data['max_pool_padding'] == 'valid'
            
                x = self._appy_max_pool_valid_2x2(x, 28, 32)

        return x
        
    
    def get_weights_and_biases(self):
        return self._weights, self._biases

    def set_weights_and_biases(self, weights, biases):
        self._weights = weights
        self._biases = biases
    
    def predict_classes(self, X, masks=None):
        logits = self._forward(X, masks)
        return np.argmax(logits,
                         axis=-1)

    def get_ignore_layers(self):
        return [bool(layer) for layer in self._max_pools]

    
def cnn2mlp(model_path, model_params=CNN_MODEL_PARAMS, verbose=False):
    cnn_model = load_model2(model_path)
    cnn_weights, cnn_biases = extract_weights(cnn_model,
                                              with_bias=True)

    mlp_weights, _, mlp_biases = extract_cnn_weights(cnn_weights,
                                                     biases=cnn_biases,
                                                     verbose=verbose,
                                                     as_sparse=True)

    return SimpleMLP(mlp_weights, mlp_biases, model_params)
