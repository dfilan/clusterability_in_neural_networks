from operator import xor
import itertools as it

import numpy as np
import pytest

from src.lesion.experimentation import _damaged_neurons_gen


@pytest.mark.parametrize('n_way, n_way_type, to_shuffle',
                         [(1, 'joint', False),
                          (1, 'joint', True),
                          (2, 'joint', False),
                          (2, 'joint', True),
                          (2, 'conditional', True),
                          (2, 'conditional', False)]) # only shuffled for double conditional
def test_damaged_neurons_gen(to_shuffle, n_way, n_way_type):
    
    layer_widths = [784, 100, 100, 100, 100, 10]

    cumsum_layer_widths = np.cumsum(layer_widths) - layer_widths[0]

    n_clusters = 7
    
    layer_labels = [np.random.randint(0, n_clusters, size=width)
                    for width in layer_widths]

    labels = np.concatenate(layer_labels)

    damaged_neurons_gen = _damaged_neurons_gen('mlp', layer_widths, labels, ignore_layers=False,
                                               to_shuffle=to_shuffle, n_way=n_way, n_way_type=n_way_type)


    def single_damage_tester(single_neurons_in_layers):
        layer_id, label, damaged_neurons, actual_layer_size = single_neurons_in_layers

        # Whether the layer size is correct
        assert layer_widths[layer_id] == actual_layer_size

        which_neurons_to_damage = np.nonzero(layer_labels[layer_id]  == label)[0]

        # Whether the number of neurons with given label is correct
        assert len(which_neurons_to_damage) == len(damaged_neurons)

        are_damaged_neurons_same_as_orginal_clustering = (which_neurons_to_damage == damaged_neurons).all()

        # Wether shuffling (or not shuffling) works
        # We want that only one of the conditions will be true, but not both!
        # except for double conditional
        if not (n_way == 2 and n_way_type == 'conditional'):
            assert xor(to_shuffle, are_damaged_neurons_same_as_orginal_clustering)


    def pair_damage_tester(first_neurons_in_layers,
                           second_neurons_in_layers):
        layer_id_1, label_1, damaged_neurons_1, _ = first_neurons_in_layers
        layer_id_2, label_2, damaged_neurons_2, _ = second_neurons_in_layers
        
        # for the same (layer, label), the damaged neurons
        # should be the same,
        # no matter if they are shuffled or not
        if (layer_id_1 == layer_id_2
            and label_1 == label_2):
            assert (damaged_neurons_1 == damaged_neurons_2).all()
        
        if n_way == 2 and n_way_type == 'conditional':
            which_neurons_to_damage_1 = np.nonzero(layer_labels[layer_id_1]
                                                   == label_1)[0]

            which_neurons_to_damage_2 = np.nonzero(layer_labels[layer_id_2]
                                                   == label_2)[0]
            
            are_damaged_neurons_same_as_orginal_clustering_1 = (which_neurons_to_damage_1 
                                                                == damaged_neurons_1).all()

            are_damaged_neurons_same_as_orginal_clustering_2 = (which_neurons_to_damage_2 
                                                                == damaged_neurons_2).all()

            # The SECOND is the the fixed in conditional,
            # so it should be as the original
            assert are_damaged_neurons_same_as_orginal_clustering_2

            # double conditional has no meaning if we work on the same layer-label
            # and it doesn't produce this 
            if not (layer_id_1 == layer_id_2
                and label_1 == label_2):
                assert xor(to_shuffle, are_damaged_neurons_same_as_orginal_clustering_1)


    for neurons_in_layers in damaged_neurons_gen:
        
        if n_way == 1:
            single_neurons_in_layers_iter = neurons_in_layers
        elif n_way == 2:
            # `neurons_in_layers` might be an iterator,
            # and we'd like to iterate over twice
            (single_neurons_in_layers_iter,
             pair_neurons_in_layers_iter) = it.tee(neurons_in_layers, 2)
        
        for single_neurons_in_layers in single_neurons_in_layers_iter:
            single_damage_tester(single_neurons_in_layers)
            
        if n_way == 2:
            first_neurons_in_layers, second_neurons_in_layers = pair_neurons_in_layers_iter
            pair_damage_tester(first_neurons_in_layers,
                               second_neurons_in_layers) 
