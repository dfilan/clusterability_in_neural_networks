"""Computing N-cut of random initialized neural networks."""

import tempfile

from src.train_nn import ex, create_mlp_layers, save_weights
from src.visualization import RANDOM_STATE
from src.utils import suppress, all_logging_disabled
from src.spectral_cluster_model import clustering_experiment

import tensorflow as tf


# we need to initialzie sacred to load the `mlp_config`
# TODO: perhaps there is a better way?
@ex.main
def run():
    pass

ex.run(named_configs=['mlp_config'])


def compute_ncut_random_init_mlp():

    model = tf.keras.Sequential(create_mlp_layers())
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_weights(model, f'{tmpdirname}/random')

        weights_path = f'{tmpdirname}/random-weights.pckl'

        with suppress(), all_logging_disabled():
            experiment_run = clustering_experiment.run(config_updates={'weights_path': weights_path,
                      'with_labels': False,
                      'with_shuffle': False,
                      'seed': RANDOM_STATE,
                      'num_clusters': 4},
                                                   named_configs=['mlp_config'])

    return experiment_run.result['ncut']