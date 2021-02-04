import numpy as np
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, MaxPooling2D
from src.spectral_cluster_model import get_conv_weight_connections
from src.cnn.extractor import expand_conv_layer, expand_pool_layer, conv_tensor_to_adj
from src.cnn.convertor import cnn2mlp


def test_expand_conv_layer():

    def tester(expected_layer, expected_output_side, **kwargs):
        layer, output_side, _ = expand_conv_layer(**kwargs)

        assert np.array_equal(expected_layer, layer)
        assert expected_output_side == output_side

    tester(np.ones((1, 1)), 1,
           kernel=np.ones((1, 1, 1, 1)), input_side=1, padding='same')

    tester(np.ones((1, 1)), 1,
           kernel=np.ones((1, 1, 1, 1)), input_side=1, padding='valid')



    tester(np.eye((9)), 3,
           kernel=np.ones((1, 1, 1, 1)), input_side=3, padding='same')

    tester(np.eye((9)), 3,
           kernel=np.ones((1, 1, 1, 1)), input_side=3, padding='valid')

    tester(np.ones((9, 1)),
           1,
           kernel=np.ones((3, 3, 1, 1)), input_side=3, padding='valid')

    tester(np.array([[1, 1, 0, 1, 1, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [0, 1, 1, 0, 1, 1, 0, 0, 0],
                     [1, 1, 0, 1, 1, 0, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 1, 1, 0, 1, 1, 0, 1, 1],
                     [0, 0, 0, 1, 1, 0, 1, 1, 0],
                     [0, 0, 0, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 1, 1, 0, 1, 1]]),
           3,
           kernel=np.ones((3, 3, 1, 1)), input_side=3, padding='same')

    tester(np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]]),
            1,
            kernel=(np.arange(9) + 1).reshape((3, 3, 1, 1)), input_side=3, padding='valid')

    tester(np.array([[5, 4, 0, 2, 1, 0, 0, 0, 0],
                     [6, 5, 4, 3, 2, 1, 0, 0, 0],
                     [0, 6, 5, 0, 3, 2, 0, 0, 0],
                     [8, 7, 0, 5, 4, 0, 2, 1, 0],
                     [9, 8, 7, 6, 5, 4, 3, 2, 1],
                     [0, 9, 8, 0, 6, 5, 0, 3, 2],
                     [0, 0, 0, 8, 7, 0, 5, 4, 0],
                     [0, 0, 0, 9, 8, 7, 6, 5, 4],
                     [0, 0, 0, 0, 9, 8, 0, 6, 5]]),
            3,
            kernel=(np.arange(9) + 1).reshape((3, 3, 1, 1)), input_side=3, padding='same')


    tester(np.array([[5, 4, 0, 2, 1, 0, 0, 0, 0],
                     [6, 5, 4, 3, 2, 1, 0, 0, 0],
                     [0, 6, 5, 0, 3, 2, 0, 0, 0],
                     [8, 7, 0, 5, 4, 0, 2, 1, 0],
                     [9, 8, 7, 6, 5, 4, 3, 2, 1],
                     [0, 9, 8, 0, 6, 5, 0, 3, 2],
                     [0, 0, 0, 8, 7, 0, 5, 4, 0],
                     [0, 0, 0, 9, 8, 7, 6, 5, 4],
                     [0, 0, 0, 0, 9, 8, 0, 6, 5]]),
            3,
            kernel=(np.arange(9) + 1).reshape((3, 3, 1, 1)), input_side=3, padding='same')

    # For testing, the indices of the test matrix are not ordered
    # by the real one K[row, col, chan_in, chan_out]
    # K[:, :, 0, 0] -> 1..9
    # K[:, :, 1, 0] -> 10..18
    # K[:, :, 0, 1] -> 19..27
    # K[:, :, 1, 1] -> 28..36
    K = (np.arange(36) + 1).reshape((2, 2, -1)).T.reshape((3, 3, 2, 2))


    tester(np.array([[ 1, 19,  0,  0,  0,  0,  0,  0],
                     [10, 28,  0,  0,  0,  0,  0,  0],
                     [ 2, 20,  1, 19,  0,  0,  0,  0],
                     [11, 29, 10, 28,  0,  0,  0,  0],
                     [ 3, 21,  2, 20,  0,  0,  0,  0],
                     [12, 30, 11, 29,  0,  0,  0,  0],
                     [ 0,  0,  3, 21,  0,  0,  0,  0],
                     [ 0,  0, 12, 30,  0,  0,  0,  0],
                     [ 4, 22,  0,  0,  1, 19,  0,  0],
                     [13, 31,  0,  0, 10, 28,  0,  0],
                     [ 5, 23,  4, 22,  2, 20,  1, 19],
                     [14, 32, 13, 31, 11, 29, 10, 28],
                     [ 6, 24,  5, 23,  3, 21,  2, 20],
                     [15, 33, 14, 32, 12, 30, 11, 29],
                     [ 0,  0,  6, 24,  0,  0,  3, 21],
                     [ 0,  0, 15, 33,  0,  0, 12, 30],
                     [ 7, 25,  0,  0,  4, 22,  0,  0],
                     [16, 34,  0,  0, 13, 31,  0,  0],
                     [ 8, 26,  7, 25,  5, 23,  4, 22],
                     [17, 35, 16, 34, 14, 32, 13, 31],
                     [ 9, 27,  8, 26,  6, 24,  5, 23],
                     [18, 36, 17, 35, 15, 33, 14, 32],
                     [ 0,  0,  9, 27,  0,  0,  6, 24],
                     [ 0,  0, 18, 36,  0,  0, 15, 33],
                     [ 0,  0,  0,  0,  7, 25,  0,  0],
                     [ 0,  0,  0,  0, 16, 34,  0,  0],
                     [ 0,  0,  0,  0,  8, 26,  7, 25],
                     [ 0,  0,  0,  0, 17, 35, 16, 34],
                     [ 0,  0,  0,  0,  9, 27,  8, 26],
                     [ 0,  0,  0,  0, 18, 36, 17, 35],
                     [ 0,  0,  0,  0,  0,  0,  9, 27],
                     [ 0,  0,  0,  0,  0,  0, 18, 36]]),
            2,
            kernel=K, input_side=4, padding='valid')


def test_expand_pool_layer():

    def tester(expected_layer, expected_output_side, **kwargs):
        layer, output_side, _ = expand_pool_layer(**kwargs)

        assert np.array_equal(expected_layer, layer)
        assert expected_output_side == output_side


    tester(np.ones((1, 1)), 1,
           pool_size=(1, 1), input_side=1, n_channels=1, with_avg=False)


    tester(np.array([[1., 0., 0., 0.],
                     [1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 1., 0., 0.],
                     [1., 0., 0., 0.],
                     [1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.],
                     [0., 0., 0., 1.],
                     [0., 0., 1., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.],
                     [0., 0., 0., 1.]]),
          2,
          pool_size=(2, 2), input_side=4, n_channels=1, with_avg=False)         


    tester(np.array([[1., 0., 0., 0.],
                     [1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 1., 0., 0.],
                     [1., 0., 0., 0.],
                     [1., 0., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 1., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.],
                     [0., 0., 0., 1.],
                     [0., 0., 1., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.],
                     [0., 0., 0., 1.]]) / 4,
          2,
          pool_size=(2, 2), input_side=4, n_channels=1, with_avg=True)


    # drop last row and last column
    tester(np.array([[1.],
                     [1.],
                     [1.],
                     [0.],
                     [1.],
                     [1.],
                     [1.],
                     [0.],
                     [1.],
                     [1.],
                     [1.],
                     [0.],
                     [0.],
                     [0.],
                     [0.],
                     [0.]]),
          1,
          pool_size=(3, 3), input_side=4, n_channels=1, with_avg=False)

    tester(np.array([[1, 0],
                     [0, 1],
                     [1, 0],
                     [0, 1],
                     [1, 0],
                     [0, 1],
                     [1, 0],
                     [0, 1]]),
          1,
          pool_size=(2, 2), input_side=2, n_channels=2, with_avg=False)

    tester(np.array([[1, 0],
                     [0, 1],
                     [1, 0],
                     [0, 1],
                     [1, 0],
                     [0, 1],
                     [1, 0],
                     [0, 1]]) / 4,
          1,
          pool_size=(2, 2), input_side=2, n_channels=2, with_avg=True)


def test_conv_tensor_to_adj():

    conv_a = np.random.randn(4, 4, 1, 1)
    conv_b = np.random.randn(5, 5, 1, 1)
    l1_a = np.linalg.norm(np.abs(np.squeeze(conv_a)), 1)
    l1_b = np.linalg.norm(np.abs(np.squeeze(conv_b)), 1)
    l2_a = np.linalg.norm(np.abs(np.squeeze(conv_a)), 2)
    l2_b = np.linalg.norm(np.abs(np.squeeze(conv_b)), 2)

    assert l1_a == conv_tensor_to_adj(conv_a, norm=1)[0][0]
    assert l1_b == conv_tensor_to_adj(conv_b, norm=1)[0][0]
    assert l2_a == conv_tensor_to_adj(conv_a, norm=2)[0][0]
    assert l2_b == conv_tensor_to_adj(conv_b, norm=2)[0][0]


def test_imagenet_conv_connections(gamma_factor=2, var_factor=4):

    model = Sequential()
    model.add(Conv2D(32, 3, activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))

    model_layers = model.layers

    conv_connections_before, _ = get_conv_weight_connections(model_layers, False, 1)

    cc = 0
    for i in range(len(model_layers)):
        if 'conv2d' in model_layers[i].name.lower():
            conv_weights = model_layers[i].get_weights()[0]
            conv_adj = conv_tensor_to_adj(conv_weights)
            assert math.isclose(np.mean(conv_adj / conv_connections_before[cc][0]['weights']), 1, rel_tol=0.001)
            cc += 1

    for i in range(len(model_layers)):
        if 'batch_normalization' in model_layers[i].name.lower():
            weights = model_layers[i].get_weights()
            weights[0] *= gamma_factor  # gamma times 2
            weights[-1] /= var_factor  # variance divided by 4
            model_layers[i].set_weights(weights)

    conv_connections_after, _ = get_conv_weight_connections(model_layers, False, 1)

    for ccb, cca in zip(conv_connections_before, conv_connections_after):
        assert math.isclose(np.mean(cca[0]['weights'] / ccb[0]['weights']), 4, rel_tol=0.01)
