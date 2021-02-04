from tempfile import mkstemp
from functools import lru_cache

import sys
import os
import tensorflow as tf
import numpy as np
import pickle
from src.spectral_cluster_model import run_clustering_imagenet, get_dense_sizes
from src.utils import splitter, compute_pvalue, load_model2, get_model_paths, suppress, all_logging_disabled
from src.lesion.experimentation import _damaged_neurons_gen, load_model2
from src.experiment_tagging import get_model_path
from src.cnn import CNN_MODEL_PARAMS, CNN_VGG_MODEL_PARAMS
from src.visualization import run_spectral_cluster
from scipy.stats import entropy
from scipy.ndimage import zoom
from IPython import display
from src.lesion.experimentation import chi2_categorical_test, combine_ps
from classification_models.keras import Classifiers
from lucid.modelzoo.vision_base import Model
import lucid.modelzoo.vision_models as models
from lucid.misc.io import show
import lucid.optvis.objectives as objectives
from lucid.optvis import param
import lucid.optvis.render as render
import lucid.optvis.transform as transform

IMAGE_SIZE = 28
IMAGE_SIZE_CIFAR10 = 32
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
IMAGE_SHAPE_CIFAR10 = (IMAGE_SIZE_CIFAR10, IMAGE_SIZE_CIFAR10, 3)


def lucid_model_factory(pb_model_path=None,
                        model_image_shape=IMAGE_SHAPE,
                        model_input_name='dense_input',
                        model_output_name='dense_4/Softmax',
                        model_image_value_range=(0, 1)):
    """Build Lucid model object."""

    if pb_model_path is None:
        _, pb_model_path = mkstemp(suffix='.pb')

    # Model.suggest_save_args()

    # Save tf.keras model in pb format
    # https://www.tensorflow.org/guide/saved_model
    Model.save(
        pb_model_path,
        image_shape=model_image_shape,
        input_name=model_input_name,
        output_names=[model_output_name],
        image_value_range=model_image_value_range)

    class MyLucidModel(Model):
        model_path = pb_model_path
        # labels_path = './lucid/mnist.txt'
        # synsets_path = 'gs://modelzoo/labels/ImageNet_standard_synsets.txt'
        # dataset = 'ImageNet'
        image_shape = model_image_shape
        # is_BGR = True
        image_value_range = model_image_value_range
        input_name = model_input_name

    lucid_model = MyLucidModel()
    lucid_model.load_graphdef()

    return lucid_model


def print_model_nodes(lucid_model):
    graph_def = tf.GraphDef()
    with open(lucid_model.model_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
        for node in graph_def.node:
            print(node.name)


def render_vis_with_loss(model, objective_f, size, optimizer=None,
                         transforms=[], thresholds=(256,), print_objectives=None,
                         relu_gradient_override=True):

    param_f = param.image(size)
    images = []
    losses = []

    with param_f.graph.as_default() as graph, tf.Session() as sess:

        T = render.make_vis_T(model, objective_f, param_f=param_f, optimizer=optimizer,
                              transforms=transforms, relu_gradient_override=relu_gradient_override)
        print_objective_func = render.make_print_objective_func(print_objectives, T)
        loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")
        tf.global_variables_initializer().run()

        for i in range(max(thresholds)+1):
            loss_, _ = sess.run([loss, vis_op])
            if i in thresholds:
                vis = t_image.eval()
                images.append(vis)
                losses.append(loss_)
                # if display:
                #     print(f'loss: {loss_}')
                #     print_objective_func(sess)
                #     show(vis)

    tf.compat.v1.reset_default_graph()

    return images[-1], losses[-1]


# @lru_cache(maxsize=None)
# def renderer(model, objective_f, threshold=512, size=IMAGE_SIZE,
#              learning_rate=0.001, obj_l1=0, obj_tv=0, obj_blur=0,
#              transform_jitter=0, transform_scale=1, transform_rotate=0,
#              param_fft=False, param_decorrelate=False):
#
#     optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
#
#     obj = objectives.as_objective(objective_f)
#     obj += obj_l1 * objectives.L1(constant=.5)
#     obj += obj_tv * objectives.total_variation()
#     obj += obj_blur * objectives.blur_input_each_step()
#
#     # Preconditioning:
#     # `fft` parameter controls spatial decorrelation
#     # `decorrelate` parameter controls channel decorrelation
#
#     # This is not like the usual behavior of lucid
#     # which has a bunch of default transforms.
#     # However, some of them doesn't work with dense layers
#     transforms = []
#
#     if transform_jitter != 0:
#         transforms.append(transform.pad(2 * transform_jitter))
#         transforms.append(transform.jitter(transform_jitter))
#
#     if transform_scale != 1:
#         transforms.append(transform.random_scale([transform_scale ** (n / 10.) for n in range(-10, 11)]))
#
#     if transform_rotate != 0:
#         transforms.append(transform.random_rotate(range(-transform_rotate, transform_rotate + 1)))
#
#     transforms.append(transform.crop_or_pad_to(size, size))
#
#     param_f = lambda: param.image(size, fft=param_fft, decorrelate=param_decorrelate)
#
#     img, loss = render.render_vis_with_loss(model, obj,
#                                             param_f=param_f,
#                                             optimizer=optimizer,
#                                             transforms=transforms,
#                                             thresholds=(threshold,))
#     return img, loss
#
#
# def render_sum_neurons(lucid_model, layer_name, neurons, zoom_factor=3, **kwargs):
#
#     if kwargs is not None:
#         kwargs = {'learning_rate': 0.1,
#                   'transform_jitter': 1,
#                   'param_fft': True}
#
#     objective_f = sum(objectives.as_objective(f'{layer_name}:{neuron}')
#                       for neuron in neurons)
#
#     img, loss = renderer(lucid_model, objective_f, **kwargs)
#
#     print(f'loss={loss}')
#     show(zoom(img, (1, zoom_factor, zoom_factor, 1)))


def make_lucid_dataset(model_tag, lucid_net, all_labels, is_unpruned, transforms=[],
                       n_random=9, min_size=5, max_prop=0.8, display=True,
                       savedir='/project/nn_clustering/datasets/', savetag=''):

    if 'cnn' in model_tag.lower():
        cnn_params = CNN_VGG_MODEL_PARAMS if 'vgg' in str(model_tag).lower() else CNN_MODEL_PARAMS
        layer_sizes = [cl['filters'] for cl in cnn_params['conv']]
        layer_names = ['conv2d/Relu'] + [f'conv2d_{i}/Relu' for i in range(1, len(layer_sizes))]
    else:  # it's an mlp
        layer_sizes = [256, 256, 256, 256]
        layer_names = ['dense/Relu'] + [f'dense_{i}/Relu' for i in range(1, len(layer_sizes))]
    if not is_unpruned:
        layer_names = ['prune_low_magnitude_' + ln for ln in layer_names]

    labels_in_layers = [np.array(lyr_labels) for lyr_labels in list(splitter(all_labels, layer_sizes))]

    max_images = []  # to be filled with images that maximize cluster activations
    random_max_images = []  # to be filled with images that maximize random units activations
    max_losses = []  # to be filled with losses
    random_max_losses = []  # to be filled with losses
    sm_sizes = []  # list of submodule sizes
    sm_layer_sizes = []
    sm_layers = []  # list of layer names
    sm_clusters = []  # list of clusters

    imsize = IMAGE_SIZE_CIFAR10 if 'vgg' in model_tag.lower() else IMAGE_SIZE

    for layer_name, labels, layer_size in zip(layer_names, labels_in_layers, layer_sizes):

        max_size = max_prop * layer_size

        for clust_i in range(max(all_labels)+1):

            sm_binary = labels == clust_i
            sm_size = sum(sm_binary)
            if sm_size <= min_size or sm_size >= max_size:  # skip if too big or small
                continue

            sm_sizes.append(sm_size)
            sm_layer_sizes.append(layer_size)
            sm_layers.append(layer_name)
            sm_clusters.append(clust_i)

            # print(f'{model_tag}, layer: {layer_name}')
            # print(f'submodule_size: {sm_size}, layer_size: {layer_size}')

            sm_idxs = [i for i in range(layer_size) if sm_binary[i]]
            max_obj = sum([objectives.channel(layer_name, unit) for unit in sm_idxs])

            max_im, max_loss = render_vis_with_loss(lucid_net, max_obj, size=imsize, transforms=transforms)
            max_images.append(max_im)
            max_losses.append(max_loss)
            if display:
                print(f'loss: {round(max_loss, 3)}')
                show(max_im)

            rdm_losses = []
            rdm_ims = []
            for _ in range(n_random):  # random max results
                rdm_idxs = np.random.choice(np.array(range(layer_size)), size=sm_size, replace=False)
                random_max_obj = sum([objectives.channel(layer_name, unit) for unit in rdm_idxs])
                random_max_im, random_max_loss = render_vis_with_loss(lucid_net, random_max_obj,
                                                                      size=imsize, transforms=transforms)
                random_max_images.append(random_max_im)
                random_max_losses.append(random_max_loss)
                rdm_ims.append(np.squeeze(random_max_im))
                rdm_losses.append(round(random_max_loss, 3))
            if display:
                print(f'random losses: {rdm_losses}')
                show(np.hstack(rdm_ims))

    max_images = np.squeeze(np.array(max_images))
    random_max_images = np.squeeze(np.array(random_max_images))
    max_losses = np.array(max_losses)
    random_max_losses = np.array(random_max_losses)

    results = {'max_images': max_images,
               'random_max_images': random_max_images,
               'max_losses': max_losses,
               'random_max_losses': random_max_losses,
               'sm_sizes': sm_sizes, 'sm_layer_sizes': sm_layer_sizes,
               'sm_layers': sm_layers, 'sm_clusters': sm_clusters}

    if is_unpruned:
        suff = '_unpruned_max_data'
    else:
        suff = '_pruned_max_data'

    with open(savedir + model_tag + suff + savetag + '.pkl', 'wb') as f:
        pickle.dump(results, f)


def evaluate_visualizations(model_tag, rep, is_unpruned, data_dir='/project/nn_clustering/datasets/'):

    if is_unpruned:
        suff = f'{rep}_unpruned_max_data.pkl'
    else:
        suff = f'{rep}_pruned_max_data.pkl'

    with open(data_dir + model_tag + suff, 'rb') as f:
        data = pickle.load(f)

    # unpack data
    max_images = data['max_images']
    random_max_images = data['random_max_images']
    max_losses = data['max_losses']
    random_max_losses = data['random_max_losses']
    sm_sizes = data['sm_sizes']
    sm_layers = data['sm_layers']
    sm_layer_sizes = data['sm_layer_sizes']
    sm_clusters = data['sm_clusters']
    n_examples = len(sm_sizes)
    n_max_min = int(len(max_images) / n_examples)
    n_random = int(len(random_max_images) / n_examples)
    input_side = max_images.shape[1]

    # flatten all inputs if mlp
    if 'mlp' in model_tag.lower():
        max_images = np.reshape(max_images, [-1, IMAGE_SIZE**2])
        random_max_images = np.reshape(random_max_images, [-1, IMAGE_SIZE**2])

    # get model
    model_dir = get_model_path(model_tag, filter_='all')[rep]
    model_path = get_model_paths(model_dir)[is_unpruned]
    model = load_model2(model_path)

    # get predictions
    max_preds = model.predict(max_images)
    random_max_preds = np.reshape(model.predict(random_max_images), (n_examples, n_random, -1))

    # get entropies
    max_entropies = np.array([entropy(pred) for pred in max_preds])
    random_max_entropies = np.array([[entropy(pred) for pred in reps] for reps in random_max_preds])

    # reshape losses
    random_max_losses = np.reshape(random_max_losses, (n_examples, n_random))

    # get percentiles
    max_percentiles_entropy = np.array([compute_pvalue(max_entropies[i], random_max_entropies[i])
                                        for i in range(len(max_entropies))])
    max_percentiles_loss = np.array([compute_pvalue(max_losses[i], random_max_losses[i], side='right')
                                     for i in range(len(max_losses))])

    # get effect sizes
    effect_factors_entropies = np.array([np.mean(random_max_entropies[i]) / max_entropies[i]
                                         for i in range(len(max_entropies)) if max_entropies[i] > 0])
    mean_effect_factor_entropy = np.nanmean(effect_factors_entropies)
    effect_factors_losses = np.array([np.mean(random_max_losses[i]) / max_losses[i]
                                      for i in range(len(max_losses)) if max_losses[i] > 0])
    mean_effect_factor_loss = np.nanmean(effect_factors_losses)

    # get pvalues
    max_chi2_p_entropy = chi2_categorical_test(max_percentiles_entropy, n_random)
    max_combined_p_entropy = combine_ps(max_percentiles_entropy, n_random)
    max_chi2_p_loss = chi2_categorical_test(max_percentiles_loss, n_random)
    max_combined_p_loss = combine_ps(max_percentiles_loss, n_random)

    results = {'percentiles': (max_percentiles_entropy,  # min_percentiles_entropy,
                               max_percentiles_loss),  # min_percentiles_loss),
               'effect_factors': (mean_effect_factor_entropy,
                                  mean_effect_factor_loss),
               'chi2_ps': (max_chi2_p_entropy,  # min_chi2_categorical_p_entropy,
                             max_chi2_p_loss),  # min_chi2_categorical_p_loss),
               'combined_ps': (max_combined_p_entropy,
                            max_combined_p_loss),
               'sm_layers': sm_layers, 'sm_sizes': sm_sizes,
               'sm_layer_sizes': sm_layer_sizes, 'sm_clusters': sm_clusters}

    return results


###################################################################################
# ImageNet
###################################################################################

IMAGE_SIZE_IMAGENET = 224

VIS_NETS = ['vgg16', 'vgg19', 'resnet50']

VGG16_LAYER_MAP = {'block1_conv1': 'conv1_1/conv1_1', 'block1_conv2': 'conv1_2/conv1_2',
                   'block2_conv1': 'conv2_1/conv2_1', 'block2_conv2': 'conv2_2/conv2_2',
                   'block3_conv1': 'conv3_1/conv3_1', 'block3_conv2': 'conv3_2/conv3_2',
                   'block3_conv3': 'conv3_3/conv3_3', 'block4_conv1': 'conv4_1/conv4_1',
                   'block4_conv2': 'conv4_2/conv4_2', 'block4_conv3': 'conv4_3/conv4_3',
                   'block5_conv1': 'conv5_1/conv5_1', 'block5_conv2': 'conv5_2/conv5_2',
                   'block5_conv3': 'conv5_3/conv5_3'}

VGG19_LAYER_MAP = {'block1_conv1': 'conv1_1/conv1_1', 'block1_conv2': 'conv1_2/conv1_2',
                   'block2_conv1': 'conv2_1/conv2_1', 'block2_conv2': 'conv2_2/conv2_2',
                   'block3_conv1': 'conv3_1/conv3_1', 'block3_conv2': 'conv3_2/conv3_2',
                   'block3_conv3': 'conv3_3/conv3_3', 'block3_conv4': 'conv3_4/conv3_4',
                   'block4_conv1': 'conv4_1/conv4_1', 'block4_conv2': 'conv4_2/conv4_2',
                   'block4_conv3': 'conv4_3/conv4_3', 'block4_conv4': 'conv4_4/conv4_4',
                   'block5_conv1': 'conv5_1/conv5_1', 'block5_conv2': 'conv5_2/conv5_2',
                   'block5_conv3': 'conv5_3/conv5_3', 'block5_conv4': 'conv5_4/conv5_4'}

RESNET50_LAYER_MAP = {'conv0': 'resnet_v1_50/conv1/Relu',
                      'stage1_unit1_sc': 'resnet_v1_50/block1/unit_1/bottleneck_v1/Relu',
                      'stage1_unit2_conv3': 'resnet_v1_50/block1/unit_2/bottleneck_v1/Relu',
                      'stage1_unit3_conv3': 'resnet_v1_50/block1/unit_3/bottleneck_v1/Relu',
                      'stage2_unit1_sc': 'resnet_v1_50/block2/unit_1/bottleneck_v1/Relu',
                      'stage2_unit2_conv3': 'resnet_v1_50/block2/unit_2/bottleneck_v1/Relu',
                      'stage2_unit3_conv3': 'resnet_v1_50/block2/unit_3/bottleneck_v1/Relu',
                      'stage2_unit4_conv3': 'resnet_v1_50/block2/unit_4/bottleneck_v1/Relu',
                      'stage3_unit1_sc': 'resnet_v1_50/block3/unit_1/bottleneck_v1/Relu',
                      'stage3_unit2_conv3': 'resnet_v1_50/block3/unit_2/bottleneck_v1/Relu',
                      'stage3_unit3_conv3': 'resnet_v1_50/block3/unit_3/bottleneck_v1/Relu',
                      'stage3_unit4_conv3': 'resnet_v1_50/block3/unit_4/bottleneck_v1/Relu',
                      'stage3_unit5_conv3': 'resnet_v1_50/block3/unit_5/bottleneck_v1/Relu',
                      'stage3_unit6_conv3': 'resnet_v1_50/block3/unit_6/bottleneck_v1/Relu',
                      'stage4_unit1_sc': 'resnet_v1_50/block4/unit_1/bottleneck_v1/Relu',
                      'stage4_unit2_conv3': 'resnet_v1_50/block4/unit_2/bottleneck_v1/Relu',
                      'stage4_unit3_conv3': 'resnet_v1_50/block4/unit_3/bottleneck_v1/Relu'}

NETWORK_LAYER_MAP = {'vgg16': VGG16_LAYER_MAP, 'vgg19': VGG19_LAYER_MAP, 'resnet50': RESNET50_LAYER_MAP}


def get_clustering_info_imagenet(model_tag, num_clusters, savedir='/project/nn_clustering/results/'):

    assert model_tag in VIS_NETS

    clustering_results = run_clustering_imagenet(model_tag, num_clusters=num_clusters,
                                                 with_shuffle=False, eigen_solver='arpack')

    layer_names = clustering_results['layer_names']
    conv_connections = clustering_results['conv_connections']
    layer_sizes = [cc[0]['weights'].shape[0] for cc in conv_connections[1:]]
    dense_sizes = get_dense_sizes(conv_connections)
    layer_sizes.extend(list(dense_sizes.values()))
    labels = clustering_results['labels']
    labels_in_layers = list(splitter(labels, layer_sizes))

    for nm, ly in zip(layer_names, layer_sizes):
        print(ly, nm)

    clustering_info = {'layers': layer_names, 'labels': labels_in_layers}

    with open(savedir + model_tag + '_clustering_info.pkl', 'wb') as f:
        pickle.dump(clustering_info, f)


def make_lucid_imagenet_dataset(model_tag, n_random=9,
                                min_size=3, max_prop=0.8, display=True,
                                infodir='/project/nn_clustering/results/',
                                savedir='/project/nn_clustering/datasets/'):

    assert model_tag in VIS_NETS

    with open(infodir + model_tag + '_clustering_info.pkl', 'rb') as f:
        clustering_info = pickle.load(f)

    layer_names = clustering_info['layers']
    labels_in_layers = [np.array(lyr_labels) for lyr_labels in clustering_info['labels']]
    layer_sizes = [len(labels) for labels in labels_in_layers]
    n_clusters = max([max(labels) for labels in labels_in_layers]) + 1

    if model_tag == 'vgg16':
        lucid_net = models.VGG16_caffe()
    elif model_tag == 'vgg19':
        lucid_net = models.VGG19_caffe()
    else:
        lucid_net = models.ResnetV1_50_slim()
    lucid_net.load_graphdef()
    layer_map = NETWORK_LAYER_MAP[model_tag]

    max_images = []  # to be filled with images that maximize cluster activations
    # min_images = []  # to be filled with images that minimize cluster activations
    random_max_images = []  # to be filled with images that maximize random units activations
    # random_min_images = []  # to be filled with images that minimize random units activations
    max_losses = []  # to be filled with losses
    # min_losses = []  # to be filled with losses
    random_max_losses = []  # to be filled with losses
    # random_min_losses = []  # to be filled with losses
    sm_sizes = []  # list of submodule sizes
    sm_layer_sizes = []
    sm_layers = []  # list of layer names
    sm_clusters = []  # list of clusters

    for layer_name, labels, layer_size in zip(layer_names, labels_in_layers, layer_sizes):

        if layer_name not in layer_map.keys():
            continue

        lucid_name = layer_map[layer_name]
        max_size = max_prop * layer_size

        for clust_i in range(n_clusters):

            sm_binary = labels == clust_i
            sm_size = sum(sm_binary)
            if sm_size <= min_size or sm_size >= max_size:  # skip if too big or small
                continue

            sm_sizes.append(sm_size)
            sm_layer_sizes.append(layer_size)
            sm_layers.append(layer_name)
            sm_clusters.append(clust_i)

            print(f'{model_tag}, layer names: {layer_name}, {lucid_name}')
            print(f'submodule_size: {sm_size}, layer_size: {layer_size}')

            sm_idxs = [i for i in range(layer_size) if sm_binary[i]]
            max_obj = sum([objectives.channel(lucid_name, unit) for unit in sm_idxs])
            # min_obj = -1 * sum([objectives.channel(lucid_name, unit) for unit in sm_idxs])

            max_im, max_loss = render_vis_with_loss(lucid_net, max_obj, size=IMAGE_SIZE_IMAGENET, thresholds=(256,))
            max_images.append(max_im)
            max_losses.append(max_loss)
            # min_im, min_loss = render_vis_with_loss(lucid_net, min_obj)
            # min_images.append(min_im)
            # min_losses.append(min_loss)
            if display:
                print(f'loss: {round(max_loss, 3)}')
                show(max_im)

            rdm_losses = []
            rdm_ims = []
            for _ in range(n_random):  # random max results
                rdm_idxs = np.random.choice(np.array(range(layer_size)), size=sm_size, replace=False)
                random_max_obj = sum([objectives.channel(lucid_name, unit) for unit in rdm_idxs])
                random_max_im, random_max_loss = render_vis_with_loss(lucid_net, random_max_obj,
                                                                      size=IMAGE_SIZE_IMAGENET,
                                                                      thresholds=(256,))
                random_max_images.append(random_max_im)
                random_max_losses.append(random_max_loss)
                rdm_losses.append(round(random_max_loss, 3))
                rdm_ims.append(np.squeeze(random_max_im))
            if display:
                print(f'random losses: {rdm_losses}')
                show(np.hstack(rdm_ims))

            # for _ in range(n_random):  # random min results
            #     rdm_idxs = np.random.choice(np.array(range(layer_size)), size=sm_size, replace=False)
            #     random_min_obj = -1 * sum([objectives.channel(lucid_name, unit) for unit in rdm_idxs])
            #     random_min_im, random_min_loss = render_vis_with_loss(lucid_net, random_min_obj)
            #     random_min_images.append(random_min_im)
            #     random_min_losses.append(random_min_loss)

    max_images = np.squeeze(np.array(max_images))
    # min_images = np.squeeze(np.array(min_images))
    random_max_images = np.squeeze(np.array(random_max_images))
    # random_min_images = np.squeeze(np.array(random_min_images))
    max_losses = np.array(max_losses)
    # min_losses = np.array(min_losses)
    random_max_losses = np.array(random_max_losses)
    # random_min_losses = np.array(random_min_losses)

    results = {'max_images': max_images,  # 'min_images': min_images,
               'random_max_images': random_max_images,  # 'random_min_images': random_min_images,
               'max_losses': max_losses, # 'min_losses': min_losses,
               'random_max_losses': random_max_losses,  # 'random_min_losses': random_min_losses,
               'sm_sizes': sm_sizes, 'sm_layer_sizes': sm_layer_sizes,
               'sm_layers': sm_layers, 'sm_clusters': sm_clusters}

    with open(savedir + model_tag + '_max_data.pkl', 'wb') as f:
        pickle.dump(results, f)


def evaluate_imagenet_visualizations(model_tag, data_dir='/project/nn_clustering/datasets/'):

    assert model_tag in VIS_NETS

    with open(data_dir + model_tag + '_max_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # unpack data
    max_images = data['max_images']
    # min_images = data['min_images']
    random_max_images = data['random_max_images']
    # random_min_images = data['random_min_images']
    max_losses = data['max_losses']
    # min_losses = data['min_losses']
    random_max_losses = data['random_max_losses']
    # random_min_losses = data['random_min_losses']
    sm_sizes = data['sm_sizes']
    sm_layers = data['sm_layers']
    sm_layer_sizes = data['sm_layer_sizes']
    sm_clusters = data['sm_clusters']
    n_examples = len(sm_sizes)
    n_random = int(len(random_max_images) / n_examples)
    input_side = max_images.shape[1]

    # get model
    net, preprocess = Classifiers.get(model_tag)  # get network object and preprocess fn
    model = net((input_side, input_side, 3), weights='imagenet')  # get network tf.keras.model

    # get predictions
    max_preds = model.predict(max_images)
    # min_preds = model.predict(min_images)
    random_max_preds = np.reshape(model.predict(random_max_images), (n_examples, n_random, -1))
    # random_min_preds = np.reshape(model.predict(random_min_images), (n_examples, n_random, -1))

    # get entropies
    max_entropies = np.array([entropy(pred) for pred in max_preds])
    # min_entropies = np.array([entropy(pred) for pred in min_preds])
    random_max_entropies = np.array([[entropy(pred) for pred in reps] for reps in random_max_preds])
    # random_min_entropies = np.array([[entropy(pred) for pred in reps] for reps in random_min_preds])

    # reshape losses
    random_max_losses = np.reshape(random_max_losses, (n_examples, n_random))
    # random_min_losses = np.reshape(random_min_losses, (n_examples, n_random))

    # get percentiles
    max_percentiles_entropy = np.array([compute_pvalue(max_entropies[i], random_max_entropies[i])
                                        for i in range(len(max_entropies))])
    # min_percentiles_entropy = np.array([compute_pvalue(min_entropies[i], random_min_entropies[i])
    #                             for i in range(len(min_entropies))])
    max_percentiles_loss = np.array([compute_pvalue(max_losses[i], random_max_losses[i], side='right')
                                     for i in range(len(max_losses))])
    # min_percentiles_loss = np.array([compute_pvalue(min_losses[i], random_min_losses[i])
    #                             for i in range(len(min_losses))])

    # get effect sizes
    effect_factor_entropy = np.mean(np.array([np.mean(random_max_entropies[i]) / max_entropies[i]
                                              for i in range(len(max_entropies))]))
    effect_factor_loss = np.mean(np.array([np.mean(random_max_losses[i]) / max_losses[i]
                                           for i in range(len(max_losses))]))

    # get pvalues
    max_chi2_p_entropy = chi2_categorical_test(max_percentiles_entropy, n_random)
    max_combined_p_entropy = combine_ps(max_percentiles_entropy, n_random)
    max_chi2_p_loss = chi2_categorical_test(max_percentiles_loss, n_random)
    max_combined_p_loss = combine_ps(max_percentiles_loss, n_random)

    results = {'percentiles': (max_percentiles_entropy,
                               max_percentiles_loss),
               'effect_factors': (effect_factor_entropy,
                                  effect_factor_loss),
               'chi2_ps': (max_chi2_p_entropy,
                             max_chi2_p_loss),
               'combined_ps': (max_combined_p_entropy,
                            max_combined_p_loss),
               'sm_layers': sm_layers, 'sm_sizes': sm_sizes,
               'sm_layer_sizes': sm_layer_sizes, 'sm_clusters': sm_clusters}

    return results
