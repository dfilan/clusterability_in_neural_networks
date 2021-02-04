"""Script for generating MNIST, MNIST-Fashion, CIFAR-10 and LINE dataests."""

import pickle
import os
import itertools
from pathlib import Path
import functools
import numpy as np
from skimage import draw
import cv2 as cv
from skimage.color import rgb2gray
from skimage.transform import resize
from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import tfrecords_writer, tfrecords_reader
from tensorflow_datasets.core import utils as tfds_utils
from tensorflow_datasets.core import splits as splits_lib
from tensorflow_datasets.core import utils as tfds_utils


def resize_grayscale(imgs_in, width, height, in_color=True):
    '''used to be called "preprocess_batch_cifar10"'''
    imgs_out = np.zeros([imgs_in.shape[0], width, height])

    for n in range(len(imgs_in)):
        if in_color:
            imgs_out[n,:,:] = rgb2gray(resize(imgs_in[n,:,:], [width, height], anti_aliasing=True))
        else:
            imgs_out[n,:,:] = resize(imgs_in[n,:,:], [width, height], anti_aliasing=True)

    imgs_out *= 255

    return imgs_out


def line_counting_data_gen(batch_size, n_possible_lines=10, width=28, height=28):
    while True:
        inputs = []
        labels = []

        for _ in range(batch_size):
            img = np.zeros((height, width))

            n_lines = np.random.randint(1, n_possible_lines + 1)
            lines = set()

            while len(lines) < n_lines:
                orientation = 'vertical' # if np.random.randint(0, 2) else 'horizontal'
                
                if orientation == 'vertical':
                    y = np.random.randint(0, height)
                    line = (0, y, width-1, y)
                elif orientation == 'horizontal':
                    x = np.random.randint(0, width)
                    line = (x, 0, x, height-1)

                if line not in lines:
                    rr, cc = draw.line(*line)
                    img[rr, cc] = 255
                    lines.add(line)
        
            inputs.append(img)
            labels.append(n_lines - 1)

        yield np.array(inputs), np.array(labels)
        # yield np.expand_dims(np.stack(inputs), axis=-1), np.stack(labels)


def circle_counting_data_gen(batch_size, n_possible_circles=10, width=28, height=28, radius=4):
    while True:
        inputs = []
        labels = []

        for _ in range(batch_size):
            img = np.zeros((height, width))

            n_circles = np.random.randint(0, n_possible_circles)
            centers = set()

            while len(centers) < n_circles:
                
                center_x = np.random.randint(0, width)
                center_y = np.random.randint(0, height)
            
                if (center_x, center_y) not in centers:
                    rr, cc = draw.circle_perimeter(center_x, center_y, radius=radius, shape=img.shape)
                    img[rr, cc] = 255
                    centers.add((center_x, center_y))
        
            inputs.append(img)
            labels.append(n_circles - 1)

        yield np.array(inputs), np.array(labels)
        # yield np.expand_dims(np.stack(inputs), axis=-1), np.stack(labels)


def get_resized_grayscale_cifar10(width, height):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = resize_grayscale(X_train, width, height, in_color=True)
    y_train = y_train[:, 0]
    X_test = resize_grayscale(X_test, width, height, in_color=True)
    y_test = y_test[:, 0]
    return (X_train, y_train), (X_test, y_test)


def halves_data_gen(X_train, y_train, X_test, y_test,
                    n_train, n_test, width, height):

    X_train_halves = np.zeros_like(X_train)
    X_test_halves = np.zeros_like(X_test)
    y_train_halves = np.zeros_like(y_train)
    y_test_halves = np.zeros_like(y_test)

    len_train = y_train.shape[0]
    len_test = y_test.shape[0]
    X_train = resize(X_train, (len_train, height, width/2))
    X_train = np.round(X_train * 255 / np.max(X_train))
    X_test = resize(X_test, (len_test, height, width/2))
    X_test = np.round(X_test * 255 / np.max(X_test))

    for train_i in range(n_train):
        rdm_i = np.random.randint(0, len_train, size=2)
        X_select = X_train[rdm_i]
        y_select = y_train[rdm_i]
        X_example = np.hstack((X_select[0], X_select[1]))
        y_example = np.sum(y_select) % 10
        X_train_halves[train_i] = X_example
        y_train_halves[train_i] = y_example

    for test_i in range(n_test):
        rdm_i = np.random.randint(0, len_test, size=2)
        X_select = X_test[rdm_i]
        y_select = y_test[rdm_i]
        X_example = np.hstack((X_select[0], X_select[1]))
        y_example = np.sum(y_select) % 10
        X_test_halves[test_i] = X_example
        y_test_halves[test_i] = y_example

    return (X_train_halves, y_train_halves), (X_test_halves, y_test_halves)


def halves_same_data_gen(X_train, y_train, X_test, y_test,
                    n_train, n_test, width, height, n_classes=10):

    X_train = resize(X_train, (n_train, height, width / 2))
    X_train = np.round(X_train * 255 / np.max(X_train))
    X_test = resize(X_test, (n_test, height, width / 2))
    X_test = np.round(X_test * 255 / np.max(X_test))

    X_train_classes = [X_train[y_train == c] for c in range(n_classes)]
    X_test_classes = [X_test[y_test == c] for c in range(n_classes)]
    n_train_class = min([len(x_class) for x_class in X_train_classes])
    n_test_class = min([len(x_class) for x_class in X_test_classes])

    X_train_halves = np.zeros((n_train, height, width))
    X_test_halves = np.zeros((n_test, height, width))

    for train_i in range(n_train):
        label = y_train[train_i]
        rdm_i = np.random.randint(0, n_train_class)
        X_train_halves[train_i] = np.hstack([X_train[train_i], X_train_classes[label][rdm_i]])

    for test_i in range(n_test):
        label = y_test[test_i]
        rdm_i = np.random.randint(0, n_test_class)
        X_test_halves[test_i] = np.hstack([X_test[test_i], X_test_classes[label][rdm_i]])

    return (X_train_halves, y_train), (X_test_halves, y_test)


def stacked_data_gen(X_train, y_train, X_test, y_test,
                     n_train, n_test, width, height, stacks=2, n_classes=10):

    X_train_stacked = np.zeros([n_train, stacks, width, height])
    X_test_stacked = np.zeros([n_test, stacks, width, height])
    y_train_stacked = np.zeros(n_train)
    y_test_stacked = np.zeros(n_test)

    for train_i in range(n_train):
        rdm_i = np.random.randint(0, n_train, size=stacks)
        X_example = X_train[rdm_i]
        y_labels = y_train[rdm_i]
        y_example = np.sum(y_labels) % n_classes
        X_train_stacked[train_i] = X_example
        y_train_stacked[train_i] = y_example

    for test_i in range(n_test):
        rdm_i = np.random.randint(0, n_test, size=stacks)
        X_example = X_test[rdm_i]
        y_labels = y_test[rdm_i]
        y_example = np.sum(y_labels) % n_classes
        X_test_stacked[test_i] = X_example
        y_test_stacked[test_i] = y_example

    return (X_train_stacked, y_train_stacked), (X_test_stacked, y_test_stacked)


def stacked_same_data_gen(X_train, y_train, X_test, y_test,
                          n_train, n_test, width, height, stacks=2, n_classes=10):

    X_train_classes = [X_train[y_train == c] for c in range(n_classes)]
    X_test_classes = [X_test[y_test == c] for c in range(n_classes)]
    n_train_class = min([len(x_class) for x_class in X_train_classes])
    n_test_class = min([len(x_class) for x_class in X_test_classes])

    X_train_stacked = np.zeros([n_train, stacks, width, height])
    X_test_stacked = np.zeros([n_test, stacks, width, height])

    for train_i in range(n_train):
        label = y_train[train_i]
        rdm_i = np.random.randint(0, n_train_class, size=stacks-1)
        example = np.concatenate([np.expand_dims(X_train[train_i], axis=0), X_train_classes[label][rdm_i]])
        X_train_stacked[train_i] = example

    for test_i in range(n_test):
        label = y_test[test_i]
        rdm_i = np.random.randint(0, n_test_class, size=stacks-1)
        example = np.concatenate([np.expand_dims(X_test[test_i], axis=0), X_test_classes[label][rdm_i]])
        X_test_stacked[test_i] = example

    return (X_train_stacked, y_train), (X_test_stacked, y_test)


def generate_random_datast(n_train, n_test, width, height):
    X_train = np.random.randint(0, 256, (60000, 28, 28))
    y_train = np.random.randint(0, 10, 60000)

    X_test = np.random.randint(0, 256, (10000, 28, 28))
    y_test = np.random.randint(0, 10, 10000)

    return (X_train, y_train), (X_test, y_test)


def polynomial_regression_dataset(n_train, n_test, n_inputs=2, coefs=(0, 1), exps=(0, 1, 2)):

    n_terms = len(exps) ** n_inputs
    n_outputs = len(coefs) ** n_terms

    np.random.seed(42)
    X_train_poly = np.random.normal(size=(n_train, n_inputs))
    X_test_poly = np.random.normal(size=(n_test, n_inputs))

    y_train_poly = np.zeros((n_train, n_outputs))
    y_test_poly = np.zeros((n_test, n_outputs))

    poly_coefs = np.zeros((n_outputs, n_terms))

    for poly_i, coef_list in enumerate(itertools.product(coefs, repeat=n_terms)):

        poly_coefs[poly_i] = np.array(coef_list)

        poly_fn = lambda inpt: sum([coef_list[term_i] *
                                    np.prod(np.array([inpt[inpt_i]**exs[inpt_i] for inpt_i in range(n_inputs)]))
                                    for term_i, exs in enumerate(itertools.product(exps, repeat=n_inputs))])

        for train_i in range(n_train):
            y_train_poly[train_i, poly_i] = poly_fn(X_train_poly[train_i])
        for test_i in range(n_test):
            y_test_poly[test_i, poly_i] = poly_fn(X_test_poly[test_i])

    return (X_train_poly, y_train_poly), (X_test_poly, y_test_poly)


def prep_imagenet_validation_data(data_dir='/project/nn_clustering/datasets/imagenet2012',
                                  val_tar='ILSVRC2012_img_val.tar'):

    # prior to running this, execute:
    # mkdir datasets/imagenet2012
    # cd dstasets/imagenet2012
    # wget [the imagenet 2012 validation tar download link]

    val_path = os.path.join(data_dir, val_tar)

    if not os.path.exists(val_path):
        raise FileNotFoundError(f'{val_path} does not exist. Manually download ILSVRC2012_img_val.tar \
        into {data_dir} and try again.')

    imagenet = tfds.image.Imagenet2012()
    dl_manager = tfds.download.DownloadManager(download_dir=data_dir)
    arch = dl_manager.iter_archive(val_path)
    val_gen = tfds.core.SplitGenerator(name=tfds.Split.VALIDATION,
                                       gen_kwargs={'archive': arch,
                                                   'validation_labels': imagenet._get_validation_labels(val_path)})
    validation_labels = imagenet._get_validation_labels(val_path)
    main_gen = imagenet._generate_examples_validation(archive=arch, labels=validation_labels)
    fname = "{}-{}.tfrecord".format('imagenet2012', val_gen.name)
    fpath = os.path.join(data_dir, fname)
    writer = tfrecords_writer.Writer(imagenet._example_specs, fpath,
                                     hash_salt=val_gen.name)
    for key, record in tfds_utils.tqdm(main_gen, unit=" examples",
                                       total=val_gen.split_info.num_examples, leave=False):
        example = imagenet.info.features.encode_example(record)
        writer.write(key, example)
    _, _ = writer.finalize()


def main(path, n_train=60000, n_test=10000,
         width: ('Image width', 'option', 'w', int, None, None)=28,
         height: ('Image height', 'option', 'ht', int, None, None)=28,
         shape: ('Task shape', 'option', 's')='line',
         random_state: ('Random state', 'option', 'r')=42):
    
    if random_state is not None:
        np.random.seed(random_state)
    
    assert shape in ('line', 'circle', 'mnist', 'small_mnist', 'halves_mnist', 'halves_same_mnist',
                     'stacked_mnist', 'stacked_same_mnist', 'cifar10', 'halves_cifar10',
                     'halves_same_cifar10', 'stacked_cifar10', 'stacked_same_cifar10',
                     'cifar10_full', 'fashion', 'halves_fashion', 'halves_same_fashion',
                     'stacked_fashion', 'stacked_same_fashion', 'random', 'poly')
    
    if shape == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif shape == 'small_mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = resize_grayscale(X_train, width, height, in_color=False)
        X_test = resize_grayscale(X_test, width, height, in_color=False)

    elif shape == 'halves_mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        (X_train, y_train), (X_test, y_test) = halves_data_gen(X_train, y_train, X_test, y_test,
                                                               n_train, n_test, width, height)

    elif shape == 'halves_same_mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        (X_train, y_train), (X_test, y_test) = halves_same_data_gen(X_train, y_train, X_test, y_test,
                                                               n_train, n_test, width, height)

    elif shape == 'stacked_mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        (X_train, y_train), (X_test, y_test) = stacked_data_gen(X_train, y_train, X_test, y_test,
                                                                n_train, n_test, width, height)

    elif shape == 'stacked_same_mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        (X_train, y_train), (X_test, y_test) = stacked_same_data_gen(X_train, y_train, X_test, y_test,
                                                                     n_train, n_test, width, height)

    elif shape == 'cifar10':
        (X_train, y_train), (X_test, y_test) = get_resized_grayscale_cifar10(width, height)

    elif shape == 'halves_cifar10':
        (X_train, y_train), (X_test, y_test) = get_resized_grayscale_cifar10(width, height)
        (X_train, y_train), (X_test, y_test) = halves_data_gen(X_train, y_train, X_test, y_test,
                                                               50000, n_test, width, height)

    elif shape == 'halves_same_cifar10':
        (X_train, y_train), (X_test, y_test) = get_resized_grayscale_cifar10(width, height)
        (X_train, y_train), (X_test, y_test) = halves_same_data_gen(X_train, y_train, X_test, y_test,
                                                               50000, n_test, width, height)

    elif shape == 'stacked_cifar10':
        (X_train, y_train), (X_test, y_test) = get_resized_grayscale_cifar10(width, height)
        (X_train, y_train), (X_test, y_test) = stacked_data_gen(X_train, y_train, X_test, y_test,
                                                                50000, n_test, width, height)

    elif shape == 'stacked_same_cifar10':
        (X_train, y_train), (X_test, y_test) = get_resized_grayscale_cifar10(width, height)
        (X_train, y_train), (X_test, y_test) = stacked_same_data_gen(X_train, y_train, X_test, y_test,
                                                                     50000, n_test, width, height)

    elif shape == 'cifar10_full':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        y_train = y_train[:, 0]
        y_test = y_test[:, 0]

    elif shape == 'fashion':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    elif shape == 'halves_fashion':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        (X_train, y_train), (X_test, y_test) = halves_data_gen(X_train, y_train, X_test, y_test,
                                                               n_train, n_test, width, height)

    elif shape == 'halves_same_fashion':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        (X_train, y_train), (X_test, y_test) = halves_same_data_gen(X_train, y_train, X_test, y_test,
                                                               n_train, n_test, width, height)

    elif shape == 'stacked_fashion':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        (X_train, y_train), (X_test, y_test) = stacked_data_gen(X_train, y_train, X_test, y_test,
                                                                n_train, n_test, width, height)

    elif shape == 'stacked_same_fashion':
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        (X_train, y_train), (X_test, y_test) = stacked_same_data_gen(X_train, y_train, X_test, y_test,
                                                                     n_train, n_test, width, height)

    elif shape == 'random':
        ((X_train, y_train),
         (X_test, y_test)) = generate_random_datast(n_train, n_test, width, height)

    elif shape == 'poly':
        ((X_train, y_train), (X_test, y_test)) = polynomial_regression_dataset(n_train, n_test)

    else:
        gen = line_counting_data_gen if shape == 'line' else circle_counting_data_gen

        train_dataset = gen(batch_size=n_train, width=width, height=height)
        X_train, y_train = next(train_dataset)

        test_dataset = gen(batch_size=n_test, width=width, height=height)
        X_test, y_test= next(test_dataset)

    assert X_train.max() >= 250 or shape == 'poly'
    assert X_train.max() <= 255 or shape == 'poly'
    assert X_test.max() >= 250 or shape == 'poly'
    assert X_test.max() <= 255 or shape == 'poly'

    with open(path, 'wb') as f:
        pickle.dump({'X_train': X_train,
                     'y_train': y_train,
                     'X_test': X_test,
                     'y_test': y_test},
                     f)


if __name__ == '__main__':
    import plac; plac.call(main)
