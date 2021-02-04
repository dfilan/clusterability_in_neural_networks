#!/usr/bin/env bash

echo "######## Generating Datasets... ######## "

mkdir -p datasets

python -m src.generate_datasets -s mnist datasets/mnist.pckl
python -m src.generate_datasets -s cifar10 datasets/cifar10.pckl
python -m src.generate_datasets -s fashion datasets/fashion.pckl

python -m src.generate_datasets -s halves_same_mnist datasets/halves_same_mnist.pckl
python -m src.generate_datasets -s halves_same_cifar10 datasets/halves_same_cifar10.pckl
python -m src.generate_datasets -s halves_same_fashion datasets/halves_same_fashion.pckl

python -m src.generate_datasets -s halves_mnist datasets/halves_mnist.pckl
python -m src.generate_datasets -s halves_cifar10 datasets/halves_cifar10.pckl
python -m src.generate_datasets -s halves_fashion datasets/halves_fashion.pckl

python -m src.generate_datasets -s stacked_same_mnist datasets/stacked_same_mnist.pckl
python -m src.generate_datasets -s stacked_same_cifar10 datasets/stacked_same_cifar10.pckl
python -m src.generate_datasets -s stacked_same_fashion datasets/stacked_same_fashion.pckl

python -m src.generate_datasets -s stacked_mnist datasets/stacked_mnist.pckl
python -m src.generate_datasets -s stacked_cifar10 datasets/stacked_cifar10.pckl
python -m src.generate_datasets -s stacked_fashion datasets/stacked_fashion.pckl

python -m src.generate_datasets -s poly datasets/poly.pckl
# python -m src.generate_datasets -s small_mnist -w 7 -ht 7 datasets/small_mnist.pckl
python -m src.generate_datasets -s cifar10_full datasets/cifar10_full.pckl
