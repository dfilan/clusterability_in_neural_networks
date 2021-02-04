#!/usr/bin/env bash

set -xe

if [[ "$1" = "datasets" ]]
then
    echo "######## Generating Datasets... ######## "

    mkdir -p datasets

    python -m src.generate_datasets -s mnist datasets/mnist.pckl
    python -m src.generate_datasets -s halves_mnist datasets/halves_mnist.pckl
    python -m src.generate_datasets -s stacked_mnist datasets/stacked_mnist.pckl
    python -m src.generate_datasets -s stacked_same_mnist datasets/stacked_same_mnist.pckl
    python -m src.generate_datasets -s cifar10 datasets/cifar10.pckl
    python -m src.generate_datasets -s cifar10_full datasets/cifar10_full.pckl
    python -m src.generate_datasets -s line datasets/line.pckl
    python -m src.generate_datasets -s fashion datasets/fashion.pckl
    python -m src.generate_datasets -s random datasets/random.pckl
    python -m src.generate_datasets -s small_mnist -w 7 -ht 7 datasets/small_mnist.pckl

#    python -m src.mix_datasets datasets/line.pckl datasets/mnist.pckl datasets/line-mnist.pckl
#    python -m src.mix_datasets datasets/line.pckl datasets/cifar10.pckl datasets/line-cifar10.pckl
    python -m src.mix_datasets datasets/mnist.pckl datasets/cifar10.pckl datasets/mnist-cifar10.pckl
    python -m src.mix_datasets datasets/mnist.pckl datasets/fashion.pckl datasets/mnist-fashion.pckl
    python -m src.mix_datasets datasets/fashion.pckl datasets/cifar10.pckl datasets/fashion-cifar10.pckl

#    python -m src.mix_datasets datasets/line.pckl datasets/mnist.pckl datasets/line-mnist-separated.pckl -s
#    python -m src.mix_datasets datasets/line.pckl datasets/cifar10.pckl datasets/line-cifar10-separated.pckl -s
    python -m src.mix_datasets datasets/mnist.pckl datasets/cifar10.pckl datasets/mnist-cifar10-separated.pckl -s
    python -m src.mix_datasets datasets/mnist.pckl datasets/fashion.pckl datasets/mnist-fashion-separated.pckl -s
    python -m src.mix_datasets datasets/fashion.pckl datasets/cifar10.pckl datasets/fashion-cifar10-separated.pckl -s

    python -m src.generate_datasets -s poly datasets/poly.pckl

    echo "######## Done! ########"

elif [[ "$1" = "models" ]]
then
    echo "######## Training Models... ########"

    mkdir -p models

    for i in {1..10}
    do
        echo "######## MLP: MNIST $i ########"
        python -m src.train_nn with mlp_config dataset_name=mnist
    done

    for i in {1..10}
    do
        echo "######## MLP: FASHION $i ########"
        python -m src.train_nn with mlp_config dataset_name=fashion
    done

    for i in {1..10}
    do
        echo "######## MLP: CIFAR-10 $i ########"
        python -m src.train_nn with mlp_config dataset_name=cifar10 pruning_epochs=40
    done

    for i in {1..10}
    do
        echo "######## MLP: LINES $i ########"
        python -m src.train_nn with mlp_config dataset_name=line
    done

    for i in {1..10}
    do
        echo "######## MLP: MNIST + DROPOUT $i ########"
        python -m src.train_nn with mlp_config dataset_name=mnist with_dropout=True
    done

    for i in {1..10}
    do
        echo "######## MLP: FASHION + DROPOUT $i ########"
        python -m src.train_nn with mlp_config dataset_name=fashion with_dropout=True
    done

    for i in {1..10}
    do
        echo "######## MLP: CIFAR-10 + DROPOUT $i ########"
        python -m src.train_nn with mlp_config dataset_name=cifar10 epochs=100 pruning_epochs=40 with_dropout=True dropout_rate=0.2
    done

    for i in {1..10}
    do
        echo "######## MLP: LINES + DROPOUT $i ########"
        python -m src.train_nn with mlp_config dataset_name=line with_dropout=True
    done

    echo "######## MLP: HALVESMNIST $i ########"
    python -m src.train_nn with mlp_config dataset_name=halves_mnist

    echo "######## MLP: HALVESMNIST + DROPOUT $i ########"
    python -m src.train_nn with mlp_config dataset_name=halves_mnist with_dropout=True

#    echo "######## MLP: LINE-MNIST ########"
#    python -m src.train_nn with mlp_config dataset_name=line-mnist
#
#    echo "######## MLP: LINE-CIFAR10 ########"
#    python -m src.train_nn with mlp_config dataset_name=line-cifar10 epochs=30 pruning_epochs=40

    echo "######## MLP: MNIST-FASHION ########"
    python -m src.train_nn with mlp_config dataset_name=mnist-fashion

    echo "######## MLP: FASHION-CIFAR10 ########"
    python -m src.train_nn with mlp_config dataset_name=fashion-cifar10 epochs=30 pruning_epochs=40

    echo "######## MLP: MNIST-CIFAR10 ########"
    python -m src.train_nn with mlp_config dataset_name=mnist-cifar10 epochs=30 pruning_epochs=40

#    echo "######## MLP: LINE-MNIST-SEPARATED ########"
#    python -m src.train_nn with mlp_config dataset_name=line-mnist-separated
#
#    echo "######## MLP: LINE-CIFAR10-SEPARATED ########"
#    python -m src.train_nn with mlp_config dataset_name=line-cifar10-separated epochs=30 pruning_epochs=40

    echo "######## MLP: MNIST-FASHION-SEPARATED ########"
    python -m src.train_nn with mlp_config dataset_name=mnist-fashion-separated

    echo "######## MLP: FASHION-CIFAR10-SEPARATED ########"
    python -m src.train_nn with mlp_config dataset_name=fashion-cifar10-separated epochs=30 pruning_epochs=40

    echo "######## MLP: MNIST-CIFAR10-SEPARATED ########"
    python -m src.train_nn with mlp_config dataset_name=mnist-cifar10-separated epochs=30 pruning_epochs=40

#    echo "######## MLP: LINE-MNIST + DROPOUT ########"
#    python -m src.train_nn with mlp_config dataset_name=line-mnist with_dropout=True
#
#    echo "######## MLP: LINE-CIFAR10 + DROPOUT ########"
#    python -m src.train_nn with mlp_config dataset_name=line-cifar10 epochs=30 pruning_epochs=40 with_dropout=True dropout_rate=0.2

    echo "######## MLP: MNIST-CIFAR10 + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=mnist-cifar10 epochs=30 pruning_epochs=40 with_dropout=True dropout_rate=0.2

    echo "######## MLP: MNIST-FASHION + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=mnist-fashion with_dropout=True

    echo "######## MLP: FASHION-CIFAR10 + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=fashion-cifar10 epochs=30 pruning_epochs=40 with_dropout=True dropout_rate=0.2

#    echo "######## MLP: LINE-MNIST-SEPARATED + DROPOUT ########"
#    python -m src.train_nn with mlp_config dataset_name=line-mnist-separated with_dropout=True
#
#    echo "######## MLP: LINE-CIFAR10-SEPARATED + DROPOUT ########"
#    python -m src.train_nn with mlp_config dataset_name=line-cifar10-separated epochs=30 pruning_epochs=40 with_dropout=True dropout_rate=0.2

    echo "######## MLP: MNIST-CIFAR10-SEPARATED + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=mnist-cifar10-separated epochs=30 pruning_epochs=40 with_dropout=True dropout_rate=0.2

    echo "######## MLP: MNIST-FASHION-SEPARATED + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=mnist-fashion-separated with_dropout=True

    echo "######## MLP: FASHION-CIFAR10-SEPARATED + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=fashion-cifar10-separated epochs=30 pruning_epochs=40 with_dropout=True dropout_rate=0.2

    echo "######## MLP: RANDOM ########"
    python -m src.train_nn with mlp_config dataset_name=random

    echo "######## MLP: RANDOM + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=random with_dropout=True

    echo "######## MLP: MMNIST-x1.5-EPOCHS unpruned ########"
    # TODO is this really unpruned?
    python -m src.train_nn with mlp_config dataset_name=mnist epochs=30

    echo "######## MLP: MMNIST-x1.5-EPOCHS unpruned + DROPOUT ########"
    # TODO is this really unpruned?
    python -m src.train_nn with mlp_config dataset_name=mnist epochs=30 with_dropout=True

    echo "######## MLP: MMNIST-x2-EPOCHS unpruned ########"
    python -m src.train_nn with mlp_config dataset_name=mnist epochs=40

    echo "######## MLP: MMNIST-x2-EPOCHS unpruned + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=mnist epochs=40 with_dropout=True

    echo "######## MLP: MMNIST-x10-EPOCHS unpruned ########"
    python -m src.train_nn with mlp_config dataset_name=mnist epochs=200

    echo "######## MLP: MMNIST-x10-EPOCHS unpruned + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=mnist epochs=200 with_dropout=True

    echo "######## MLP: RANDOM-x50-EPOCHS unpruned ########"
    python -m src.train_nn with mlp_config dataset_name=random epochs=1000

    echo "######## MLP: RANDOM-x50-EPOCHS unpruned + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=random epochs=1000 with_dropout=True

    echo "######## MLP: RANDOM-OVERFITTING ########"
    python -m src.train_nn with mlp_config dataset_name=random epochs=100 pruning_epochs=100 shuffle=False n_train=3000

    echo "######## MLP: RANDOM-OVERFITTING + DROPOUT########"
    python -m src.train_nn with mlp_config dataset_name=random epochs=100 pruning_epochs=100 shuffle=False n_train=3000 with_dropout=True

    echo "######## CNN- MNIST ########"
    python -m src.train_nn with cnn_config dataset_name=mnist

    echo "######## CNN- CIFAR10 ########"
    python -m src.train_nn with cnn_config dataset_name=cifar10

    echo "######## CNN- LINE ########"
    python -m src.train_nn with cnn_config dataset_name=line

    echo "######## CNN- FASHION ########"
    python -m src.train_nn with cnn_config dataset_name=fashion

    echo "######## CNN- MNIST + DROPOUT ########"
    python -m src.train_nn with cnn_config dataset_name=mnist with_dropout=True

    echo "######## CNN- CIFAR10 + DROPOUT ########"
    python -m src.train_nn with cnn_config dataset_name=cifar10 with_dropout=True

    echo "######## CNN- LINE + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=line with_dropout=True

    echo "######## CNN- FASHION + DROPOUT ########"
    python -m src.train_nn with cnn_config dataset_name=fashion with_dropout=True

    echo "######## Done! ########"

fi
