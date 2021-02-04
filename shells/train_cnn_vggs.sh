#!/usr/bin/env bash

for i in {1..5}
do

    echo "######## CNN_VGG: CIFAR10 ########"
    python -m src.train_nn with cnn_vgg_config dataset_name=cifar10_full
    echo "######## CNN_VGG: CIFAR10+L1REG########"
    python -m src.train_nn with cnn_vgg_config dataset_name=cifar10_full with_l1reg=True
    echo "######## CNN_VGG: CIFAR10+L2REG########"
    python -m src.train_nn with cnn_vgg_config dataset_name=cifar10_full with_l2reg=True
    echo "######## CNN_VGG: CIFAR10+DROPOUT########"
    python -m src.train_nn with cnn_vgg_config dataset_name=cifar10_full with_dropout=True
    echo "######## CNN_VGG: CIFAR10+L2REG+DROPOUT########"
    python -m src.train_nn with cnn_vgg_config dataset_name=cifar10_full with_l2reg=True with_dropout=True
    echo "######## CNN_VGG: CIFAR10+MOD-INIT########"
    python -m src.train_nn with cnn_vgg_config dataset_name=cifar10_full init_modules=10

done