#!/usr/bin/env bash

for i in {1..5}
do

    echo "######## CNN: MNIST ########"
    python -m src.train_nn with cnn_config dataset_name=mnist
    echo "######## CNN: CIFAR10 ########"
    python -m src.train_nn with cnn_config dataset_name=cifar10
    echo "######## CNN: FASHION ########"
    python -m src.train_nn with cnn_config dataset_name=fashion

    echo "######## CNN: MNIST+DROPOUT ########"
    python -m src.train_nn with cnn_config dataset_name=mnist with_dropout=True
#     echo "######## CNN: CIFAR10+DROPOUT ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_same_cifar10 with_dropout=True
    echo "######## CNN: FASHION+DROPOUT ########"
    python -m src.train_nn with cnn_config dataset_name=fashion with_dropout=True

    echo "######## CNN: MNIST+L1REG ########"
    python -m src.train_nn with cnn_config dataset_name=mnist with_l1reg=True
#     echo "######## CNN: CIFAR10+L1REG ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_same_cifar10 with_l1reg=True
    echo "######## CNN: FASHION+L1REG ########"
    python -m src.train_nn with cnn_config dataset_name=fashion with_l1reg=True

    echo "######## CNN: MNIST+L2REG ########"
    python -m src.train_nn with cnn_config dataset_name=mnist with_l2reg=True
#     echo "######## CNN: CIFAR10+L2REG ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_same_cifar10 with_l2reg=True
    echo "######## CNN: FASHION+L2REG ########"
    python -m src.train_nn with cnn_config dataset_name=fashion with_l2reg=True

    echo "######## CNN: CNN-MOD-INIT-MNIST ########"
    python -m src.train_nn with cnn_config dataset_name=mnist init_modules=10
#     echo "######## CNN: CNN-MOD-INIT-CIFAR10 ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_same_cifar10 init_modules=10
    echo "######## CNN: CNN-MOD-INIT-FASHION ########"
    python -m src.train_nn with cnn_config dataset_name=fashion init_modules=10

    echo "######## CNN: STACKED-SAME-MNIST ########"
    python -m src.train_nn with cnn_config dataset_name=stacked_same_mnist
#     echo "######## CNN: STACKED-SAME-CIFAR10 ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_same_cifar10
    echo "######## CNN: STACKED-SAME-FASHION ########"
    python -m src.train_nn with cnn_config dataset_name=stacked_same_fashion
#
#     echo "######## CNN: STACKED-SAME-MNIST+DROPOUT ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_mnist with_dropout=True
# #     echo "######## CNN: STACKED-SAME-CIFAR10+DROPOUT ########"
# #     python -m src.train_nn with cnn_config dataset_name=stacked_same_cifar10 with_dropout=True
#     echo "######## CNN: STACKED-SAME-FASHION+DROPOUT ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_fashion with_dropout=True
#
#     echo "######## CNN: STACKED-SAME-MNIST+L1REG ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_same_mnist with_l1reg=True
# #     echo "######## CNN: STACKED-SAME-CIFAR10+L1REG ########"
# #     python -m src.train_nn with cnn_config dataset_name=stacked_same_cifar10 with_l1reg=True
#     echo "######## CNN: STACKED-SAME-FASHION+L1REG ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_same_fashion with_l1reg=True
#
#     echo "######## CNN: STACKED-SAME-MNIST+L2REG ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_same_mnist with_l2reg=True
# #     echo "######## CNN: STACKED-SAME-CIFAR10+L2REG ########"
# #     python -m src.train_nn with cnn_config dataset_name=stacked_same_cifar10 with_l2reg=True
#     echo "######## CNN: STACKED-SAME-FASHION+L2REG ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_same_fashion with_l2reg=True

#     echo "######## CNN: CNN-MOD-INIT-STACKED-SAME-MNIST ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_same_mnist init_modules=10
# #     echo "######## CNN: CNN-MOD-INIT-STACKED-SAME-CIFAR10 ########"
# #     python -m src.train_nn with cnn_config dataset_name=stacked_same_cifar10 init_modules=10
#     echo "######## CNN: CNN-MOD-INIT-STACKED-SAME-FASHION ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_same_fashion init_modules=10

    echo "######## CNN: STACKED-MNIST ########"
    python -m src.train_nn with cnn_config dataset_name=stacked_mnist
#     echo "######## CNN: STACKED-CFIFAR10 ########"
#     python -m src.train_nn with cnn_config dataset_name=stacked_cifar10
    echo "######## CNN: STACKED-FASHION ########"
    python -m src.train_nn with cnn_config dataset_name=stacked_fashion

    echo "######## CNN: MNIST+LUCID ########"
    python -m src.train_nn with cnn_config dataset_name=mnist lucid=True
    echo "######## CNN: MNIST+DROPOUT+LUCID ########"
    python -m src.train_nn with cnn_config dataset_name=mnist with_dropout=True lucid=True
    echo "######## CNN: MOD-INIT-MNIST+LUCID ########"
    python -m src.train_nn with cnn_config dataset_name=mnist init_modules=10 lucid=True
    echo "######## CNN: MOD-INIT-MNIST + DROPOUT+LUCID ########"
    python -m src.train_nn with cnn_config dataset_name=mnist init_modules=10 with_dropout=True lucid=True

done

echo "######## CNN: MNIST+DROPOUT ########"
python -m src.train_nn with cnn_config dataset_name=mnist with_dropout=True extract_activations=True
# echo "######## CNN: CIFAR10+DROPOUT ########"
# python -m src.train_nn with cnn_config dataset_name=cifar10 with_dropout=True extract_activations=True
echo "######## CNN: FASHION+DROPOUT ########"
python -m src.train_nn with cnn_config dataset_name=fashion with_dropout=True extract_activations=True

echo "######## CNN: STACKED-SAME-MNIST ########"
python -m src.train_nn with cnn_config dataset_name=stacked_same_mnist extract_activations=True
echo "######## CNN: STACKED-MNIST ########"
python -m src.train_nn with cnn_config dataset_name=stacked_mnist extract_activations=True
echo "######## CNN: STACKED-SAME-MNIST +DROPOUT ########"
python -m src.train_nn with cnn_config dataset_name=stacked_same_mnist with_dropout=True extract_activations=True
echo "######## CNN: STACKED-MNIST +DROPOUT########"
python -m src.train_nn with cnn_config dataset_name=stacked_mnist with_dropout=True extract_activations=True