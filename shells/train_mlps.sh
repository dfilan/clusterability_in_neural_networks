#!/usr/bin/env bash

# train networks without extracting activations
for i in {1..5}
do

    echo "######## MLP: MNIST ########"
    python -m src.train_nn with mlp_config dataset_name=mnist
    echo "######## MLP: FASHION ########"
    python -m src.train_nn with mlp_config dataset_name=fashion
#     echo "######## MLP: CIFAR10 ########"
#     python -m src.train_nn with mlp_config dataset_name=cifar10 pruning_epochs=40

    echo "######## MLP: MNIST + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=mnist with_dropout=True
#     echo "######## MLP: CIFAR10 + DROPOUT ########"
#     python -m src.train_nn with mlp_config dataset_name=cifar10 with_dropout=True
    echo "######## MLP: FASHION + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=fashion with_dropout=True

    echo "######## MLP: MNIST + L1REG ########"
    python -m src.train_nn with mlp_config dataset_name=mnist with_l1reg=True
#     echo "######## MLP: CIFAR10 + L1REG ########"
#     python -m src.train_nn with mlp_config dataset_name=cifar10 with_l1reg=True
    echo "######## MLP: FASHION + L1REG ########"
    python -m src.train_nn with mlp_config dataset_name=fashion with_l1reg=True

    echo "######## MLP: MNIST + L2REG ########"
    python -m src.train_nn with mlp_config dataset_name=mnist with_l2reg=True
#     echo "######## MLP: CIFAR10 + L2REG ########"
#     python -m src.train_nn with mlp_config dataset_name=cifar10 with_l2reg=True
    echo "######## MLP: FASHION + L2REG ########"
    python -m src.train_nn with mlp_config dataset_name=fashion with_l2reg=True

    echo "######## MLP: MOD-INIT-MNIST ########"
    python -m src.train_nn with mlp_config dataset_name=mnist init_modules=10
#     echo "######## MLP: MOD-INIT-CIFAR10 ########"
#     python -m src.train_nn with mlp_config dataset_name=cifar10 init_modules=10
    echo "######## MLP: MOD-INIT-FASHION ########"
    python -m src.train_nn with mlp_config dataset_name=fashion init_modules=10

    echo "######## MLP: HALVES-SAME-MNIST ########"
    python -m src.train_nn with mlp_config dataset_name=halves_same_mnist
#     echo "######## MLP: HALVES-SAME-CIFAR10 ########"
#     python -m src.train_nn with mlp_config dataset_name=halves_same_cifar10
    echo "######## MLP: HALVES-SAME-FASHION ########"
    python -m src.train_nn with mlp_config dataset_name=halves_same_fashion
#
#     echo "######## MLP: HALVES-SAME-MNIST + DROPOUT ########"
#     python -m src.train_nn with mlp_config dataset_name=halves_same_mnist with_dropout=True
# #     echo "######## MLP: HALVES-SAME-CIFAR10 + DROPOUT ########"
# #     python -m src.train_nn with mlp_config dataset_name=halves_same_cifar10 with_dropout=True
#     echo "######## MLP: HALVES-SAME-FASHION + DROPOUT ########"
#     python -m src.train_nn with mlp_config dataset_name=halves_same_fashion with_dropout=True
#
#     echo "######## MLP: HALVES-SAME-MNIST + L1REG ########"
#     python -m src.train_nn with mlp_config dataset_name=halves_same_mnist with_l1reg=True
# #     echo "######## MLP: HALVES-SAME-CIFAR10 + L1REG ########"
# #     python -m src.train_nn with mlp_config dataset_name=halves_same_cifar10 with_l1reg=True
#     echo "######## MLP: HALVES-SAME-FASHION + L1REG ########"
#     python -m src.train_nn with mlp_config dataset_name=halves_same_fashion with_l1reg=True
#
#     echo "######## MLP: HALVES-SAME-MNIST + L2REG ########"
#     python -m src.train_nn with mlp_config dataset_name=halves_same_mnist with_l2reg=True
# #     echo "######## MLP: HALVES-SAME-CIFAR10 + L2REG ########"
# #     python -m src.train_nn with mlp_config dataset_name=halves_same_cifar10 with_l2reg=True
#     echo "######## MLP: HALVES-SAME-FASHION + L2REG ########"
#     python -m src.train_nn with mlp_config dataset_name=halves_same_fashion with_l2reg=True
#
#     echo "######## MLP: MOD-INIT-HALVES-SAME-MNIST ########"
#     python -m src.train_nn with mlp_config dataset_name=halves_same_mnist init_modules=10
# #     echo "######## MLP: MOD-INIT-HALVES-SAME-CIFAR10 ########"
# #     python -m src.train_nn with mlp_config dataset_name=halves_same_cifar10 init_modules=10
#     echo "######## MLP: MOD-INIT-HALVES-SAME-FASHION ########"
#     python -m src.train_nn with mlp_config dataset_name=halves_same_fashion init_modules=10

    echo "######## MLP: HALVES-MNIST ########"
    python -m src.train_nn with mlp_config dataset_name=halves_mnist
#     echo "######## MLP: HALVES-CIFAR10 ########"
#     python -m src.train_nn with mlp_config dataset_name=halves_cifar10
    echo "######## MLP: HALVES-FASHION ########"
    python -m src.train_nn with mlp_config dataset_name=halves_fashion

    echo "######## MLP: MNIST+LUCID ########"
    python -m src.train_nn with mlp_config dataset_name=mnist epochs=30 lucid=True
    echo "######## MLP: MNIST + DROPOUT + LUCID ########"
    python -m src.train_nn with mlp_config dataset_name=mnist epochs=30 with_dropout=True lucid=True
    echo "######## MLP: MOD-INIT-MNIST + LUCID ########"
    python -m src.train_nn with mlp_config dataset_name=mnist init_modules=10 epochs=30 lucid=True
    echo "######## MLP: MOD-INIT-MNIST + DROPOUT + LUCID ########"
    python -m src.train_nn with mlp_config dataset_name=mnist init_modules=10 with_dropout=True \
    epochs=30 lucid=True

    echo "######## MLP: POLY ########"
    python -m src.train_nn with mlp_regression_config dataset_name=poly
    echo "######## MLP: POLY + L1REG########"
    python -m src.train_nn with mlp_regression_config dataset_name=poly with_l1reg=True
    echo "######## MLP: POLY + L2REG########"
    python -m src.train_nn with mlp_regression_config dataset_name=poly with_l2reg=True

done

echo "######## MLP: MNIST DROPOUT ########"
python -m src.train_nn with mlp_config dataset_name=mnist with_dropout=True extract_activations=True
echo "######## MLP: FASHION DROPOUT ########"
python -m src.train_nn with mlp_config dataset_name=fashion with_dropout=True extract_activations=True
# echo "######## MLP: CIFAR10 DROPOUT ########"
# python -m src.train_nn with mlp_config dataset_name=cifar10 epochs=100 pruning_epochs=40 \
# with_dropout=True extract_activations=True

echo "######## MLP: HALVES-SAME-MNIST ########"
python -m src.train_nn with mlp_config dataset_name=halves_same_mnist extract_activations=True
echo "######## MLP: HALVES-MNIST ########"
python -m src.train_nn with mlp_config dataset_name=halves_mnist extract_activations=True
echo "######## MLP: HALVES-SAME-MNIST+DROPOUT ########"
python -m src.train_nn with mlp_config dataset_name=halves_same_mnist with_dropout=True extract_activations=True
echo "######## MLP: HALVES-MNIST+DROPOUT ########"
python -m src.train_nn with mlp_config dataset_name=halves_mnist with_dropout=True extract_activations=True

# Train some nets with all the stops except dropout (cuz dropout doesn't play well with others)
# echo "######## MLP: HALVES-MNIST+MOD-INIT+L1REG ########"
# python -m src.train_nn with mlp_config dataset_name=halves_mnist with_l1reg=True init_modules=10 \
# extract_activations=True
# echo "######## MLP: HALVES-FASHION+MOD-INIT+L1REG ########"
# python -m src.train_nn with mlp_config dataset_name=halves_fashion with_l1reg=True init_modules=10 \
# extract_activations=True
# echo "######## MLP: HALVES-CIFAR10+MOD-INIT+L1REG ########"
# python -m src.train_nn with mlp_config dataset_name=halves_cifar10 with_l1reg=True init_modules=10 \
# epochs=40 pruning_epochs=40 extract_activations=True
#
# echo "######## CNN: STACKED-MNIST+MOD-INIT+L1REG ########"
# python -m src.train_nn with cnn_config dataset_name=stacked_mnist with_l1reg=True init_modules=10 \
# extract_activations=True
# echo "######## CNN: STACKED-FASHION+MOD-INIT+L1REG ########"
# python -m src.train_nn with cnn_config dataset_name=stacked_fashion with_l1reg=True init_modules=10 \
# extract_activations=True
# echo "######## CNN: STACKED-CIFAR10+MOD-INIT+L1REG ########"
# python -m src.train_nn with cnn_config dataset_name=stacked_cifar10 with_l1reg=True init_modules=10 \
# extract_activations=True
