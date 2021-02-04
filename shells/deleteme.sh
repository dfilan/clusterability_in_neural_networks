#!/usr/bin/env bash

# train networks without extracting activations
for i in {1..5}
do

    echo "######## MLP: MNIST + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=mnist with_dropout=True
    echo "######## MLP: CIFAR10 + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=cifar10 with_dropout=True
    echo "######## MLP: FASHION + DROPOUT ########"
    python -m src.train_nn with mlp_config dataset_name=fashion with_dropout=True

    echo "######## MLP: MNIST + L1REG ########"
    python -m src.train_nn with mlp_config dataset_name=mnist with_l1reg=True
    echo "######## MLP: CIFAR10 + L1REG ########"
    python -m src.train_nn with mlp_config dataset_name=cifar10 with_l1reg=True
    echo "######## MLP: FASHION + L1REG ########"
    python -m src.train_nn with mlp_config dataset_name=fashion with_l1reg=True

    echo "######## MLP: MNIST + L2REG ########"
    python -m src.train_nn with mlp_config dataset_name=mnist with_l2reg=True
    echo "######## MLP: CIFAR10 + L2REG ########"
    python -m src.train_nn with mlp_config dataset_name=cifar10 with_l2reg=True
    echo "######## MLP: FASHION + L2REG ########"
    python -m src.train_nn with mlp_config dataset_name=fashion with_l2reg=True

    echo "######## MLP: MOD-INIT-MNIST ########"
    python -m src.train_nn with mlp_config dataset_name=mnist init_modules=10
    echo "######## MLP: MOD-INIT-CIFAR10 ########"
    python -m src.train_nn with mlp_config dataset_name=cifar10 init_modules=10
    echo "######## MLP: MOD-INIT-FASHION ########"
    python -m src.train_nn with mlp_config dataset_name=fashion init_modules=10



    echo "######## CNN: MNIST+DROPOUT ########"
    python -m src.train_nn with cnn_config dataset_name=stacked_same_mnist with_dropout=True
    echo "######## CNN: CIFAR10+DROPOUT ########"
    python -m src.train_nn with cnn_config dataset_name=stacked_same_cifar10 with_dropout=True
    echo "######## CNN: FASHION+DROPOUT ########"
    python -m src.train_nn with cnn_config dataset_name=stacked_same_fashion with_dropout=True

    echo "######## CNN: MNIST+L1REG ########"
    python -m src.train_nn with cnn_config dataset_name=stacked_same_mnist with_l1reg=True
    echo "######## CNN: CIFAR10+L1REG ########"
    python -m src.train_nn with cnn_config dataset_name=stacked_same_cifar10 with_l1reg=True
    echo "######## CNN: FASHION+L1REG ########"
    python -m src.train_nn with cnn_config dataset_name=stacked_same_fashion with_l1reg=True

    echo "######## CNN: MNIST+L2REG ########"
    python -m src.train_nn with cnn_config dataset_name=stacked_same_mnist with_l2reg=True
    echo "######## CNN: CIFAR10+L2REG ########"
    python -m src.train_nn with cnn_config dataset_name=stacked_same_cifar10 with_l2reg=True
    echo "######## CNN: FASHION+L2REG ########"
    python -m src.train_nn with cnn_config dataset_name=stacked_same_fashion with_l2reg=True

    echo "######## CNN: CNN-MOD-INIT-MNIST ########"
    python -m src.train_nn with cnn_config dataset_name=stacked_same_mnist init_modules=10
    echo "######## CNN: CNN-MOD-INIT-CIFAR10 ########"
    python -m src.train_nn with cnn_config dataset_name=stacked_same_cifar10 init_modules=10
    echo "######## CNN: CNN-MOD-INIT-FASHION ########"
    python -m src.train_nn with cnn_config dataset_name=stacked_same_fashion init_modules=10

done

echo "######## MLP: MNIST DROPOUT ########"
python -m src.train_nn with mlp_config dataset_name=mnist with_dropout=True extract_activations=True
echo "######## MLP: FASHION DROPOUT ########"
python -m src.train_nn with mlp_config dataset_name=fashion with_dropout=True extract_activations=True
echo "######## MLP: CIFAR10 DROPOUT ########"
python -m src.train_nn with mlp_config dataset_name=cifar10 epochs=100 pruning_epochs=40 \
with_dropout=True extract_activations=True

echo "######## CNN: MNIST+DROPOUT ########"
python -m src.train_nn with cnn_config dataset_name=mnist with_dropout=True extract_activations=True
echo "######## CNN: CIFAR10+DROPOUT ########"
python -m src.train_nn with cnn_config dataset_name=cifar10 with_dropout=True extract_activations=True
echo "######## CNN: FASHION+DROPOUT ########"
python -m src.train_nn with cnn_config dataset_name=fashion with_dropout=True extract_activations=True