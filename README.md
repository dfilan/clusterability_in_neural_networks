# Clusterability in Neural Networks

## Results

1. [Clustering and p-value plots](notebooks/mlp-plots.ipynb)
2. Lesion test: [MNIST](notebooks/mlp-double-lesion-test-MNIST.ipynb), [MNIST+Dropout](notebooks/mlp-double-lesion-test-MNIST+DROPOUT.ipynb), [Fashion](notebooks/mlp-double-lesion-test-FASHION.ipynb), [Fashion+Dropout](notebooks/mlp-double-lesion-test-FASHION+DROPOUT.ipynb)
3. Learning curves [notebook](notebooks/mlp-learning-curve.ipynb)


## Instructions

We use `make` with a `Makefile` to automate the project.

1. `make datasets` - Build all datasets (deterministic),
2. `make models` - Train all NN models, both MLP and CNN.
3. `make test` - Run tests (with `pytest`).
4. `make mlp-clustering` - Run the notebook `notebooks/mlp-clustering.ipynb` that cluster all MLP models (including alternative explanation ones), and save the results as a table into `results/mlp-clustering.csv`.
5. `make mlp-lesion` - Run the notebook `notebooks/mlp-lesion-test.ipynb` that perform the lesion test on all standard MLP models, and save the results as a table into `results/mlp-lesion.xlsx`.
6. `make mlp-double-lesion` - Run the notebook `notebooks/mlp-double-lesion-test.ipynb` that perform the double lesion test on all standard MLP models.
7. `make mlp-learning-curve` - Run the notebook `notebooks/mlp-learning-curve.ipynb` that plot the learning curves to selected set of MLP models.
8. `make mlp-clustering-stability` - Run the notebook `notebooks/mlp-clustering-stability.ipynb` that train and cluster multiple trained instanced of all of the MLP models (including alternative explanation ones), and save the results as a table into `results/mlp-clustering-stability-statistic.csv` (NOTE: read the comment in the notebook about `src/train_nn.py` before running it).
9. `make mlp-plots` - Run the notebook `nootebooks/mlp-plots.ipynb` that generates many of the plots from the NeurIPS 2020 paper.
10. `make mlp-clustering-stability-n-clusters` - Run the notebook `notebooks/mlp-clustering-stability-different-n_clusters.ipynb` tha cluster multiple trained instanced of all of the MLP models (including alternative explanation ones) with various number of clusters (K=2, 7, 10), and save the results as a table into `results/mlp-clustering-stability-statistic-K.csv` (NOTE: read the comment in the notebook about `src/train_nn.py` before running it).

## Research Environment Setup

Requirements: Python 3.7 (It might work with an earlier version, but it wasn't tested)

There are two options to set up the environment:

1. Using a Python virtual environment
2. Using a Docker container

### 1. Python Virtual Environment

1. Clone this repository

2. Install `graphviz`
   1. Ubuntu/Debian: `apt intall graphviz`
n   2. MacOS: `brew install graphviz`

3. Install with `pipenv install --dev`

4. On MacOs **only**, you will need to install `pygraphviz` separatly:
   `pipenv run pip install pygraphviz --install-option="--include-path=/usr/local/Cellar/`
   
5. To enter the virtual environment, type `pipenv shell`

6. To set up the dependencies and finish, type `cd nn_clustering` and `pipenv install --system`

### 2. Docker

Useful: [Lifecycle of Docker Container](https://medium.com/@nagarwal/lifecycle-of-docker-container-d2da9f85959)

#### Building the image (done **ONCE** per machine)

Clone the repository and change to the `devops` directory.

```bash
docker build -t humancompatibleai/clusterability_in_neural_networks .
```

#### Creating a container (done **ONCE** by one user per one machine)

In other words, if you run this commands on, for example, `svm` - you don't need to do so again. You're supposed then to have a *container* there already.

First, you need a port number to your Jupyter Netbook - pick up a random number (with your favoriate generator) in the range 8000-8500.
We pick up a random number so you don't collide with existing notebooks on that machine.

First run: 

- **Remove the comments before**, and 
- **Replace** `<PORT NUMBER>` with your random port number (also in the instructions that will come later)

```bash
docker run \
-it \
-p <PORT NUMBER>:8888 \
--rm \
--name nn_clustering-$(whoami) \
--runtime=nvidia \  # REMOVE, if you don't have GPU
humancompatibleai/nn-clustering:latest \
bash
```

And then type

```bash
bash build.sh
```

NB: to leave the container, use ctrl-P ctrl-Q. Typing `exit` will destroy the container.

#### Running the container

```bash
docker exec \
-it nn_clustering-$(whoami) \
bash
```

#### Running a Jupyter Notebook

Run this command:

```bash
jupyter notebook --allow-root --no-browser --ip=0.0.0.0 --port=8888
```

If the container is on your computer, just open the browser at the address: `http://localhost:8888`

If the container is on another server, you should forward the 8888 port through ssh on your personal machine with the command:

```bash
ssh -N -L localhost:8888:localhost:<PORT NUMBER> -i <PATH TO SSH PUBLIC KEY>  <USERNAME>@<SERVER ADDRESS>
```

After doing this, you can then open the jupyter notebook in your browser.

These notebooks were created back when the code lived in a directory called `nn_clustering`. Now that it lives in a differently-named directory, you might have to find-and-replace.

#### `tmux`

It is advised to learn how to use tmux, and run the Jupyter Notebook on a separate window: https://github.com/tmux/tmux/wiki


### 3. AWS S3

To upload files or directories:

```bash
aws s3 cp --recursive <local> s3://nn-clustering/<remote>
```
