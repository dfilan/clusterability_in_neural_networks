# <NOTEBOOK_PATH> <OUTPUT_PATH>
PAPERMILL := papermill --cwd ./notebooks
# NBX := jupyter nbconvert --ExecutePreprocessor.timeout=-1 --execute --to notebook --inplace

.PHONY: clean-datasets clean-models clean-all mkdir-results models test all
.PHONY: mlp-clustering mlp-lesion mlp-double-lesion make mlp-plots

clean-datasets:
	rm -rf datasets

clean-models:
	rm -rf training_runs_dir models

clean-all: clean-datasets clean-models

mkdir-results:
	mkdir -p results

datasets:
	bash prepare_all.sh datasets

models:
	bash prepare_all.sh models

# Running `pytest src` causes to the weird `sacred` and `tensorflow` import error:
# ImportError: Error while finding loader for 'tensorflow' (<class 'ValueError'>: tensorflow.__spec__ is None)
# https://github.com/IDSIA/sacred/issues/493
test:
	pytest src/tests/test_lesion.py
	pytest src/tests/test_utils.py
	pytest src/tests/test_cnn.py
	pytest src/tests/test_spectral_clustering.py

all: datasets models test mlp-analysis

mlp-clustering: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-clustering.ipynb ./notebooks/mlp-clustering.ipynb

mlp-clustering-reps-mixed: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-clustering-reps-mixed.ipynb ./notebooks/mlp-clustering-reps-mixed.ipynb

mlp-clustering-stability: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-clustering-stability.ipynb ./notebooks/mlp-clustering-stability.ipynb

mlp-clustering-stability-n-clusters: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-clustering-stability-different-n_clusters.ipynb ./notebooks/mlp-clustering-stability-different-n_clusters-K=2.ipynb -p N_CLUSTERS 2
	$(PAPERMILL) ./notebooks/mlp-clustering-stability-different-n_clusters.ipynb ./notebooks/mlp-clustering-stability-different-n_clusters-K=7.ipynb -p N_CLUSTERS 7
	$(PAPERMILL) ./notebooks/mlp-clustering-stability-different-n_clusters.ipynb ./notebooks/mlp-clustering-stability-different-n_clusters-K=10.ipynb -p N_CLUSTERS 10
    
mlp-learning-curve: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-learning-curve.ipynb ./notebooks/mlp-learning-curve.ipynb

halves-mnist: mkdir-results
	$(PAPERMILL) ./notebooks/halves_mnist.ipynb ./notebooks/halves_mnist.ipynb

mlp-lesion: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-lesion-test.ipynb ./notebooks/mlp-lesion-test.ipynb

mlp-lesion-mixed-dsets: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-lesion-test-mixed-dsets.ipynb ./notebooks/mlp-lesion-test-mixed-dsets.ipynb

mlp-activations: mkdir-results
	$(PAPERMILL) ./notebooks/mlp_activations.ipynb ./notebooks/mlp_activations.ipynb

# Using 10 clusters
mlp-lesion-TEN: mkdir-results
	$(PAPERMILL) ./notebooks/mlp-lesion-test-TEN.ipynb ./notebooks/mlp-lesion-test-TEN.ipynb

mlp-double-lesion: mkdir-results
	# $(PAPERMILL) ./notebooks/mlp-double-lesion-test.ipynb ./notebooks/mlp-double-lesion-test-MNIST.ipynb -p MODEL_TAG MNIST
	# $(PAPERMILL) ./notebooks/mlp-double-lesion-test.ipynb ./notebooks/mlp-double-lesion-test-MNIST+DROPOUT.ipynb -p MODEL_TAG MNIST+DROPOUT
	# $(PAPERMILL) ./notebooks/mlp-double-lesion-test.ipynb ./notebooks/mlp-double-lesion-test-FASHION.ipynb -p MODEL_TAG FASHION
	$(PAPERMILL) ./notebooks/mlp-double-lesion-test.ipynb ./notebooks/mlp-double-lesion-test-FASHION+DROPOUT.ipynb -p MODEL_TAG FASHION+DROPOUT

mlp-plots:
	$(PAPERMILL) ./notebooks/mlp-plots.ipynb ./notebooks/mlp-plots.ipynb

mlp-analysis: mlp-clustering mlp-clustering-stability mlp-clustering-stability-n-clusters mlp-learning-curve mlp-lesion mlp-double-lesion mlp-plots

polynomial: mkdir-results
	$(PAPERMILL) ./notebooks/polynomial.ipynb ./notebooks/polynomial.ipynb

modular-init: mkdir-results
	$(PAPERMILL) ./notebooks/modular_init.ipynb ./notebooks/modular_init.ipynb

stacked-mnist: mkdir-results
	$(PAPERMILL) ./notebooks/stacked_mnist.ipynb ./notebooks/stacked_mnist.ipynb

cnn-clustering: mkdir-results
	$(PAPERMILL) ./notebooks/cnn-clustering.ipynb ./notebooks/cnn-clustering.ipynb

cnn-lesion: mkdir-results
	$(PAPERMILL) ./notebooks/cnn-lesion-test.ipynb ./notebooks/cnn-lesion-test.ipynb

cnn-activations: mkdir-results
	$(PAPERMILL) ./notebooks/cnn_activations.ipynb ./notebooks/cnn_activations.ipynb

cnn-cifar10-full: mkdir-results
	$(PAPERMILL) ./notebooks/cifar10_cnn.ipynb ./notebooks/cifar10_cnn.ipynb

lucid-make-dataset-mlp: mkdir-results
	$(PAPERMILL) ./notebooks/lucid_make_dataset_mlp.ipynb ./notebooks/lucid_make_dataset_mlp.ipynb

lucid-make-dataset-cnn: mkdir-results
	$(PAPERMILL) ./notebooks/lucid_make_dataset_cnn.ipynb ./notebooks/lucid_make_dataset_cnn.ipynb

lucid-make-dataset-cnn-vgg: mkdir-results
	$(PAPERMILL) ./notebooks/lucid_make_dataset_cnn_vgg.ipynb ./notebooks/lucid_make_dataset_cnn_vgg.ipynb

lucid-make-dataset-imagenet: mkdir-results
	$(PAPERMILL) ./notebooks/lucid_make_dataset_imagenet.ipynb ./notebooks/lucid_make_dataset_imagenet.ipynb

lucid-eval: mkdir-results
	$(PAPERMILL) ./notebooks/lucid_evaluation.ipynb ./notebooks/lucid_evaluation.ipynb

lucid-results-all: mkdir-results
	$(PAPERMILL) ./notebooks/lucid_results_all.ipynb ./notebooks/lucid_results_all.ipynb

imagenet-clustering: mkdir-results
	$(PAPERMILL) ./notebooks/imagenet_clustering.ipynb ./notebooks/imagenet_clustering.ipynb

lesion-results-mlp: mkdir-results
	$(PAPERMILL) ./notebooks/lesion_results_mlp.ipynb ./notebooks/lesion_results_mlp.ipynb

lesion-results-cnn: mkdir-results
	$(PAPERMILL) ./notebooks/lesion_results_cnn.ipynb ./notebooks/lesion_results_cnn.ipynb

lesion-results-cnn-vgg: mkdir-results
	$(PAPERMILL) ./notebooks/lesion_results_cnn_vgg.ipynb ./notebooks/lesion_results_cnn_vgg.ipynb

lesion-results-imagenet: mkdir-results
	$(PAPERMILL) ./notebooks/lesion_results_imagenet.ipynb ./notebooks/lesion_results_imagenet.ipynb

imagenet-clustering-info: mkdir-results
	$(PAPERMILL) ./notebooks/imagenet_clustering_info.ipynb ./notebooks/imagenet_clustering_info.ipynb

imagenet-lucid-make-dataset: mkdir-results
	$(PAPERMILL) ./notebooks/imagenet_lucid_make_dataset.ipynb ./notebooks/imagenet_lucid_make_dataset.ipynb

imagenet-lucid-eval: mkdir-results
	$(PAPERMILL) ./notebooks/imagenet_lucid_evaluation.ipynb ./notebooks/imagenet_lucid_evaluation.ipynb

cluster-vis-corr: mkdir-results
	$(PAPERMILL) ./notebooks/cluster_vis_corr.ipynb ./notebooks/cluster_vis_corr.ipynb

clustering-factors-mlp: mkdir-results
	$(PAPERMILL) ./notebooks/clustering_factors_mlp.ipynb ./notebooks/clustering_factors_mlp.ipynb

clustering-factors-clust-grad: mkdir-results
	$(PAPERMILL) ./notebooks/clustering_factors_clust_grad.ipynb ./notebooks/clustering_factors_clust_grad.ipynb

clustering-factors-cnn: mkdir-results
	$(PAPERMILL) ./notebooks/clustering_factors_cnn.ipynb ./notebooks/clustering_factors_cnn.ipynb

clustering-factors-cnn-vgg: mkdir-results
	$(PAPERMILL) ./notebooks/clustering_factors_cnn_vgg.ipynb ./notebooks/clustering_factors_cnn_vgg.ipynb

clustering-factors-imagenet: mkdir-results
	$(PAPERMILL) ./notebooks/clustering_factors_imagenet.ipynb ./notebooks/clustering_factors_imagenet.ipynb

plotting-main: mkdir-results
	$(PAPERMILL) ./notebooks/plotting_main.ipynb ./notebooks/plotting_main.ipynb

l1reg-clustering: mkdir-results
	$(PAPERMILL) ./notebooks/l1reg_clustering.ipynb ./notebooks/l1reg_clustering.ipynb

activations-tables: mkdir-results
	$(PAPERMILL) ./notebooks/activations_tables.ipynb ./notebooks/activations_tables.ipynb

combined-factors: mkdir-results
	$(PAPERMILL) ./notebooks/combined_factors.ipynb ./notebooks/combined_factors.ipynb

random-init-ncuts: mkdir-results
	$(PAPERMILL) ./notebooks/random_init_ncuts.ipynb ./notebooks/random_init_ncuts.ipynb
