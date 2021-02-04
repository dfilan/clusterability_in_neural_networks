import sys
import itertools
import time

from pathos.multiprocessing import ProcessPool

from src.spectral_cluster_model import (weights_to_graph,
                                        delete_isolated_ccs_refactored)

import numpy as np
import tensorflow as tf
import scipy.sparse
from scipy.sparse.linalg import eigsh

def adj_to_laplacian_and_degs(adj_mat_csr):
    """
    Takes in:
    adj_mat_csr, a sparse adjacency matrix in CSR format
    Returns:
    a tuple of two elements:
    the first element is the normalised laplacian matrix Lsym in CSR format
    the second element is the degree vector as a numpy array.
    """
    num_rows = adj_mat_csr.shape[0]
    degree_vec = np.squeeze(np.asarray(adj_mat_csr.sum(axis=0)))
    inv_sqrt_degrees = np.reciprocal(np.sqrt(degree_vec))
    inv_sqrt_deg_col = np.expand_dims(inv_sqrt_degrees, axis=1)
    inv_sqrt_deg_row = np.expand_dims(inv_sqrt_degrees, axis=0)
    result = adj_mat_csr.multiply(inv_sqrt_deg_row)
    result = result.multiply(inv_sqrt_deg_col)
    return scipy.sparse.identity(num_rows, format='csr') - result, degree_vec


def get_dy_dW_aux(layer, mat_list, degree_list, widths, pre_sums, dy_dL):
    mat = mat_list[layer]
    grad = np.zeros(shape=mat.shape)
    # grad will be the gradient of the eigenvalue sum with respect to mat
    # we will fill in the elements of grad one-by-one in the following loop
    # (and then a global multiplication afterwards)
    for m,n in itertools.product(range(widths[layer]),
                                 range(widths[layer + 1])):
        # NB: this is two nested for loops. But m_contributions doesn't depend
        # on n, and similarly n_contributions doesn't depend on m.
        # so refactoring this loop will cause a massive speedup.
        # I, Daniel Filan, didn't do this myself because by the time I noticed
        # this problem, I was moving to a different codebase.
        e_m = pre_sums[layer] + m # this is the index of neuron m in dy_dL
        e_n = pre_sums[layer + 1] + n # same for neuron n
        # e is for "embedding" in the above variable names

        # there are four batches of contributions: neurons that feed to m,
        # neurons that m feeds to, neurons that feed to n, and neurons that
        # n feeds to. The calculation deals with the first two batches, then
        # the next two batches.
        m_contributions = 0

        if layer != 0:
            abs_weights_to = np.abs(mat_list[layer - 1][:, m])
            degrees_prev_layer = degree_list[
                pre_sums[layer - 1] : pre_sums[layer]
            ] ** (-0.5)
            dy_dL_terms = dy_dL[e_m, pre_sums[layer - 1] : pre_sums[layer]]
            x = np.multiply(abs_weights_to, degrees_prev_layer)
            m_contributions += 0.5 * np.dot(x, dy_dL_terms)

        abs_weights_from = np.abs(mat[m,:])
        degrees_next_layer = degree_list[
            pre_sums[layer + 1] : pre_sums[layer + 2]
        ] ** (-0.5)
        dy_dL_terms = dy_dL[e_m, pre_sums[layer + 1] : pre_sums[layer + 2]]
        x = np.multiply(abs_weights_from, degrees_next_layer)
        m_contributions += 0.5 * np.dot(x, dy_dL_terms)
        m_contributions *= degree_list[e_m] ** (-1.5)

        # the next two batches:
        n_contributions = 0

        abs_weights_to = np.abs(mat[:,n])
        degrees_prev_layer = degree_list[
            pre_sums[layer] : pre_sums[layer + 1]
        ] ** (-0.5)
        dy_dL_terms = dy_dL[e_n, pre_sums[layer] : pre_sums[layer + 1]]
        x = np.multiply(abs_weights_to, degrees_prev_layer)
        n_contributions += 0.5 * np.dot(x, dy_dL_terms)

        if layer + 3 < len(pre_sums):
            abs_weights_from = np.abs(mat_list[layer + 1][n,:])
            degrees_next_layer = degree_list[
                pre_sums[layer + 2] : pre_sums[layer + 3]
            ] ** (-0.5)
            dy_dL_terms = dy_dL[
                e_n, pre_sums[layer + 2] : pre_sums[layer + 3]
            ]
            x = np.multiply(abs_weights_from, degrees_next_layer)
            n_contributions += 0.5 * np.dot(x, dy_dL_terms)

        n_contributions *= degree_list[e_n] ** (-1.5)

        gradient_term = m_contributions + n_contributions
        gradient_term -= (((degree_list[e_m] * degree_list[e_n]) ** (-0.5))
                          * dy_dL[e_m, e_n])
        grad[m,n] = gradient_term
    # now, we need to point-wise multiply with sign(mat) to back-prop thru
    # the absolute value function that takes the weights to the adj mat.
    grad = np.multiply(grad, np.sign(mat))
    return grad.astype(np.float32)

def get_dy_dW_np(degree_list, mat_list, dy_dL, num_workers=1):
    """
    Takes in: 
    degree_list (array-like), which is an array of degrees of each node.
    adj_mat (sparse array), the adjacency matrix. should have shape 
    (len(degree_list), len(degree_list)).
    mat_list, a list of numpy arrays of the weight matrices
    dy_dL (rank 2 np ndarray), a num_eigenvalues * len(degree_list) 
    * len(degree_list) tensor of the derivatives of the eigenvalue with respect 
    to the laplacian entries
    Returns:
    grad_list, an array of numpy arrays representing the derivatives of the 
    eigenvalues with respect to the individual weights.
    """
    # we're going to be dividing by degrees later, so none of them can be zero
    assert np.all(degree_list != 0), "Some degrees were zero in get_dy_dW_np!"
    widths = [mat.shape[0] for mat in mat_list]
    widths.append(mat_list[-1].shape[1])
    cumulant = np.cumsum(widths)
    pre_sums = np.insert(cumulant,0,0)
    # pre_sums[i] is the number of neurons before layer i
    num_neurons = cumulant[-1]
    num_mats = len(mat_list)
    assert num_neurons == len(degree_list), "Different ways of reckoning the number of neurons give different results"
    assert num_neurons == dy_dL.shape[0], "Different ways of reckoning the number of neurons give different results"
    assert num_neurons == dy_dL.shape[1], "Different ways of reckoning the number of neurons give different results"
    with ProcessPool(nodes=num_workers) as p:
        grad_list = p.map(get_dy_dW_aux,
                          range(num_mats),
                          [mat_list] * num_mats,
                          [degree_list] * num_mats,
                          [widths] * num_mats,
                          [pre_sums] * num_mats,
                          [dy_dL] * num_mats)
    return grad_list


def invert_layer_masks_np(mat, mask_rows, mask_cols):
    """
    Takes a numpy array mat, and two lists of booleans.
    Returns a numpy array which, if masked by the lists, would produce 
    the input. The entries that would be masked are input as 0.0.
    """
    assert mat.shape[0] == len(list(filter(None, mask_rows)))
    assert mat.shape[1] == len(list(filter(None, mask_cols)))
    for (row, mask_bool) in enumerate(mask_rows):
        if not mask_bool:
            mat = np.insert(mat, row, 0.0, axis=0)
    for (col, mask_bool) in enumerate(mask_cols):
        if not mask_bool:
            mat = np.insert(mat, col, 0.0, axis=1)
    return mat


# now: to glue it all together
def make_eigenval_function(num_eigs):
    def top_k_eigenvals_etc(*args):
        w_mat_array = [mat for mat in args]
        adj_mat_csr = weights_to_graph(w_mat_array)
        my_tup = delete_isolated_ccs_refactored(w_mat_array, adj_mat_csr)
        w_mat_unified, adj_mat_unified, _, layer_masks = my_tup
        lap_mat_csr, degree_vec = adj_to_laplacian_and_degs(adj_mat_unified)
        evals, evecs = eigsh(lap_mat_csr, num_eigs + 1, sigma=-1.0, which='LM')
        evecs = np.transpose(evecs)
        # ^ makes evecs (num eigenvals) * (size of lap mat)
        outers = []
        for i in range(num_eigs):
            outers.append(np.outer(evecs[i+1], evecs[i+1]))
        assert len(w_mat_unified) == len(layer_masks)
        zipped_thing = []
        for i in range(len(w_mat_unified)):
            zipped_thing += [w_mat_unified[i], layer_masks[i]]
        return_list = [
            evals[1:num_eigs+1].astype(dtype=np.float32), # eigenvalues
            np.array(outers, dtype=np.float32), # outer product of each
                                                # eigenvalue with itself
            degree_vec.astype(dtype=np.float32) # vector of degrees
        ]
        for i, thing in enumerate(zipped_thing):
            if i % 2 == 0:
                return_list.append(np.array(thing, dtype=np.float32))
                # weight matrix with isolated ccs removed
            else:
                return_list.append(np.array(thing, dtype=np.bool))
                # node mask that turns original weight matrix into
                # w_mat_unified
        return tuple(return_list)
    return top_k_eigenvals_etc


def make_grad_comp_function(num_workers=1):
    def grad_comp_np(dy, outers, degree_vec, *layer_mat_zip_thing):
        mat_list = []
        layer_masks = []
        for i, obj in enumerate(layer_mat_zip_thing):
            if i % 2 == 0:
                mat_list.append(obj)
            else:
                layer_masks.append(obj)
        adj_mat_csr = weights_to_graph(mat_list)
        dy_dL = np.tensordot(dy, outers, [[0], [0]])
        penult_grad = get_dy_dW_np(degree_vec, mat_list, dy_dL, num_workers)
        assert len(layer_masks) == len(penult_grad), "penult_grad different length than expected"
        layer_masks.append([True] * (penult_grad[-1].shape[1]))
        final_grad = []
        for (i, grad) in enumerate(penult_grad):
            fat_grad = invert_layer_masks_np(
                grad, layer_masks[i], layer_masks[i+1]
            )
            final_grad.append(fat_grad)
        return final_grad
    return grad_comp_np

def make_eigenval_op(num_eigs, num_workers=1):
    @tf.custom_gradient
    def top_k_eigenvals_op(*args):
        tensor_list = [arg for arg in args]
        num_tensors = len(tensor_list)
        big_tup = tf.numpy_function(
            make_eigenval_function(num_eigs),
            tensor_list,
            ([tf.float32, tf.float32, tf.float32]
             + ([tf.float32, tf.bool] * num_tensors)
            ),
            name="eigenval_comp"
        )
        evals  = big_tup[0]
        outers = big_tup[1]
        degs   = big_tup[2]
        big_zip_thing = big_tup[3:]
        def grad(dy):
            return tf.numpy_function(
                make_grad_comp_function(num_workers),
                [dy, outers, degs] + big_zip_thing,
                [tf.float32] * len(tensor_list),
                name="gradient_comp"
            )
        return evals, grad
    return top_k_eigenvals_op

