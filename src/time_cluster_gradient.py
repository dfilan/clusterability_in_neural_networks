import time

from src.cluster_gradient import make_eigenval_op

import numpy as np
import tensorflow as tf

input_dim = 100
hidden_dim = 200
output_dim = 10

random_mat_numpy_0 = np.random.rand(input_dim, hidden_dim)
random_mat_numpy_1 = np.random.rand(hidden_dim, hidden_dim)
random_mat_numpy_2 = np.random.rand(hidden_dim, hidden_dim)
random_mat_numpy_3 = np.random.rand(hidden_dim, output_dim)

# eigenvals_numpy, outers, deg_vec = make_eigenval_function(3)(random_mat_numpy_1,
#                                                              random_mat_numpy_2,
#                                                              random_mat_numpy_3)

# fake_dy = eigenvals_numpy
# my_grad_np = grad_comp_np(fake_dy, outers, deg_vec, [random_mat_numpy_1,
#                                                      random_mat_numpy_2,
#                                                      random_mat_numpy_3])
# print("eigenvals shape:", eigenvals_numpy.shape)
# print("my_grad_np shapes:", [x.shape for x in my_grad_np])

random_mat_0 = tf.Variable(random_mat_numpy_0, dtype=tf.float32,
                           name="variable_0")
random_mat_1 = tf.Variable(random_mat_numpy_1, dtype=tf.float32,
                           name="variable_1")
random_mat_2 = tf.Variable(random_mat_numpy_2, dtype=tf.float32,
                           name="variable_2")
random_mat_3 = tf.Variable(random_mat_numpy_3, dtype=tf.float32,
                           name="variable_3")
top_3_eigenvals = make_eigenval_op(3, 1)(
    random_mat_0, random_mat_1, random_mat_2, random_mat_3
)
# print_eigs_shape = tf.print("shape of top_3_eigenvals",
#                             tf.shape(top_3_eigenvals),
#                             output_stream=sys.stdout)
# print(top_3_eigenvals.get_shape().as_list())
# with tf.control_dependencies([print_eigs_shape]):
eigenval_sum = tf.reduce_sum(top_3_eigenvals)
optimiser = tf.compat.v1.train.GradientDescentOptimizer(1.0)
train = optimiser.minimize(eigenval_sum)

init = tf.initializers.global_variables()

with tf.Session() as sess:
    sess.run(init)
    print("initial eigenval_sum:", eigenval_sum.eval())
    for step in range(3):
        sess.run(train)
        print("step:", step)
        print("eigenval_sum:", eigenval_sum.eval())
