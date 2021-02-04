import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from tensorflow_model_optimization.sparsity import keras as sparsity
from src.utils import load_model2
import pdb
import cv2

# TODOs (or, at least, things Daniel thinks would be nice to have done):
# - Actually have code for getting a cluster mask from a model (maybe this
#   involves storing the results of clustering on S3).
# - Instead of optimising the image, let image = tanh(x), and optimise x.
#   Then, instead of doing projected gradient ascent, you can do normal
#   gradient ascent, and maybe it works better (I got this idea from Sam Toyer)
# - Regularise the images to be 'natural' - various ways of doing this.
# - Instead of producing one image, produce n images that all maximise
#   activations but also are diverse.

# Relevant background reference: distill.pub/2017/feature-visualization/

# NB: this code is relying on both the weights and the architecture of a
# trained model being stored in the h5 file.
model_path = "./models/10112019/cifar10_mlp_20epochs/cifar10-mlp-pruned.h5"
image_dir = "./images/"
# NB: these paths have to be relative to where the command is run
my_num_iters = 3e4
final_shape = (28,28)

# function that takes a model, layer, and cluster mask, and returns an image
# that maximises the norm of the nodes in the given layer and cluster after
# being run through the given model.
def visualize_cluster_layer(model, layer_name, cluster_mask,
                            num_iters=my_num_iters, print_interval=1e3,
                            step_size=1e-4):
    # adapted from https://keras.io/examples/deep_dream/
    # this is projected gradient ascent
    # start with random image on unit sphere
    # compute gradient of ||cluster_mask . model[layer_name] ||_2^2
    # - which is going to involve defining a loss tensor/layer/whatever
    # update image, then project back onto unit sphere
    # NB: return value is a numpy array

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    my_layer = layer_dict[layer_name]

    # define l2 gain
    mask_tensor = tf.constant(cluster_mask)
    norm = K.sum(K.square(tf.multiply(my_layer.output, mask_tensor)))
    objective = norm / K.prod(K.cast(K.shape(my_layer.output), 'float32'))
    input_grad = K.gradients(objective, model.input)[0]

    # weird keras magic to nicely get our objective value and gradients
    outputs = [objective, input_grad]
    fetch_obj_and_grads = K.function([model.input], outputs)

    # make our initial random image
    input_shape = model.input.shape.as_list()[1:]
    input_shape = [1] + input_shape
    image = np.random.normal(loc=0.0, scale=1.0, size=input_shape)
    image /= np.linalg.norm(image.flatten())

    # making a dummy zero gradient value, so that at every iteration (including
    # the first), we can look at the difference between the new and the 'old'
    # gradient
    grad_val = np.zeros(image.shape)
    
    for i in range(int(num_iters)):
        # we're going to check how much our gradient is changing from step to
        # step for debugging purposes
        old_grad = grad_val
        # actually getting the gradient as a numpy array
        obj_val, grad_val = fetch_obj_and_grads([image])
        delta_grad = grad_val - old_grad
        # updating and projecting the image
        image += step_size * grad_val
        image /= np.linalg.norm(image.flatten())
        
        # print diagnostic info every so often
        if i % print_interval == 0:
            print("Objective value at iteration", i, ":", obj_val)
            grad_norm = np.sqrt(np.sum(grad_val**2))
            print("Norm of gradient:", grad_norm)
            # NB: the norm of the gradient will be big throughout the training
            # run, since it will be pushing the image away from the sphere,
            # since if the image values are big, the activations will be big.
            dot_product = np.dot(grad_val.flatten(), image.flatten())
            sphere_grad = grad_val - dot_product * image
            # thinking of the gradient as a vector based at the image vector,
            # sphere_grad is projecting that vector on the tangent space of the
            # sphere at the image vector. this should basically be the
            # difference between your image vector and the result after you add
            # the gradient and then project back onto the sphere.
            print("Norm of component of gradient that lies along the sphere:",
                  np.sqrt(np.sum(sphere_grad**2)))
            print("Norm of difference in gradient vector from last step to" +
                  " this step:", np.sqrt(np.sum(delta_grad**2)))

    return image

# loads the model. NB: this relies on both the weights and the architecture
# being stored in the h5 file.
my_model = load_model2(model_path)
# parameters to visualize_cluster_layer
my_index = 3
my_name = my_model.layers[my_index].name
output_shape = my_model.layers[my_index].output.shape.as_list()[1:]
# DF didn't actually get around to using real cluster masks.
no_mask = np.ones(output_shape, dtype=np.float32)

my_image = visualize_cluster_layer(my_model, my_name, no_mask)

# utility function to convert the image into something suitable for a png file
def normalize_image(image):
    image -= np.amin(image)
    image /= np.amax(image)
    image = (255 * image).astype('uint8')
    return image

# save the image in png format.
normalized_image = normalize_image(my_image[0])
file_name = ("dreamed_image_projected_" + str(my_num_iters) + "_iters_layer_"
             + str(my_index) + ".png")
image_loc = image_dir + file_name
cv2.imwrite(image_loc, np.reshape(normalized_image, final_shape))
print("Wrote image to " + image_loc)
