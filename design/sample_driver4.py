"""write meeee
"""
import optimus

# --------------------
# 1. Create Components
# --------------------
# Inputs
# - - - - - - -
input_data = optimus.Input(
    name='image',
    shape=(None, 1, 28, 28))

class_labels = optimus.Input(
    name='label',
    shape=(None,),
    dtype='int32')

decay = optimus.Input(
    name='decay_param',
    shape=None)

sparsity = optimus.Input(
    name='sparsity_param',
    shape=None)

learning_rate = optimus.Input(
    name='learning_rate',
    shape=None)

# Nodes
# - - - - - - -
conv = optimus.Conv3D(
    name='conv',
    input_shape=input_data.shape,
    weight_shape=(15, 1, 9, 9),
    pool_shape=(2, 2),
    act_type='relu')

affine = optimus.Affine(
    name='affine',
    input_shape=conv.output.shape,
    output_shape=(None, 512,),
    act_type='relu')

classifier = optimus.Likelihood(
    name='classifier',
    input_shape=affine.output.shape,
    n_out=10,
    act_type='linear')

# Losses
# - - - - -
nll = optimus.NegativeLogLikelihood(
    name="negloglikelihood")

conv_decay = optimus.L2Magnitude(
    name='weight_decay')

affine_sparsity = optimus.L1Magnitude(
    name="feature_sparsity")

# --------------------
# 2. Define Graphs
# --------------------
# Define connection map as (from, to) tuples.
# These should all be Ports; the ConnectionManager handles the abstraction
# - - - - - - - - - - - - - - - - - - - - - -
transform_edges = [
    (input_data, conv.input),
    (conv.output, affine.input),
    (affine.output, classifier.input)]

loss_edges = transform_edges + [
    (classifier.output, nll.likelihood),
    (class_labels, nll.target_idx),
    (conv.weights, conv_decay.input),
    (decay, conv_decay.weight),
    (affine.output, affine_sparsity.input),
    (sparsity, affine_sparsity.weight)]

# Build the actual functions
# - - - - - - - - - - - - - -
train = optimus.Graph(
    name='train',
    inputs=[input_data, class_labels, decay, sparsity, learning_rate],
    nodes=[conv, affine, classifier],
    edges=loss_edges,
    outputs=[optimus.Graph.TOTAL_LOSS],
    losses=[nll, conv_decay, affine_sparsity],
    # constraints=[optimus.L2UnitNorm(conv.weights)],
    update_param=learning_rate)

# Define other functions now for using outside the main driver.
# transform = optimus.Graph(
#     name='transform',
#     inputs=[input_data],
#     nodes=[conv, affine, classifier],
#     edges=transform_edges,
#     outputs=[classifier.output, affine.output])

# loss = optimus.Graph(
#     name='loss',
#     inputs=[input_data, class_labels, decay, sparsity],
#     nodes=[conv, affine, classifier],
#     edges=loss_edges,
#     outputs=[optimus.Graph.TOTAL_LOSS, conv.output, nll.cost, conv_decay.cost],
#     losses=[nll, conv_decay, affine_sparsity])


# --------------------
# 3. Create Data
# --------------------
dset = optimus.File("/Users/ejhumphrey/Desktop/mnist_train.hdf5")
source = optimus.Queue(dset, batch_size=50, refresh_prob=0)

driver = optimus.Driver(
    name="mnist_training",
    graph=train,
    output_directory="/Volumes/megatron/optimus/",
    log_file="trainer.log")

hyperparams = {learning_rate.name: 0.01,
               sparsity.name: 0.01,
               decay.name: 0.01}

driver.fit(source, hyperparams=hyperparams, max_iter=5000, save_freq=100)

optimus.save(transform, "/Volumes/megatron/optimus/")
optimus.save(loss, "/Volumes/megatron/optimus/")
