"""write meeee
"""
import optimus

# --------------------
# 1. Create Components
# --------------------
# Inputs
# - - - - - - -
input_data = optimus.Input(
    name='images',
    shape=(None, 1, 28, 28))

class_labels = optimus.Input(
    name='labels',
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
    weight_shape=(15, 9, 9),
    pool_shape=(2, 2),
    act_type='relu')

affine = optimus.Affine(
    name='affine',
    input_shape=conv.output.shape,
    output_shape=(512,),
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

total_loss = optimus.Accumulator(
    name="total_loss")

# Outputs
# - - - - - - -
posterior = optimus.Output(
    name='posterior')

affine_out = optimus.Output(
    name='features')

# Assemble everything
canvas = optimus.Canvas(
    inputs=[input_data, class_labels, decay, sparsity, learning_rate],
    nodes=[conv, affine, classifier],
    losses=[nll, conv_decay, affine_sparsity, total_loss],
    outputs=[posterior, affine_out])

# --------------------
# 2. Define Graphs
# --------------------
# Define connection map as (from, to) tuples.
# Note: These should all be Ports.
# - - - - - - - - - - - - - - - - - - - - - -
transform_edges = [
    (input_data, conv.input),
    (conv.output, affine.input),
    (affine.output, classifier.input),
    (affine.output, affine_out),
    (classifier.output, posterior)]

loss_edges = transform_edges + [
    (classifier.output, nll.likelihood),
    (class_labels, nll.target_idx),
    (conv.weights, conv_decay.input),
    (decay, conv_decay.weight),
    (affine.output, affine_sparsity.input),
    (sparsity, affine_sparsity.weight),
    (nll.cost, total_loss.input_list),
    (conv_decay.cost, total_loss.input_list),
    (affine_sparsity.cost, total_loss.input_list)]

# Build the actual functions
# - - - - - - - - - - - - - -
transform = optimus.Graph(
    name='transform',
    canvas=canvas,
    edges=transform_edges,
    outputs=[posterior])

loss = optimus.Graph(
    name='loss',
    canvas=canvas,
    edges=loss_edges,
    outputs=[total_loss.cost])

train = optimus.Graph(
    name='train',
    canvas=canvas,
    edges=loss_edges,
    outputs=[total_loss.cost],
    loss=total_loss.cost,
    # constraints=[optimus.L2UnitNorm(conv.weights)],
    update_param=learning_rate)

# --------------------
# 3. Create Data
# --------------------
dset = optimus.File("/Users/ejhumphrey/Desktop/mnist_train.hdf5")
source = optimus.Queue(dset, batch_size=50, refresh_prob=0)

driver = optimus.Driver(
    name="mnist_training",
    graph=train,
    log_file="trainer.log",
    save_freq=100)

hyperparams = {learning_rate.name: 0.01,
               sparsity.name: 0.01,
               decay.name: 0.01}

driver.fit(source, hyperparams=hyperparams, max_iter=5000)
