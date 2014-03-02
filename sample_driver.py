"""write meeee
"""
import optimus

# --------------------
# 1. Create Components
# --------------------
# Inputs
# - - - - - - -
input_data = optimus.Input(
    name='input_data',
    shape=(1, 28, 28))

class_labels = optimus.Input(
    name='class_labels',
    shape=[],
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

# Outputs
# - - - - - - -
posterior = optimus.Output(
    name='posterior')

affine_out = optimus.Output(
    name='features')

# Assemble everything
modules = optimus.Canvas(
    inputs=[input_data, class_labels, decay, sparsity, learning_rate],
    nodes=[conv, affine, classifier],              # Differentiable
    losses=[nll, conv_decay, affine_sparsity],     # Non-differentiable
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
    (decay, conv_decay.parameter),
    (affine.output, affine_sparsity.input),
    (sparsity, affine_sparsity.parameter)]

# Build the actual functions
# - - - - - - - - - - - - - -
transform = optimus.Graph(
    name='transform',
    modules=modules,
    edges=transform_edges,
    outputs=[posterior.output])

loss = optimus.Graph(
    name='loss',
    modules=modules,
    edges=loss_edges,
    losses=[nll.cost],  # Losses are a collection of Ports
    outputs="losses")

train = optimus.Graph(
    name='train',
    modules=modules,
    edges=loss_edges,
    losses=[nll.cost, conv_decay.cost, affine_sparsity.cost],
    constraints=[optimus.UnitL2Norm(conv.weights)],
    outputs="losses",
    updates=[(param, learning_rate) for param in modules.params])

# --------------------
# 3. Create Data
# --------------------
fh_train = optimus.File("/Users/ejhumphrey/Desktop/mnist_train.hdf5")
train_source = optimus.Factory(fh_train, batch_size=50, refresh_prob=0)

train_inputs = optimus.DataServer([
    optimus.Variable(
        input_data, value=train_source.values, update=train_source.buffer),
    optimus.Variable(class_labels, value=train_source.labels),
    optimus.Constant(sparsity, value=0),
    optimus.Constant(decay, value=0),
    optimus.Constant(learning_rate, value=0.01)])

trainer = optimus.Driver(
    name="mnist_training",
    graph=train,
    data=train_inputs)

fh_valid = optimus.File("/Users/ejhumphrey/Desktop/mnist_valid.hdf5")
valid_source = optimus.Factory(fh_valid, batch_size=50, refresh_prob=0)

inputs_valid = optimus.DataServer([
    optimus.Variable(
        input_data, value=valid_source.values, update=valid_source.buffer),
    optimus.Variable(class_labels, value=valid_source.labels)])

validator = optimus.Driver(
    name="mnist_validation",
    graph=loss,
    data=inputs_valid)

trainer.run(max_iter=5000, print_freq=25)
