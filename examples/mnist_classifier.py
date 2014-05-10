"""write meeee"""
import optimus
from optimus.examples.mnist import load_mnist

# 1.1 Create Inputs
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

# 1.2 Create Nodes
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

classifier = optimus.Softmax(
    name='classifier',
    input_shape=affine.output.shape,
    n_out=10,
    act_type='linear')

# 1.1 Create Losses
nll = optimus.NegativeLogLikelihood(
    name="negloglikelihood")

conv_decay = optimus.L2Magnitude(
    name='weight_decay')

affine_sparsity = optimus.L1Magnitude(
    name="feature_sparsity")

# 2. Define Edges
connect_manager = optimus.ConnectionManager([
    (input_data, conv.input),
    (conv.output, affine.input),
    (affine.output, classifier.input),
    (classifier.output, nll.likelihood),
    (class_labels, nll.target_idx),
    (conv.weights, conv_decay.input),
    (decay, conv_decay.weight),
    (affine.output, affine_sparsity.input),
    (sparsity, affine_sparsity.weight)])

update_manager = optimus.ConnectionManager([
    (learning_rate, conv.weights),
    (learning_rate, conv.bias),
    (learning_rate, affine.weights),
    (learning_rate, affine.bias),
    (learning_rate, classifier.weights),
    (learning_rate, classifier.bias)])

train = optimus.Graph(
    name='mnist_3layer',
    inputs=[input_data, class_labels, decay, sparsity, learning_rate],
    nodes=[conv, affine, classifier],
    connections=connect_manager.connections,
    outputs=[optimus.Graph.TOTAL_LOSS, classifier.output],
    losses=[nll, conv_decay, affine_sparsity],
    updates=update_manager.connections)

optimus.random_init(classifier.weights)

# 3. Create Data
dset = load_mnist("/Users/ejhumphrey/Desktop/mnist.pkl")[0]
source = optimus.Queue(dset, batch_size=50, refresh_prob=0.0, cache_size=5000)

driver = optimus.Driver(graph=train, name='example_classifier')

hyperparams = {
    learning_rate.name: 0.02,
    sparsity.name: 0.001,
    decay.name: 0.1}

driver.fit(source, hyperparams=hyperparams, max_iter=5000, save_freq=25)
