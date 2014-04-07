"""write meeee
"""

import optimus


driver = optimus.Driver()

x_in = driver.inputs.create(
    name='x_in',
    shape=(1, 28, 28))

conv0 = driver.nodes.add(
    name='conv0',
    node=optimus.network.Conv3D.simple(
        input_shape=driver.inputs.x_in.shape,
        weight_shape=(15, 9, 9),
        pool_shape=(2, 2),
        act_type='relu')
    )

affine0 = optimus.network.Affine.simple(
    input_shape=conv0.outputs.z_out.shape,
    output_shape=(512,),
    act_type='relu')

softmax0 = optimus.network.Softmax.simple(
    input_shape=(512,),
    n_out=10,
    act_type='linear')

posterior = driver.create_output(name='posterior')
affine0_out = driver.create_output(name='affine0_out')


# Define edges as (from, to) tuples.
# - - - - - - - - - - - - - - - - - -
edges = [(x_in, conv0.inputs.x_in),
         (conv0.outputs.z_out, affine0.inputs.x_in),
         (affine0.outputs.z_out, softmax0.inputs.x_in),
         (affine0.outputs.z_out, affine0_out),
         (softmax0.outputs.z_out, posterior)]

# Define losses over the network.
# - - - - - - - - - - - - - - - -
class_idx = optimus.network.Input('class_idx', shape=[])
nll = optimus.network.NegativeLogLikelihood(
    posterior=posterior,
    target_idx=class_idx)

conv0_lambda = optimus.network.Input('conv0_lambda', shape=None)
conv0_decay = optimus.network.L2Norm(
    variable=conv0.params.weight,
    weight=conv0_lambda)

affine0_lambda = optimus.network.Input('affine0_lambda', shape=None)
affine0_sparsity = optimus.network.L1Norm(
    variable=affine0_out,
    weight=affine0_lambda)

losses = [nll, conv0_decay, affine0_sparsity]

# conv_norm = optimus.UnitL2Norm('conv0.weights')
# constraints = optimus.Constraints([conv_norm], graph.params)

