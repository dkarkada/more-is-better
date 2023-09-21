import numpy as np

import neural_tangents as nt
from neural_tangents import stax

def FC_kernel(depth, batchsz=20):
    width = 1 # irrelevant to kernel calculation, so just set to 1
    layers = [stax.Dense(width, W_std=np.sqrt(2), b_std=0), stax.Relu()] * depth
    layers += [stax.Dense(width, W_std=1, b_std=0)]
    _, _, kernel_fn = stax.serial(*layers)
    kernel_fn = nt.batch(kernel_fn, batch_size=batchsz, store_on_device=False)
    return kernel_fn

import functools
def MyrtleNTK(depth, batchsz=20, input_size=32):
    W_std, b_std = np.sqrt(1), 0.
    architectures = {5: [2, 1, 1], 7: [2, 2, 2], 10: [3, 3, 3]}
    layer_nums = architectures[depth]
    width = 1 # irrelevant to kernel calculation, so just set to 1
    activ = stax.Relu()
    layers = []
    num_final_pools = int(np.log2(input_size)) - 2
    assert num_final_pools >= 0
    conv = functools.partial(stax.Conv, W_std=W_std, b_std=b_std, padding='SAME')

    layers += [conv(width, (3, 3)), activ] * layer_nums[0]
    layers += [stax.AvgPool((2, 2), strides=(2, 2))]
    layers += [conv(width, (3, 3)), activ] * layer_nums[1]
    layers += [stax.AvgPool((2, 2), strides=(2, 2))]
    layers += [conv(width, (3, 3)), activ] * layer_nums[2]
    layers += [stax.AvgPool((2, 2), strides=(2, 2))] * num_final_pools

    layers += [stax.Flatten(), stax.Dense(10, W_std, b_std)]

    _, _, kernel_fn =  stax.serial(*layers)
    kernel_fn = nt.batch(kernel_fn, batch_size=batchsz, store_on_device=False)
    return kernel_fn
