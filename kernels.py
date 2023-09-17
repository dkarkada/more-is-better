import torch
import pickle
import os
import numpy as np

import neural_tangents as nt
from neural_tangents import stax

# need to use numpy for cpu eigendecomp -- pytorch 32bit-interface can't handle large matrices 
# see https://github.com/pytorch/pytorch/issues/92141


def _gpu_eigd(K, y, n):
    # K, y numpy arrays
    K = torch.from_numpy(K).cuda()
    y = torch.from_numpy(y).cuda()
    eigvals, eigvecs = torch.linalg.eigh(K)
    eigvals = eigvals.cpu().numpy()

    # eigvals are now on cpu; eigvecs still on gpu
    eigvals /= n
    eigvecs *= np.sqrt(n)

    eigcoeffs = (1/n) * eigvecs.T @ y
    eigvecs = eigvecs.cpu().numpy()
    eigcoeffs = eigcoeffs.cpu().numpy()
    torch.cuda.empty_cache()
    return eigvals, eigvecs, eigcoeffs


def _cpu_eigd(K, y, n):
    # K, y numpy arrays
    eigvals, eigvecs = np.linalg.eigh(K)

    eigvals /= n
    eigvecs *= np.sqrt(n)

    eigcoeffs = (1/n) * eigvecs.T @ y
    return eigvals, eigvecs, eigcoeffs


def eigendecomp(K, y):
    n = K.shape[0]
    assert K.shape == (n, n)
    assert y.shape[0] == n and y.ndim == 2
    K, y = K.astype(np.float64), y.astype(np.float64)
    eigd = _gpu_eigd if n <= 20000 else _cpu_eigd
    eigvals, eigvecs, eigcoeffs = eigd(K, y, n)
    
    # Sort in descending eigval order
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[::-1]
    eigcoeffs = eigcoeffs[::-1]
    
    return eigvals, eigvecs, eigcoeffs

def FC_kernel(depth, batchsz=20):
    width = 1 # irrelevant to kernel calculation, so just set to 1
    layers = [stax.Dense(width, W_std=np.sqrt(2), b_std=0), stax.Relu()] * depth
    layers += [stax.Dense(width, W_std=1, b_std=0)]
    _, _, kernel_fn = stax.serial(*layers)
    kernel_fn = nt.batch(kernel_fn, batch_size=batchsz, store_on_device=False)
    return kernel_fn

import functools
def MyrtleNTK(depth, batchsz=20):
    W_std, b_std = np.sqrt(1), 0.
    architectures = {5: [2, 1, 1], 7: [2, 2, 2], 10: [3, 3, 3]}
    layer_nums = architectures[depth]
    width = 1 # irrelevant to kernel calculation, so just set to 1
    activ = stax.Relu()
    layers = []
    conv = functools.partial(stax.Conv, W_std=W_std, b_std=b_std, padding='SAME')

    layers += [conv(width, (3, 3)), activ] * layer_nums[0]
    layers += [stax.AvgPool((2, 2), strides=(2, 2))]
    layers += [conv(width, (3, 3)), activ] * layer_nums[1]
    layers += [stax.AvgPool((2, 2), strides=(2, 2))]
    layers += [conv(width, (3, 3)), activ] * layer_nums[2]
    layers += [stax.AvgPool((2, 2), strides=(2, 2))] * 3

    layers += [stax.Flatten(), stax.Dense(10, W_std, b_std)]

    _, _, kernel_fn =  stax.serial(*layers)
    kernel_fn = nt.batch(kernel_fn, batch_size=batchsz, store_on_device=False)
    return kernel_fn
