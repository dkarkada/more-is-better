import torch
import pickle
import os
import numpy as np

import neural_tangents as nt
from neural_tangents import stax

from jax import jit

def eigendecomp(load_dir, filename):
    
    with open(f"{load_dir}/{filename}.file", 'rb') as f:
        kernel_data = pickle.load(f)
    K, y = kernel_data['K'], kernel_data['y']
    n = K.shape[0]
    assert K.shape == (n, n)
    assert y.shape[0] == n and y.ndim == 2
    print(f"eigendecomposing n={n} {filename}... ", end='')
    K = torch.from_numpy(K).cuda()
    y = torch.from_numpy(y).cuda()
    eigvals, eigvecs = torch.linalg.eigh(K)
    eigvals = eigvals.cpu().numpy()

    # eigvals are now on CPU; eigvecs still on device
    eigvals /= n
    eigvecs *= np.sqrt(n)

    eigcoeffs = (1/n) * eigvecs.T @ y.type(torch.float)
    eigvecs = eigvecs.cpu().numpy()
    eigcoeffs = eigcoeffs.cpu().numpy()
    
    # Sort in descending eigval order
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[::-1]
    eigcoeffs = eigcoeffs[::-1]

    eigendata = {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "eigcoeffs": eigcoeffs
    }
    with open(f"{load_dir}/{filename}--eigendata.file", 'wb') as f:
        pickle.dump(eigendata, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("done")

    del K, y, eigvals, eigvecs, eigcoeffs
    torch.cuda.empty_cache()


def save_kernel(K, y, filename, kernel_dir):
    data = {
        'K': K,
        'y': y
    }
    with open(f"{kernel_dir}/{filename}.file", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def compute_FCNTK(X, nHL=2):
    width = 1 # irrelevant to kernel calculation, so just set to 1
    layers = [stax.Dense(width, W_std=np.sqrt(2), b_std=0.1), stax.Relu()] * nHL
    layers += [stax.Dense(width, W_std=1, b_std=0)]
    _, _, kernel_fn = stax.serial(*layers)
    # kernel_fn = jit(kernel_fn, static_argnames='get')
    kernel_fn = nt.batch(kernel_fn, batch_size=10, store_on_device=False)
    K = kernel_fn(X, get='ntk').block_until_ready()
    return np.array(K)


def compute_NNGPK(X, nHL=1, nonlin="relu"):
    width = 1 # irrelevant to kernel calculation, so just set to 1
    assert nonlin in ["relu", "erf"]
    activ = stax.Relu() if nonlin == "relu" else stax.Erf()
    layers = [stax.Dense(width, W_std=np.sqrt(2), b_std=0), activ] * nHL
    layers += [stax.Dense(width, W_std=1, b_std=0)]
    _, _, kernel_fn = stax.serial(*layers)
    # kernel_fn = jit(kernel_fn, static_argnames='get')
    kernel_fn = nt.batch(kernel_fn, batch_size=10, store_on_device=False)
    K = kernel_fn(X, get='nngp').block_until_ready()
    return np.array(K)

import functools
def compute_MyrtleNTK(X, depth):
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
    # kernel_fn = jit(kernel_fn, static_argnames='get')
    kernel_fn = nt.batch(kernel_fn, batch_size=10, store_on_device=False)
    K = kernel_fn(X, get='ntk').block_until_ready()
    return np.array(K)

# myrtle cntk loader wants these
from tensorflow.io import gfile
import itertools
import concurrent.futures

## MYRTLE CNTK FETCH AUTHENTICATION
from google.colab import auth
try:
    auth.authenticate_user()
except:
    print('Colab authentication failed!!')
    
def fetch_Myrtle_NTK(n):
    filedir = 'gs://neural-tangents-kernels/infinite-uncertainty/kernels/myrtle-10/clean'
    assert gfile.exists(filedir), f"File path {filedir} doesn't exist"
    filepath = os.path.join(filedir, 'ntk')

    assert n % 5000 == 0
    max_idx = int(np.ceil(n / 5000))
    all_idxs = list(
        filter(
            lambda x: x[0] <= x[1],
            itertools.product(range(max_idx), range(max_idx))))
    K = np.zeros(shape=(n, n), dtype=np.float32)

    def _update_kernel_from_indices(index):
        row, col = index
        # Loading based on 60k x 60k matrix
        with gfile.GFile(f'{filepath}-{row}-{col}', 'rb') as f:
            val = np.load(f).astype(np.float32)

        K[row*5000:(row+1)*5000, col*5000:(col+1)*5000] = val
        if col > row:
            K[col*5000:(col+1)*5000, row*5000:(row+1)*5000] = val.T

    with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
        executor.map(_update_kernel_from_indices, all_idxs)

    # load labels
    with gfile.GFile(os.path.join(filedir, 'labels'), 'rb') as f:
        labels = np.load(f)
    # change encoding from zero-mean [-0.1, 0.9] to one-hot [0, 1]
    y = labels[:max_idx*5000].round()
    return K, y