import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import jit

import torch
import torch.nn.functional as F
import torchvision

import neural_tangents as nt
from neural_tangents import stax

import os
import sys

import subprocess
if os.path.isdir('more-is-better'):
    print('deleting old repo... ', end='')
    subprocess.run(["rm", "-rf", "more-is-better"])
subprocess.run(["git", "clone", "-q", "https://github.com/dkarkada/more-is-better.git"])
sys.path.insert(0,'more-is-better')
print('cloned repo')

from kernels import compute_MyrtleNTK
from ImageData import ImageData
from utils import save, load

EXPT_NUM = 1
DO_50K = False

cifar10 = ImageData('cifar10')

if EXPT_NUM == 1:
    expt = "cntk5-clean"
    dataset = cifar10.get_dataset(50000, flatten=False)
    msg = "Myrtle depth-5 CNTK @ vanilla CIFAR10"

kernel_dir = "/scratch/bbjr/dkarkada/kernel_matrices"
work_dir = f"{kernel_dir}/{expt}"
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

# 20k matrix, 50k matrix, done flags for each (per block), dataset, expt description
# K blocked in 5k chunks

metadata = load(f"{work_dir}/metadata.file")
if metadata is None:
    metadata = {
        "flags_20k": np.zeros(4, 4),
        "flags_50k": np.zeros(10, 10),
        "dataset": dataset,
        "msg": msg
    }
    save(metadata, f"{work_dir}/metadata.file")
    with open("readme.txt", 'w') as f:
        f.write(msg)

def set_block(K, block, idx, fn):
    i, j = idx
    K[i*5000:(i+1)*5000, j*5000:(j+1)*5000] = block
    if i != j:
        K[j*5000:(j+1)*5000, i*5000:(i+1)*5000] = block.T
    save(K, fn)

    n = K.shape[0]
    metadata[f"flags_{n//1000}k"][i, j] = 1
    metadata[f"flags_{n//1000}k"][j, i] = 1
    save(metadata, f"{work_dir}/metadata.file")

    print(f"set block ({i}, {j}) of {n//1000}k kernel.")

n = 50000 if DO_50K else 20000
K_fn = f"{work_dir}/CNTK_{n//1000}k.npy"
K = load(K_fn)
if K is None:
    K = np.zeros(n, n)
    save(K, K_fn)
assert K.shape == (n, n)

# copy results from other kernel matrix, if it exists
n_other = 20000 if DO_50K else 50000
flags_other = metadata[f"flags_{n_other//1000}k"]
# check that the other kernel matrix is even partially computed
if flags_other.any():
    K_other = load(f"{work_dir}/CNTK_{n_other//1000}k.npy")
    assert K_other is not None
    for (i, j), done_other in np.ndenumerate(flags_other):
        # check that block is within bounds in this matrix (e.g. if other is larger)
        if i*5000<n and j*5000<n:
            done = metadata[f"flags_{n//1000}k"][i, j]
            # check that other block is computed AND this one is not. Then copy
            if done_other and not done:
                block = K_other[i*5000:(i+1)*5000, j*5000:(j+1)*5000]
                set_block(K, block, (i, j), K_fn)

# iterate over blocks
X_full, _ = metadata["dataset"]
flags = metadata[f"flags_{n//1000}k"]
for (i, j), done in np.ndenumerate(flags):
    if done or i > j:
        continue
    print(f"starting ({i}, {j}) block... ", end='')
    X_i = X_full[i*5000:(i+1)*5000]
    X_j = X_full[j*5000:(j+1)*5000]
    # check code here
    block = compute_MyrtleNTK()
    set_block(K, block, (i, j), K_fn)
    assert metadata[f"flags_{n//1000}k"][i, j] == 1
    assert metadata[f"flags_{n//1000}k"][j, i] == i
    print()
    print()
