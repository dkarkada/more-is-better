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

kernel_dir = "/scratch/bbjr/dkarkada/kernel_matrices"
cifar10 = ImageData('cifar10')

def get_expt(expt_num):
    if expt_num == 1:
        expt = "cntk5-clean"
        dataset = cifar10.get_dataset(50000, flatten=False)
        msg = "Myrtle depth-5 CNTK @ vanilla CIFAR10"
    return expt, dataset, msg

expt, dataset, msg = get_expt(1)
work_dir = f"{kernel_dir}/{expt}"
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

# 20k matrix, 50k matrix, done flags for each (per block), dataset, expt description
# K blocked in 5k chunks

metadata = load(f"{work_dir}/metadata.file")
if metadata is None:
    metadata = {
        "20k_flags": np.zeros(4, 4),
        "50k_flags": np.zeros(10, 10),
        "dataset": dataset,
        "msg": msg
    }
    save(metadata, f"{work_dir}/metadata.file")
    with open("readme.txt", 'w') as f:
        f.write(msg)
    
K_20k = load(f"{work_dir}/CNTK_20k.npy")
if K_20k is None:
    K_20k = np.zeros(20000, 20000)
    save(K_20k, f"{work_dir}/CNTK_20k.npy")
assert K_20k.shape == (20000, 20000)
    
K_50k = load(f"{work_dir}/CNTK_50k.npy")
if K_50k is None:
    K_50k = np.zeros(50000, 50000)
    save(K_50k, f"{work_dir}/CNTK_50k.npy")
assert K_50k.shape == (50000, 50000)

# iterate over blocks