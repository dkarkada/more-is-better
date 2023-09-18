import numpy as np
from jax import numpy as jnp

import os
import sys
import time

start_time = time.time()

sys.path.insert(0, 'more-is-better')

from kernels import MyrtleNTK
from imagedata import ImageData, blockwise_shuffle, shuffle_frac
from utils import save, load

args = sys.argv

EXPT_NUM = int(args[1])
print(f"Running experiment {EXPT_NUM}")

cifar10 = ImageData('cifar10')
RNG = np.random.default_rng(seed=42)

if EXPT_NUM == 0:
    expt = "cntk5-clean"
    dataset = cifar10.get_dataset(50000, flatten=False)
    depth = 5
    msg = "Myrtle depth-5 CNTK @ vanilla CIFAR10"
    do_50k = True
if EXPT_NUM == 1:
    expt = "cntk10-clean"
    dataset = cifar10.get_dataset(50000, flatten=False)
    depth = 10
    msg = "Myrtle depth-10 CNTK @ vanilla CIFAR10"
    do_50k = False
if EXPT_NUM in [2, 3, 4, 5, 6]:
    sizes = {
        2: 2, 3: 3, 4: 4, 5: 5, 6: 8,
    }
    sz = sizes[EXPT_NUM]
    expt = f"cntk10-{sz}px-block-shuffle"
    X, y = cifar10.get_dataset(50000, flatten=False)
    X = np.array([blockwise_shuffle(img, RNG, block_size=sz)
                  for img in X])
    dataset = X, y
    depth = 10
    msg = f"Myrtle depth-10 CNTK @ CIFAR10, block-shuffled (blocksize {sz}px)"
    do_50k = False
if EXPT_NUM in [7, 8, 9]:
    corruption_fracs = {
        7: 0.2, 8: 0.5, 9: 1,
    }
    frac = corruption_fracs[EXPT_NUM]
    expt = f"cntk10-{frac*100:.0f}-frac-shuffle"
    X, y = cifar10.get_dataset(50000, flatten=False)
    X = np.array([shuffle_frac(img, RNG, corrupt_fraction=frac)
                  for img in X])
    dataset = X, y
    depth = 10
    msg = f"Myrtle depth-10 CNTK @ CIFAR10, {frac*100:.0f}% shuffled"
    do_50k = False

kernel_dir = "/scratch/bbjr/dkarkada/kernel-matrices"
work_dir = f"{kernel_dir}/cifar10/{expt}"
assert os.path.exists(work_dir)

# 20k matrix, 50k matrix, done flags for each (per block), dataset, expt description
# K blocked in 10k chunks
chunk = 10000

metadata = load(f"{work_dir}/metadata.file")
assert metadata is not None

K = load(f"{work_dir}/cntk-50k.npy")
if K is None:
    K = load(f"{work_dir}/cntk-20k.npy")
assert K is not None
n = K.shape[0]

# iterate over blocks
dataset = metadata["dataset"]
save(dataset, f"{work_dir}/dataset.file")
flags = metadata[f"flags_{n//1000}k"]
kernel_fn = MyrtleNTK(depth)
for (i, j), done in np.ndenumerate(flags):
    if not done or i > j:
        continue
    print(f"saving block ({i}, {j})... ", end='')
    block = K[i*chunk:(i+1)*chunk, j*chunk:(j+1)*chunk]    
    save(block, f"{work_dir}/tile-{i}-{j}.npy")
    print("done")
