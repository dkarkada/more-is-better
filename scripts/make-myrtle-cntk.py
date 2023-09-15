import numpy as np
from jax import numpy as jnp

import os
import sys
import time

start_time = time.time()

sys.path.insert(0, 'more-is-better')

from kernels import MyrtleNTK
from ImageData import ImageData
from utils import save, load

EXPT_NUM = 1

cifar10 = ImageData('cifar10')

if EXPT_NUM == 1:
    expt = "cntk5-clean"
    dataset = cifar10.get_dataset(50000, flatten=False)
    depth = 5
    msg = "Myrtle depth-5 CNTK @ vanilla CIFAR10"
    do_50k = True

kernel_dir = "/scratch/bbjr/dkarkada/kernel-matrices"
work_dir = f"{kernel_dir}/{expt}"
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

# 20k matrix, 50k matrix, done flags for each (per block), dataset, expt description
# K blocked in 10k chunks
chunk = 10000

metadata = load(f"{work_dir}/metadata.file")
if metadata is None:
    metadata = {
        "flags_20k": np.zeros((2, 2)),
        "flags_50k": np.zeros((5, 5)),
        "dataset": dataset,
        "msg": msg
    }
    save(metadata, f"{work_dir}/metadata.file")
    with open(f"{work_dir}/readme.txt", 'w') as f:
        f.write(msg)

def set_block(K, block, idx, fn):
    i, j = idx
    K[i*chunk:(i+1)*chunk, j*chunk:(j+1)*chunk] = block
    if i != j:
        K[j*chunk:(j+1)*chunk, i*chunk:(i+1)*chunk] = block.T
    save(K, fn)

    n = K.shape[0]
    metadata[f"flags_{n//1000}k"][i, j] = 1
    metadata[f"flags_{n//1000}k"][j, i] = 1
    save(metadata, f"{work_dir}/metadata.file")

n = 50000 if do_50k else 20000
K_fn = f"{work_dir}/cntk-{n//1000}k.npy"
K = load(K_fn)
if K is None:
    K = np.zeros((n, n))
    save(K, K_fn)
assert K.shape == (n, n)

# copy results from other kernel matrix, if it exists
n_other = 20000 if do_50k else 50000
flags_other = metadata[f"flags_{n_other//1000}k"]
# check that the other kernel matrix is even partially computed
if flags_other.any():
    K_other = load(f"{work_dir}/cntk-{n_other//1000}k.npy")
    assert K_other is not None
    for (i, j), done_other in np.ndenumerate(flags_other):
        # check that block is within bounds in this matrix (e.g. if other is larger)
        if i*chunk<n and j*chunk<n:
            done = metadata[f"flags_{n//1000}k"][i, j]
            # check that other block is computed AND this one is not. Then copy
            if done_other and not done:
                block = K_other[i*chunk:(i+1)*chunk, j*chunk:(j+1)*chunk]
                set_block(K, block, (i, j), K_fn)
                print(f"copied block ({i}, {j})")

# iterate over blocks
X_full, _ = metadata["dataset"]
X_full = jnp.array(X_full)
flags = metadata[f"flags_{n//1000}k"]
kernel_fn = MyrtleNTK(depth)
for (i, j), done in np.ndenumerate(flags):
    if done or i > j:
        continue
    print(f"starting block ({i}, {j})... ", end='')
    X_i = X_full[i*chunk:(i+1)*chunk]
    X_j = X_full[j*chunk:(j+1)*chunk]
    
    args = (X_i,) if i == j else (X_i, X_j)
    block = kernel_fn(*args, get='ntk').block_until_ready()
    # block = jnp.einsum('ajkl,bjkl->ab', X_i, X_j)
    block = np.array(block)
    assert block.shape == (chunk, chunk)
    set_block(K, block, (i, j), K_fn)
    assert metadata[f"flags_{n//1000}k"][i, j] == 1
    assert metadata[f"flags_{n//1000}k"][j, i] == 1
    print("done")
print(f"all done. hours elapsed: {(time.time()-start_time)/3600:.2f}")
