import numpy as np
from jax import numpy as jnp

import os
import sys
import time

start_time = time.time()

sys.path.insert(0, 'more-is-better')

from kernels import FC_kernel
from imagedata import ImageData
from exptdetails import ExptDetails
from utils import save, load

args = sys.argv

DATASET_NAME = str(args[1]).lower()
NUM_TILES = int(args[2])
EXPT_NUM = 1
DEPTH = 1

expt_details = ExptDetails(EXPT_NUM, DEPTH, DATASET_NAME)
expt_name = expt_details.expt_name
print(f"Computing {NUM_TILES}-tile {expt_name} @ {DATASET_NAME}")

kernel_dir = "/scratch/bbjr/dkarkada/kernel-matrices"
work_dir = f"{kernel_dir}/{DATASET_NAME}/{expt_name}"
if not os.path.exists(work_dir):
    print(f"Making directory {work_dir}")
    os.makedirs(work_dir)
with open(f"{work_dir}/readme.txt", 'w') as f:
    f.write(expt_details.msg)

dataset = load(f"{work_dir}/dataset.file")
if dataset is None:
    print("Generating dataset... ", end='')
    data_generator = ImageData(DATASET_NAME, work_dir=kernel_dir)
    dataset = data_generator.get_dataset(50000, flatten=True)
    save(dataset, f"{work_dir}/dataset.file")
    print("done")

# K tiled in 10k chunks
tile_size = 10000

# iterate over tiles
X_full, _ = dataset
X_full = jnp.array(X_full)
img_size = X_full.shape[1]
kernel_fn = FC_kernel(depth=1)
for i, j in np.ndindex(NUM_TILES, NUM_TILES):
    tile_fn = f"{work_dir}/tile-{i}-{j}.npy"
    if (load(tile_fn) is not None) or i > j:
        print(f"skipping tile ({i}, {j})")
        continue
    print(f"computing tile ({i}, {j})... ", end='')
    X_i = X_full[i*tile_size:(i+1)*tile_size]
    X_j = X_full[j*tile_size:(j+1)*tile_size]
    
    args = (X_i,) if i == j else (X_i, X_j)
    tile = kernel_fn(*args, get='nngp').block_until_ready()
    tile = np.array(tile, dtype=np.float32)
    assert not np.isnan(tile.sum())
    assert tile.shape == (tile_size, tile_size)
    save(tile, tile_fn)
    print("done")
print(f"all done. hours elapsed: {(time.time()-start_time)/3600:.2f}")
