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

DATASET_NAME = str(args[1])
EXPT_NUM = int(args[2])
DEPTH = int(args[3])
NUM_TILES = int(args[4])

DATASET_NAME = DATASET_NAME.lower()
assert DATASET_NAME in ['cifar10', 'cifar100', 'emnist',
                        'mnist', 'imagenet32', 'imagenet64']
print(f"Computing {NUM_TILES}-tile cntk @ {DATASET_NAME}: expt {EXPT_NUM}")

RNG = np.random.default_rng(seed=42)

expt_details = {}
if EXPT_NUM == 0:
    expt = f"myrtle{DEPTH}-clean"
    msg = f"Myrtle depth-{DEPTH} CNTK @ vanilla {DATASET_NAME}"
if EXPT_NUM in [10, 11, 12, 13, 14]:
    sizes = {
        10: 2, 11: 3, 12: 4, 13: 5, 14: 8,
    }
    sz = sizes[EXPT_NUM]
    expt_details["block size"] = sz
    expt = f"myrtle{DEPTH}-{sz}px-block-shuffle"
    msg = f"Myrtle depth-{DEPTH} CNTK @ {DATASET_NAME}, block-shuffled (blocksize {sz}px)"
if EXPT_NUM in [20, 21, 22]:
    corruption_fracs = {
        20: 0.2, 21: 0.5, 22: 1,
    }
    frac = corruption_fracs[EXPT_NUM]
    expt_details["corruption frac"] = frac
    expt = f"myrtle{DEPTH}-{frac*100:.0f}-frac-shuffle"
    msg = f"Myrtle depth-{DEPTH} CNTK @ {DATASET_NAME}, {frac*100:.0f}% shuffled"

kernel_dir = "/scratch/bbjr/dkarkada/kernel-matrices"
work_dir = f"{kernel_dir}/{DATASET_NAME}/{expt}"
if not os.path.exists(work_dir):
    print(f"Making directory {work_dir}")
    os.makedirs(work_dir)
with open(f"{work_dir}/readme.txt", 'w') as f:
    f.write(msg)

dataset = load(f"{work_dir}/dataset.file")
if dataset is None:
    print("Generating dataset... ", end='')
    data_generator = ImageData(DATASET_NAME)
    X, y = data_generator.get_dataset(50000, flatten=False)
    if EXPT_NUM in [10, 11, 12, 13, 14]:
        sz = expt_details["block size"]
        X = np.array([blockwise_shuffle(img, RNG, block_size=sz)
                      for img in X])
    if EXPT_NUM in [20, 21, 22]:
        frac = expt_details["corruption frac"]
        X = np.array([shuffle_frac(img, RNG, corrupt_fraction=frac)
                    for img in X])
    dataset = X, y
    save(dataset, f"{work_dir}/dataset.file")
    print("done")
    

# K tiled in 10k chunks
tile_size = 10000

# iterate over tiles
X_full, _ = dataset
X_full = jnp.array(X_full)
kernel_fn = MyrtleNTK(DEPTH)
for i, j in np.ndindex(NUM_TILES, NUM_TILES):
    tile_fn = f"{work_dir}/tile-{i}-{j}.npy"
    if (load(tile_fn) is not None) or i > j:
        print(f"skipping tile ({i}, {j})")
        continue
    print(f"computing tile ({i}, {j})... ", end='')
    X_i = X_full[i*tile_size:(i+1)*tile_size]
    X_j = X_full[j*tile_size:(j+1)*tile_size]
    
    args = (X_i,) if i == j else (X_i, X_j)
    tile = kernel_fn(*args, get='ntk').block_until_ready()
    tile = np.array(tile)
    assert tile.shape == (tile_size, tile_size)
    save(tile, tile_fn)
    print("done")
print(f"all done. hours elapsed: {(time.time()-start_time)/3600:.2f}")
