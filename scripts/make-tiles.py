import numpy as np

import os
import sys
import time

start_time = time.time()

sys.path.insert(0, 'more-is-better')

from utils import save, load
from exptdetails import ExptDetails

args = sys.argv

DATASET_NAME = str(args[1])
EXPT_NUM = int(args[2])
DEPTH = int(args[3])

DATASET_NAME = DATASET_NAME.lower()
assert DATASET_NAME in ['cifar10', 'cifar100', 'emnist',
                        'mnist', 'imagenet32', 'imagenet64']

expt_details = ExptDetails(EXPT_NUM, DEPTH, DATASET_NAME)
expt_name = expt_details.expt_name
print(f"tiling {expt_name} cntk @ {DATASET_NAME}")

kernel_dir = "/scratch/bbjr/dkarkada/kernel-matrices"
work_dir = f"{kernel_dir}/{DATASET_NAME}/{expt_name}"
assert os.path.exists(work_dir), work_dir

# 20k matrix, 50k matrix, done flags for each (per block), dataset, expt description
# K blocked in 10k chunks
sz = 10000

metadata = load(f"{work_dir}/metadata.file")
assert metadata is not None

K = load(f"{work_dir}/cntk-50k.npy")
if K is None:
    print("50k not found; loading 20k")
    K = load(f"{work_dir}/cntk-20k.npy")
assert K is not None
n = K.shape[0]

# iterate over blocks
dataset = metadata["dataset"]
save(dataset, f"{work_dir}/dataset.file")
flags = metadata[f"flags_{n//1000}k"]
for (i, j), done in np.ndenumerate(flags):
    if not done or i > j:
        continue
    print(f"saving tile ({i}, {j})... ", end='')
    tile = K[i*sz:(i+1)*sz, j*sz:(j+1)*sz]    
    save(tile, f"{work_dir}/tile-{i}-{j}.npy")
    print("done")
