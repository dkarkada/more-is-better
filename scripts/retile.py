import numpy as np

import os
import sys
import time

start_time = time.time()

sys.path.insert(0, 'more-is-better')

from utils import load, load_kernel
from exptdetails import ExptDetails

args = sys.argv

RNG = np.random.default_rng()

DATASET_NAME = str(args[1])
EXPT_NUM = int(args[2])
DEPTH = int(args[3])
N = int(args[4])

N_SIZES = 80
N_TRIALS = 10
MAX_SIZE = 28000

DATASET_NAME = DATASET_NAME.lower()
assert DATASET_NAME in ['cifar10', 'cifar100', 'emnist',
                        'mnist', 'imagenet32', 'imagenet64']

expt_details = ExptDetails(EXPT_NUM, DEPTH, DATASET_NAME)
expt_name = expt_details.expt_name
print(f"Eigenanalysis of {expt_name} cntk @ {DATASET_NAME}")
    
kernel_dir = "/scratch/bbjr/dkarkada/kernel-matrices"
work_dir = f"{kernel_dir}/{DATASET_NAME}/{expt_name}"
assert os.path.exists(work_dir), work_dir

dataset = load(f"{work_dir}/dataset.file")
assert dataset is not None
_, y = dataset

K = load_kernel(N, work_dir)
print(round(K.nbytes / 1024**2, 2))
del K, y
print(f"all done. hours elapsed: {(time.time()-start_time)/3600:.2f}")

