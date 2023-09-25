import numpy as np

import os
import sys
import time

start_time = time.time()

sys.path.insert(0, 'more-is-better')

from eigsolver import eigsolve
from utils import save, load, load_kernel
from exptdetails import ExptDetails

args = sys.argv

RNG = np.random.default_rng()

DATASET_NAME = str(args[1])
EXPT_NUM = int(args[2])
DEPTH = int(args[3])
N = int(args[4])

DATASET_NAME = DATASET_NAME.lower()
assert DATASET_NAME in ['cifar10', 'cifar100', 'svhn', 'emnist',
                        'mnist', 'imagenet32', 'imagenet64']

expt_details = ExptDetails(EXPT_NUM, DEPTH, DATASET_NAME)
expt_name = expt_details.expt_name
print(f"Eigensolving {expt_name} cntk @ {DATASET_NAME}")

decomp_sizes = [50, 500, 5000, 20000, 30000, 50000]
    
kernel_dir = "/scratch/bbjr/dkarkada/kernel-matrices"
work_dir = f"{kernel_dir}/{DATASET_NAME}/{expt_name}"
assert os.path.exists(work_dir), work_dir

dataset = load(f"{work_dir}/dataset.file")
assert dataset is not None
_, y = dataset

K = load_kernel(N, work_dir)
assert np.allclose(K, K.T), np.sum((K-K.T))**2

eigdata = load(f"{work_dir}/eigdata.file")
eigdata = {} if eigdata is None else eigdata

for n in decomp_sizes:
    if n > K.shape[0]:
        print(f"Skipping n={n}: too big")
        continue
    if (n in eigdata):
        print(f"Skipping n={n}: already done")
        continue
    print(f"Eigensolving n={n}... ")
    eigvals, eigvecs, eigcoeffs = eigsolve(K[:n, :n], y[:n])
    eigdata[n] = {
        "eigvals": eigvals,
        "eigcoeffs": eigcoeffs
    }
    save(eigdata, f"{work_dir}/eigdata.file")
    save(eigvecs, f"{work_dir}/eigvecs-{n}.npy")    
    print("\tdone.")
    del eigvals, eigvecs, eigcoeffs

del K, y
print(f"all done. hours elapsed: {(time.time()-start_time)/3600:.2f}")
