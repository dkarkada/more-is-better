import numpy as np
from jax import numpy as jnp

import os
import sys

sys.path.insert(0, 'more-is-better')

from kernels import eigendecomp
from utils import save, load

EXPT_NUM = 1
DO_50K = False

if EXPT_NUM == 1:
    expt = "cntk5-clean"
    
kernel_dir = "/scratch/bbjr/dkarkada/kernel-matrices"
work_dir = f"{kernel_dir}/{expt}"
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

metadata = load(f"{work_dir}/metadata.file")
assert metadata is not None
X_full, y_full = metadata["dataset"]

n = 50000 if DO_50K else 20000
K = load(f"{work_dir}/CNTK_{n//1000}k.npy")
assert K is not None
assert K.shape == (n, n)
y = y_full[:n]

print(f"Eigendecomposing n={n} {expt}... ", end='')
eigvals, eigvecs, eigcoeffs = eigendecomp(K, y)
print("done. Saving... ", end='')
eigdata = {
    "eigvals": eigvals,
    "eigcoeffs": eigcoeffs
}
save(eigdata, f"{work_dir}/eigdata-{n//1000}k.file")
save(eigvecs, f"{work_dir}/eigvecs-{n//1000}k.npy")
print("done.")

del K, y, eigvals, eigvecs, eigcoeffs