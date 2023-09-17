import numpy as np
from jax import numpy as jnp

import os
import sys
import time

start_time = time.time()

sys.path.insert(0, 'more-is-better')

from kernels import eigendecomp
from utils import save, load

EXPT_NUM = 1

if EXPT_NUM == 1:
    expt = "cntk5-clean"
    decomp_sizes = [50, 500, 5000, 50000]
    
kernel_dir = "/scratch/bbjr/dkarkada/kernel-matrices"
work_dir = f"{kernel_dir}/{expt}"
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

metadata = load(f"{work_dir}/metadata.file")
assert metadata is not None
_, y = metadata["dataset"]

K = load(f"{work_dir}/cntk-50k.npy")
if K is None:
    K = load(f"{work_dir}/cntk-20k.npy")
assert K is not None
assert max(decomp_sizes) <= K.shape[0]

eigdata = load(f"{work_dir}/eigdata.file")
eigdata = {} if eigdata is None else eigdata

print(f"Expt: {expt}")
for n in decomp_sizes:
    if (n in eigdata) and (load(f"{work_dir}/eigvecs-{n//1000}k.npy") is not None):
        print(f"Skipping n={n}")
        continue
    print(f"Eigendecomposing n={n}... ", end='')
    eigvals, eigvecs, eigcoeffs = eigendecomp(K[:n, :n], y[:n])
    eigdata[n] = {
        "eigvals": eigvals,
        "eigcoeffs": eigcoeffs
    }
    save(eigdata, f"{work_dir}/eigdata.file")
    save(eigvecs, f"{work_dir}/eigvecs-{n//1000}k.npy")
    print("done.")

del K, y, eigvals, eigvecs, eigcoeffs
print(f"all done. hours elapsed: {(time.time()-start_time)/3600:.2f}")
