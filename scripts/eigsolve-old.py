import numpy as np

import os
import sys
import time

start_time = time.time()

sys.path.insert(0, 'more-is-better')

from eigsolver import eigsolve
from utils import save, load

args = sys.argv

EXPT_NUM = int(args[1])
print(f"Eigendecomposing experiment {EXPT_NUM}")

if EXPT_NUM == 0:
    expt = "cntk5-clean"
    decomp_sizes = [20, 200, 2000, 20000, 30000]
if EXPT_NUM == 1:
    expt = "cntk10-clean"
    decomp_sizes = [20, 200, 2000, 20000]
if EXPT_NUM in [2, 3, 4, 5, 6]:
    sizes = {
        2: 2, 3: 3, 4: 4, 5: 5, 6: 8,
    }
    sz = sizes[EXPT_NUM]
    expt = f"cntk10-{sz}px-block-shuffle"
    decomp_sizes = [20, 200, 2000, 20000]
if EXPT_NUM in [7, 8, 9]:
    corruption_fracs = {
        7: 0.2, 8: 0.5, 9: 1,
    }
    frac = corruption_fracs[EXPT_NUM]
    expt = f"cntk10-{frac*100:.0f}-frac-shuffle"
    decomp_sizes = [20, 200, 2000, 20000]
    
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
assert np.allclose(K, K.T)
assert max(decomp_sizes) <= K.shape[0]

eigdata = load(f"{work_dir}/eigdata.file")
eigdata = {} if eigdata is None else eigdata

print(f"Expt: {expt}")
for n in decomp_sizes:
    if (n in eigdata) and (load(f"{work_dir}/eigvecs-{n//1000}k.npy") is not None):
        print(f"Skipping n={n}")
        # print("jk")
        continue
    print(f"Eigensolving n={n}... ", end='')
    eigvals, eigvecs, eigcoeffs = eigsolve(K[:n, :n], y[:n])
    eigdata[n] = {
        "eigvals": eigvals,
        "eigcoeffs": eigcoeffs
    }
    save(eigdata, f"{work_dir}/eigdata.file")
    save(eigvecs, f"{work_dir}/eigvecs-{n//1000}k.npy")
    print("done.")

del K, y, eigvals, eigvecs, eigcoeffs
print(f"all done. hours elapsed: {(time.time()-start_time)/3600:.2f}")
