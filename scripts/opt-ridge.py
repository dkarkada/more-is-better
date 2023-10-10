import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')

import torch

import os
import sys
import time

start_time = time.time()

sys.path.insert(0, 'more-is-better')

from utils import save, load, load_kernel
from theory import krr
from exptdetails import ExptDetails
from ExperimentResults import ExperimentResults

args = sys.argv

RNG = np.random.default_rng()

DATASET_NAME = str(args[1]).lower()
EXPT_NUM = int(args[2])
DEPTH = int(args[3])
N = int(args[4])

N_TRIALS = 5
N_RIDGES = 50
N_TEST = 8000

expt_details = ExptDetails(EXPT_NUM, DEPTH, DATASET_NAME)
expt_name = expt_details.expt_name
print(f"Optimal ridge: {expt_name} @ {DATASET_NAME}")
    
kernel_dir = "/scratch/bbjr/dkarkada/kernel-matrices"
work_dir = f"{kernel_dir}/{DATASET_NAME}/{expt_name}"
assert os.path.exists(work_dir), work_dir

print("loading data and kernel... ", end='')
# load dataset
dataset = load(f"{work_dir}/dataset.file")
assert dataset is not None
_, y = dataset
y = y[:N+N_TEST]
# load kernel
K = load_kernel(N+N_TEST, work_dir)
assert np.allclose(K, K.T), np.sum((K-K.T))**2
# to GPU
K = torch.from_numpy(K).cuda()
y = torch.from_numpy(y).cuda()
print('done')

# these eigvals aren't normalized; lambda ~ ridge here (c.f. theory, lambda~ridge/N)
eigvals = torch.linalg.eigvalsh(K[:N, :N]).cpu().numpy()

log_max_ridge = round(np.log10(eigvals.max())) + 2
log_min_ridge = round(np.log10(eigvals.max()/(N**1.5))) - 1
print(f"max eigval {eigvals.max():.1e}, min eigval {eigvals.min():.1e}")
print(f"ridge ranging from 10^{log_min_ridge} to 10^{log_max_ridge}")
ridges = np.logspace(log_min_ridge, log_max_ridge, base=10, num=N_RIDGES)
noise_rels = np.array([0, 1, 5, 50])  # relative noise level

# do ridgeless noiseless KR
_, base_mse = krr(K, y, n_train=N, ridge=1e-10)

meta = {"base_mse": base_mse}
axes = [
    ("trial", np.arange(N_TRIALS)),
    ("noise", noise_rels),
    ("ridge", ridges),
    ("result", ["train_mse", "test_mse"])
]
expt = ExperimentResults(axes, f"{work_dir}/optridge.expt", meta)

for trial in expt.get_axis('trial'):
    print(f"trial {trial}: ", end='')
    idxs = torch.randperm(K.size()[0]).cuda()
    K = K[idxs[:, None], idxs[None, :]]
    y = y[idxs]
    for noise_relative in expt.get_axis('noise'):
        print('.', end='')
        noise_absolute = noise_relative * base_mse
        y_noise = torch.normal(0, 1, size=y.size(), dtype=torch.float32).cuda()
        y_noise *= np.sqrt(noise_absolute / y_noise.size()[-1])
        y_corrupted = y + y_noise
        y_corrupted /= torch.linalg.norm(y_corrupted, dim=1, keepdim=True)
        for ridge in expt.get_axis('ridge'):
            train_mse, test_mse = krr(K, y_corrupted, n_train=N, ridge=ridge)
            result = [train_mse, test_mse]
            expt.write(result, trial=trial, noise=noise_relative, ridge=ridge)
            torch.cuda.empty_cache()
    print()

del K, y
torch.cuda.empty_cache()
print(f"all done. hours elapsed: {(time.time()-start_time)/3600:.2f}")