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

args = sys.argv

RNG = np.random.default_rng()

DATASET_NAME = str(args[1])
EXPT_NUM = int(args[2])
DEPTH = int(args[3])
N = int(args[4])

N_RIDGES = 50
N_TEST = 10000

DATASET_NAME = DATASET_NAME.lower()
assert DATASET_NAME in ['cifar10', 'cifar100', 'svhn', 'emnist',
                        'mnist', 'imagenet32', 'imagenet64']

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
# binarize
# class_1 = y[:, [0, 4, 6, 8, 9]].sum(axis=1)
# class_2 = y[:, [1, 2, 3, 5, 7]].sum(axis=1)
# y = np.stack([class_1, class_2]).T
# load kernel
K = load_kernel(N+N_TEST, work_dir)
assert np.allclose(K, K.T), np.sum((K-K.T))**2
# to GPU
K = torch.from_numpy(K).cuda()
y = torch.from_numpy(y).cuda()
print('done')

# these eigvals aren't normalized; lambda ~ ridge here (c.f. theory, lambda~ridge/N)
eigvals = torch.linalg.eigvalsh(K[:N, :N]).cpu().numpy()

log_max_ridge = int(np.log10(eigvals.max())) + 3
log_min_ridge = int(np.log10(eigvals.min())) - 3
print(f"max eigval {eigvals.max():.1e}, min eigval {eigvals.min():.1e}")
print(f"ridge ranging from 10^{log_min_ridge} to 10^{log_max_ridge}")
ridges = np.logspace(log_min_ridge, log_max_ridge, base=10, num=N_RIDGES)
noise_rels = np.array([0, 1, 5, 50])  # relative noise level

# do ridgeless noiseless KR
_, base_mse = krr(K, y, n_train=N, ridge=1e-10)

test_mses = {noise: np.zeros(N_RIDGES) for noise in noise_rels}
train_mses = {noise: np.zeros(N_RIDGES) for noise in noise_rels}
for noise_relative in noise_rels:
    print(f"relative noise {noise_relative}: ", end='')
    noise_absolute = noise_relative * base_mse
    y_noise = torch.normal(0, 1, size=y.size(), dtype=torch.float32).cuda()
    y_noise *= torch.sqrt(noise_absolute / y_noise.size()[-1])
    y_corrupted = y + y_noise
    y_corrupted /= torch.linalg.norm(y_corrupted, dim=1, keepdim=True)
    for i, ridge in enumerate(ridges):
        print('.', end='')
        train_mse, test_mse = krr(K, y_corrupted, n_train=N, ridge=ridge)        
        test_mses[noise_relative][i] = test_mse
        train_mses[noise_relative][i] = train_mse
        torch.cuda.empty_cache()
    print()

results = {
    "ridges": ridges,
    "noise_rels": noise_rels,
    "test_mses": test_mses,
    "train_mses": train_mses,
}
save(results, f"{work_dir}/optridge.file")

del K, y
torch.cuda.empty_cache()
print(f"all done. hours elapsed: {(time.time()-start_time)/3600:.2f}")