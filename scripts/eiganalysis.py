import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')

import torch

import os
import sys
import time

start_time = time.time()

sys.path.insert(0, 'more-is-better')

from utils import save, load, load_kernel, int_logspace
from theory import calc_kappa, krr
from exptdetails import ExptDetails

args = sys.argv

RNG = np.random.default_rng()

DATASET_NAME = str(args[1])
EXPT_NUM = int(args[2])
DEPTH = int(args[3])
N = int(args[4])

N_SIZES = 80
N_TRIALS = 10
MAX_SIZE = 29000

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
assert np.allclose(K, K.T), np.sum((K-K.T))**2
assert K.shape[0] >= MAX_SIZE + 2000

eigdata = load(f"{work_dir}/eigdata.file")
assert eigdata is not None, "Must compute eigdata first"
num_eigvals = max(eigdata.keys())
eigvals = eigdata[num_eigvals]["eigvals"]
assert MAX_SIZE < num_eigvals

kappa_estimates = np.zeros(N_SIZES)
true_kappas = np.zeros(N_SIZES)
test_mses = np.zeros((N_TRIALS, N_SIZES))
train_mses = np.zeros((N_TRIALS, N_SIZES))
sizes = int_logspace(0, np.log10(MAX_SIZE), num=N_SIZES, base=10)
K = torch.from_numpy(K).cuda()
y = torch.from_numpy(y).cuda()
for i, n in enumerate(sizes):
    print(f"Starting size {n}... ", end='')
    K_sub = K[:n, :n]
    kap_estim = 1 / torch.linalg.pinv(K_sub, atol=1e-30, rtol=0, hermitian=True).trace()
    kappa_estimates[i] = kap_estim.cpu().numpy()
    true_kappas[i] = calc_kappa(n, eigvals)
    for trial in range(N_TRIALS):
        idxs = RNG.choice(N, size=(n+2000), replace=False)
        K_sub, y_sub = K[idxs[:, None], idxs[None, :]], y[idxs]
        train_mse, test_mse = krr(K_sub, y_sub, n_train=n, ridge=0)
        test_mses[trial, i] = test_mse
        train_mses[trial, i] = train_mse
        torch.cuda.empty_cache()
    print("\tdone.")

noise = torch.normal(0, 1, size=(N, 1), dtype=torch.float32).cuda()
_, test_mse_noise = krr(K, noise, n_train=MAX_SIZE)

eigstats = {
    "eigvals": eigvals,
    "sizes": sizes,
    "kappa_estimates": kappa_estimates,
    "true_kappas": true_kappas,
    "test_mse_noise": test_mse_noise,
    "test_mses": test_mses,
    "train_mses": train_mses,
}
save(eigstats, f"{work_dir}/eigstats.file")

del K, y
print(f"all done. hours elapsed: {(time.time()-start_time)/3600:.2f}")

