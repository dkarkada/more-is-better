import numpy as np
import torch
import os
import sys
import time

start_time = time.time()

sys.path.insert(0, 'more-is-better')

from utils import save, load, load_kernel, int_logspace
from exptdetails import ExptDetails

args = sys.argv

RNG = np.random.default_rng()

DATASET_NAME = str(args[1])
EXPT_NUM = int(args[2])
DEPTH = int(args[3])
N = int(args[4])

N_SIZES = 10
N_TRIALS = 1
MAX_SIZE = 18000


def rkrr(K, y, n_train):
    y_train = y[:n_train]
    y_test = y[n_train:]
    K_train = K[:n_train, :n_train]
    K_test = K[:, :n_train]

    y_hat = K_test @ torch.linalg.inv(K_train) @ y_train
    # train error
    y_hat_train = y_hat[:n_train]
    train_mse = ((y_train - y_hat_train) ** 2).sum(axis=1).mean()
    # test error
    y_hat_test = y_hat[n_train:]
    test_mse = ((y_test - y_hat_test) ** 2).sum(axis=1).mean()
    return train_mse, test_mse


DATASET_NAME = DATASET_NAME.lower()
assert DATASET_NAME in ['cifar10', 'cifar100', 'emnist',
                        'mnist', 'imagenet32', 'imagenet64']

expt_details = ExptDetails(EXPT_NUM, DEPTH, DATASET_NAME)
expt_name = expt_details.expt_name
print(f"Eigendebug of {expt_name} cntk @ {DATASET_NAME}")
    
kernel_dir = "/scratch/bbjr/dkarkada/kernel-matrices"
work_dir = f"{kernel_dir}/{DATASET_NAME}/{expt_name}"
assert os.path.exists(work_dir), work_dir

dataset = load(f"{work_dir}/dataset.file")
assert dataset is not None
_, y = dataset

K = load_kernel(N, work_dir)
assert np.allclose(K, K.T), np.sum((K-K.T))**2
assert K.shape[0] >= MAX_SIZE + 1000
K = torch.from_numpy(K).cuda()

eigdata = load(f"{work_dir}/eigdata.file")
assert eigdata is not None, "Must compute eigdata first"
eigvals = eigdata[max(eigdata.keys())]["eigvals"]

kappa_estimates = np.zeros(N_SIZES)
true_kappas = np.zeros(N_SIZES)
test_mses = np.zeros((N_TRIALS, N_SIZES))
train_mses = np.zeros((N_TRIALS, N_SIZES))
sizes = int_logspace(0, np.log10(MAX_SIZE), num=N_SIZES, base=10)
for i, n in enumerate(sizes):
    print(f"Starting size {n}... ", end='', flush=True)
    K_sub = K[:n, :n]
    # kappa_estimates[i] = estimate_kappa(K_sub)
    # true_kappas[i] = calc_kappa(n, eigvals)
    for trial in range(N_TRIALS):
        idxs = RNG.choice(N, size=(n+1000), replace=False)
        K_sub, y_sub = K[idxs[:, None], idxs[None, :]], y[idxs]
        train_mse, test_mse = rkrr(K_sub, y_sub, n_train=n)
        test_mses[trial, i] = test_mse
        train_mses[trial, i] = train_mse
    print("\tdone.")

eigstats = {
    "sizes": sizes,
    "kappa_estimates": kappa_estimates,
    "true_kappas": true_kappas,
    "test_mses": test_mses,
    "train_mses": train_mses
}
save(eigstats, f"{work_dir}/eigdebug.file")

del K, y
print(f"all done. hours elapsed: {(time.time()-start_time)/3600:.2f}")

