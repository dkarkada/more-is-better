import numpy as np
import jax
jax.config.update('jax_platform_name', 'cpu')

import torch
import torch.nn.functional as torchfun

import os
import sys
import time

start_time = time.time()

sys.path.insert(0, 'more-is-better')

from utils import save, load
from theory import rf_krr_risk_theory
from imagedata import ImageData
from ExperimentResults import ExperimentResults

args = sys.argv

RNG = np.random.default_rng()

# n_train up to 10000, k up to 10000

N_SIZES = 31
N_TRIALS = 15
N_RIDGES = 31

print(f"RF expt: gaussian")

kernel_dir = "/scratch/bbjr/dkarkada/kernel-matrices"
work_dir = f"{kernel_dir}/cifar10/fc1-nngpk"
assert os.path.exists(work_dir), work_dir

def get_gaussian_dataset_closure(eigcoeffs, noise_var):
    m = len(eigcoeffs)

    def get_gaussian_dataset(n):
        X = torch.normal(0, 1, size=(n, m))
        y = X @ eigcoeffs + torch.normal(0, noise_var, size=n)
        y = y[:, None]
        return X, y

    return get_gaussian_dataset

def get_gaussian_feature_map_closure(eigvals):
    in_dim = len(eigvals)

    def get_gaussian_feature_map():
        proj = torch.normal(0, 1, size=(in_dim, in_dim)) / torch.sqrt(in_dim)
        F = torch.einsum('ij,j->ij', proj, torch.sqrt(eigvals))
        def gaussian_feature_map(X):
            return (F @ X.T).T
        return gaussian_feature_map

    return get_gaussian_feature_map

# ensure eigcoeffs are torch tensor
# gen eigvals and eigcoeffs


#continue from here

# put on CPU to speed up theory calculation
cpus = jax.devices("cpu")
eigvals = jax.device_put(eigvals, cpus[0])
eigcoeffs = jax.device_put(eigcoeffs, cpus[0])

get_dataset = get_cifar10_dataset_closure()



kk = int_logspace(1, 4, base=10, num=40)
kk_peak = int_logspace(2, 3, base=10, num=30)
kk = onp.unique(onp.concatenate((kk, kk_peak),0))
n_trains = [256]
ridges = onp.logspace(-12, 12, base=2, num=25)

theory_axes = [
    ("n", n_trains),
    ("k", kk),
    ("ridge", ridges),
    ("result", ["kappa", "gamma", "test_mse"])
]
theory = ExperimentResults(theory_axes, f"{expt_dir}/theory-{expt_name}.file", meta)
run_rf_theory(theory, eigvals, eigcoeffs, noise_var=0) # 4m, 20 jun
def run_rf_theory(theory, eigvals, eigcoeffs, noise_var):
    n_trains = theory.get_axis("n")
    kk = theory.get_axis("k")
    ridges = theory.get_axis("ridge")

    for n in n_trains:
        for k in kk:
            print('.', end='')
            for ridge in ridges:
                mse, kappa, gamma = rf_krr_risk_theory(eigvals, eigcoeffs, n, k, ridge, noise_var)
                theory.write(kappa, n=n, k=k, ridge=ridge, result="kappa", save_after=False)
                theory.write(gamma, n=n, k=k, ridge=ridge, result="gamma", save_after=False)
                theory.write(mse, n=n, k=k, ridge=ridge, result="test_mse", save_after=False)
        theory.save()
        print()




kk = int_logspace(1, 4, base=10, num=31)
n_trains = [256]
ridges = onp.logspace(-12, 12, base=2, num=25)
trials = onp.arange(25)

axes = [
    ("trial", trials),
    ("n", n_trains),
    ("k", kk),
    ("ridge", ridges),
    ("result", ["train_mse", "test_mse"])
]

expt = ExperimentResults(axes, f"{expt_dir}/expt-{expt_name}.file")
run_rf_expt(expt, get_dataset, get_relu_feature_map, RNG)
# expt = ExperimentResults.load(f"{expt_dir}/expt-{expt_name}.file")
def run_rf_expt(expt, get_dataset, get_feature_map, rng):
    trials = expt.get_axis("trial")
    n_trains = expt.get_axis("n")
    kk = expt.get_axis("k")
    ridges = expt.get_axis("ridge")

    for n in n_trains:
        if expt.is_written(n=n):
            continue
        for trial in trials:
            print('.', end='')
            if expt.is_written(trial=trial, n=n):
                print(f"skipping trial {trial}, n {n}")
                continue
            train_X, train_y, test_X, test_y = get_dataset(n)
            assert train_y.ndim == 2 and test_y.ndim == 2
            feature_map = get_feature_map()
            train_features = feature_map(train_X)
            test_features = feature_map(test_X)
            assert train_features.shape[0] == n

            for k in kk:
                num_features = train_features.shape[-1]
                keep_inds = rng.choice(num_features, size=k, replace=False)
                train_mses, test_mses = rf_krr(train_features, test_features, keep_inds,
                                               train_y, test_y, ridges)
                result = onp.array([train_mses, test_mses]).T
                expt.write(result, n=n, k=k, trial=trial)
        print()



results = {
    "ridges": ridges,
    "noises": noises,
    "test_mses": test_mses,
    "train_mses": train_mses,
}
save(results, f"{work_dir}/optridge.file")

del K, y
torch.cuda.empty_cache()
print(f"all done. hours elapsed: {(time.time()-start_time)/3600:.2f}")