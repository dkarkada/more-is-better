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

from utils import load, int_logspace
from theory import rf_krr_risk_theory, rf_krr
from exptdetails import ExptDetails
from ExperimentResults import ExperimentResults

args = sys.argv

RNG = np.random.default_rng()

# n_train up to 10000, k up to 10000

IMITATE = False
if IMITATE:
    BINARIZATION = [[0, 1, 7, 8, 9], [2, 3, 4, 5, 6]]
    NOISE_VAR = 0
    ID = "gaussian-imitation"
if not IMITATE:
    ALPHA = 1.5
    BETA = 1.5
    NOISE_VAR = 0.5
    ID = "gaussian"
M = 10000

N_THRY_PTS = 20 # 60
N_EXPT_PTS = 11 # 31
N_TRIALS = 11 # 45
N_RIDGES = 31
assert N_THRY_PTS >= 10

DATASET_NAME = 'cifar10'

print(f"RF expt: gaussian")

expt_details = ExptDetails(1, 1, DATASET_NAME)
expt_name = expt_details.expt_name
kernel_dir = "/scratch/bbjr/dkarkada/kernel-matrices"
work_dir = f"{kernel_dir}/{DATASET_NAME}/{expt_name}"
assert os.path.exists(work_dir), work_dir

def get_gaussian_dataset_closure(eigcoeffs, noise_var=0):
    m = len(eigcoeffs)

    def get_gaussian_dataset(n):
        X = torch.normal(0, 1, size=(n, m)).cuda()
        y = X @ eigcoeffs + torch.normal(0, noise_var, size=(n, 1)).cuda()
        return X, y

    return get_gaussian_dataset

def get_gaussian_feature_map_closure(eigvals):
    in_dim = len(eigvals)

    def get_gaussian_feature_map():
        proj = torch.normal(0, 1, size=(in_dim, in_dim)).cuda() / np.sqrt(in_dim)
        F = torch.einsum('ij,j->ij', proj, torch.sqrt(eigvals))
        def gaussian_feature_map(X):
            return (F @ X.T).T
        return gaussian_feature_map

    return get_gaussian_feature_map

# gen eigvals and eigcoeffs
if IMITATE:
    eigdata = load(f"{work_dir}/eigdata.file")
    assert eigdata is not None, "Must compute eigdata first"
    num_eigvals = max(eigdata.keys())
    eigvals = eigdata[num_eigvals]["eigvals"][:M]
    eigcoeffs = eigdata[num_eigvals]["eigcoeffs"][:M]
    eigcoeffs = eigcoeffs[:, BINARIZATION[0]].sum(axis=1) \
                - eigcoeffs[:, BINARIZATION[1]].sum(axis=1)
else:
    idxs = 1 + np.arange(M)
    eigvals = idxs ** -ALPHA
    eigcoeffs = np.sqrt(idxs ** -BETA)

# put on CPU to speed up theory calculation
cpus = jax.devices("cpu")
eigvals = jax.device_put(eigvals, cpus[0])
eigcoeffs = jax.device_put(eigcoeffs, cpus[0])

## THEORY CURVES

def do_theory(theory):
    for n in theory.get_axis("n"):
        for k in theory.get_axis("k"):
            print('.', end='')
            for ridge in theory.get_axis("ridge"):
                mse, kappa, gamma = rf_krr_risk_theory(eigvals, eigcoeffs, n,
                                                       k, ridge, NOISE_VAR)
                results = [mse, kappa, gamma]
                theory.write(results, n=n, k=k, ridge=ridge)
    print()
    

vary_dim = int_logspace(1, 4, base=10, num=N_THRY_PTS)
vary_dim_peak = int_logspace(2, 3, base=10, num=(N_THRY_PTS-10))
vary_dim = np.unique(np.concatenate((vary_dim, vary_dim_peak),0))
fixed_dim = [256]
ridges = np.logspace(-3, 2, base=10, num=N_RIDGES)

# n = 256, varying k
axes = [
    ("n", fixed_dim),
    ("k", vary_dim),
    ("ridge", ridges),
    ("result", ["test_mse", "kappa", "gamma"])
]
theory_n256 = ExperimentResults(axes, f"{work_dir}/theory-{ID}-n256.expt")
print("Starting theory n=256")
do_theory(theory_n256)
print("done.")

# k = 256, varying n
axes = [
    ("n", vary_dim),
    ("k", fixed_dim),
    ("ridge", ridges),
    ("result", ["test_mse", "kappa", "gamma"])
]
theory_k256 = ExperimentResults(axes, f"{work_dir}/theory-{ID}-k256.expt")
print("Starting theory k=256")
do_theory(theory_k256)
print("done.")


## EXPT CURVES

# ensure eigcoeffs are torch tensor
eigvals = torch.from_numpy(np.array(eigvals)).cuda()
eigcoeffs = torch.from_numpy(np.array(eigcoeffs)).cuda()

get_dataset = get_gaussian_dataset_closure(eigcoeffs, NOISE_VAR)
get_gaussian_feature_map = get_gaussian_feature_map_closure(eigvals)

def do_expt(expt):
    for trial in expt.get_axis("trial"):
        for n in expt.get_axis("n"):
            X, y = get_dataset(n+1000)
            feature_map = get_gaussian_feature_map()
            features = feature_map(X)
            assert features.shape[0] == n + 1000
            for k in expt.get_axis("k"):
                print('.', end='')
                ridges = expt.get_axis("ridge")
                train_mses, test_mses = rf_krr(features, y, n, k, ridges, RNG)
                expt.write(train_mses, n=n, k=k, trial=trial, result="train_mse")
                expt.write(test_mses, n=n, k=k, trial=trial, result="test_mse")
        print()


vary_dim = int_logspace(1, 4, base=10, num=N_EXPT_PTS)
trials = np.arange(N_TRIALS)

# n = 256, varying k
axes = [
    ("trial", trials),
    ("n", fixed_dim),
    ("k", vary_dim),
    ("ridge", ridges),
    ("result", ["train_mse", "test_mse"])
]
expt = ExperimentResults(axes, f"{work_dir}/expt-{ID}-n256.expt")
print("Starting expt n=256")
do_expt(expt)
print("done.")

# k = 256, varying n
axes = [
    ("trial", trials),
    ("n", vary_dim),
    ("k", fixed_dim),
    ("ridge", ridges),
    ("result", ["train_mse", "test_mse"])
]
expt = ExperimentResults(axes, f"{work_dir}/expt-{ID}-k256.expt")
print("Starting expt k=256")
do_expt(expt)
print("done.")

torch.cuda.empty_cache()
print(f"all done. hours elapsed: {(time.time()-start_time)/3600:.2f}")