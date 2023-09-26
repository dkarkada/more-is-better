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

from utils import save, load, int_logspace
from theory import rf_krr_risk_theory, rf_krr
from imagedata import ImageData
from exptdetails import ExptDetails
from ExperimentResults import ExperimentResults

args = sys.argv

RNG = np.random.default_rng()

# n_train up to 10000, k up to 10000

N_THRY_PTS = 12 # 40
N_EXPT_PTS = 11 # 31
N_TRIALS = 1 # 15
N_RIDGES = 3 # 31
assert N_THRY_PTS >= 10

DATASET_NAME = 'cifar10'
CIFAR_SZ = 3072
BINARIZATION = [[0, 1, 7, 8, 9], [2, 3, 4, 5, 6]]

print(f"RF expt: cifar10")

expt_details = ExptDetails(1, 1, DATASET_NAME)
expt_name = expt_details.expt_name
kernel_dir = "/scratch/bbjr/dkarkada/kernel-matrices"
work_dir = f"{kernel_dir}/{DATASET_NAME}/{expt_name}"
assert os.path.exists(work_dir), work_dir

dataset = load(f"{work_dir}/dataset.file")
if dataset is None:
    print("Generating dataset... ", end='')
    data_generator = ImageData(DATASET_NAME, classes=BINARIZATION, work_dir=kernel_dir)
    dataset = data_generator.get_dataset(50000, flatten=True)
    save(dataset, f"{work_dir}/dataset.file")
    print("done")
X, y = dataset
# binarize
y = y[:, BINARIZATION[0]].sum(axis=1) - y[:, BINARIZATION[1]].sum(axis=1)
y = y[:, None]
assert y.ndim == 2
X = torch.from_numpy(X).cuda()
y = torch.from_numpy(y).cuda()
N = len(X)


def get_dataset(n):
    idxs = RNG.choice(N, size=n, replace=False)
    return X[idxs], y[idxs]

def get_relu_feature_map():
    w = 10000 # width
    # W = torch.from_numpy(np.sqrt(2/CIFAR_SZ) * RNG.standard_normal(size=(w, CIFAR_SZ))).cuda()
    W = torch.sqrt(2/CIFAR_SZ) * torch.normal(0, 1, size=(w, CIFAR_SZ))
    def relu_feature_map(X):
        # factor sqrt(1/w) to ensure kernel is O(1)
        WX = torch.sqrt(1/w) * (W @ X.T).T
        return torchfun.relu_(WX)
    return relu_feature_map

# ensure eigcoeffs are torch tensor

meta = {
    "binarization": BINARIZATION,
}

eigdata = load(f"{work_dir}/eigdata.file")
assert eigdata is not None, "Must compute eigdata first"
num_eigvals = max(eigdata.keys())
eigvals = eigdata[num_eigvals]["eigvals"]
eigcoeffs = eigdata[num_eigvals]["eigcoeffs"]
eigcoeffs = eigcoeffs[:, BINARIZATION[0]].sum(axis=1) \
             - eigcoeffs[:, BINARIZATION[1]].sum(axis=1)
eigcoeffs /= np.linalg.norm(eigcoeffs)

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
                mse, kappa, gamma = rf_krr_risk_theory(eigvals, eigcoeffs, n, k, ridge)
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
theory_n256 = ExperimentResults(axes, f"{work_dir}/theory-n256.expt", meta)
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
theory_k256 = ExperimentResults(axes, f"{work_dir}/theory-k256.expt", meta)
print("Starting theory k=256")
do_theory(theory_k256)
print("done.")


## EXPT CURVES

def do_expt(expt):
    for trial in expt.get_axis("trial"):
        for n in expt.get_axis("n"):
            print('.', end='')
            X, y = get_dataset(n+1000)
            feature_map = get_relu_feature_map()
            features = feature_map(X)
            assert features.shape[0] == n + 1000
            for k in expt.get_axis("k"):
                ridges = expt.get_axis("ridge")
                train_mses, test_mses = rf_krr(features, y, n, k, ridges, RNG)
                result = [train_mses, test_mses]
                expt.write(result, n=n, k=k, trial=trial)
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
expt = ExperimentResults(axes, f"{work_dir}/expt-n256.expt")
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
expt = ExperimentResults(axes, f"{work_dir}/expt-k256.expt")
print("Starting expt k=256")
do_expt(expt)
print("done.")

torch.cuda.empty_cache()
print(f"all done. hours elapsed: {(time.time()-start_time)/3600:.2f}")