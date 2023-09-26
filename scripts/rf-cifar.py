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
from theory import rf_krr_risk_theory
from imagedata import ImageData
from exptdetails import ExptDetails
from ExperimentResults import ExperimentResults

args = sys.argv

RNG = np.random.default_rng()

# n_train up to 10000, k up to 10000

N_SIZES = 31
N_TRIALS = 15
N_RIDGES = 31

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

#continue from here

# put on CPU to speed up theory calculation
cpus = jax.devices("cpu")
eigvals = jax.device_put(eigvals, cpus[0])
eigcoeffs = jax.device_put(eigcoeffs, cpus[0])



vary_dim = int_logspace(1, 4, base=10, num=40)
vary_dim_peak = int_logspace(2, 3, base=10, num=30)
vary_dim = np.unique(np.concatenate((vary_dim, vary_dim_peak),0))
fixed_dim = [256]
ridges = np.logspace(-12, 12, base=2, num=25)

theory_n256_axes = [
    ("n", n_trains),
    ("k", kk),
    ("ridge", ridges),
    ("result", ["test_mse", "kappa", "gamma"])
]
theory_n256 = ExperimentResults(theory_n256_axes, f"{work_dir}/theory-n256.file", meta)

for n in n_trains:
    for k in kk:
        print('.', end='')
        for ridge in ridges:
            mse, kappa, gamma = rf_krr_risk_theory(eigvals, eigcoeffs, n, k, ridge, noise_var=0)
            results = [mse, kappa, gamma]
            theory_n256.write(results, n=n, k=k, ridge=ridge)
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