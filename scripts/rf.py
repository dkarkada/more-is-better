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
from eigsolver import eigsolve
from exptdetails import ExptDetails
from ExperimentResults import ExperimentResults

args = sys.argv

RNG = np.random.default_rng()

DATASET_NAME = str(args[1])
EXPT_NUM = int(args[2])
DEPTH = int(args[3])
N = int(args[4])

N_SIZES = 31
N_TRIALS = 15
N_RIDGES = 

DATASET_NAME = DATASET_NAME.lower()
assert DATASET_NAME in ['cifar10', 'cifar100', 'emnist',
                        'mnist', 'imagenet32', 'imagenet64']

expt_details = ExptDetails(EXPT_NUM, DEPTH, DATASET_NAME)
expt_name = expt_details.expt_name
print(f"Optimal ridge: {expt_name} @ {DATASET_NAME}")
    
kernel_dir = "/scratch/bbjr/dkarkada/kernel-matrices"
work_dir = f"{kernel_dir}/{DATASET_NAME}/{expt_name}"
assert os.path.exists(work_dir), work_dir

dataset = load(f"{work_dir}/dataset.file")
assert dataset is not None
_, y = dataset

RNG = onp.random.default_rng(seed=42)

## START EXPERIMENT

def get_cifar10_dataset_closure():
    cifar10 = ImageData("cifar10", classes=binarization)

    def get_cifar10_dataset(n):
        train_X, train_y, test_X, test_y = cifar10.get_dataset(n, n_test=1000, rng=RNG)
        return (jnp.array(train_X), jnp.array(train_y),
                jnp.array(test_X), jnp.array(test_y))

    return get_cifar10_dataset


def get_relu_feature_map():
    w = 10000 # width
    W = jnp.array(onp.sqrt(2/3072) * RNG.standard_normal(size=(w, 3072)))
    def relu_feature_map(X):
        # factor sqrt(1/w) to ensure kernel is O(1)
        WX = jnp.sqrt(1/w) * (W @ X.T).T
        return jax.nn.relu(WX)
    return relu_feature_map



binarization = [[0, 1, 7, 8, 9], [2, 3, 4, 5, 6]]
meta = {
    "binarization": binarization,
}

with open(f"{kernel_dir}/20k/eigendata.file", 'rb') as f:
    eigendata = pickle.load(f)
eigvals = eigendata["relu_kernel_cifar10"]["eigvals"]
eigcoeffs = eigendata["relu_kernel_cifar10"]["eigcoeffs"]
eigcoeffs = eigcoeffs[:, binarization[0]].sum(axis=1) \
             - eigcoeffs[:, binarization[1]].sum(axis=1)
eigcoeffs /= onp.linalg.norm(eigcoeffs)
idxs = 1 + onp.arange(len(eigvals))

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
                mse, kappa, gamma = rf_risk_prediction(eigvals, eigcoeffs, n, k, ridge, noise_var)
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