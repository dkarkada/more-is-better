import numpy as np

import torch

import os
import sys
import time

start_time = time.time()

sys.path.insert(0, 'more-is-better')

from utils import save, load, load_kernel
from eigsolver import eigsolve
from exptdetails import ExptDetails

args = sys.argv

RNG = np.random.default_rng()

DATASET_NAME = str(args[1])
EXPT_NUM = int(args[2])
DEPTH = int(args[3])
N = int(args[4])

N_RIDGES = 10

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

K = load_kernel(N+1000, work_dir)
assert np.allclose(K, K.T), np.sum((K-K.T))**2

eigvals, eigvecs, eigcoeffs = eigsolve(K[:N, :N], y[:N])

log_max_ridge = int(np.log10(eigvals.max())) + 3
log_min_ridge = int(np.log10(eigvals.min())) - 3
print(f"max eigval {eigvals.max():.5f}, min eigval {eigvals.min():.5f}")
print(f"ridge ranging from 10^{log_min_ridge} to 10^{log_max_ridge}")
ridges = np.logspace(log_min_ridge, log_max_ridge, base=10, num=N_RIDGES)
noises = np.array([0, 0.5, 5])

K = torch.from_numpy(K).cuda()
y = torch.from_numpy(y).cuda()
K_train, K_test = K[:N, :N], K[:, :N]
y_train, y_test = y[:N], y[N:N+1000]

# do ridgeless noiseless KR
y_hat = K_test @ torch.linalg.inv(K_train) @ y_train
y_hat_test = y_hat[N:]
base_mse = ((y_test - y_hat_test) ** 2).sum(axis=1).mean()

test_mses = {noise: {} for noise in noises}
train_mses = {noise: {} for noise in noises}
eye = torch.eye(N, dtype=torch.float32).cuda()
for noise_relative in noises:
    print(f"Noise {noise_relative}: ", end='')
    # TODO do the noise thing
    noise_absolute = noise_relative * base_mse
    y_noise = torch.normal(0, torch.sqrt(noise_absolute),
                           size=y_train.size(), dtype=torch.float32)    
    for ridge in ridges:
        print('.', end='')
        K_inv = torch.linalg.inv(K_train + ridge*eye)
        y_hat = K_test @ K_inv @ (y_train + y_noise)
        # train error
        y_hat_train = y_hat[:N]
        train_mse = ((y_train - y_hat_train) ** 2).sum(axis=1).mean()
        train_mse = train_mse.cpu().numpy()
        # test error
        y_hat_test = y_hat[N:]
        test_mse = ((y_test - y_hat_test) ** 2).sum(axis=1).mean()
        test_mse = test_mse.cpu().numpy()
        
        test_mses[noise_relative][ridge] = test_mse
        train_mses[noise_relative][ridge] = train_mse
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