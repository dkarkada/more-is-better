import pickle
import os
import sys

sys.path.insert(0,'more-is-better')

from kernels import MyrtleNTK
from ImageData import ImageData
import time

def save(results):
    with open(results['fn'], 'wb') as f:
        pickle.dump(results, f)

RESULTS = {
    "fn": "/scratch/bbjr/dkarkada/results.pickle"
}

print('starting')
cifar10 = ImageData('cifar10')
kernel_fn = MyrtleNTK(depth=10)
for n in [400, 800, 1600, 3200]:
    X, y = cifar10.get_dataset(n, flatten=False)
    start = time.time()
    kernel_fn(X, get='ntk').block_until_ready()
    elapsed = time.time() - start
    RESULTS[n] = elapsed
    print(f"{n}: {elapsed}")
    save(RESULTS)
