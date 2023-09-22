import numpy as np

import os
import sys

sys.path.insert(0, 'more-is-better')

from utils import save, load

# chunk size 5000 -> 10000

expt = "cntk5-clean"

kernel_dir = "/scratch/bbjr/dkarkada/kernel-matrices"
work_dir = f"{kernel_dir}/{expt}"
assert os.path.exists(work_dir)


metadata = load(f"{work_dir}/metadata.file")
assert metadata is not None

def convert(old_flags):
    sz = old_flags.shape[0]
    flags = np.zeros((sz//2, sz//2))
    for (i, j) in np.ndindex(*flags.shape):
        chunk = old_flags[i*2:(i+1)*2, j*2:(j+1)*2]
        flags[i, j] = chunk.all()
    return flags

metadata["flags_20k"] = convert(metadata["flags_20k"])
metadata["flags_50k"] = convert(metadata["flags_50k"])
print(metadata["flags_20k"])
print(metadata["flags_50k"])

save(metadata, f"{work_dir}/metadata.file")