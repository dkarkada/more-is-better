import numpy as np

import os
import sys
import time

start_time = time.time()

sys.path.insert(0, 'more-is-better')

from imagedata import ImageData
from exptdetails import ExptDetails
from utils import save, load

args = sys.argv

DATASET_NAME = str(args[1])
EXPT_NUM = int(args[2])
DEPTH = int(args[3])

DATASET_NAME = DATASET_NAME.lower()
assert DATASET_NAME in ['cifar10', 'cifar100', 'emnist',
                        'mnist', 'imagenet32', 'imagenet64']

expt_details = ExptDetails(EXPT_NUM, DEPTH, DATASET_NAME)
expt_name = expt_details.expt_name
print(f"remaking datatset: {expt_name} cntk @ {DATASET_NAME}")

kernel_dir = "/scratch/bbjr/dkarkada/kernel-matrices"
work_dir = f"{kernel_dir}/{DATASET_NAME}/{expt_name}"
if not os.path.exists(work_dir):
    print(f"Making directory {work_dir}")
    os.makedirs(work_dir)
with open(f"{work_dir}/readme.txt", 'w') as f:
    f.write(expt_details.msg)

dataset = load(f"{work_dir}/dataset.file")
if dataset is not None:
    print("Going to replace dataset.")

print("Generating dataset... ", end='')
data_generator = ImageData(DATASET_NAME, work_dir=kernel_dir)
dataset = data_generator.get_dataset(50000, flatten=False)
dataset = expt_details.modify_dataset(dataset)
print(dataset[0].dtype(), dataset[1].dtype())
save(dataset, f"{work_dir}/dataset.file")
print("done")