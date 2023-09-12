import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def int_logspace(low, high, num, base):
    arr = np.logspace(low, high, num=num, base=base).astype(int)
    for i in range(1, len(arr)):
        if arr[i] <= arr[i-1]:
            arr[i] = arr[i-1] + 1
    return arr

def rcsetup():
    np.set_printoptions(suppress=True)
    plt.rc("figure", dpi=120, facecolor=(1, 1, 1))
    plt.rc("font", family='stixgeneral', size=12)
    plt.rc("axes", titlesize=12)
    plt.rc("axes", facecolor=(1, .99, .95))
    plt.rc("mathtext", fontset='cm')

def save(obj, fn):
    if fn.endswith('.npy'):
        assert isinstance(obj, np.ndarray)
        np.save(fn, obj)
        return
    with open(fn, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(fn):
    if not os.path.isfile(fn):
        print(f"{fn} not found")
        return None
    if fn.endswith('.npy'):
        obj = np.load(fn)
        return obj
    with open(fn, 'rb') as handle:
        obj = pickle.load(handle)
    return obj   