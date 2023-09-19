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
        return None
    if fn.endswith('.npy'):
        obj = np.load(fn)
        return obj
    with open(fn, 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def load_kernel(n, work_dir):
    K = np.zeros((n, n))
    sz = 10000
    def get_start_end(x):
        start = x*sz
        end = (x+1)*sz if x < n//sz else n
        return start, end
    for i, j in np.ndindex(n//sz + 1, n//sz + 1):
        i_start, i_end = get_start_end(i)
        j_start, j_end = get_start_end(j)
        delta_i, delta_j = i_end-i_start, j_end-j_start
        if delta_i * delta_j == 0:
            continue
        tile = load(f"{work_dir}/tile-{i}-{j}.npy")
        assert tile is not None, f"Tile {i} {j} not yet computed!"
        assert tile.shape == (sz, sz)
        tile = tile[:delta_i, :delta_j]
        K[i_start:i_end, j_start:j_end] = tile
        if i != j:
            K[j_start:j_end, i_start:i_end] = tile.T
    return K
