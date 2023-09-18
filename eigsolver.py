import torch
import numpy as np
import scipy

# need to use numpy for cpu eigendecomp -- pytorch 32bit-interface can't handle large matrices 
# see https://github.com/pytorch/pytorch/issues/92141
# see https://github.com/numpy/numpy/issues/13956
# scipy eigh calls dsyevr, better (?) than numpy which uses dsyevd


def _gpu_eigs(K, y, n):
    # K, y numpy arrays
    K = torch.from_numpy(K).cuda()
    y = torch.from_numpy(y).cuda()
    eigvals, eigvecs = torch.linalg.eigh(K)
    eigvals = eigvals.cpu().numpy()

    # eigvals are now on cpu; eigvecs still on gpu
    eigvals /= n
    eigvecs *= np.sqrt(n)

    eigcoeffs = (1/n) * eigvecs.T @ y
    eigvecs = eigvecs.cpu().numpy()
    eigcoeffs = eigcoeffs.cpu().numpy()
    torch.cuda.empty_cache()
    return eigvals, eigvecs, eigcoeffs


def _cpu_eigs(K, y, n):
    # K, y numpy arrays
    eigvals, eigvecs = scipy.linalg.eigh(K)

    eigvals /= n
    eigvecs *= np.sqrt(n)

    eigcoeffs = (1/n) * eigvecs.T @ y
    return eigvals, eigvecs, eigcoeffs


def eigsolve(K, y):
    n = K.shape[0]
    assert K.shape == (n, n)
    assert y.shape[0] == n and y.ndim == 2
    K, y = K.astype(np.float64), y.astype(np.float64)
    eigs = _gpu_eigs if n < 32767 else _cpu_eigs  # 32767 = 2^15 - 1
    eigvals, eigvecs, eigcoeffs = eigs(K, y, n)
    
    # Sort in descending eigval order
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[::-1]
    eigcoeffs = eigcoeffs[::-1]
    
    return eigvals, eigvecs, eigcoeffs