import jax.numpy as jnp
from jax import jit
import jaxopt
import numpy as np
import torch

@jit
def rf_krr_risk_theory(eigvals, eigcoeffs, n, k, ridge, noise_var=0):
    """
    eigvals (jax array @ CPU): eigenvalues
    eigcoeffs (jax array @ CPU): eigenbasis coefficients of target function
    """
    # alright, gotta solve these coupled equations...
    # okay, so could do it with a nested solve.
    # first, choose kappa. (fixing kappa always gives a positive gamma.)
    # then find the gamma that satisfies one equation.
    # plug it into the other equation.
    # return the error signal and optimize kappa to make it zero.
    # this is like finding the intersection of two curves in a plane
    #     by first optimizing to find them, then optimizing the
    #     difference between them to be zero.

    # works fastest if eigvals and eigcoeffs are jax arrays that live on CPU

    # this should equal n
    def sum_1(kappa, gamma):
        eigenlrns = eigvals / (eigvals + gamma)
        return eigenlrns.sum() + ridge / kappa

    # this should equal k
    def sum_2(kappa, gamma):
        eigenlrns = eigvals / (eigvals + gamma)
        return eigenlrns.sum() + k * kappa / gamma

    # inner loop: sum_1 = n
    gamma_optimizer = jaxopt.Bisection(lambda gamma, kappa: sum_1(kappa, gamma) - n,
                                       1e-15, 1e15, maxiter=200, check_bracket=False, jit=True)

    # outer loop: sum_2 = k
    def kappa_error_signal(kappa):
        gamma, _ = gamma_optimizer.run(kappa=kappa)
        return sum_2(kappa, gamma) - k

    # optimize kappa in logspace
    logkap_optimizer = jaxopt.Bisection(lambda logkap: kappa_error_signal(10**logkap),
                                        -15, 15, maxiter=200, check_bracket=False, jit=True)
    logkap, _ = logkap_optimizer.run()
    kappa = 10**logkap
    gamma, _ = gamma_optimizer.run(kappa=kappa)

    eigenlrns = eigvals / (eigvals + gamma)
    z = eigenlrns.sum()
    q = (eigenlrns ** 2).sum()

    x = (q * (k - 2 * z) + z ** 2) / (n * (k - q))
    overfitting_coeff = 1 / (1 - x)

    bias_coeffs = (1 - eigenlrns) - kappa * eigenlrns / (eigvals + gamma) * k / (k - q)

    risk_pred = overfitting_coeff * (bias_coeffs @ eigcoeffs**2 + noise_var)

    return risk_pred, kappa, gamma


@jit
def rf_krr(train_features, test_features, keep_inds, train_y, test_y, ridges):
    num_features = train_features.shape[-1]
    k = len(keep_inds)
    scale = jnp.sqrt(num_features/k)
    train_RF = scale * train_features[:, keep_inds]
    test_RF = scale * test_features[:, keep_inds]
    K_test = test_RF @ train_RF.T
    K_train = train_RF @ train_RF.T
    n = K_train.shape[0]
    train_mses, test_mses = [], []
    for ridge in ridges:
        # run regression
        K_inv = jnp.linalg.inv(K_train + ridge*jnp.eye(n))
        # train mse
        y_hat = K_train @ K_inv @ train_y
        train_mse = ((train_y - y_hat) ** 2).sum(axis=1).mean()
        train_mses.append(train_mse.astype(float))
        # test mse
        y_hat = K_test @ K_inv @ train_y
        test_mse = ((test_y - y_hat) ** 2).sum(axis=1).mean()
        test_mses.append(test_mse.astype(float))
    return train_mses, test_mses


def get_gaussian_dataset_closure(RNG, eigcoeffs, noise_var):
    noise_amp = np.sqrt(noise_var)
    m = len(eigcoeffs)

    def get_gaussian_dataset(n):
        X = RNG.standard_normal(size=(n, m))        
        y = X @ eigcoeffs + noise_amp*RNG.standard_normal(size=n)
        y = y[:, None]

        return (jnp.array(X), jnp.array(y))

    return get_gaussian_dataset


def get_gaussian_feature_map_closure(RNG, eigvals):
    in_dim = len(eigvals)

    def get_gaussian_feature_map():
        proj = RNG.standard_normal(size=(in_dim, in_dim)) / jnp.sqrt(in_dim)
        F = jnp.einsum('ij,j->ij', proj, jnp.sqrt(eigvals))
        def gaussian_feature_map(X):
            return (F @ X.T).T
        return gaussian_feature_map

    return get_gaussian_feature_map


from functools import partial


@partial(jit, static_argnames=('n_train'))
def krr_jax(K, y, ridge, n_train):
    y_train = y[:n_train]
    y_test = y[n_train:]
    K_train = K[:n_train, :n_train]
    K_test = K[:, :n_train]

    regularizer = ridge * jnp.eye(n_train)
    y_hat = K_test @ jnp.linalg.inv(K_train + regularizer) @ y_train
    # train error
    y_hat_train = y_hat[:n_train]
    train_mse = ((y_train - y_hat_train) ** 2).sum(axis=1).mean()
    # test error
    y_hat_test = y_hat[n_train:]
    test_mse = ((y_test - y_hat_test) ** 2).sum(axis=1).mean()
    return train_mse, test_mse


def krr(K, y, n_train, ridge=0):
    K_train, K_test = K[:n_train, :n_train], K[:, :n_train]
    y_train, y_test = y[:n_train], y[n_train:]
    
    if ridge == 0:
        alpha = torch.linalg.lstsq(K_train, y_train).solution
    else:
        eye = torch.eye(n_train, dtype=torch.float32).cuda()
        alpha = torch.linalg.lstsq(K_train + ridge*eye, y_train).solution
    y_hat = K_test @ alpha
    # train error
    y_hat_train = y_hat[:n_train]
    train_mse = ((y_train - y_hat_train) ** 2).sum(axis=1).mean()
    train_mse = train_mse.cpu().numpy()
    # test error
    y_hat_test = y_hat[n_train:]
    test_mse = ((y_test - y_hat_test) ** 2).sum(axis=1).mean()
    test_mse = test_mse.cpu().numpy()
    return train_mse, test_mse


@jit
def calc_kappa(n, eigvals, ridge=0):
    def lrn_sum(kappa):
        eiglearns = eigvals / (eigvals + kappa)
        return eiglearns.sum()
    optimizer = jaxopt.Bisection(lambda kap: lrn_sum(kap) + ridge / kap - n,
                                 1e-15, 1e10, maxiter=200, check_bracket=False, jit=True)
    kappa, _ = optimizer.run()
    return kappa


@jit
def krr_risk_theory(n, eigcoeffs, eigvals, ridge=0, noise_var=0):
    """Get theoretical quantities of interest for a given learning problem and dataset size.
    n (float): training dataset size
    eigcoeffs (jax or numpy array): the coefficients of the target function in the
        eigenbasis (ordered by decreasing eigenvalue)
    eigvals (jax or numpy array): kernel eigenvalues in decreasing order
    ridge (float): ridge parameter. Default: 0
    noise_var (float): The variance of the noise. Default: 0

    Returns: dict{kappa, overfitting_coeff, train_mse, test_mse}
    """
    
    kappa = calc_kappa(n, eigvals, ridge)
    eiglearns = eigvals / (eigvals + kappa)
    e0 = n / (n - (eiglearns**2).sum())

    # compute mse
    test_mse = e0 * (((1-eiglearns)**2 * eigcoeffs**2).sum() + noise_var)
    train_mse = (ridge / (n * kappa))**2 * test_mse

    return {
        "kappa": kappa,
        "overfitting_coeff": e0,
        "train_mse": train_mse,
        "test_mse": test_mse,
    }
        