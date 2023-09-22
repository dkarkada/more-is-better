from functools import partial

@partial(jit, static_argnames=['n_train'])
def krr(K, y, ridges, idxs, n_train):
    train_idxs = idxs[:n_train]
    test_idxs = idxs[n_train:]
    y_test = y[test_idxs]
    y_train = y[train_idxs]
    K_test = K[idxs[:, None], train_idxs[None, :]]
    K_train = K[train_idxs[:, None], train_idxs[None, :]]
    eye = jnp.eye(len(train_idxs))
    train_mses, test_mses = [], []
    for ridge in ridges:
        K_inv = jnp.linalg.inv(K_train + ridge*eye)
        y_hat = K_test @ K_inv @ y_train
        # train error
        y_hat_train = y_hat[:n_train]
        train_mse = ((y_train - y_hat_train) ** 2).sum(axis=1).mean()
        train_mses.append(train_mse.astype(float))
        # test error
        y_hat_test = y_hat[n_train:]
        test_mse = ((y_test - y_hat_test) ** 2).sum(axis=1).mean()
        test_mses.append(test_mse.astype(float))
    return train_mses, test_mses

def krr_expt(expt, n_test, rng, load_dir):
    n_trains = expt.get_axis("n")
    ridges = expt.get_axis("ridge")
    kernel_names = expt.get_axis("kernel")
    with open(f"{load_dir}/eigendata.file", 'rb') as f:
        eigendata = pickle.load(f)

    for kn in kernel_names:
        if expt.is_written(kernel=kn):
            print(f"Already computed {kn}")
            continue
        print(f"starting {kn}: ", end='')
        with open(f"{load_dir}/{kn}.file", 'rb') as f:
            kernel_data = pickle.load(f)
        K = jnp.array(kernel_data['K'])
        K /= (eigendata[kn]["eigvals"]).max()
        y = jnp.array(kernel_data['y'])
        n_total = K.shape[0]
        assert max(n_trains) + n_test <= n_total
        for n_train in n_trains:
            if expt.is_written(kernel=kn, n=n_train):
                print(f"Already computed n={n_train} on {kn}")
                continue
            print('.', end='')
            n = n_train + n_test
            for trial in trials:
                idxs = rng.choice(n_total, size=n, replace=False)
                train_mses, test_mses = krr(K, y, ridges, idxs, n_train)
                result = onp.array([train_mses, test_mses]).T
                expt.write(result, n=n_train, kernel=kn, trial=trial)
        print()
        for x in jax.devices()[0].client.live_buffers():
            if x.size > 5000:
                x.delete()
                
DO_SMALL = False
expt_name = f"ridge-error-{'small' if DO_SMALL else 'large'}-n"

RNG = onp.random.default_rng(seed=42)

n_trains = int_logspace(5, 14, base=2, num=10)
n_trains = n_trains[:7] if DO_SMALL else n_trains[7:]
ridges = onp.logspace(-8, 4, base=10, num=100)
trials = onp.arange(25 if DO_SMALL else 5)
kernels = ["fcntk_2hl_mnist", "fcntk_2hl_cifar10", "relu_nngpk_4hl_cifar10",
           "myrtle_ntk_cifar10",]

axes = [
    ("n", n_trains),
    ("ridge", ridges),
    ("kernel", kernels),
    ("trial", trials),
    ("result", ["train_mse", "test_mse"])
]
expt = ExperimentResults(axes, f"{expt_dir}/expt-{expt_name}.file")
# expt = ExperimentResults.load(f"{expt_dir}/expt-{expt_name}.file")
krr_expt(expt, n_test=1000, rng=RNG, load_dir=f"{kernel_dir}/20k")
expt = ExperimentResults.load(f"{expt_dir}/expt-{expt_name}.file")