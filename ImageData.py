import torch
import torch.nn.functional as F
import torchvision

import numpy as np

class ImageData():

    dataset_dict = {
        'mnist': torchvision.datasets.MNIST,
        'fmnist': torchvision.datasets.FashionMNIST,
        'cifar10': torchvision.datasets.CIFAR10,
        'cifar100': torchvision.datasets.CIFAR100,
    }

    def __init__(self, dataset_name, classes=None):
        """
        dataset_name (str): one of  'mnist', 'fmnist', 'cifar10', 'cifar100'
        classes (iterable): a list of groupings of old class labels that each constitute a new class.
            e.g. [[0,1], [8]] on MNIST would be a binary classification problem where the first class
            consists of samples of 0's and 1's and the second class has samples of 8's
        """

        assert dataset_name in self.dataset_dict
        self.name = dataset_name
        self.dataset = self.dataset_dict[dataset_name]

        def get_xy(dataset):
            x = dataset.data.numpy() if self.name not in ['cifar10','cifar100'] else dataset.data
            y = dataset.targets.numpy() if self.name not in ['cifar10','cifar100'] else dataset.targets
            n_classes = int(max(y)) + 1

            if classes is not None:
                # convert old class labels to new
                converter = -1 * np.ones(n_classes)
                for new_class, group in enumerate(classes):
                    group = [group] if type(group) == int else group
                    for old_class in group:
                        converter[old_class] = new_class
                # remove datapoints not in new classes
                mask = (converter[y] >= 0)
                x = x[mask]
                y = converter[y][mask]
                # update n_classes
                n_classes = int(max(y)) + 1

            # normalize globally (correct for the overall mean and std)
            x = (x - x.mean())/x.std()

            # onehot encoding, unless binary classification (+1,-1)
            if n_classes != 2:
                y = F.one_hot(torch.Tensor(y).long()).numpy()
            else:
                y = 2*y - 1
                y = y[:, None] #reshape

            # add a dummy channel dimension to MNIST and FMNIST
            if self.name in ['mnist', 'fmnist']:
                x = x[:,:,:,None]

            return x, y

        raw_train = self.dataset_dict[self.name](root='./data', train=True, download=True, transform=None)
        raw_test = self.dataset_dict[self.name](root='./data', train=False, download=True, transform=None)

        # cut and convert raw datasets
        self.train_X, self.train_y = get_xy(raw_train)
        self.test_X, self.test_y = get_xy(raw_test)

    def get_dataset(self, n_train, n_test=None, rng=None):
        """Generate an image dataset.

        n_train (int): the trainset size, must be at least 2.
        n_test (int): the testset size. Testset may overlap trainset. Default: full image test set
        rng (numpy RNG): numpy RNG state for random sampling.

        Returns: train_X, train_y, test_X, test_y
        """

        train_X, train_y = self.train_X, self.train_y
        test_X, test_y = self.test_X, self.test_y

        # get training and test subset
        if rng is None:
            train_X, train_y = train_X[:n_train], train_y[:n_train]
            if n_test is not None:
                test_X, test_y = test_X[:n_test], test_y[:n_test]
        else:
            train_idxs = rng.choice(len(train_X), size=int(n_train), replace=False)
            train_X, train_y = train_X[train_idxs], train_y[train_idxs]
            if n_test is not None:
                test_idxs = rng.choice(len(test_X), size=int(n_test), replace=False)
                test_X, test_y = test_X[test_idxs], test_y[test_idxs]
        assert len(train_X) == n_train
        if n_test:
            assert len(test_X) == n_test

        # flatten
        train_X, test_X = train_X.reshape((len(train_X), -1)), test_X.reshape((len(test_X), -1))

        return train_X, train_y, test_X, test_y