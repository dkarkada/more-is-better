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

    def __init__(self, dataset_name, classes=None, binarize=False):
        """
        dataset_name (str): one of  'mnist', 'fmnist', 'cifar10', 'cifar100'
        classes (iterable): a list of groupings of old class labels that each constitute a new class.
            e.g. [[0,1], [8]] on MNIST would be a binary classification problem where the first class
            consists of samples of 0's and 1's and the second class has samples of 8's
        binarize (boolean): whether to use +1/-1 label encoding. Ignored if num_classes!=2
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
            if n_classes == 2 and binarize:
                y = 2*y - 1
                y = y[:, None] #reshape
            else:
                y = F.one_hot(torch.Tensor(y).long()).numpy()

            # add a dummy channel dimension to MNIST and FMNIST
            if self.name in ['mnist', 'fmnist']:
                x = x[:,:,:,None]

            return x, y

        raw_train = self.dataset_dict[self.name](root='./data', train=True, download=True, transform=None)
        raw_test = self.dataset_dict[self.name](root='./data', train=False, download=True, transform=None)

        # cut and convert raw datasets
        self.train_X, self.train_y = get_xy(raw_train)
        self.test_X, self.test_y = get_xy(raw_test)

    def get_dataset(self, n, rng=None, get="train"):
        """Generate an image dataset.

        n (int): the dataset size
        rng (numpy RNG): numpy RNG state for random sampling. Default: None
        get (str): either "train" or "test." Default: "train"

        Returns: tuple (X, y) such that X.shape = (n, d_in), y.shape = (n, d_out)
        """
        
        assert int(n) == n
        n = int(n)
        assert n > 0
        assert get in ["train", "test"]
        if get == "train":
            full_X, full_y = self.train_X, self.train_y
        else:
            full_X, full_y = self.test_X, self.test_y
        
        # get subset
        idxs = slice(n) if rng is None else rng.choice(len(full_X), size=n, replace=False)
        X, y = full_X[idxs], full_y[idxs]
        assert len(X) == n

        # flatten
        X = X.reshape((len(X), -1))

        return X, y