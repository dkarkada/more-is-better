import numpy as np
from imagedata import blockwise_shuffle, shuffle_frac


class ExptDetails:
    block_sizes = {
        10: 2, 11: 3, 12: 4, 13: 5, 14: 8,
    }
    corruption_fracs = {
        20: 0.2, 21: 0.5, 22: 1,
    }
    
    def __init__(self, expt_num, depth, dataset_name):
        self.expt_num = expt_num
        self.depth = depth
        self.dataset_name = dataset_name
        if expt_num == 0:
            expt_name = f"myrtle{depth}-clean"
            msg = f"Myrtle depth-{depth} CNTK @ vanilla {dataset_name}"
        if expt_num in [10, 11, 12, 13, 14]:
            sz = self.block_sizes[expt_num]
            expt_name = f"myrtle{depth}-{sz}px-block-shuffle"
            msg = f"Myrtle depth-{depth} CNTK @ {dataset_name}, block-shuffled (blocksize {sz}px)"
        if expt_num in [20, 21, 22]:
            frac = self.corruption_fracs[expt_num]
            expt_name = f"myrtle{depth}-{frac*100:.0f}pct-shuffle"
            msg = f"Myrtle depth-{depth} CNTK @ {dataset_name}, {frac*100:.0f}% shuffled"
        self.expt_name = expt_name
        self.msg = msg
        
    def modify_dataset(self, dataset):
        X, y = dataset
        RNG = np.random.default_rng(seed=42)
        if self.expt_num in [10, 11, 12, 13, 14]:
            sz = self.block_sizes[self.expt_num]
            X = np.array([blockwise_shuffle(img, RNG, block_size=sz)
                        for img in X])
        if self.expt_num in [20, 21, 22]:
            frac = self.corruption_fracs[self.expt_num]
            X = np.array([shuffle_frac(img, RNG, corrupt_fraction=frac)
                        for img in X])
        return X, y