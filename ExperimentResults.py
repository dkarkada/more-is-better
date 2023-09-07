import numpy as np
import pickle


class ExperimentResults():

    def __init__(self, axes, filename, metadata=None):
        self.filename = filename
        self.metadata = metadata
        self.axes = axes.copy()
        shape = [len(axis) for (_, axis) in axes]
        self.results = np.zeros(shape=shape)
        self.written = np.zeros(shape=shape)

        converter = lambda vals: {val: idx for idx, val in enumerate(vals)}
        self.idx_converter = {ax_name: converter(vals) for (ax_name, vals) in self.axes}

    def _to_idxs(self, **kwargs):
        idxs = []
        for (ax_name, _) in self.axes:
            if ax_name in kwargs:
                ax_val = kwargs[ax_name]
                idx = self.idx_converter[ax_name].get(ax_val)
                assert idx is not None, f"{ax_name}={ax_val}"
                idxs.append(slice(idx, idx+1))
            else:
                idxs.append(slice(None))
        return tuple(idxs)

    def is_written(self, **kwargs):
        idxs = self._to_idxs(**kwargs)
        return np.all(self.written[idxs])

    def get(self, stats_axis=None, **kwargs):
        idxs = self._to_idxs(**kwargs)
        if not np.all(self.written[idxs]):
            print("warning: not all values have been written to")
        result = self.results[idxs]
        if stats_axis:
            ax = self.get_axis(stats_axis, get_idx=True)
            mean = result.mean(axis=ax).squeeze()
            std = result.std(axis=ax).squeeze()
            return mean, std
        return result.squeeze()

    def get_axis(self, axis_name, get_idx=False):
        for i, (ax_name, vals) in enumerate(self.axes):
            if ax_name == axis_name:
                return i if get_idx else vals
        print(f"Axis '{axis_name}' not found.")
        return False

    def write(self, write_vals, save_after=True, **kwargs):
        idxs = self._to_idxs(**kwargs)
        idxs = tuple([slc if slc.start is None else slc.start for slc in idxs])
        hole_shape, fill_shape = np.shape(self.results[idxs]), np.shape(write_vals)
        if hole_shape != fill_shape:
            print(f"Bad shape: writing into shape = {hole_shape}, # writes = {fill_shape}.")
            return
        self.results[idxs] = write_vals
        self.written[idxs] = True
        if save_after:
            self.save()

    def print_axes(self):
        for (ax_name, axis) in self.axes:
            print(f"{ax_name}: {axis}")

    def save(self):
        with open(self.filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(filename):
        with open(filename, 'rb') as handle:
            return pickle.load(handle)

        
