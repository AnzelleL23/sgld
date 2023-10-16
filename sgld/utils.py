import numpy as np
import numpy.lib.stride_tricks as npst
import os
import matplotlib.pyplot as plt

def lossrate(t, a, b, gamma):
    return a * np.power(b, gamma) / np.power((t + b), gamma)


def apply_rolling1D(seq, func, window=20):
    """
    this applies the function 'func' to seq over rolling
    time windows of size 'window'
    note: seq is 1D
    """
    stride = seq.strides[0]
    sequence_strides = npst.as_strided(seq, shape=[len(seq) - window + 1, window], strides=[stride, stride])
    return func(sequence_strides)


def apply_rolling2D(seq, func, window=20, stack=True):
    """
    here seq is of dimension time x examples
    """
    result = []
    for row in seq.T:
        row_result = apply_rolling1D(row, func, window)
        result.append(row_result)

    if stack:
        return np.vstack(result).T
    else:
        return result


def make_dict(experiment_dir, basedir):
    """
    returns dictionary based on elements in the given directory
    """
    results = {}
    for filename in os.listdir(os.path.join(basedir, experiment_dir)):
        if filename[-3:] == 'npy':
            path = os.path.join(basedir, experiment_dir, filename)
            tesults[filename[:-4]] = np.load(path)

    return results


def state_dict_histo_2_numpy(state_dict_histo):
    history_np = []
    for timestep_data in state_dict_histo:
        history_timestep = []
        for key, val in timestep_data.items():
            history_timestep.append(val.ravel())

        history_np.append(np.hstack(history_timestep))

    return np.vstack(history_np)

def visualise_samples(data_loader, num_samples):
    for batch_idx, (data, target) in enumerate(data_loader):
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(data[i].numpy().squeeze(), cmap='gray')
            plt.title(f'Target: {target[i].item()}')
            plt.axis('off')
        plt.show()
        break  # Break after one batch