import numpy as np

def _pad_batch_2D(batch, value = 0):
    max_batch = max([len(x) for x in batch])
    batch = [n + [value] * (max_batch - len(n)) for n in batch]
    batch = np.asarray(batch)
    return batch

def _pad_batch_3D(batch, value = 0):
    max_2nd_D = max([len(x) for x in batch])
    max_3rd_D = max([len(c) for n in batch for c in n])
    batch = [n + ([[]] * (max_2nd_D - len(n))) for n in batch]
    batch = [[c + [value] * (max_3rd_D - len(c)) for c in sample] for sample in batch]
    batch = np.asarray(batch)
    return batch

def _pad_batch_4D(batch, value = 0):
    max_2nd_D = max([len(x) for x in batch])
    max_3rd_D = max([len(c) for n in batch for c in n])
    max_4th_D = max([len(s) for n in batch for c in n for s in c] or [value])
    batch = [n + ([[]] * (max_2nd_D - len(n))) for n in batch]
    batch = [[c + ([[]] * (max_3rd_D - len(c))) for c in sample] for sample in batch]
    batch = [[[s + [value] * (max_4th_D - len(s)) for s in c] for c in sample] for sample in batch]
    batch = np.asarray(batch)
    return batch