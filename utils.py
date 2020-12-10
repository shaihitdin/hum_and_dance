import numpy as np
import time

bone_list = [[1, 2],[4, 5], [5, 6], [6, 7], [4, 13], [13, 14], [14, 15], [15, 16], [3, 8], [8, 9], [9, 10], [10, 12], [10, 11], [3, 17], [17, 18], [18, 19], [19, 21], [19, 20],[7,23],[16,22]]
bone_list = np.array(bone_list) - 1

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
