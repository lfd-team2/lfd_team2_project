import numpy as np


def zscore_normalize(data, clipping=False, clip_min_percentile=0, clip_max_percentile=100):
    if (clipping):
        data = clipping_normalize(
            data, clip_min_percentile, clip_max_percentile)

    columns_mean = np.mean(data, axis=0)
    columns_std = np.std(data, axis=0)
    normalized_data = (data - columns_mean) / columns_std
    return normalized_data, columns_mean, columns_std


# Normalizing the data by cramming them in between 0 and 1
def minmax_normalize(data, clipping=False, clip_min_percentile=0, clip_max_percentile=100):
    if (clipping):
        data = clipping_normalize(
            data, clip_min_percentile, clip_max_percentile)

    columns_min = np.min(data, axis=0)
    columns_max = np.max(data, axis=0)
    normalized_data = (data - columns_min) / (columns_max - columns_min)
    return normalized_data, columns_min, columns_max


# Normalizing the data by clipping them in their percentiles (between 0 and 100)
def clipping_normalize(data, clip_min_percentile, clip_max_percentile):
    clip_min = np.percentile(data, clip_min_percentile)
    clip_max = np.percentile(data, clip_max_percentile)
    return np.clip(data, clip_min, clip_max)
