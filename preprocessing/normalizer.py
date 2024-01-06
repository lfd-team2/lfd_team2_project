import numpy as np


def zscore_normalize(data, clipping=False, clip_min_percentile=0, clip_max_percentile=100):
    if (clipping):
        data = clipping_normalize(
            data, clip_min_percentile, clip_max_percentile)

    columns_mean = np.nanmean(data, axis=0)
    columns_std = np.nanstd(data, axis=0)
    # If the standard deviation is 0, then we set it to 1
    columns_std[columns_std == 0] = 1
    normalized_data = (data - columns_mean) / columns_std
    return normalized_data, columns_mean, columns_std


# Normalizing the data by cramming them in between 0 and 1
def minmax_normalize(data, clipping=False, clip_min_percentile=0, clip_max_percentile=100):
    if (clipping):
        data = clipping_normalize(
            data, clip_min_percentile, clip_max_percentile)

    columns_min = np.nanmin(data, axis=0)
    columns_max = np.nanmax(data, axis=0)
    columns_minmax = columns_max - columns_min
    # If the min and max values are the same, then we set the max value to 1
    columns_minmax[columns_minmax == 0] = 1
    normalized_data = (data - columns_min) / columns_minmax
    return normalized_data, columns_min, columns_minmax


# Normalizing the data by clipping them in their percentiles (between 0 and 100)
def clipping_normalize(data, clip_min_percentile=0, clip_max_percentile=100):
    # Get the min and max values from the data by their percentiles
    clip_min = np.nanpercentile(data, clip_min_percentile)
    clip_max = np.nanpercentile(data, clip_max_percentile)
    # We want to leave nans as they are
    nan_mask = np.isnan(data)
    def clipper(x): return np.clip(x, clip_min, clip_max)
    # Apply the clipper function to the data with leaving the nan values as they are
    clipped_data = np.where(nan_mask, data, clipper(data))
    return clipped_data
