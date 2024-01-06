# Modified version of pca from the homework 1

import numpy as np
import preprocessing.normalizer as nmr


def pca(data, num_components=2, is_normalized=True, mode='zscore', clipping=False, clip_min_percentile=0, clip_max_percentile=100):
    # Apply normalization if needed
    min_vector
    max_vector
    mean_vector
    std_vector

    if (not is_normalized):
        if (mode == 'zscore'):
            data, mean_vector, std_vector = nmr.zscore_normalize(
                data, clipping, clip_min_percentile, clip_max_percentile)
        if (mode == 'minmax'):
            # Normalizing the data by min max method
            data, min_vector, max_vector = nmr.minmax_normalize(
                data, clipping, clip_min_percentile, clip_max_percentile)

    # Sorting eigenvector and values in the descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]

    components = eigenvectors[:, 0:num_components]
