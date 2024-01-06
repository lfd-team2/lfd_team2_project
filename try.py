import numpy as np
import pandas as pd
import preprocessing.normalizer as nmr
import preprocessing.imputer as imp

data_frame = pd.read_csv('data/aps_failure_training_set.csv', na_values='na')
data_frame_values = data_frame.drop(columns=['id', 'class']).values

normalized_data = nmr.zscore_normalize(
    data_frame_values, clipping=True, clip_min_percentile=2.5, clip_max_percentile=97.5)

imputed_data = imp.imputeRandomFill(normalized_data[0])

imputed_data = imp.imputeConstant(normalized_data[0], 0)

imputed_data = imp.imputeMostFrequent(normalized_data[0])

imputed_data = imp.imputeMedian(normalized_data[0])

imputed_data = imp.imputeMean(normalized_data[0])

imputed_data = imp.imputeKNN(normalized_data[0])
