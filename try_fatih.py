import numpy as np
import pandas as pd
import preprocessing.normalizer as nmr
import preprocessing.imputer as imp
from preprocessing.pca import pca
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler


data_frame = pd.read_csv('data/aps_failure_training_set.csv', na_values='na')

training_data = data_frame.drop(columns=['id', 'class']).values
training_class = data_frame['class'].values

# Normalizing the data
training_data, mean, std = nmr.zscore_normalize(
    training_data, clipping=True, clip_min_percentile=2.5, clip_max_percentile=97.5)

# Imputing the data
training_data = imp.imputeMedian(training_data)

# Undersampling the data
undersampler = RandomUnderSampler(random_state=0, sampling_strategy='majority')
training_data, training_class = undersampler.fit_resample(
    training_data, training_class)

# Applying PCA
training_data, eigenvectors, eigenvalues = pca(training_data, 70)
pca_components = eigenvectors[:, 0:70]

random_forest = RandomForestClassifier(
    n_estimators=100, random_state=0, max_depth=10)

random_forest.fit(training_data, training_class)

# Testing the model
test_frame = pd.read_csv('data/aps_failure_test_set.csv', na_values='na')
test_data = test_frame.drop(columns=['id']).values

# Normalizing the data
# test_data, mean, std = nmr.zscore_normalize(test_data)
test_data = nmr.clipping_normalize(test_data, 2.5, 97.5)
test_data = nmr.applyZscoreNormalization(test_data, mean, std)

# Imputing the data
test_data = imp.imputeMedian(test_data)

# Applying PCA
test_data = np.dot(test_data, pca_components)


# Predicting the test data
test_class = random_forest.predict(test_data)

ids = test_frame['id'].to_list()
resullt_frame = pd.DataFrame({'id': ids, 'class': test_class})
resullt_frame.to_csv(
    'data/result_apply_training.csv', index=False)
