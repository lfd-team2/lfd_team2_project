import numpy as np
import pandas as pd
import preprocessing.normalizer as nmr
import preprocessing.imputer as imp
from preprocessing.pca import pca
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler


data_frame = pd.read_csv('data/aps_failure_training_set.csv', na_values='na')

# cols_to_drop = []
# for col in data_frame.columns:
#    if data_frame[col].isna().mean() > 0.70:
#        cols_to_drop.append(col)

training_data = data_frame.drop(columns=['id', 'class']).values
training_class = data_frame['class'].values

# Dropping the columns with more than 70% missing values

# Undersampling the data
undersampler = RandomUnderSampler(
    random_state=42, sampling_strategy='majority')

training_data, training_class = undersampler.fit_resample(
    training_data, training_class)

# Imputing the data
training_data, min, minmax = nmr.minmax_normalize(
    training_data)
training_data = imp.imputeMedian(training_data)

random_forest = RandomForestClassifier(
    n_estimators=100, random_state=42)

random_forest.fit(training_data, training_class)

# Testing the model
test_frame = pd.read_csv('data/aps_failure_test_set.csv', na_values='na')
test_data = test_frame.drop(columns=['id']).values

# Imputing the data
test_data = nmr.applyMinmaxNormalization(test_data, min, minmax)
test_data = imp.imputeMedian(test_data)

# Predicting the test data
test_class = random_forest.predict(test_data)

ids = test_frame['id'].to_list()
resullt_frame = pd.DataFrame({'id': ids, 'class': test_class})
resullt_frame.to_csv(
    'data/result_apply_training.csv', index=False)
