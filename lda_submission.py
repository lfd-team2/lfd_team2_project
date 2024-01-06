import numpy as np
import pandas as pd
import preprocessing.normalizer as nmr
import preprocessing.imputer as imp
from preprocessing.lda import lda
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import time

data_frame = pd.read_csv('data/aps_failure_training_set.csv', na_values='na')

cols_to_drop = []
# for col in data_frame.columns:
#     if data_frame[col].isna().mean() > 0.70:
#         cols_to_drop.append(col)

training_data = data_frame.drop(columns=['id', 'class'] + cols_to_drop).values
training_class = data_frame['class'].values


def random_seed(): return int(time.time())


lda_size = 75

# Trying different normalization methods
test_frame = pd.read_csv('data/aps_failure_test_set.csv', na_values='na')
test_data = test_frame.drop(columns=['id'] + cols_to_drop).values
test_ids = test_frame['id'].values

nrm_strat = 0
imp_strat = 0

if nrm_strat == 0:
    strategy = 'zscore'
    training_data, mean, std = nmr.zscore_normalize(
        training_data)
    test_fold = nmr.applyZscoreNormalization(test_data, mean, std)
else:
    strategy = 'minmax'
    training_data, min, minmax = nmr.minmax_normalize(
        training_data)
    test_fold = nmr.applyMinmaxNormalization(test_data, min, minmax)

if imp_strat == 0:
    impute = 'mean'
    training_data = imp.imputeMean(training_data)
    test_fold = imp.imputeMean(test_fold)
elif imp_strat == 1:
    impute = 'median'
    training_data = imp.imputeMedian(training_data)
    test_fold = imp.imputeMedian(test_fold)
elif imp_strat == 2:
    impute = 'most_frequent'
    training_data = imp.imputeMostFrequent(training_data)
    test_fold = imp.imputeMostFrequent(test_fold)
elif imp_strat == 3:
    impute = 'zero'
    training_data = imp.imputeConstant(training_data, 0)
    test_fold = imp.imputeConstant(test_fold, 0)
else:
    impute = 'random'
    training_data = imp.imputeRandomFill(training_data)
    test_fold = imp.imputeRandomFill(test_fold)

print(f"current_iter: {strategy} {impute}")

projected_data, components, component_weights = lda(
    training_data, training_class, lda_size)

# Undersampling the data
undersampler = RandomUnderSampler(
    random_state=random_seed(), sampling_strategy='majority')

projected_data, training_class = undersampler.fit_resample(
    projected_data, training_class)

test_fold = np.dot(test_fold, components)

# Random forest classifier
random_forest = RandomForestClassifier(
    n_estimators=100, random_state=random_seed())

random_forest.fit(projected_data, training_class)

# Testing the model

predict_labels = random_forest.predict(test_fold)

# Calculating the accuracy

file_name = (
    f'{strategy}_{impute}_random_forest_{lda_size}_component_lda')

ids = test_frame['id'].to_list()
resullt_frame = pd.DataFrame({'id': ids, 'class': predict_labels})
resullt_frame.to_csv(
    f'data/automated_tests/{file_name}.csv', index=False)

# Logistic regression classifier
logistic_regression = LogisticRegression(
    random_state=random_seed(), max_iter=1000)

logistic_regression.fit(projected_data, training_class)

# Testing the model
predict_labels = logistic_regression.predict(test_fold)

file_name = (
    f'{strategy} {impute} logistic_regression_{lda_size}_component_lda')

ids = test_frame['id'].to_list()
resullt_frame = pd.DataFrame({'id': ids, 'class': predict_labels})
resullt_frame.to_csv(
    f'data/automated_tests/{file_name}.csv', index=False)


# Normalizing && imputing the data
training_data, mean, std = nmr.minmax_normalize(
    training_data)

training_data = imp.imputeMedian(training_data)

projected_data, components, component_weights = lda(
    training_data, training_class, 3)

# Getting the projected data as classes
class_neg_data = projected_data[np.where(training_class == 'neg')]
class_pos_data = projected_data[np.where(training_class == 'pos')]
