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
for col in data_frame.columns:
    if data_frame[col].isna().mean() > 0.70:
        cols_to_drop.append(col)

training_data = data_frame.drop(columns=['id', 'class'] + cols_to_drop).values
training_class = data_frame['class'].values


def random_seed(): return int(time.time())

# Trying different normalization methods


metrics_outputs = []
for i in range(2):
    for j in range(5):
        if i == 0:
            strategy = 'zscore'
            training_data, mean, std = nmr.zscore_normalize(
                training_data)
        else:
            strategy = 'minmax'
            training_data, min, minmax = nmr.minmax_normalize(
                training_data)

        if j == 0:
            impute = 'mean'
            training_data = imp.imputeMean(training_data)
        elif j == 1:
            impute = 'median'
            training_data = imp.imputeMedian(training_data)
        elif j == 2:
            impute = 'most_frequent'
            training_data = imp.imputeMostFrequent(training_data)
        elif j == 3:
            impute = 'zero'
            training_data = imp.imputeConstant(training_data, 0)
        else:
            impute = 'random'
            training_data = imp.imputeRandomFill(training_data)

        print(f"current_iter: {i} {j} {strategy} {impute}")

        projected_data, components, component_weights = lda(
            training_data, training_class, 70)

        train_fold, test_fold, train_labels, test_labels = train_test_split(
            projected_data, training_class, test_size=0.2, random_state=random_seed())

        # Undersampling the data
        undersampler = RandomUnderSampler(
            random_state=random_seed(), sampling_strategy='majority')

        train_fold, train_labels = undersampler.fit_resample(
            train_fold, train_labels)

        # Random forest classifier
        random_forest = RandomForestClassifier(
            n_estimators=100, random_state=random_seed())

        random_forest.fit(train_fold, train_labels)

        # Testing the model

        predict_labels = random_forest.predict(test_fold)

        # Calculating the accuracy
        accuracy_rf = metrics.accuracy_score(test_labels, predict_labels)
        presicion_rf = metrics.precision_score(
            test_labels, predict_labels, pos_label='pos')
        recall_rf = metrics.recall_score(
            test_labels, predict_labels, pos_label='pos')
        f1_rf = metrics.f1_score(
            test_labels, predict_labels, pos_label='pos')

        metrics_outputs.append(f'{strategy} {impute} random forest accuracy: {accuracy_rf}, precision: \
            {presicion_rf}, recall: {recall_rf}, f1: {f1_rf}')

        # Logistic regression classifier
        logistic_regression = LogisticRegression(
            random_state=random_seed(), max_iter=100)

        logistic_regression.fit(train_fold, train_labels)

        # Testing the model
        predict_labels = logistic_regression.predict(test_fold)

        # Calculating the accuracy
        accuracy_lr = metrics.accuracy_score(test_labels, predict_labels)
        presicion_lr = metrics.precision_score(
            test_labels, predict_labels, pos_label='pos')
        recall_lr = metrics.recall_score(
            test_labels, predict_labels, pos_label='pos')
        f1_lr = metrics.f1_score(
            test_labels, predict_labels, pos_label='pos')

        metrics_outputs.append(f'{strategy} {impute} logistic regression accuracy: {accuracy_lr}, precision: \
                {presicion_lr}, recall: {recall_lr}, f1: {f1_lr}')

with open('data/lda_metrics.txt', 'w') as file:
    file.write('\n'.join(metrics_outputs))


# Normalizing && imputing the data
training_data, mean, std = nmr.minmax_normalize(
    training_data)

training_data = imp.imputeMedian(training_data)

projected_data, components, component_weights = lda(
    training_data, training_class, 3)

# Getting the projected data as classes
class_neg_data = projected_data[np.where(training_class == 'neg')]
class_pos_data = projected_data[np.where(training_class == 'pos')]

# # Plotting the data
# plt.scatter(class_neg_data[:, 0], class_neg_data[:, 1],
#             c='r', s=[0.7], alpha=0.5, label='neg')
# plt.scatter(class_pos_data[:, 0], class_pos_data[:, 1],
#             c='b', s=[0.7], alpha=0.5, label='pos')
# plt.legend()
# plt.show()
