import numpy as np
import pandas as pd
import preprocessing.normalizer as nmr
import preprocessing.imputer as imp
from preprocessing.fa import fa
from preprocessing.pca import pca
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

training_data, mean, std = nmr.zscore_normalize(training_data)
training_data = imp.imputeMean(training_data)


training_data, eigenvalues, eigenvectors = fa(training_data, 2)


neg_indices = np.where(training_class == 'neg')
neg_values = training_data[neg_indices]

pos_indices = np.where(training_class == 'pos')
pos_values = training_data[pos_indices]


plt.scatter(pos_values[:, 0], pos_values[:, 1],
            c='r', s=0.5, alpha=0.5, label='pos_factor')

plt.scatter(neg_values[:, 0], neg_values[:, 1],
            c='b', s=0.5, alpha=0.5, label='neg_factor')
plt.legend
plt.show()
