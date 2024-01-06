import numpy as np
import pandas as pd
import preprocessing.normalizer as nmr
import preprocessing.imputer as imp
from preprocessing.lda import lda
import matplotlib.pyplot as plt


data_frame = pd.read_csv('data/aps_failure_training_set.csv', na_values='na')

cols_to_drop = []
for col in data_frame.columns:
    if data_frame[col].isna().mean() > 0.70:
        cols_to_drop.append(col)

training_data = data_frame.drop(columns=['id', 'class'] + cols_to_drop).values
training_class = data_frame['class'].values

# Normalizing && imputing the data
training_data, mean, std = nmr.zscore_normalize(
    training_data)

training_data = imp.imputeConstant(training_data, fill_value=0)

projected_data, components, component_weights = lda(
    training_data, training_class, 3)

# Getting the projected data as classes
class_neg_data = projected_data[np.where(training_class == 'neg')]
class_pos_data = projected_data[np.where(training_class == 'pos')]

# Plotting the data
plt.scatter(class_neg_data[:, 0], class_neg_data[:, 1],
            c='r', s=[0.7], alpha=0.5, label='neg')
plt.scatter(class_pos_data[:, 0], class_pos_data[:, 1],
            c='b', s=[0.7], alpha=0.5, label='pos')
plt.legend()
plt.show()
