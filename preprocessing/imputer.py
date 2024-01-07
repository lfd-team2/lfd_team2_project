import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer


# Impute the missing values with usage of the knn algorithm:
# Works very slow, so it is not recommended to use it
def imputeKNN(data, n_neighbors=3):
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='uniform',
                         metric='nan_euclidean', missing_values=np.nan, copy=True)
    imputed_data = imputer.fit_transform(data)
    return imputed_data


# Impute the missing values with usage of the mean value:
def imputeMean(data):
    imputer = SimpleImputer(missing_values=np.nan,
                            strategy='mean', copy=True)
    imputed_data = imputer.fit_transform(data)
    return imputed_data


# Impute the missing values with usage of the median value:
def imputeMedian(data):
    imputer = SimpleImputer(missing_values=np.nan,
                            strategy='median', copy=True)
    imputed_data = imputer.fit_transform(data)
    return imputed_data


# Impute the missing values with usage of the most frequent value:
def imputeMostFrequent(data):
    imputer = SimpleImputer(missing_values=np.nan,
                            strategy='most_frequent', copy=True)
    imputed_data = imputer.fit_transform(data)
    return imputed_data


# Impute the missing values with usage of the constant value:
def imputeConstant(data, fill_value=0):
    imputer = SimpleImputer(missing_values=np.nan, strategy='constant',
                            fill_value=fill_value, copy=True)
    imputed_data = imputer.fit_transform(data)
    return imputed_data


# Impute the missing values with a random noise:
def imputeRandomFill(data):
    random_data = np.random.randn(data.shape[0], data.shape[1])
    imputed_data = np.where(np.isnan(data), random_data, data)
    return imputed_data
