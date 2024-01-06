# Modified version of pca from the homework 1
import numpy as np


# Provide normalized data to the pca function
def pca(data, num_components=2):
    # Calculating the covariance matrix
    covariance_matrix = np.cov(data, rowvar=False)
    # Calculating the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    print(covariance_matrix.shape)

    # Sorting eigenvector and values in the descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]

    # Selecting the first num_components eigenvectors
    components = eigenvectors[:, 0:num_components]

    # Projecting the data on the components
    projected_data = np.dot(data, components)

    return projected_data, eigenvectors, eigenvalues
