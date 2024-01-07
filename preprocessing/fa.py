import numpy as np


def fa(data, num_factors):
    AAt = np.dot(data.T, data)

    # Decompose using SVD
    u, s, v = np.linalg.svd(AAt)
    print(u.shape, s.shape, v.shape)

    # Get the eigenvalues and eigenvectors
    eigenvalues = s
    eigenvectors = u

    # Sort the eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Get the first num_factors eigenvectors
    eigenvectors = eigenvectors[:, :num_factors]

    # Get the factor loadings
    factor_loadings = np.dot(data, eigenvectors)

    return factor_loadings, eigenvalues, eigenvectors
