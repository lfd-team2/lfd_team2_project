import numpy as np


def lda(data, labels, num_components):
    # data: (n_samples x n_features)
    # labels: (n_samples)
    # num_components: number of linear discriminants to return
    # returns: (n_samples x num_components), components, eigenvalues

    unique_classes = np.unique(labels)
    num_features = data.shape[1]
    num_samples = data.shape[0]

    # Split data per class
    data_per_class = {}
    data_size_per_class = {}

    for class_label in unique_classes:
        class_indices = np.where(labels == class_label)
        class_data = data[class_indices]
        data_per_class[class_label] = class_data
        data_size_per_class[class_label] = class_data.shape[0]

    # Overall mean
    mean_overall = np.mean(data, axis=0) / num_samples

    # Class means
    means_per_class = {}
    for class_label in unique_classes:
        class_data = data_per_class[class_label]
        class_data_size = data_size_per_class[class_label]
        means_per_class[class_label] = np.mean(
            class_data, axis=0) / class_data_size

    # Within class scatter
    Sw = np.zeros((num_features, num_features))
    for class_label in unique_classes:
        class_data = data_per_class[class_label]
        class_mean = means_per_class[class_label]
        # Sizes of the matrices after subtraction is N_samples x N_features
        # Therefore the transpose of the first matrix is taken
        subtracted_class_data = class_data - class_mean
        Sw += np.dot(subtracted_class_data.T, subtracted_class_data)

    # Between class scatter
    Sb = np.zeros((num_features, num_features))
    for class_label in unique_classes:
        class_mean = means_per_class[class_label]
        mean_difference = class_mean - mean_overall
        # Sizes of the matrices after outer product is N_features x N_features
        Sb += data_size_per_class[class_label] * \
            np.outer(mean_difference, mean_difference)

    # Get the inverse of Sw, sometimes it comes as a singular matrix
    try:
        Sw_inverse = np.linalg.inv(Sw)
    except np.linalg.LinAlgError:
        # If the matrix is singular, then add a small value to the diagonal
        Sw_inverse = np.linalg.inv(Sw + 0.0001 * np.eye(num_features))

    Sw_inverse = np.real(Sw_inverse)

    # Get eigenvalues and eigenvectors of Sw_inverse * Sb
    SwSb = np.dot(Sw_inverse, Sb)
    eigenvalues, eigenvectors = np.linalg.eig(SwSb)

    # Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices][:num_components]
    eigenvectors = eigenvectors[:, sorted_indices][:, :num_components]

    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Project the data onto eigenvectors
    projected_data = np.dot(data, eigenvectors)

    return projected_data, eigenvectors, eigenvalues
