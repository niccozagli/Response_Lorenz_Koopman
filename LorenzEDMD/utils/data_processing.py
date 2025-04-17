import numpy as np
from typing import Tuple

def normalise_data_chebyshev(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize each column of data to [-1, 1].

    Parameters:
        data: (n_samples, d) input array.

    Returns:
        scaled_data: normalized data in [-1, 1]
        data_min: minimum values per column (shape: d,)
        data_max: maximum values per column (shape: d,)
    """
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    scaled = 2 * (data - data_min) / (data_max - data_min) - 1
    return scaled, data_min, data_max
 
def get_spectral_properties(K : np.ndarray):
    """
    Returns the sorted (decreasing orders in terms of absolute value) of eigenvalues
    and eigenvectors of the Koopman matrix
    """
    eigenvalues, eigenvectors = np.linalg.eig(K)
    # Sort indices by decreasing magnitude of eigenvalues
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]

    # Apply sorting
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return eigenvalues , eigenvectors





