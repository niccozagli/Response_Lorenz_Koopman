from typing import List, Tuple
from itertools import product
import numpy as np
from scipy.special import eval_chebyt
from LorenzEDMD.utils.data_processing import normalise_data_chebyshev
from tqdm import tqdm

def chebyshev_indices(degree : int, dim : int = 3) -> List[Tuple[int,...]]:
    return [i for i in product(range(degree + 1), repeat=dim) if sum(i) <= degree]

def evaluate_dictionary_point(scaled_data: np.ndarray, indices: List[Tuple[int, int, int]]) -> np.ndarray:
    """
    Evaluate Chebyshev tensor dictionary at a single point x.

    Parameters:
        x: 3D point (shape: (3,))
        indices: list of basis multi-indices

    Returns:
        Feature vector phi(x) as a 1D array.
    """
   
    return np.array([
        eval_chebyt(i1, scaled_data[0]) *
        eval_chebyt(i2, scaled_data[1]) *
        eval_chebyt(i3, scaled_data[2])
        for (i1, i2, i3) in indices
    ])

def evaluate_dictionary_batch(
    scaled_data: np.ndarray,  # shape (T, 3)
    indices: List[Tuple[int, int, int]]
) -> np.ndarray:
    """
    Evaluate tensorized Chebyshev dictionary using scipy's eval_chebyt.

    Parameters:
        X: (T, 3) array of input points (scaled to [-1, 1])
        indices: list of (i, j, k) tuples for the Chebyshev polynomial degrees

    Returns:
        Psi_X: (T, N) array with dictionary evaluations at each point
    """
    T = scaled_data.shape[0]
    N = len(indices)
    Psi = np.empty((T, N), dtype=np.float64)

    for n, (i, j, k) in enumerate(indices):
        Psi[:, n] = (
            eval_chebyt(i, scaled_data[:, 0]) *
            eval_chebyt(j, scaled_data[:, 1]) *
            eval_chebyt(k, scaled_data[:, 2])
        )

    return Psi

def perform_edmd_chebyshev(
    scaled_data : np.ndarray,
    degree: int
) -> Tuple[np.ndarray, List[Tuple[int, int, int]], Tuple[np.ndarray, np.ndarray]]:
    """
    Perform EDMD using tensorized Chebyshev polynomials.

    Parameters:
        ys: (n_samples, 3) array of trajectory data.
        degree: maximum total degree for dictionary functions.

    Returns:
        K: Koopman operator matrix (n_basis, n_basis)
        indices: list of multi-indices used
        (data_min, data_max): tuple for de-normalization
    """
    
    indices = chebyshev_indices(degree)
    n_basis = len(indices)

    G = np.zeros((n_basis, n_basis))
    A = np.zeros((n_basis, n_basis))
    L = len(scaled_data) - 1

    for x, y in tqdm(zip(scaled_data[:-1], scaled_data[1:]), total=L, desc="EDMD"):
        phi_x = evaluate_dictionary_point(x, indices)
        phi_y = evaluate_dictionary_point(y, indices)
        G += np.outer(phi_x, phi_x) / L
        A += np.outer(phi_x, phi_y) / L

    K = K = np.linalg.solve(G, A) #np.linalg.pinv(G,hermitian=True) @ A
    return K, indices

# def get_koopman_eigenfunction(eigenvector: np.ndarray, indices: List[Tuple[int, int, int]], x: np.ndarray) -> np.ndarray:
#     """
#     Reconstruct the Koopman eigenfunction from one eigenvector and the dictionary of Chebyshev functions.

#     Parameters:
#         eigenvector: Eigenvector of K (columns correspond to eigenfunctions).
#         indices: List of indices for the Chebyshev polynomials.
#         scaled_data: Data used to evaluate the Chebyshev polynomials.

#     Returns:
#         eigenfunction:  Koopman eigenfunction.
#     """
#     psi = evaluate_dictionary_point(x = x, indices=indices)
#     eigenfunction = eigenvector @ psi 
#     return eigenfunction

def evaluate_koopman_eigenfunctions_batch(
    scaled_data: np.ndarray,           # shape (T, 3)
    indices : Tuple[np.ndarray, np.ndarray],
    eigenvectors: np.ndarray     # shape (N, M)
) -> np.ndarray:
    """
    Evaluate Koopman eigenfunctions on a batch of data.

    Returns:
        Eigenfunction values: (T, M) complex array
    """
    Psi_X = evaluate_dictionary_batch(scaled_data,indices)
    return Psi_X @ eigenvectors  # shape (T, M)


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
