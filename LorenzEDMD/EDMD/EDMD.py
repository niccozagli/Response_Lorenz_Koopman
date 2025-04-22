from typing import List, Tuple, Optional
from itertools import product
import numpy as np
from scipy.special import eval_chebyt
from LorenzEDMD.utils.data_processing import normalise_data_chebyshev, get_spectral_properties
from tqdm import tqdm
from LorenzEDMD.utils.load_config import get_edmd_settings
from LorenzEDMD.config import EDMDSettings
from sklearn.neighbors import KernelDensity


EDMD_SETTINGS = get_edmd_settings()

def chebyshev_indices(degree : int,
    dim : int = 3
) -> List[Tuple[int,int,int]]:
    indices =  [i for i in product(range(degree + 1), repeat=dim) if sum(i) <= degree]
    return indices

######### EDMD CLASS FOR CHEBYSHEV ##########

class EDMD_CHEB:
    def __init__(self,
    edmd_settings_handler : EDMDSettings = EDMD_SETTINGS
    ):
        self.flight_time = edmd_settings_handler.flight_time
        self.degree = edmd_settings_handler.degree
        self.indices = chebyshev_indices(edmd_settings_handler.degree)
        self.G = None
        self.A = None
        self.K = None
        
    def _evaluate_dictionary_point(self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate Chebyshev tensor dictionary at a single point x.

        Parameters:
            x: 3D point (shape: (3,))
            indices: list of basis multi-indices

        Returns:
            Feature vector phi(x) as a 1D array.
        """
        return np.array([
            eval_chebyt(i1, x[0]) *
            eval_chebyt(i2, x[1]) *
            eval_chebyt(i3, x[2])
            for (i1, i2, i3) in self.indices
        ])
        

    def perform_edmd_chebyshev(
        self,
        scaled_data : np.ndarray,
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
        n_basis = len(self.indices)
        G = np.zeros((n_basis, n_basis))
        A = np.zeros((n_basis, n_basis))

        X , Y = self._create_edmd_snapshots(scaled_data)
        L = X.shape[0]

        for x, y in tqdm(zip(X,Y), total=L, desc="EDMD"):
            phi_x = self._evaluate_dictionary_point(x)
            phi_y = self._evaluate_dictionary_point(y)
            G += np.outer(phi_x, phi_x) / L
            A += np.outer(phi_x, phi_y) / L

        K = K = np.linalg.solve(G, A) #np.linalg.pinv(G,hermitian=True) @ A
        self.G = G
        self.A = A
        self.K = K
        
        return K

    def evaluate_koopman_eigenfunctions_batch(
        self,
        scaled_data: np.ndarray,           # shape (T, 3)
        eigenvectors: np.ndarray     # shape (N, M)
    ) -> np.ndarray:
        """
        Evaluate Koopman eigenfunctions on a batch of data.

        Returns:
            Eigenfunction values: (T, M) complex array
        """
        Psi_X = self.evaluate_dictionary_batch(scaled_data)
        return Psi_X @ eigenvectors  # shape (T, M)
    
    def evaluate_koopman_eigenfunctions_reduced(
        self,
        scaled_data: np.ndarray,           # shape (T, 3)
        tsvd_regulariser 
    ) -> np.ndarray:
        """
        Evaluate Koopman eigenfunctions from reduced-space representation.

        Parameters:
            scaled_data: (T, 3) array of inputs (scaled to [-1, 1])
            reduced_eigenvectors: (r, M) Koopman eigenvectors in reduced space
            Ur: (N, r) truncated left singular vectors (from TSVD)

        Returns:
            Eigenfunction values: (T, M) complex array
        """
        if tsvd_regulariser.Ur is not None and tsvd_regulariser.reduced_right_eigvecs is not None:
            Psi_X = self.evaluate_dictionary_batch(scaled_data)  # (T, N)
            Psi_reduced = Psi_X @ tsvd_regulariser.Ur                             # (T, r)
            return Psi_reduced @ tsvd_regulariser.reduced_right_eigvecs         # (T, M)
        else: print("You need to first apply the TSVD decomposition")

    def _create_edmd_snapshots(
        self,
        scaled_data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates snapshot matrices X and Y from scaled_data for EDMD with flight time tau.

        Parameters:
            scaled_data: array of shape (N, d)

        Returns:
            X: shape (N - tau, d), states at time t
            Y: shape (N - tau, d), states at time t + tau * Δt
        """
        if self.flight_time < 1:
            raise ValueError("Flight time tau must be at least 1.")

        N = scaled_data.shape[0]
        if self.flight_time >= N:
            raise ValueError(f"tau={self.flight_time} is too large for the dataset of length {N}.")

        X = scaled_data[:-self.flight_time]
        Y = scaled_data[self.flight_time:]
        return X, Y

    def evaluate_dictionary_batch(
        self,
        scaled_data: np.ndarray,  # shape (T, 3)
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
        N = len(self.indices)
        Psi = np.empty((T, N), dtype=np.float64)

        for n, (i, j, k) in enumerate(self.indices):
            Psi[:, n] = (
                eval_chebyt(i, scaled_data[:, 0]) *
                eval_chebyt(j, scaled_data[:, 1]) *
                eval_chebyt(k, scaled_data[:, 2])
            )

        return Psi
    

############# REGULARISATION CLASSES ###############

class Tikhonov():
    def __init__(self,alpha : float = 1e-7):
        self.alpha = alpha

    def tikhonov(self,edmd : EDMD_CHEB) -> np.ndarray:
        if edmd.G is not None and edmd.A is not None:
            K = np.linalg.solve(edmd.G+self.alpha *np.eye(edmd.G.shape[0]),edmd.A)
            return K
        else:
            print("You need to evaluate the EDMD matrix G and A first")

class TSVD():
    def __init__(self,rel_threshold : float = 1e-6):
        self.rel_threshold = rel_threshold
        self.Ur = None
        self.Sr = None
        self.Kreduced = None
        self.reduced_right_eigvecs = None
        self.reduced_left_eigvecs = None
        self.eigenvalues = None

    def decompose(self,edmd: EDMD_CHEB):
        if edmd.G is not None and edmd.A is not None:
            U, S, Vt = np.linalg.svd(edmd.G, full_matrices=False)
            r = np.sum(S > self.rel_threshold *S[0])
            Ur = U[:,:r] 
            Sr_inv = np.diag(1 / S[:r])
            K_reduced = Sr_inv @ (Ur.T @ edmd.A @ Ur)

            self.Ur = Ur
            self.Sr = S[:r]
            self.Kreduced = K_reduced
            return K_reduced
        
    def get_spectral_properties(self):
        if self.Kreduced is not None:
            eigenvalues , left_eigenvectors, right_eigenvectors = get_spectral_properties(self.Kreduced) #right_eigenvectors , left_eigenvectors
            self.reduced_right_eigvecs = right_eigenvectors
            self.reduced_left_eigvecs = left_eigenvectors
            self.eigenvalues = eigenvalues
        else: print("You need to first run the decompose method")

    def map_eigenvectors(self,eigenvectors):
        if self.Ur is not None:
            return self.Ur @ eigenvectors
        else: print("You should first perform the TSVD decomposition")
         
