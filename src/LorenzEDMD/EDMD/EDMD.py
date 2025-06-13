from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from typing import DefaultDict, Dict, List, Tuple, cast

import numpy as np
from scipy.special import eval_chebyt, eval_chebyu
from tqdm import tqdm

from LorenzEDMD.config import EDMDSettings
from LorenzEDMD.utils.data_processing import find_index, get_spectral_properties
from LorenzEDMD.utils.load_config import get_edmd_settings

EDMD_SETTINGS = get_edmd_settings()


def chebyshev_indices(degree: int, dim: int = 3) -> List[Tuple[int, int, int]]:
    indices = [
        cast(Tuple[int, int, int], i)
        for i in product(range(degree + 1), repeat=dim)
        if sum(i) <= degree
    ]
    return indices


######### EDMD CLASS ##########


class BaseEDMD(ABC):
    def __init__(self, flight_time: int):
        self.flight_time = flight_time
        self.G = None
        self.A = None
        self.K = None

    @abstractmethod
    def evaluate_dictionary_batch(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate_dictionary_point(self, x: np.ndarray) -> np.ndarray:
        pass

    def _create_edmd_snapshots(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.flight_time < 1:
            raise ValueError("Flight time must be >= 1.")
        N = data.shape[0]
        if self.flight_time >= N:
            raise ValueError(
                f"Flight time = {self.flight_time} is too large for data length {N}."
            )
        return data[: -self.flight_time], data[self.flight_time :]

    def perform_edmd(self, data: np.ndarray, batch_size: int = 10_000) -> np.ndarray:

        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        X, Y = self._create_edmd_snapshots(data)
        N = X.shape[0]
        n_features = self.evaluate_dictionary_batch(X[:1]).shape[1]

        G = np.zeros((n_features, n_features))
        A = np.zeros((n_features, n_features))

        for start in tqdm(range(0, N, batch_size)):
            end = min(start + batch_size, N)
            X_batch = X[start:end]
            Y_batch = Y[start:end]

            Phi_X = self.evaluate_dictionary_batch(X_batch)
            Phi_Y = self.evaluate_dictionary_batch(Y_batch)

            G += Phi_X.T @ Phi_X
            A += Phi_X.T @ Phi_Y

        L = N  # total number of snapshot pairs
        G /= L
        A /= L

        self.G = G
        self.A = A
        self.K = np.linalg.solve(G, A)
        return self.K


# ---------------------- Chebyshev EDMD ----------------------


class EDMD_CHEB(BaseEDMD):
    def __init__(self, edmd_settings_handler: EDMDSettings = EDMD_SETTINGS):
        super().__init__(edmd_settings_handler.flight_time)
        self.degree = edmd_settings_handler.degree
        self.indices = chebyshev_indices(self.degree)

    def evaluate_dictionary_point(self, x: np.ndarray) -> np.ndarray:
        return np.array(
            [
                eval_chebyt(i1, x[0]) * eval_chebyt(i2, x[1]) * eval_chebyt(i3, x[2])
                for (i1, i2, i3) in self.indices
            ]
        )

    def evaluate_dictionary_batch(self, data: np.ndarray) -> np.ndarray:
        T = data.shape[0]
        N = len(self.indices)
        Psi = np.empty((T, N), dtype=np.float64)
        for n, (i, j, k) in enumerate(self.indices):
            Psi[:, n] = (
                eval_chebyt(i, data[:, 0])
                * eval_chebyt(j, data[:, 1])
                * eval_chebyt(k, data[:, 2])
            )
        return Psi

    def _chebyshev_U_to_T_matrix(self, N: int) -> np.ndarray:
        """
        Create a matrix M such that:
        M[n, m] gives the coefficient of T_m in U_n(x)
        Returns an (N x N) matrix.
        """
        M = np.zeros((N, N))
        for n in range(N):
            for m in range(0, n + 1, 2):
                M[n, m] = 2
        M[0, 0] = 1  # U_0(x) = T_0(x)
        return M

    def spectral_derivative_tensor_chebyshev_explicit(
        self, c_flat: np.ndarray, direction: int
    ) -> np.ndarray:
        """
        Compute the spectral derivative of a tensorized Chebyshev T_n basis expansion
        along the specified direction (0: x, 1: y, 2: z), using the exact formula:

            d/dx T_n(x) = n * U_{n-1}(x)
            U_n(x) =
                2 * sum_{j odd > 0}^{n} T_j(x),           if n is odd
                2 * sum_{j even ≥ 0}^{n} T_j(x) - 1,      if n is even

        Parameters:
        - c_flat: (N,) array of Chebyshev T-basis coefficients (flattened)
        - direction: int in {0,1,2} for derivative direction

        Returns:
        - dc_flat: (N,) array of T-basis coefficients of the derivative
        """
        indices = self.indices
        index_map = {idx: n for n, idx in enumerate(indices)}

        dc_dict: DefaultDict[Tuple[int, int, int], float] = defaultdict(float)

        for n, (i, j, k) in enumerate(indices):
            coeff = c_flat[n]

            # Select degree in the direction
            degs = [i, j, k]
            deg = degs[direction]

            if deg == 0:
                continue  # T_0 -> 0

            scale = deg  # from derivative rule
            U_index = deg - 1

            # Loop over T_j terms in expansion of U_{deg-1}
            if U_index % 2 == 0:  # even
                for m in range(0, U_index + 1, 2):
                    new_degs = list(degs)
                    new_degs[direction] = m
                    new_idx = cast(Tuple[int, int, int], tuple(new_degs))
                    if new_idx in index_map:
                        dc_dict[new_idx] += coeff * scale * 2
                # subtract the constant 1 term
                new_degs = list(degs)
                new_degs[direction] = 0
                new_idx = cast(Tuple[int, int, int], tuple(new_degs))
                if new_idx in index_map:
                    dc_dict[new_idx] -= coeff * scale

            else:  # odd
                for m in range(1, U_index + 1, 2):
                    new_degs = list(degs)
                    new_degs[direction] = m
                    new_idx = cast(Tuple[int, int, int], tuple(new_degs))
                    if new_idx in index_map:
                        dc_dict[new_idx] += coeff * scale * 2

        # Build the output flat array
        dc_flat = np.zeros_like(c_flat)
        for idx, val in dc_dict.items():
            dc_flat[index_map[idx]] = val

        return dc_flat

    def build_derivative_matrix(self, direction: int) -> np.ndarray:
        """
        Constructs the matrix A^{(direction)} such that:
        A @ c = coefficients of d/dx_i f(x), when f(x) = sum c_n psi_n(x)

        Parameters:
        - direction: 0 for x, 1 for y, 2 for z

        Returns:
        - A: (N, N) differentiation matrix in the Chebyshev dictionary
        """
        N = len(self.indices)
        A = np.zeros((N, N))
        I = np.eye(N)
        for i in range(N):
            A[:, i] = self.spectral_derivative_tensor_chebyshev_explicit(
                I[:, i], direction
            )
        return A

    def evaluate_koopman_eigenfunctions_batch(
        self, data: np.ndarray, eigenvectors: np.ndarray
    ) -> np.ndarray:
        Psi_X = self.evaluate_dictionary_batch(data)
        return Psi_X @ eigenvectors

    def evaluate_koopman_eigenfunctions_reduced(
        self, data: np.ndarray, tsvd_regulariser: TSVD
    ) -> np.ndarray:
        if (
            tsvd_regulariser.Ur is None
            or tsvd_regulariser.reduced_right_eigvecs is None
        ):
            raise ValueError("TSVD decomposition must be applied first.")
        Psi_X = self.evaluate_dictionary_batch(data)
        Psi_reduced = Psi_X @ tsvd_regulariser.Ur
        return Psi_reduced @ tsvd_regulariser.reduced_right_eigvecs

    def get_decomposition_observables(self) -> Dict[str, np.ndarray]:
        dictionary_decomposition = {}

        # Decomposing f(x,y,z) = z
        degree_cheb = (0, 0, 1)
        index = find_index(self.indices, degree_cheb)
        projections_cheb_dictionary = np.zeros(len(self.indices))
        projections_cheb_dictionary[index] = 1

        dictionary_decomposition["z"] = projections_cheb_dictionary

        # Decomposing f(x,y,z) = x^2
        degree_cheb = (2, 0, 0)
        index2 = find_index(self.indices, degree_cheb)
        degree_cheb = (0, 0, 0)
        index0 = find_index(self.indices, degree_cheb)
        projections_cheb_dictionary = np.zeros(len(self.indices))
        projections_cheb_dictionary[index0] = 1 / 2
        projections_cheb_dictionary[index2] = 1 / 2

        dictionary_decomposition["x^2"] = projections_cheb_dictionary

        # Decomposing f(x,y,z) = y^2
        degree_cheb = (0, 2, 0)
        index2 = find_index(self.indices, degree_cheb)
        projections_cheb_dictionary = np.zeros(len(self.indices))
        projections_cheb_dictionary[index0] = 1 / 2
        projections_cheb_dictionary[index2] = 1 / 2

        dictionary_decomposition["y^2"] = projections_cheb_dictionary

        # Decomposing f(x,y,z) = z^2
        degree_cheb = (0, 0, 2)
        index2 = find_index(self.indices, degree_cheb)
        projections_cheb_dictionary = np.zeros(len(self.indices))
        projections_cheb_dictionary[index0] = 1 / 2
        projections_cheb_dictionary[index2] = 1 / 2

        dictionary_decomposition["z^2"] = projections_cheb_dictionary

        # Decomposing f(x,y,z) = xy
        degree_cheb = (1, 1, 0)
        index2 = find_index(self.indices, degree_cheb)
        projections_cheb_dictionary = np.zeros(len(self.indices))
        projections_cheb_dictionary[index2] = 1

        dictionary_decomposition["xy"] = projections_cheb_dictionary

        return dictionary_decomposition


############# REGULARISATION CLASSES ###############


class Tikhonov:
    def __init__(self, alpha: float = 1e-7):
        self.alpha = alpha

    def tikhonov(self, edmd: EDMD_CHEB) -> np.ndarray:
        if edmd.G is not None and edmd.A is not None:
            K = np.linalg.solve(edmd.G + self.alpha * np.eye(edmd.G.shape[0]), edmd.A)
            return K
        else:
            raise ValueError(
                "Matrices G and A are not set. Run `perform_edmd()` first."
            )


class TSVD:
    def __init__(self, rel_threshold: float = 1e-6):
        self.rel_threshold = rel_threshold
        self.Ur = None
        self.Sr = None
        self.Kreduced = None
        self.reduced_right_eigvecs = None
        self.reduced_left_eigvecs = None
        self.eigenvalues = None

    def decompose(self, edmd: EDMD_CHEB):
        if edmd.G is not None and edmd.A is not None:
            U, S, Vt = np.linalg.svd(edmd.G, full_matrices=False)
            r = np.sum(S > self.rel_threshold * S[0])
            Ur = U[:, :r]
            Sr_inv = np.diag(1 / S[:r])
            K_reduced = Sr_inv @ (Ur.T @ edmd.A @ Ur)

            self.Ur = Ur
            self.Sr = S[:r]
            self.Kreduced = K_reduced
            return K_reduced

    def get_spectral_properties(self):
        if self.Kreduced is not None:
            eigenvalues, right_eigenvectors, left_eigenvectors = (
                get_spectral_properties(self.Kreduced)
            )  # right_eigenvectors , left_eigenvectors
            self.reduced_right_eigvecs = right_eigenvectors
            self.reduced_left_eigvecs = left_eigenvectors
            self.eigenvalues = eigenvalues
        else:
            raise RuntimeError(
                "You must call `decompose()` before computing spectral properties."
            )

    def map_eigenvectors(self, eigenvectors):
        if self.Ur is not None:
            return self.Ur @ eigenvectors
        else:
            RuntimeError("You must call `decompose()` before mapping eigenvectors.")


# class EDMD_CHEB:
#     def __init__(self,
#     edmd_settings_handler : EDMDSettings = EDMD_SETTINGS
#     ):
#         self.flight_time = edmd_settings_handler.flight_time
#         self.degree = edmd_settings_handler.degree
#         self.indices = chebyshev_indices(edmd_settings_handler.degree)
#         self.G = None
#         self.A = None
#         self.K = None

#     def _evaluate_dictionary_point(self,
#         x: np.ndarray
#     ) -> np.ndarray:
#         """
#         Evaluate Chebyshev tensor dictionary at a single point x.

#         Parameters:
#             x: 3D point (shape: (3,))
#             indices: list of basis multi-indices

#         Returns:
#             Feature vector phi(x) as a 1D array.
#         """
#         return np.array([
#             eval_chebyt(i1, x[0]) *
#             eval_chebyt(i2, x[1]) *
#             eval_chebyt(i3, x[2])
#             for (i1, i2, i3) in self.indices
#         ])


#     def perform_edmd_chebyshev(
#         self,
#         scaled_data : np.ndarray,
#     ) -> Tuple[np.ndarray, List[Tuple[int, int, int]], Tuple[np.ndarray, np.ndarray]]:
#         """
#         Perform EDMD using tensorized Chebyshev polynomials.

#         Parameters:
#             ys: (n_samples, 3) array of trajectory data.
#             degree: maximum total degree for dictionary functions.

#         Returns:
#             K: Koopman operator matrix (n_basis, n_basis)
#             indices: list of multi-indices used
#             (data_min, data_max): tuple for de-normalization
#         """
#         n_basis = len(self.indices)
#         G = np.zeros((n_basis, n_basis))
#         A = np.zeros((n_basis, n_basis))

#         X , Y = self._create_edmd_snapshots(scaled_data)
#         L = X.shape[0]

#         for x, y in tqdm(zip(X,Y), total=L, desc="EDMD"):
#             phi_x = self._evaluate_dictionary_point(x)
#             phi_y = self._evaluate_dictionary_point(y)
#             G += np.outer(phi_x, phi_x) / L
#             A += np.outer(phi_x, phi_y) / L

#         K = np.linalg.solve(G, A) #np.linalg.pinv(G,hermitian=True) @ A
#         self.G = G
#         self.A = A
#         self.K = K

#         return K

#     def evaluate_koopman_eigenfunctions_batch(
#         self,
#         scaled_data: np.ndarray,           # shape (T, 3)
#         eigenvectors: np.ndarray     # shape (N, M)
#     ) -> np.ndarray:
#         """
#         Evaluate Koopman eigenfunctions on a batch of data.

#         Returns:
#             Eigenfunction values: (T, M) complex array
#         """
#         Psi_X = self.evaluate_dictionary_batch(scaled_data)
#         return Psi_X @ eigenvectors  # shape (T, M)

#     def evaluate_koopman_eigenfunctions_reduced(
#         self,
#         scaled_data: np.ndarray,           # shape (T, 3)
#         tsvd_regulariser : TSVD
#     ) -> np.ndarray:
#         """
#         Evaluate Koopman eigenfunctions from reduced-space representation.

#         Parameters:
#             scaled_data: (T, 3) array of inputs (scaled to [-1, 1])
#             reduced_eigenvectors: (r, M) Koopman eigenvectors in reduced space
#             Ur: (N, r) truncated left singular vectors (from TSVD)

#         Returns:
#             Eigenfunction values: (T, M) complex array
#         """
#         if tsvd_regulariser.Ur is not None and tsvd_regulariser.reduced_right_eigvecs is not None:
#             Psi_X = self.evaluate_dictionary_batch(scaled_data)  # (T, N)
#             Psi_reduced = Psi_X @ tsvd_regulariser.Ur                             # (T, r)
#             return Psi_reduced @ tsvd_regulariser.reduced_right_eigvecs         # (T, M)
#         else:
#             raise RuntimeError("TSVD decomposition has not been applied. Call `decompose()` and `get_spectral_properties()` first.")

#     def _create_edmd_snapshots(
#         self,
#         scaled_data: np.ndarray,
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Creates snapshot matrices X and Y from scaled_data for EDMD with flight time tau.

#         Parameters:
#             scaled_data: array of shape (N, d)

#         Returns:
#             X: shape (N - tau, d), states at time t
#             Y: shape (N - tau, d), states at time t + tau * Δt
#         """
#         if self.flight_time < 1:
#             raise ValueError("Flight time tau must be at least 1.")

#         N = scaled_data.shape[0]
#         if self.flight_time >= N:
#             raise ValueError(f"tau={self.flight_time} is too large for the dataset of length {N}.")

#         X = scaled_data[:-self.flight_time]
#         Y = scaled_data[self.flight_time:]
#         return X, Y

#     def evaluate_dictionary_batch(
#         self,
#         scaled_data: np.ndarray,  # shape (T, 3)
#     ) -> np.ndarray:
#         """
#         Evaluate tensorized Chebyshev dictionary using scipy's eval_chebyt.

#         Parameters:
#             X: (T, 3) array of input points (scaled to [-1, 1])
#             indices: list of (i, j, k) tuples for the Chebyshev polynomial degrees

#         Returns:
#             Psi_X: (T, N) array with dictionary evaluations at each point
#         """
#         T = scaled_data.shape[0]
#         N = len(self.indices)
#         Psi = np.empty((T, N), dtype=np.float64)

#         for n, (i, j, k) in enumerate(self.indices):
#             Psi[:, n] = (
#                 eval_chebyt(i, scaled_data[:, 0]) *
#                 eval_chebyt(j, scaled_data[:, 1]) *
#                 eval_chebyt(k, scaled_data[:, 2])
#             )

#         return Psi
