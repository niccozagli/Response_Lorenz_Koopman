from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import product
from typing import DefaultDict, Dict, List, Optional, Tuple, cast

import numpy as np
from scipy.special import eval_chebyt, eval_chebyu
from tqdm import tqdm

from LorenzEDMD.config import EDMDSettings
from LorenzEDMD.dynamical_system.Lorenz import lorenz63
from LorenzEDMD.utils.data_processing import (
    Koopman_correlation_function,
    find_index,
    get_spectral_properties,
)
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
        self.rel_threshold: float = rel_threshold
        self.Ur: Optional[np.ndarray] = None
        self.Sr: Optional[np.ndarray] = None
        self.Kreduced: Optional[np.ndarray] = None
        self.reduced_right_eigvecs: Optional[np.ndarray] = None
        self.reduced_left_eigvecs: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None
        self.Gr: Optional[np.ndarray] = None
        self.lambdas: Optional[np.ndarray] = None

    def decompose(self, edmd: EDMD_CHEB):
        if edmd.G is not None and edmd.A is not None:
            U, S, Vt = np.linalg.svd(edmd.G, full_matrices=False)
            r = np.sum(S > self.rel_threshold * S[0])
            Ur = U[:, :r]
            Sr_inv = np.diag(1 / S[:r])
            K_reduced = Sr_inv @ (Ur.T @ edmd.A @ Ur)

            self.Ur = cast(np.ndarray, Ur)
            self.Sr = cast(np.ndarray, S[:r])
            self.Gr = cast(np.ndarray, np.diag(S[:r]))
            self.Kreduced = cast(np.ndarray, K_reduced)
            return K_reduced

    def get_spectral_properties(self):
        if self.Kreduced is not None:
            eigenvalues, right_eigenvectors, left_eigenvectors = (
                get_spectral_properties(self.Kreduced)
            )  # right_eigenvectors , left_eigenvectors
            self.reduced_right_eigvecs = cast(np.ndarray, right_eigenvectors)
            self.reduced_left_eigvecs = cast(np.ndarray, left_eigenvectors)
            self.eigenvalues = cast(np.ndarray, eigenvalues)
        else:
            raise RuntimeError(
                "You must call `decompose()` before computing spectral properties."
            )

    def find_continuous_time_eigenvalues(
        self, lorenz_model: lorenz63, edmd: EDMD_CHEB
    ) -> None:
        if self.eigenvalues is not None:
            lambdas = np.log(self.eigenvalues) / (
                lorenz_model.dt * lorenz_model.tau * edmd.flight_time
            )
            self.lambdas = cast(np.ndarray, lambdas)

    def project_reduced_space(self, dictionary_projections) -> np.ndarray:
        if self.Ur is not None:
            return self.Ur.conj().T @ dictionary_projections
        else:
            raise RuntimeError(
                "You must call `decompose()` before mapping eigenvectors."
            )


class Projection_Koopman_Space:
    def __init__(self, threshold_lambda: float = -2):
        self.threshold: float = threshold_lambda
        self.lambdas: Optional[np.ndarray] = None
        self.Vn: Optional[np.ndarray] = None
        self.Gn: Optional[np.ndarray] = None
        self.Gr: Optional[np.ndarray] = None

    def set_subspace(self, tsvd: TSVD) -> None:
        if tsvd.lambdas is not None and tsvd.reduced_right_eigvecs is not None:
            lambdas = tsvd.lambdas
            indx = np.where(np.real(lambdas) > self.threshold)[0]

            lambdas_good = lambdas[indx]
            Vn = tsvd.reduced_right_eigvecs[:, indx]
            Gn = Vn.T.conj() @ tsvd.Gr @ Vn

            self.lambdas = cast(np.ndarray, lambdas_good)
            self.Vn = cast(np.ndarray, Vn)
            self.Gn = cast(np.ndarray, Gn)
            self.Gr = cast(np.ndarray, tsvd.Gr)
        else:
            raise RuntimeError(
                "The tsvd should be performed to get continuous time eigenvalues and right eigenvectors!"
            )

    def project_to_koopman_space(
        self, reduced_svd_projections: np.ndarray
    ) -> np.ndarray:
        if self.Gn is not None and self.Vn is not None and self.Gr is not None:
            return (
                np.linalg.pinv(self.Gn)
                @ self.Vn.conj().T
                @ self.Gr
                @ reduced_svd_projections
            )
        else:
            raise RuntimeError(
                "You must call `set_subspace()` before mapping eigenvectors."
            )

    def reconstruct_correlation_function(
        self, coefficients_f: np.ndarray, coefficients_g: np.ndarray
    ):
        if self.Gn is not None:
            Koopman_reconstruction = lambda t: Koopman_correlation_function(
                t=t,
                M=self.Gn,
                alpha1=coefficients_f,
                alpha2=coefficients_g,
                eigenvalues=self.lambdas,
            )
            return Koopman_reconstruction
        else:
            raise RuntimeError(
                "You must call `set_subspace()` before mapping eigenvectors."
            )
