"""
Lorenz.py

This file contains the settings class for the dynamical system
"""

from typing import List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from LorenzEDMD.config import ModelSettings
from LorenzEDMD.utils.load_config import get_model_settings

MODEL_settings = get_model_settings()


class lorenz63:
    def __init__(
        self,
        model_settings_handler: ModelSettings = MODEL_settings,
    ):
        self.rho: float = model_settings_handler.rho
        self.sigma: float = model_settings_handler.sigma
        self.beta: float = model_settings_handler.beta
        self.noise: float = model_settings_handler.noise

        self.t_span: Tuple[float, float] = (
            model_settings_handler.tmin,
            model_settings_handler.tmax,
        )
        self.dt: float = model_settings_handler.dt
        self.tau: int = model_settings_handler.tau
        self.transient: float = model_settings_handler.transient

        self.y0: List[float] = [1, 0.5, 2]

        self.trajectory: Optional[np.ndarray] = None

    def _drift(self, t, Y):
        sigma = self.sigma
        rho = self.rho
        beta = self.beta
        x, y, z = Y
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    def _diffusion(self, t, Y):
        diffusion = self.noise * np.eye(3)
        return diffusion

    def integrate_EM(self, show_progress: bool = True, rng: np.random.Generator = None):
        t0, tf = self.t_span
        n_steps = int((tf - t0) / self.dt)
        ts = np.linspace(t0, tf, n_steps + 1)

        tsave = ts[: -1 : self.tau]
        ysave = np.zeros((len(tsave), 3))

        yold = self.y0

        if rng is None:
            rng = np.random.default_rng()

        index = 0
        for i in tqdm(range(n_steps), disable=not show_progress):
            t = ts[i]
            f = self._drift(t=t, Y=yold)
            g = self._diffusion(t=t, Y=yold)
            dW = rng.normal(0, np.sqrt(self.dt), size=3)

            ynew = yold + f * self.dt + g @ dW

            if np.mod(i, self.tau) == 0:
                ysave[index, :] = ynew
                index += 1

            yold = ynew.copy()

        ind_transient = np.where(tsave >= self.transient)[0][0]
        tsave = tsave[ind_transient:]
        ysave = ysave[ind_transient:, :]

        self.trajectory = (tsave, ysave)
        return tsave, ysave

    def plot3d_trajectory(self, figsize=(8, 6), color="royalblue", alpha=0.8, lw=0.7):
        if self.trajectory is None:
            raise ValueError("No trajectory found. Run integrate() first.")

        ts, ys = self.trajectory

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(ys[:, 0], ys[:, 1], ys[:, 2], color=color, lw=lw, alpha=alpha)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("Lorenz Attractor")

        plt.tight_layout()
        plt.show()
