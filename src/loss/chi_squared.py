import numpy as np
from typing import Optional

from .loss_function import LossFunction


class ChiSquaredLoss(LossFunction):
    # Chi-squared (or weighted MSE) loss function
    def __init__(self, measurements_amount: int, measurement_variance: float | np.ndarray | None = None) -> None:
        if isinstance(measurement_variance, float):
            self._weights = np.eye(measurements_amount) / measurement_variance
        elif isinstance(measurement_variance, np.ndarray):
            self._weights = np.diag(1 / measurement_variance)
        else:
            self._weights = np.eye(measurements_amount)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        residuals = y_true - y_pred
        return np.float_(residuals.T @ self._weights @ residuals)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
        residuals = y_true - y_pred
        return jacobian.T @ self._weights @ residuals

    def hessian(self, jacobian: np.ndarray) -> np.ndarray:
        return jacobian.T @ self._weights @ jacobian
