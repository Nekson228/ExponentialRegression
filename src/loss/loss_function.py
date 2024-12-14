from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class LossFunction(ABC):
    def __repr__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the loss function given the residuals
        :param y_true: true values
        :param y_pred: predicted values
        """
        pass

    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss function
        :param y_true: true values
        :param y_pred: predicted values
        :param jacobian: jacobian of the model
        """
        pass

    @abstractmethod
    def hessian(self, jacobian: np.ndarray) -> np.ndarray:
        """
        Compute the hessian of the loss function
        :param jacobian: jacobian of the model
        """
        pass
