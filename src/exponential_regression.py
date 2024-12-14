from typing import Self, Optional

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array

from .loss import LossFunction, Chi2Loss


class ExponentialRegression(BaseEstimator, RegressorMixin):
    GRADIENT_TOL: float = 1e-3
    COEFFICIENTS_TOL: float = 1e-3
    CHI_SQR_REDUCED_TOL: float = 1e-1
    STEP_ACCEPTANCE: float = 1e-1

    REG_INIT: float = 1e-1
    REG_INCREASE_FACTOR: float = 11.
    REG_DECREASE_FACTOR: float = 9.
    REG_MIN: float = 1e-7
    REG_MAX: float = 1e7

    def __init__(self,
                 n_terms: int = 1,
                 max_iter: int = 1000,
                 gradient_tol: float = GRADIENT_TOL,
                 coefficients_tol: float = COEFFICIENTS_TOL,
                 chi2_reduced_tol: float = CHI_SQR_REDUCED_TOL,
                 step_acceptance: float = STEP_ACCEPTANCE,
                 reg_init: float = REG_INIT,
                 loss_function: LossFunction = None) -> None:
        self.n_terms = n_terms
        self.max_iter = max_iter

        self.gradient_tol = gradient_tol
        self.coefficients_tol = coefficients_tol
        self.chi2_reduced_tol = chi2_reduced_tol
        self.step_acceptance = step_acceptance

        self.reg_init = reg_init

        self.loss_function = loss_function

        self.regularization_: None | float = None
        self.lambda_: None | np.ndarray = None
        self.omega_: None | np.ndarray = None

    def fit(self,
            data: np.ndarray,
            target: np.ndarray,
            initial_lambda: Optional[np.ndarray] = None, initial_omega: Optional[np.ndarray] = None) -> Self:
        data, target = check_X_y(data, target, ensure_2d=False)
        t = data.ravel()

        if initial_lambda is not None and initial_omega is not None:
            self.lambda_ = initial_lambda
            self.omega_ = initial_omega
        else:
            self._init_parameters(self.n_terms)

        self.regularization_ = self.reg_init

        new_y_pred = self._model(t)
        is_step_accepted = True
        hessian = gradient = None

        for _ in range(self.max_iter):
            y_pred = new_y_pred
            if is_step_accepted:
                jacobian = self._jacobian(t)
                gradient = self.loss_function.gradient(
                    target, y_pred, jacobian)
                loss = self.loss_function.loss(target, y_pred)

                if self._check_convergence(gradient, self.lambda_, self.omega_, loss, data.size):
                    break

                hessian = self.loss_function.hessian(jacobian)

            hessian = self._regularize_hessian(hessian)

            delta = np.linalg.solve(hessian, gradient)

            is_step_accepted, new_y_pred = self._accept_step(
                t, target, y_pred, delta, gradient)
            if is_step_accepted:
                self._decrease_regularization()
            else:
                self._increase_regularization()
        else:
            print(f"Failed to converge after {self.max_iter} iterations")

        return self

    def _init_parameters(self, n_terms: int) -> None:
        self.lambda_ = np.ones(n_terms)
        self.omega_ = np.ones(n_terms)

    def _jacobian(self, t: np.ndarray) -> np.ndarray:
        exp_terms = np.exp(np.outer(t, self.omega_))
        jacobian_lambda = exp_terms
        jacobian_omega = exp_terms * self.lambda_ * t[:, np.newaxis]
        jacobian = np.hstack((jacobian_lambda, jacobian_omega))
        return jacobian

    def _regularize_hessian(self, hessian: np.ndarray) -> np.ndarray:
        return hessian + np.eye(hessian.shape[0]) * self.regularization_

    def _accept_step(self, t: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, delta: np.ndarray,
                     gradient: np.ndarray) -> tuple[bool, np.ndarray]:
        new_y_pred = self._model(t,
                                 self.lambda_ + delta[:self.n_terms],
                                 self.omega_ + delta[self.n_terms:])

        chi_sqr = self.loss_function.loss(y_true, y_pred)
        new_chi_sqr = self.loss_function.loss(y_true, new_y_pred)

        rho = (
                (chi_sqr - new_chi_sqr) /
                np.abs(delta.T @ (self.regularization_ * delta + gradient))
        )

        if rho > self.step_acceptance:
            self.lambda_ += delta[:self.n_terms]
            self.omega_ += delta[self.n_terms:]
            return True, new_y_pred
        return False, y_pred

    def _increase_regularization(self) -> None:
        self.regularization_ *= self.REG_INCREASE_FACTOR
        self.regularization_ = min(self.regularization_, self.REG_MAX)

    def _decrease_regularization(self) -> None:
        self.regularization_ /= self.REG_DECREASE_FACTOR
        self.regularization_ = max(self.regularization_, self.REG_MIN)

    def _check_convergence(self, gradient: np.ndarray, lambda_delta: np.ndarray, omega_delta: np.ndarray,
                           loss: float, measurements_amount: int) -> bool:
        return (
                np.max(np.abs(gradient)) < self.gradient_tol or
                np.max(np.abs(lambda_delta / self.lambda_)) < self.coefficients_tol or
                np.max(np.abs(omega_delta / self.omega_)) < self.coefficients_tol or
                loss / (measurements_amount -
                        gradient.shape[0]) < self.chi2_reduced_tol
        )

    def predict(self, data: np.ndarray) -> np.ndarray:
        data = check_array(data, ensure_2d=False)
        t = data.ravel()
        return self._model(t)

    def _model(self, t: np.ndarray, lambda_: np.ndarray = None, omega_: np.ndarray = None) -> np.ndarray:
        lambda_ = self.lambda_ if lambda_ is None else lambda_
        omega_ = self.omega_ if omega_ is None else omega_

        exp_terms = np.exp(np.outer(t, omega_))
        return exp_terms @ lambda_
