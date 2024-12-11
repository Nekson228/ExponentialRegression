import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from src.exponential_regression import ExponentialRegression
from src.loss.chi_squared import ChiSquaredLoss


def main():
    measurements = 100
    measurement_errors = 1 / 20
    x = np.linspace(0, 10, measurements).reshape(-1, 1)
    y = (2 * np.exp(-0.25 * x) + (-5) * np.exp(-2 * x)).ravel()
    y += np.random.normal(0, measurement_errors, measurements)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    loss = ChiSquaredLoss(measurements, measurement_errors)

    er = ExponentialRegression(n_terms=2, loss_function=loss)
    er.fit(x, y, initial_lambda=np.array([1., -1.]), initial_omega=np.array([-1., -1.]))
    coefficients = np.hstack((er.lambda_, er.omega_))
    print(coefficients)

    plt.scatter(x, y)
    plt.plot(x, er.predict(x), color='red')
    plt.show()


if __name__ == '__main__':
    main()
