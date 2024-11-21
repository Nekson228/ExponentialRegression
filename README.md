# ExponentialRegression

ETU 2024 CompMath exponential regression implementation

Given $(X, Y)$, where $X$ is 1-dimensional values and $Y$ is 1-dimensional target.

Consider $p$ is amount of exponential terms so we are going to fit:

$$
f(x) =\sum_{i=1}^p\lambda_i\alpha_i^x
$$

Assuming $\forall i \:\alpha_i > 0$ we can rewrite $f(x)$ as:

$$
f(x)=\sum_{i=1}^p\lambda_i\exp(\ln(\alpha_i)x) =
\sum_{i=1}^p\lambda_i\exp(\beta_ix)
$$

Since $\forall i\:\beta_i\in\mathbb{R}$ 

$$
f(x)=\sum_{i=1}^p\lambda_i\exp(-\omega_ix)
$$