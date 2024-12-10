# Exponential Regression

ETU 2024 CompMath exponential sum fitting implementation

Given $(t_i, y_i)_{i=1}^n$, where $X$ is 1-dimensional values and $Y$ is 1-dimensional target.

Consider $p$ is amount of exponential terms so we are going to fit:

$$
f: X \to Y; \:
f(t, \textbf{p}) =\sum_{i=1}^p\lambda_i\alpha_i^t
$$

Where $\textbf{p} = (\lambda_1, \ldots, \lambda_p, \alpha_1, \ldots, \alpha_p)$

Assuming $\forall i \:\alpha_i > 0$ we can rewrite $f(t, \textbf{p})$ as:

$$
f(t, \textbf{p})=\sum_{i=1}^p\lambda_i\exp(\ln(\alpha_i)t) =
\sum_{i=1}^p\lambda_i\exp(\omega_it)
$$

Finally, $\textbf{p} = (\lambda_1, \ldots, \lambda_p, \omega_1, \ldots, \omega_p)$ – are parameters to fit

Loss function to optimize is MSE:

$$
L(\textbf{p}) = \sum_{i=1}^n(y_i-f(t_i, \textbf{p}))^2
$$

The method to optimize will be Levenberg-Marquardt algorith used in built-in curve_fit optimizer for scipy.

## Levenberg-Marquardt algorithm (LMA)

Like other numeric minimization algorithms, the Levenberg–Marquardt algorithm is an iterative procedure.
To start a minimization, the user has to provide an initial guess for the parameter vector $\textbf{p}^T=\begin{pmatrix}1, 1, \dots, 1\end{pmatrix}$ will work fine;
in cases with multiple minima, the algorithm converges to the global minimum only if the initial guess is already somewhat close to the final solution.

In each iteration step, the parameter vector
$\textbf{p}$ is replaced by a new estimate $\textbf{p} + \mathbf{\Delta}$ 
To determine $\mathbf{\Delta}$ the function $f(t_i, \textbf{p} + \mathbf{\Delta})$ is approximated by its linearization:

$$
f(t_i, \textbf{p} + \mathbf{\Delta})\approx f(t_i, \textbf{p})+\mathbf{J}_i\mathbf{\Delta}
$$

where 

$$
\mathbf{J}_i = \frac{\partial f\left (t_i,  \mathbf{p}\right )}{\partial  \mathbf{p}}
$$

is the gradient of $f$ with respect to $\mathbf{p}$.

So, for our problem $\forall j\le p:$

$$
\mathbf{J}_{ij}=\frac{\partial f\left (t_i,  \mathbf{p}\right )}{\partial  \mathbf{\lambda_j}} = \exp(\omega_jt_i),
$$

$$
\mathbf{J}_{ij+p}=\frac{\partial f\left (t_i,  \mathbf{p}\right )}{\partial  \mathbf{\omega_j}} = \lambda_jt_i\exp(\omega_jt_i).
$$

The loss function has its minimum at a zero gradient with respect to $\textbf{p}$. The above first-order approximation of $f\left (t_i,  \mathbf{p} + \boldsymbol\Delta\right )$ gives

$$
L\left ( \mathbf{p} + \boldsymbol\Delta\right ) \approx \sum_{i=1}^p \left [y_i - f\left (t_i,  \mathbf{p}\right ) - \mathbf J_i \boldsymbol\Delta\right ]^2
$$

or in vector notation,

$$
\begin{align}
 L\left ( \mathbf{p} + \boldsymbol\Delta\right ) &\approx \left \|\mathbf y - \mathbf f\left ( \mathbf{p}\right ) - \mathbf J\boldsymbol\Delta\right \|_2^2\notag\\
  &= \left [\mathbf y - \mathbf f\left ( \mathbf{p}\right ) - \mathbf J\boldsymbol\Delta \right ]^{\mathrm T}\left [\mathbf y - \mathbf f\left ( \mathbf{p}\right ) - \mathbf J\boldsymbol\Delta\right ]\notag\\
  &= \left [\mathbf y - \mathbf f\left ( \mathbf{p}\right )\right ]^{\mathrm T}\left [\mathbf y - \mathbf f\left ( \mathbf{p}\right )\right ] - \left [\mathbf y - \mathbf f\left ( \mathbf{p}\right )\right ]^{\mathrm T} \mathbf J \boldsymbol\Delta - \left (\mathbf J \boldsymbol\Delta\right )^{\mathrm T} \left [\mathbf y - \mathbf f\left ( \mathbf{p}\right )\right ] + \boldsymbol\Delta^{\mathrm T} \mathbf J^{\mathrm T} \mathbf J \boldsymbol\Delta\notag\\
  &= \left [\mathbf y - \mathbf f\left ( \mathbf{p}\right )\right ]^{\mathrm T}\left [\mathbf y - \mathbf f\left ( \mathbf{p}\right )\right ] - 2\left [\mathbf y - \mathbf f\left ( \mathbf{p}\right )\right ]^{\mathrm T} \mathbf J \boldsymbol\Delta + \boldsymbol\Delta^{\mathrm T} \mathbf J^{\mathrm T} \mathbf J\boldsymbol\Delta.\notag
\end{align}
$$

Taking the derivative of this approximation of $L\left ( \mathbf{p} + \boldsymbol\Delta\right )$
 with respect to ⁠$\Delta$⁠ and setting the result to zero gives

 $$
 \left (\mathbf J^{\mathrm T} \mathbf J\right )\boldsymbol\Delta = \mathbf J^{\mathrm T}\left [\mathbf y - \mathbf f\left ( \mathbf{p}\right )\right ],
 $$

 The above expression obtained for ⁠$\mathbf{p}$ comes under the Gauss–Newton method. The Jacobian matrix as defined above is not (in general) a square matrix, but a rectangular matrix of size $m \times n$, where $n$ is the number of parameters (size of the vector $\mathbf{p}$). The matrix multiplication $\boldsymbol{J}^T\boldsymbol{J}$yields the required $n\times n$
 square matrix and the matrix-vector product on the right hand side yields a vector of size $n$. The result is a set of $n$ linear equations, which can be solved for ⁠$\boldsymbol{\Delta}$.

 Levenberg's contribution is to replace this equation by a "damped version":

 $$
 \left (\mathbf J^{\mathrm T} \mathbf J + \lambda\mathbf E\right ) \boldsymbol\Delta = \mathbf J^{\mathrm T}\left [\mathbf y - \mathbf f\left ( \mathbf{p}\right )\right],
 $$

The (non-negative) damping factor ⁠$\lambda$⁠ is adjusted at each iteration. If reduction of ⁠$L$ is rapid, a smaller value can be used, bringing the algorithm closer to the Gauss–Newton algorithm:
$$
\mathbf{\Delta}\approx[\mathbf{J}^T\mathbf{J}]^{-1}\mathbf{J}^{T}[\mathbf y - \mathbf f\left ( \mathbf{p}\right )]
$$

whereas if an iteration gives insufficient reduction in the residual, ⁠$\lambda$ can be increased, giving a step closer to the gradient-descent direction:

$$
\mathbf{\Delta}\approx\lambda^{-1}\mathbf{J}^{T}[\mathbf y - \mathbf f\left ( \mathbf{p}\right )]
$$

To make the solution scale invariant Marquardt's algorithm solved a modified problem with each component of the gradient scaled according to the curvature. This provides larger movement along the directions where the gradient is smaller, which avoids slow convergence in the direction of small gradient. Fletcher in his 1971 paper *A modified Marquardt subroutine for non-linear least squares* simplified the form, replacing the identity matrix ⁠$E$ with the diagonal matrix consisting of the diagonal elements of ⁠$\mathbf{J}^T\mathbf{J}$:

$$
\left [\mathbf J^{\mathrm T} \mathbf J + \lambda \operatorname{diag}\left (\mathbf J^{\mathrm T} \mathbf J\right )\right ] \boldsymbol\Delta = \mathbf J^{\mathrm T}\left [\mathbf y - \mathbf f\left (\boldsymbol p\right )\right ].
$$

### Choice of damping parameter

An effective strategy for the control of the damping parameter, called delayed gratification, consists of increasing the parameter by a small amount for each uphill step, and decreasing by a large amount for each downhill step. The idea behind this strategy is to avoid moving downhill too fast in the beginning of optimization, therefore restricting the steps available in future iterations and therefore slowing down convergence. An increase by a factor of 2 and a decrease by a factor of 3 has been shown to be effective in most cases, while for large problems more extreme values can work better, with an increase by a factor of 1.5 and a decrease by a factor of 5.

## Implementation details

As performance of implementation relied only on theoretical approaches was not good enough to use it as a solution, in resulting implementation few improvements were made.

### Loss function

Loss function was changed to $\chi^2$ loss since it is often used to curve-fitting problems. 
It is defined as:
$$
\chi^2(\boldsymbol p) = \sum_{i=1}^n\left(\frac{y_i-f(t_i, \boldsymbol p)}{\sigma_i}\right)^2 = \left[\mathbf y - \mathbf f\left ( \mathbf{p}\right )\right ]^T\boldsymbol{W}\left[\mathbf y - \mathbf f\left ( \mathbf{p}\right )\right ],
$$
where $\boldsymbol{W} = \operatorname{diag}\left(\frac{1}{\sigma_1^2}, \ldots, \frac{1}{\sigma_n^2}\right)$ is a weight matrix of variances of each measurement. In practice, it is used to give more weight to the measurements with smaller errors. 

The update formula for $\mathbf{\Delta}$ is adjusted accordingly to reflect the change in loss function::

$$
\left (\mathbf J^{\mathrm T} \boldsymbol{W} \mathbf J + \lambda \mathbf{E} \right ) \boldsymbol\Delta = \mathbf J^{\mathrm T}\boldsymbol{W}\left [\mathbf y - \mathbf f\left ( \boldsymbol p\right )\right ].
$$

### Step acceptance

Previously, step was accepted if loss function decreased, else it was rejected and damping parameter was increased. Now, the step is accepted if the metric $\rho$ is greater than a user-defined threshold $\epsilon_4 > 0$ (`step-acceptance` in code). This metric is a measure of actual reduction in $\chi^2$ compared to the the improvement of an LMA step. 

$$
\begin{align}
\rho &= \frac{\chi^2(\boldsymbol p) - \chi^2(\boldsymbol p + \boldsymbol\Delta)}
{|(\boldsymbol{y}-\boldsymbol{\hat{y}})^T\mathbf{W}(\boldsymbol{y}-\boldsymbol{\hat{y}}) - (\boldsymbol{y}-\boldsymbol{\hat{y}}-\mathbf{J\Delta})^T\mathbf{W}(\boldsymbol{y}-\boldsymbol{\hat{y}}-\mathbf{J\Delta})|}\notag\\
&=\frac{\chi^2(\boldsymbol p) - \chi^2(\boldsymbol p + \boldsymbol\Delta)}
{|\mathbf{\Delta}^T(\lambda\mathbf{\Delta} + \mathbf{J}^T\mathbf{W}(\boldsymbol{y}-\boldsymbol{\hat{y}}))|}\notag\\
\end{align}
$$

where $\boldsymbol{\hat{y}} = \mathbf{f}(\boldsymbol{p})$.

This metric of step acceptance was proposed by H.B. Nielson in his 1999 paper [3].

Chosen value for $\epsilon_4$ is $10^{-1}$.

### Update strategy

Damping parameter and model parameters are updated according to the following rules:

If $\rho > \epsilon_4$: $\lambda = \max[\lambda/L_\downarrow,\:10^{-7}],\:\mathbf{p} \leftarrow\mathbf{p} + \mathbf{\Delta}$<br>
otherwise: $\lambda = \min[\lambda L_\uparrow,\:10^{7}]$

where $L_\downarrow\approx9$ and $L_\uparrow\approx11$ are fixed constants (`REG_DECREASE_FACTOR` and `REG_INCREASE_FACTOR` in code). These values were chosen based on the paper [2].

### Convergence criteria

The algorithm stops when *one* of the following conditions is satisfied:

- Convergence in the gradient norm: $\operatorname{max}|\mathbf{J}^T\mathbf{W}(\boldsymbol{y}-\boldsymbol{\hat{y}})| < \epsilon_1$ (`gradient_tol` in code)
- Convergence in coefficients: $\operatorname{max}|{\mathbf{\Delta}}/\mathbf{p}| < \epsilon_2$ (`coefficients_tol` in code)
- Convergence in (reduced) $\chi^2$: $\chi^2_{\nu}=\chi^2/(m-n) < \epsilon_3$ (`chi2_red_tol` in code)

where $\epsilon_1 = 10^{-3}$, $\epsilon_2 = 10^{-3}$, $\epsilon_3 = 10^{-1}$ are user-defined thresholds.

### Initial guess

In nonlinear least squares problems the $\chi^2(\mathbf{p})$ loss function may have multiple local minima. In such cases, the LMA may converge to a poor fit. If this happens, the user can try to provide a better initial guess for the parameters, for example by random/grid search or data inspection.


# References

1. [Wikipedia contributors. *Levenberg–Marquardt algorithm*. Wikipedia, The Free Encyclopedia.](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm).
2. [H.P. Gavin, *The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems*. 2020](https://people.duke.edu/~hpgavin/ce281/lm.pdf).
3. [H.B. Nielson, *Damping Parameter in Marquardt's method*. 1999](https://www2.imm.dtu.dk/documents/ftp/tr99/tr05_99.pdf).
