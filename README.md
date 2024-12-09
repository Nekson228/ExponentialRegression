# Exponential Regression

ETU 2024 CompMath exponential sum fitting implementation

Given $(X, Y)$, where $X$ is 1-dimensional values and $Y$ is 1-dimensional target.

Consider $p$ is amount of exponential terms so we are going to fit:

$$
f: X \to Y; \:
f(t, \textbf{p}) =\sum_{i=1}^p\lambda_i\alpha_i^t
$$

Where $\textbf{p} = (\lambda_1, \ldots, \lambda_p, \alpha_1, \ldots, \alpha_p)$

Assuming $\forall i \:\alpha_i > 0$ we can rewrite $f(t, \textbf{p})$ as:

$$
f(t, \textbf{p})=\sum_{i=1}^p\lambda_i\exp(\ln(\alpha_i)t) =
\sum_{i=1}^p\lambda_i\exp(\beta_it)
$$

Since $\forall i\:\beta_i\in\mathbb{R}$

$$
f(t)=\sum_{i=1}^p\lambda_i\exp(-\omega_it)
$$

Finally,$\textbf{p} = (\lambda_1, \ldots, \lambda_p, \omega_1, \ldots, \omega_p)$ – are parameters to fit

Loss function to optimize is MSE:

$$
L(\textbf{p}) = \sum_{i=1}^p(y_i-f(t_i, \textbf{p}))^2
$$

The method to optimize will be Levenberg-Marquardt algorith used in built-in curve fit optimizer for scikit-learn.

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

So $\forall j\le p:$

$$
\mathbf{J}_{ij}=\frac{\partial f\left (t_i,  \mathbf{p}\right )}{\partial  \mathbf{\lambda_j}} = \exp(-\omega_jx),
$$

$$
\mathbf{J}_{ij+p}=\frac{\partial f\left (t_i,  \mathbf{p}\right )}{\partial  \mathbf{\omega_j}} = -\omega_j\exp(-\omega_jx).
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

 The above expression obtained for ⁠$ \mathbf{p}$ comes under the Gauss–Newton method. The Jacobian matrix as defined above is not (in general) a square matrix, but a rectangular matrix of size $m \times n$, where $n$ is the number of parameters (size of the vector $ \mathbf{p}$). The matrix multiplication $\boldsymbol{J}^T\boldsymbol{J}$yields the required $n\times n$
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