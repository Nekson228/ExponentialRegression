# Экспоненциальная регрессия

Даны $(X, Y)$, где $X$ — одномерные значения, а $Y$ — одномерная целевая переменная.

Рассмотрим $p$ — количество экспоненциальных членов, тогда мы подбираем функцию:

$$
f: X \to Y; \:
f(t, \textbf{p}) =\sum_{i=1}^p\lambda_i\alpha_i^t
$$

где $\textbf{p} = (\lambda_1, \ldots, \lambda_p, \alpha_1, \ldots, \alpha_p)$

Предполагая, что $\forall i \:\alpha_i > 0$, можно переписать $f(t, \textbf{p})$ как:

$$
f(t, \textbf{p})=\sum_{i=1}^p\lambda_i\exp(\ln(\alpha_i)t) =
\sum_{i=1}^p\lambda_i\exp(\omega_it),
$$

где $\textbf{p} = (\lambda_1, \ldots, \lambda_p, \omega_1, \ldots, \omega_p)$ — параметры, которые необходимо подобрать.

Функция потерь для оптимизации — это MSE:

$$
L(\textbf{p}) = \sum_{i=1}^p(y_i-f(t_i, \textbf{p}))^2.
$$

Метод оптимизации — алгоритм Левенберга — Марквардта, встроенный в функцию curve_fit библиотеки scipy.

## Алгоритм Левенберга — Марквардта (LMA)

Подобно другим численным методам минимизации, алгоритм Левенберга — Марквардта является итеративной процедурой. Для начала минимизации необходимо задать начальное приближение для вектора параметров. Начальное значение $\textbf{p}^T=\begin{pmatrix}1, 1, \dots, 1\end{pmatrix}$ подходит в большинстве случаев; в задачах с множеством локальных минимумов алгоритм сходится к глобальному минимуму, только если начальное приближение достаточно близко к решению.

На каждом шаге итерации вектор параметров $\textbf{p}$ заменяется новой оценкой $\textbf{p} + \mathbf{\Delta}$. Чтобы определить $\mathbf{\Delta}$, функция $f(t_i, \textbf{p} + \mathbf{\Delta})$ линеаризуется:

$$
f(t_i, \textbf{p} + \mathbf{\Delta})\approx f(t_i, \textbf{p})+\mathbf{J}_i\mathbf{\Delta},
$$

где 

$$
\mathbf{J}_i = \frac{\partial f\left (t_i,  \mathbf{p}\right )}{\partial  \mathbf{p}}
$$

— это градиент $f$ по параметрам $\mathbf{p}$.

Таким образом $\forall j\le p:$

$$
\mathbf{J}_{ij}=\frac{\partial f\left (t_i,  \mathbf{p}\right )}{\partial  \mathbf{\lambda_j}} = \exp(\omega_jt_i),
$$

$$
\mathbf{J}_{ij+p}=\frac{\partial f\left (t_i,  \mathbf{p}\right )}{\partial  \mathbf{\omega_j}} = \lambda_jt_i\exp(\omega_jt_i).
$$

Функция потерь достигает минимума, когда её градиент по $\textbf{p}$ равен нулю. Для первого приближения $f\left (t_i,  \mathbf{p} + \boldsymbol\Delta\right )$:

$$
L\left ( \mathbf{p} + \boldsymbol\Delta\right ) \approx \sum_{i=1}^p \left [y_i - f\left (t_i,  \mathbf{p}\right ) - \mathbf J_i \boldsymbol\Delta\right ]^2
$$

или в векторной форме:

$$
L\left ( \mathbf{p} + \boldsymbol\Delta\right ) \approx \|\mathbf y - \mathbf f\left ( \mathbf{p}\right ) - \mathbf J\boldsymbol\Delta\|_2^2.
$$

Взяв производную от $L\left ( \mathbf{p} + \boldsymbol\Delta\right )$ по $\Delta$ и приравняв её к нулю, получим:

$$
\left (\mathbf J^{\mathrm T} \mathbf J\right )\boldsymbol\Delta = \mathbf J^{\mathrm T}\left [\mathbf y - \mathbf f\left ( \mathbf{p}\right )\right ].
$$

Выражение выше соответствует методу Гаусса–Ньютона. Матрица Якоби $\mathbf{J}$ обычно не квадратная, а прямоугольная размерности $m \times n$, где $n$ — количество параметров. Перемножение $\boldsymbol{J}^T\boldsymbol{J}$ дает квадратную матрицу размерности $n \times n$. Результат — это система из $n$ линейных уравнений, решаемая для $\boldsymbol{\Delta}$.

Вклад Левенберга заключается в использовании регуляризованной версии уравнения:

$$
\left (\mathbf J^{\mathrm T} \mathbf J + \lambda\mathbf E\right ) \boldsymbol\Delta = \mathbf J^{\mathrm T}\left [\mathbf y - \mathbf f\left ( \mathbf{p}\right )\right],
$$

где $\lambda$ — коэффициент регуляризации, настраиваемый на каждой итерации. Если снижение $L$ быстрое, значение $\lambda$ уменьшается, приближая алгоритм к методу Гаусса–Ньютона:

$$
\mathbf{\Delta}\approx[\mathbf{J}^T\mathbf{J}]^{-1}\mathbf{J}^{T}[\mathbf y - \mathbf f\left ( \mathbf{p}\right )],
$$

иначе $\lambda$ увеличивается, приближая шаг к направлению градиентного спуска:

$$
\mathbf{\Delta}\approx\lambda^{-1}\mathbf{J}^{T}[\mathbf y - \mathbf f\left ( \mathbf{p}\right )].
$$

Чтобы сделать решение инвариантным к масштабу, алгоритм Марквардта решал модифицированную задачу, в которой каждая компонента градиента масштабировалось в соответствии с кривизной. Это обеспечивает более значительные изменения вдоль направлений с меньшим градиентом, что позволяет избежать медленной сходимости в этих направлениях. Флетчер в своей статье 1971 года *A modified Marquardt subroutine for non-linear least squares* упростил эту формулу, заменив единичную матрицу $E$ диагональной матрицей, состоящей из диагональных элементов $\mathbf{J}^T\mathbf{J}$:

$$
\left [\mathbf J^{\mathrm T} \mathbf J + \lambda \operatorname{diag}\left (\mathbf J^{\mathrm T} \mathbf J\right )\right ] \boldsymbol\Delta = \mathbf J^{\mathrm T}\left [\mathbf y - \mathbf f\left (\boldsymbol p\right )\right ].
$$

### Выбор коэффициента регуляризации

Эффективной стратегией управления $\lambda$ является "отложенное вознаграждение": увеличение $\lambda$ после неудачного шага и снижение после успешного. Например, увеличение в 2 раза и снижение в 3 раза работает в большинстве случаев, для задач с большим числом параметров лучше брать более агрессивные значения, например, увеличение в 1.5 раза и снижение в 5 раз.