# Экспоненциальная регрессия

Даны $(t_i, y_i)_{i=1}^n$, где $X$ — одномерные значения, а $Y$ — одномерная целевая переменная.

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

Эффективной стратегией управления $\lambda$ является "отложенное вознаграждение": увеличение $\lambda$ после неудачного шага и снижение после успешного. Это позволяет алгоритму быстро сходиться к решению, когда он находится вблизи минимума, и медленно исследовать пространство параметров, когда он находится далеко от минимума.

## Детали реализации

Так как производительность реализации, основанной исключительно на теоретических выкладках, оказалась недостаточной для практического применения, в итоговом решении были внесены несколько улучшений.

### Функция потерь

Функция потерь была изменена на $\chi^2$, так как она часто используются в задачах аппроксимации кривых. Она определяется следующим образом:
$$
\chi^2(\boldsymbol p) = \sum_{i=1}^n\left(\frac{y_i-f(t_i, \boldsymbol p)}{\sigma_i}\right)^2 = \left[\mathbf y - \mathbf f\left ( \mathbf{p}\right )\right ]^T\boldsymbol{W}\left[\mathbf y - \mathbf f\left ( \mathbf{p}\right )\right ],
$$
где $\boldsymbol{W} = \operatorname{diag}\left(\frac{1}{\sigma_1^2}, \ldots, \frac{1}{\sigma_n^2}\right)$ — матрица весов, зависящих от дисперсий каждого измерения. На практике она используется для увеличения веса измерений с меньшими ошибками.

Формула обновления для $\mathbf{\Delta}$ была скорректирована, чтобы учитывать изменение функции потерь:

$$
\left (\mathbf J^{\mathrm T} \boldsymbol{W} \mathbf J + \lambda \mathbf{E} \right ) \boldsymbol\Delta = \mathbf J^{\mathrm T}\boldsymbol{W}\left [\mathbf y - \mathbf f\left ( \boldsymbol p\right )\right ].
$$

### Принятие шага

Ранее шаг принимался, если функция потерь уменьшалась, иначе он отклонялся, а коэффициент регуляризации увеличивался. Теперь шаг принимается, если метрика $\rho$ больше порогового значения $\epsilon_4 > 0$ (`step-acceptance` в коде). Эта метрика измеряет фактическое уменьшение $\chi^2$ по сравнению с улучшением, достигаемым шагом метода Левенберга-Марквардта.

$$
\begin{align}
\rho &= \frac{\chi^2(\boldsymbol p) - \chi^2(\boldsymbol p + \boldsymbol\Delta)}
{|(\boldsymbol{y}-\boldsymbol{\hat{y}})^T\mathbf{W}(\boldsymbol{y}-\boldsymbol{\hat{y}}) - (\boldsymbol{y}-\boldsymbol{\hat{y}}-\mathbf{J\Delta})^T\mathbf{W}(\boldsymbol{y}-\boldsymbol{\hat{y}}-\mathbf{J\Delta})|}\notag\\
&=\frac{\chi^2(\boldsymbol p) - \chi^2(\boldsymbol p + \boldsymbol\Delta)}
{|\mathbf{\Delta}^T(\lambda\mathbf{\Delta} + \mathbf{J}^T\mathbf{W}(\boldsymbol{y}-\boldsymbol{\hat{y}}))|}\notag\\
\end{align}
$$

где $\boldsymbol{\hat{y}} = \mathbf{f}(\boldsymbol{p})$.

Эта метрика для принятия шага была предложена Нильсеном в его статье 1999 года [3].

Выбранное значение для $\epsilon_4$ — $10^{-1}$.

### Стратегия обновления

Коэффициент регуляризации и параметры модели обновляются согласно следующим правилам:

Если $\rho > \epsilon_4$: $\lambda = \max[\lambda/L_\downarrow,\:10^{-7}],\:\mathbf{p} \leftarrow\mathbf{p} + \mathbf{\Delta}$<br>
иначе: $\lambda = \min[\lambda L_\uparrow,\:10^{7}]$

где $L_\downarrow\approx9$ и $L_\uparrow\approx11$ — фиксированные константы (`REG_DECREASE_FACTOR` и `REG_INCREASE_FACTOR` в коде). Эти значения были выбраны на основе статьи [2].

### Критерии сходимости

Алгоритм останавливается, когда выполняется *одно* из следующих условий:

- Сходимость по норме градиента: $\operatorname{max}|\mathbf{J}^T\mathbf{W}(\boldsymbol{y}-\boldsymbol{\hat{y}})| < \epsilon_1$ (`gradient_tol` в коде)
- Сходимость по коэффициентам: $\operatorname{max}|{\mathbf{\Delta}}/\mathbf{p}| < \epsilon_2$ (`coefficients_tol` в коде)
- Сходимость по (редуцированному) $\chi^2$: $\chi^2_{\nu}=\chi^2/(m-n) < \epsilon_3$ (`chi2_red_tol` в коде)

где $\epsilon_1 = 10^{-3}$, $\epsilon_2 = 10^{-3}$, $\epsilon_3 = 10^{-1}$ — пороговые значения, заданные пользователем.

### Начальное приближение

В задачах нелинейных наименьших квадратов функция потерь $\chi^2(\mathbf{p})$ может иметь множество локальных минимумов. В таких случаях метод Левенберга-Марквардта может сходиться к неудовлетворительному решению. Если это происходит, пользователь может попытаться задать лучшее начальное приближение для параметров, например, с помощью случайного поиска, или поиска по сетке, либо путем анализа данных.

# Источники

1. [Wikipedia contributors. *Levenberg–Marquardt algorithm*. Wikipedia, The Free Encyclopedia.](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm).
2. [H.P. Gavin, *The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems*. 2020](https://people.duke.edu/~hpgavin/ce281/lm.pdf).
3. [H.B. Nielson, *Damping Parameter in Marquardt's method*. 1999](https://www2.imm.dtu.dk/documents/ftp/tr99/tr05_99.pdf).
