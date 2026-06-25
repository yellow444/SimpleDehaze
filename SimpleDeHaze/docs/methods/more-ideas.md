# Ещё кандидаты для реализации

Подборка edge-preserving подходов, которые хорошо ложатся на ту же задачу - уточнение карты
$t$ (или пост-обработку $J$) без полного Matting Laplacian. Часть методов established,
часть - практичные исследовательские идеи для этого проекта.

> **Уже реализованы** как методы в GUI: **WLS** (как Matting-WLS), **Total Variation**
> (`DCP - Total Variation`), **Domain Transform** (`DCP - Domain Transform`) и **Fast Global
> Smoother** (`DCP - Fast Global Smoother`). Остальные ниже - кандидаты.

| Метод | Основа | Память | Скорость | Когда брать |
|---|---|---|---|---|
| [WLS](#1-wls--weighted-least-squares) | взвеш. МНК | $O(N)$ | средняя | уже есть как Matting-WLS |
| [Fast Global Smoother](#2-fast-global-smoother-fgs) | быстрая WLS-аппроксимация | низкая | высокая | OpenCV/ximgproc-кандидат |
| [Domain Transform](#3-domain-transform) | геодезика 1D, сепарабельно | низкая | очень высокая | реальное время, видео |
| [Fast Bilateral Solver](#4-fast-bilateral-solver) | билатеральная сетка + малая система | низкая | высокая | edge-aware, устойчиво |
| [Anisotropic Diffusion](#5-anisotropic-diffusion-peronamalik) | PDE Перона-Малика | низкая | средняя | контролируемое сглаживание |
| [Total Variation](#6-total-variation-tv) | $\ell_1$-градиент, primal-dual | низкая | высокая | без матриц, резкие границы |
| [Confidence Hybrid](#7-confidence-weighted-hybrid-dcp--haze-lines) | смесь оценок $t$ | низкая | средняя | идея для проекта |
| [Optimal Transport](#8-optimal-transport) | перенос распределений | выше | ниже | цвет/тон, экзотика |

---

## 1. WLS - Weighted Least Squares

Edge-preserving сглаживание (Farbman, 2008): найти $t$, гладкую там, где кадр гладкий, и
сохраняющую скачки там, где у $I$ край. Минимизируем

$$\sum_x \Bigl[(t_x-\tilde t_x)^2 + \lambda\bigl(a_{x,h}\,(\partial_h t)_x^2 + a_{x,v}\,(\partial_v t)_x^2\bigr)\Bigr],\quad
a = \bigl(|\partial \log I| ^{\gamma} + \epsilon\bigr)^{-1}$$

Веса $a$ малы на краях яркости $\log I$ -> там градиент $t$ штрафуется слабее. Сводится к
разреженной 5-точечной системе (намного легче полного матового лапласиана). В проекте уже
есть близкий matrix-free вариант: [`Refiners.Wls`](../../Methods/Refiners.cs) с итерациями
взвешенного Якоби и GPU-версией. Ref: Farbman et al., SIGGRAPH 2008.

## 2. Fast Global Smoother (FGS)

Min et al. (2014): быстрый аппроксиматор WLS/edge-aware сглаживания через последовательность
1D-задач. Практически это хороший кандидат вместо собственного WLS-итератора: он сохраняет
края, работает быстро и есть в OpenCV `ximgproc` как `FastGlobalSmootherFilter`.

$$E(t)=\sum_x (t_x-\tilde t_x)^2 + \lambda \sum_{(x,y)} w_{xy}(t_x-t_y)^2$$

**Плюс:** ближе к WLS-качеству, чем guided filter, но проще и быстрее полного sparse solve.
**Минус:** нужно проверить доступность биндинга в текущей версии Emgu.CV/ximgproc.
Заменяет уточнение $t$. Ref: Min, Choi, Lu, Ham, Sohn, Do, *Fast Global Image Smoothing
Based on Weighted Least Squares*, IEEE TIP 2014.

## 3. Domain Transform

Gastal & Oliveira (2011): edge-aware фильтрация за $O(N)$ через 'геодезическое' 1D-преобразование
координат, в котором цветовые расстояния учитываются как удлинение оси. Фильтр применяется
**сепарабельно** (несколько проходов по строкам и столбцам):

$$ct(u) = \int_0^u \Bigl(1 + \tfrac{\sigma_s}{\sigma_r}\textstyle\sum_c |I_c'(s)|\Bigr)\,ds$$

затем по этой деформированной координате - простое рекурсивное сглаживание. **Плюс:**
очень быстро, реально для видео, тривиально на GPU. **Минус:** сепарабельность даёт лёгкую
анизотропию. Заменяет уточнение $t$. Ref: Gastal & Oliveira, SIGGRAPH 2011.

## 4. Fast Bilateral Solver

Barron & Poole (2016): получает edge-aware результат, решая **маленькую** систему в
билатеральной сетке (bilateral grid), а не $N\times N$ в пикселях. Минимизирует

$$\min_{t}\ \tfrac12\sum_{i,j} \hat W_{i,j}(t_i-t_j)^2 + \sum_i \lambda_i (t_i - \tilde t_i)^2$$

где сглаживающий аффинитет $\hat W$ задан в bilateral-space (где соседство = близость по
$(x,y,I)$). Система маленькая (число занятых ячеек сетки << $N$) -> быстро и мало памяти.
**Плюс:** качество близко к bilateral/Laplacian при высокой скорости; есть карта
доверия $\lambda_i$. Заменяет уточнение $t$. Ref: Barron & Poole, ECCV 2016.

## 5. Anisotropic Diffusion (Perona-Malik)

PDE-сглаживание, тормозящееся на краях:

$$\frac{\partial t}{\partial \tau} = \operatorname{div}\bigl(c(\lVert\nabla I\rVert)\,\nabla t\bigr),\qquad
c(s) = \exp\!\bigl(-(s/K)^2\bigr)$$

Коэффициент диффузии $c$ мал на сильных градиентах $I$ -> края сохраняются. Явная схема -
несколько итераций поэлементно (как [Beltrami](beltrami-flow.md), но проще: скалярный $c$).
**Плюс:** простая управляемая регуляризация, $O(N)$/итерация, GPU. **Минус:** итеративно,
подбор $K$ и числа шагов. Ref: Perona & Malik, TPAMI 1990.

## 6. Total Variation (TV)

Сглаживание с $\ell_1$-штрафом на градиент сохраняет резкие границы (в отличие от $\ell_2$):

$$\min_{t}\ \tfrac12\lVert t-\tilde t\rVert^2 + \lambda\,\mathrm{TV}(t),\qquad
\mathrm{TV}(t)=\sum_x \lVert\nabla t_x\rVert$$

Решается **matrix-free** алгоритмом primal-dual (Chambolle-Pock): только градиент/дивергенция
и проекции - хорошо ложится на GPU, память $O(N)$. Веса можно сделать направляющими по $I$
(weighted-TV). **Плюс:** резкие границы без матриц, отлично параллелится. **Минус:** возможен
'ступенчатый' (staircase) эффект на плавных градиентах. Заменяет уточнение $t$.
Ref: Rudin-Osher-Fatemi 1992; Chambolle-Pock 2011.

## 7. Confidence-weighted Hybrid DCP + Haze-Lines

Практичная идея именно для SimpleDeHaze: объединить две уже реализованные оценки $t$ -
DCP и Haze-Lines - через карты доверия.

1. Считаем $\tilde t_{dcp}$ и $\tilde t_{hl}$.
2. Строим доверие DCP: ниже на ярких/малонасыщенных областях и небе, выше на текстурных
   участках с нормальным dark-channel.
3. Строим доверие Haze-Lines: выше у бинов с большим числом пикселей и устойчивым `r_max`,
   ниже у пустых/редких направлений.
4. Смешиваем:

$$t = \frac{w_{dcp}\,t_{dcp}+w_{hl}\,t_{hl}}{w_{dcp}+w_{hl}+\epsilon}$$

5. Финально уточняем Guided Filter / WLS / MST.

Это не 'известный метод из статьи', а инженерная гипотеза: взять сильную сторону DCP
(надёжность на текстурных объектах) и Haze-Lines (лучше ведёт себя в цветовой геометрии)
и явно подавить их слабые зоны.

## 8. Optimal Transport (бонус, экзотика)

Туман сдвигает и 'сжимает' распределение цветов к $A$. Восстановление можно поставить как
**перенос** наблюдаемого цветового распределения к целевому (haze-free) с минимальной
стоимостью Вассерштейна, локально регуляризованным по пространству. Дорого и нетривиально,
но даёт интересный взгляд 'дехейзинг = транспорт гистограммы цвета'. Зрелость: research.
Связь - с [Color Cube / Haze-Lines](color-cube-projection.md). Ref: работы по
OT в обработке цвета/тона.

---

## Как это встраивать в проект

Пункты 1-6 - прямая замена уточнения $t$: на вход грубая карта из
[`DehazeCore.RawTransmission`](../../Methods/DehazeCore.cs), на выход - уточнённая. FGS,
Domain Transform и TV проще всего довести до интерактива; WLS/Bilateral Solver дают качество
ближе к [Matting Laplacian](laplacian-matting.md) при меньшей цене. Пункт 7 - гибрид уже
имеющихся оценок и может жить отдельным `IDeHazeMethod`.
