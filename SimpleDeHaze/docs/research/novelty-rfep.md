# Novelty Note: Boundary-Constrained Prior Fusion (быв. RFEP-DCP)

> **Документ устарел частично.** Актуальная и обязательная к соблюдению версия формулировок -
> [`NOVELTY.md`](../../../NOVELTY.md) в корне репозитория. Ниже сохранён исходный текст с правками.
>
> Главное изменение: граница `t_box` **не является вкладом** - это boundary constraint
> Meng et al., ICCV 2013. Кандидат в вклад - другая граница, выведенная для фактически
> используемой chroma-safe формулы восстановления (`DehazeCore.ChromaSafeLowerBound`).

## Короткий ответ

В репозитории есть кандидат на самостоятельный новый вклад, но это **не** та формулировка,
с которой начинался документ:

**допустимое множество трансмиссии для chroma-safe восстановления**
(`J_c = A_c + d̄/max(t,t_min) + δ_c/max(t,chromaFloor)`), где оно оказывается отрезком, а не лучом.

До RFEP уже существовали:

- DCP: dark-channel prior + atmospheric model.
- Boundary Constraint and Contextual Regularization: transmission constraints from the image
  formation model plus optimization.
- PF-DCP: multi-scale DCP fusion.
- RSVT: bright-region correction in HSV saturation/value space.
- Fast Guided Filter: subsampled guided refinement.

RFEP-DCP не должен заявляться как “мы первые придумали ограничения на transmission”.
Безопасная формулировка до литературной проверки chroma-safe варианта и внешней абляции:

> We implement a boundary-constrained projection layer before and after fast edge-aware
> refinement. Its standard RGB-box constraint is prior work (Meng et al., ICCV 2013); the
> chroma-safe admissible interval is treated as a testable candidate contribution, not an
> established novelty claim.

## Что именно новое относительно известных prior-based методов

| Известная линия | Что уже есть | Что добавляет RFEP |
|---|---|---|
| DCP | `t=1-omega*dark(I/A)` и guided refinement | RFEP не доверяет `t` слепо: перед делением на `t` проверяется физическая допустимость `J` |
| Boundary Constraint | lower/upper constraints на transmission + global regularization | RFEP делает constraint как дешёвый projection layer после multi-prior fusion, без тяжёлого variable splitting |
| RSVT / HSV bright prior | bright-region handling в saturation/value space | RFEP использует robust `V-S` quantile prior как один из кандидатов `t`, но не заменяет им recovery |
| PF-DCP | multi-scale transmission fusion | RFEP ортогонален масштабу: projection можно применить к DCP, BRACE или PF-SFGF |
| Gradient-domain recovery | post-recovery Poisson solve | RFEP действует раньше: предотвращает физически невозможное деление до recovery |

## Основные формулы

Atmospheric inversion:

$$
J_c(x)=\frac{I_c(x)-A_c}{t(x)}+A_c.
$$

RGB feasibility:

$$
0\le J_c(x)\le 1.
$$

Per-channel lower bound:

$$
t_{box,c}(x)=
\begin{cases}
\frac{I_c(x)-A_c}{1-A_c+\varepsilon}, & I_c(x)>A_c,\\
\frac{A_c-I_c(x)}{A_c+\varepsilon}, & I_c(x)<A_c,\\
0, & I_c(x)=A_c.
\end{cases}
$$

Envelope:

$$
t_{box}(x)=\max_c t_{box,c}(x).
$$

Projection:

$$
t_{proj}(x)=t_{mix}(x)+\rho\max(t_{box}(x)-t_{mix}(x),0).
$$

## Claims safe enough for Habr

- Это полностью non-ML метод.
- Он воспроизводим и прозрачен: каждая стадия имеет формулу.
- Он добавляет физическую “проверку допустимости” перед опасным делением на `t`.
- Он хорошо подходит для инженерного объяснения failure cases: клиппинг, цветные ореолы,
  переусиление белых/ярких областей.

## Claims that need stronger experiments before arXiv submission

- “Outperforms prior-based baselines” - нужно проверить на RESIDE SOTS / I-HAZE / O-HAZE.
- “Reduces halo artifacts” - нужен отдельный halo/edge score или human study.
- “State-of-the-art among non-ML methods” - нужен набор внешних baselines: BCCR, PF-DCP,
  non-local haze-lines, RSVT, Tarel.

## Literature anchors

- He, Sun, Tang: Dark Channel Prior.
- Meng et al.: Boundary Constraint and Contextual Regularization.
- He, Sun: Fast Guided Filter.
- Liang et al.: PF-DCP.
- Tran, Park: RSVT bright-region saturation/value prior.

RFEP should be presented as an **experimental engineering composition**. A scientific novelty
claim requires literature review and the registered ablations described in `NOVELTY.md`.
