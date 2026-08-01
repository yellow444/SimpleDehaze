# Главный вывод

HSV не нужно исключать из физического дехейзинга. Проблема лишь в том, что **обычный HSV нельзя без поправок подставить в атмосферную модель вместо RGB**. Но после нормализации изображения на атмосферный свет возникает точное представление, в котором:

[
H_I=H_J,
\qquad
C_I=tC_J,
\qquad
V_I-1=t(V_J-1),
]

где (C=S\cdot V=\max(R,G,B)-\min(R,G,B)).

То есть правильная рабочая система — не стандартные (H,S,V), а:

> **airlight-normalized HCV: Hue, абсолютная Chroma и Value.**

На этой основе я предлагаю основной новый метод:

> **HCV-A²CR-UTAW**
> *Airlight-Normalized Hue–Chroma–Value Recovery with Uncertainty-Aware Dual Gains, Convex Gamut Constraints and Transmission-Aware Stationary Wavelets.*

Это значительно сильнее как научный вклад, чем:

* просто заменить Laplacian pyramid на DWT;
* добавить HSV-карту к DCP;
* обработать цвет в Lab после обычного восстановления;
* вручную связать усиление полос со значением (t).

---

# Что обнаружено в SimpleDeHaze

Исходная концепция проекта — интерпретируемый training-free конвейер: оценка атмосферного света, построение transmission, guided refinement и RGB-восстановление. Это соответствует первоначальной статье и документации проекта. 

В публичном `master` метод `HsvCapMethod` уже использует:

[
d=0.121779+0.959710V-0.780245S,
\qquad
t=e^{-\beta d}.
]

Но HSV здесь применяется **только для оценки глубины и transmission**. После этого выполняется обычное поканальное BGR-восстановление:

[
J_c=\frac{I_c-A_c}{t}+A_c.
]

Следовательно, текущий CAP-HSV не является полноценным HSV-recovery: оттенок, насыщенность и яркость не участвуют в регуляризации обратной задачи.

В текущем `DehazeCore.Normalize` изображение лишь делится на 255 — точного преобразования sRGB → linear RGB нет. А `Recover` делит среднее RGB-отклонение и цветовой остаток через разные floors. Это уже похоже на раннюю форму A²CR, но пока остаётся эвристикой в gamma-кодированном RGB.

Также есть рассинхронизация версий: публичный реестр `master` ещё не содержит `TransScaleLaplacianMethod` и `FractalHsvMethod`, тогда как переданный архив их содержит. В публичном реестре сейчас присутствуют CAP-HSV, DCP-варианты, local airlight, spectral-adaptive и enhancement-методы.

## Что делает архивный Transmission-Scale Laplacian

В переданном архиве `TransScaleLaplacianMethod.cs`:

1. вычисляет (t_{\text{CAP}});
2. вычисляет (t_{\text{DCP}});
3. берёт:

[
t=\min(t_{\text{CAP}},t_{\text{DCP}});
]

4. уточняет карту guided filter;
5. восстанавливает RGB с локальным (A(x));
6. переводит результат в 8-bit Lab;
7. строит Gaussian/Laplacian pyramid;
8. регулирует полосы функцией:

[
gate_l(x)=
smoothstep(t_l(x);t_{\mathrm{lo}},t_{\mathrm{hi}})
(1-s_l)+s_l;
]

9. применяет вручную заданные `gFine`, `gMid`, `gCoarse`.

Инженерно это разумная конструкция. Научные слабости сейчас такие:

* `min(CAP,DCP)` наследует наиболее агрессивную ошибку любого prior;
* `tLo`, `tHi` и gains подбираются вручную;
* uncertainty карты (t) отсутствует;
* `PyrDown/PyrUp` делает representation зависимым от сдвига;
* physical recovery смешан с 8-bit Lab-постобработкой;
* Laplacian/Gaussian multiscale dehazing уже опубликован.

Работа Li, Shu и Zheng уже использует Laplacian/Gaussian pyramids, различные способы удаления дымки и подавления шума на разных уровнях, в том числе для предотвращения усиления шума в небе. Поэтому сама идея «разные масштабы обрабатывать по-разному в зависимости от восстанавливаемости» уже имеет близкий prior art. ([arXiv][1])

---

# Почему HSV действительно иногда лучше DCP

Ваше наблюдение корректно.

DCP предполагает, что в локальном фрагменте чистого изображения хотя бы один RGB-канал какого-либо пикселя близок к нулю. Это часто работает на сложных естественных сценах, но ломается на:

* небе;
* белых стенах;
* светлых автомобилях;
* равномерных поверхностях;
* источниках света;
* простых минималистичных изображениях.

Ошибки DCP на ярких поверхностях и необходимость отдельной обработки bright objects давно отмечались в литературе. ([IET Research][2])

CAP использует другую корреляцию:

[
\text{дымка} \Rightarrow V\uparrow,\quad S\downarrow.
]

На простом изображении эта корреляция может описывать сцену значительно лучше dark-channel assumption. Поэтому CAP-HSV часто даёт:

* более гладкую карту глубины;
* более естественное небо;
* меньше блоков;
* меньше локального затемнения;
* лучше визуальную цветность.

Оригинальный CAP именно по (V) и (S) строит линейную модель глубины, после чего получает transmission и восстанавливает сцену через atmospheric scattering model. ([ueaeprints.uea.ac.uk][3])

Но CAP имеет обратную проблему: белый или серый близкий объект тоже обладает высоким (V) и низким (S). Поэтому правильное решение — **не выбирать между CAP и DCP**, а измерять их согласие и расхождение.

---

# Что уже существует в HSV и где остаётся место для нового метода

Просто заявить «физический HSV-дехейзинг» уже нельзя:

* в 2022 году был опубликован метод, выводящий transmission через saturation и brightness после нормализации на atmospheric light; ([IET Research][4])
* RSVT использует наблюдение о малом изменении hue и перемещении hazy/clear точек в плоскости saturation–value; ([arXiv][5])
* в CVPR 2026 опубликована Physics-Guided HSV Decomposition Network, специально направленная на разделение цветовых компонентов и снижение chromatic distortion; ([openaccess.thecvf.com][6])
* Haze-Lines уже рассматривает геометрию цветов относительно atmospheric light в RGB. ([openaccess.thecvf.com][7])

Поэтому новизна должна звучать не как «мы применили HSV», а так:

> **Мы получили точное представление scalar atmospheric model в airlight-normalized HCV, вывели два uncertainty-aware recovery gain и точную выпуклую область допустимых gains, гарантирующую RGB-gamut без post-hoc clipping.**

В целевом поиске прямого аналога именно такой совместной конструкции я не обнаружил. Это пока кандидат на новизну, а не юридическая или библиографическая гарантия.

---

# Новая математика: airlight-normalized HCV

Работаем в **линейном RGB**:

[
I_c=tJ_c+(1-t)A_c.
]

Предположим (A_c>0). Нормализуем каждый канал на соответствующую компоненту atmospheric light:

[
X_c=\frac{I_c}{A_c},
\qquad
Y_c=\frac{J_c}{A_c}.
]

Получаем:

[
X_c=tY_c+(1-t)
]

или:

[
\boxed{X=\mathbf 1+t(Y-\mathbf 1)}.
]

Это одна и та же положительная affine transform для трёх каналов. Она сохраняет порядок каналов.

Определим:

[
V(z)=\max_c z_c,
]

[
m(z)=\min_c z_c,
]

[
C(z)=V(z)-m(z).
]

Тогда:

[
V_X=1+t(V_Y-1),
]

[
m_X=1+t(m_Y-1),
]

следовательно:

[
\boxed{V_X-1=t(V_Y-1)},
]

[
\boxed{C_X=tC_Y}.
]

Hue зависит от отношений межканальных разностей. Общий offset исчезает в разностях, а scale (t) сокращается:

[
\boxed{H_X=H_Y}
]

для пикселей вне серой оси.

Я численно проверил эти равенства на одном миллионе случайных наборов (A,J,t). Максимальные ошибки оказались порядка floating-point precision:

| Проверка                    | Максимальная ошибка |
| --------------------------- | ------------------: |
| (V_X-1=t(V_Y-1))            | (1.33\cdot10^{-15}) |
| (C_X=tC_Y)                  | (1.78\cdot10^{-15}) |
| circular hue equality       | (2.33\cdot10^{-13}) |
| обратное RGB-восстановление | (1.84\cdot10^{-14}) |

Это не результат обучения и не статистический prior — это прямое следствие atmospheric model.

---

## Почему именно HCV, а не стандартный HSV

В HSV:

[
S=\frac{C}{V}.
]

Поэтому:

[
S_X=
\frac{tC_Y}{1+t(V_Y-1)},
]

и в общем случае:

[
S_X\neq tS_Y.
]

Именно из-за этого наивное восстановление:

[
S_J=\frac{S_I}{t}
]

математически неправильно.

Но при известном (t) стандартный HSV тоже можно восстановить точно через HCV:

[
V_Y=1+\frac{V_X-1}{t},
]

[
C_Y=\frac{S_XV_X}{t},
]

[
S_Y=\frac{S_XV_X}{t+V_X-1},
]

[
H_Y=H_X.
]

Вычислительно устойчивее хранить (H,C,V), а (S=C/V) рассчитывать только на выходе.

---

# HCV-A²CR: два коэффициента вместо одного (1/t)

Классическая инверсия использует один и тот же gain для яркости, цвета, полезного сигнала и шума:

[
\frac1t.
]

В HCV вводятся два коэффициента:

[
\boxed{
\hat V_Y=1+g_V(V_X-1)
}
]

и

[
\boxed{
\hat C_Y=g_CC_X
}
]

при:

[
\hat H_Y=H_X.
]

После этого HCV преобразуется обратно в нормализованный RGB:

[
\hat J=A\odot \hat Y.
]

Классическая физическая инверсия остаётся частным случаем:

[
g_V=g_C=\frac1t.
]

А безопасный fallback:

[
g_V=g_C=1
]

возвращает исходное изображение:

[
\hat J=I.
]

Это означает, что алгоритм не обязан агрессивно восстанавливать информацию, которой по факту нет.

---

## Аналитические gains из риска

Для координаты (k\in{V,C}) вводим:

[
R_k(g)=
S_k\left[(g\bar t-1)^2+g^2\sigma_t^2\right]
+N_kg^2
+U_k(g-1)^2,
]

где:

* (S_k) — мощность полезного сигнала;
* (N_k) — мощность шума;
* (\bar t) — итоговая transmission;
* (\sigma_t^2) — её неопределённость;
* (U_k) — неопределённость atmospheric light и координатной модели;
* (g=1) — identity fallback.

Минимум получается в закрытой форме:

[
\boxed{
g_k^*=
\frac{S_k\bar t+U_k}
{S_k(\bar t^2+\sigma_t^2)+N_k+U_k}
}.
]

После этого:

[
g_k=
\operatorname{clip}
\left(
g_k^*,
1,
\frac1{t_{\min}}
\right).
]

Получается нужное поведение:

* точная карта (t), сильный сигнал и слабый шум → (g\approx1/t);
* сильное расхождение priors → (g\to1);
* плотная дымка и цветовой шум → (g_C\to1);
* надёжный цветной объект → (g_C) растёт;
* value и chroma восстанавливаются независимо.

Это заменяет ручной `chromaFloor`.

---

## Защита hue возле серой оси

Hue плохо определён, когда (C\approx0). Вводится confidence:

[
q_H=
\frac{C_X^2}
{C_X^2+N_C+\lambda_H\sigma_t^2+\varepsilon}.
]

Тогда:

[
\boxed{
g_C^{eff}=1+q_H(g_C-1)
}.
]

В цветной области (q_H\approx1). В сером шумном тумане (q_H\approx0), поэтому метод не создаёт цветные пятна.

---

# Точная защита gamut без clipping

При фиксированном hue каждый нормализованный RGB-канал можно записать:

[
Y_c=V_Y-\kappa_c(H)C_Y,
\qquad
0\leq\kappa_c\leq1.
]

Тогда:

[
\hat J_c=
A_c
\left[
1+g_V(V_X-1)-\kappa_cg_CC_X
\right].
]

Требование:

[
0\leq\hat J_c\leq1
]

даёт шесть линейных ограничений относительно двух неизвестных (g_V,g_C):

[
0\leq
1+g_V(V_X-1)-\kappa_cg_CC_X
\leq\frac1{A_c}.
]

Их пересечение — выпуклый многоугольник.

Важно:

[
(g_V,g_C)=(1,1)
]

всегда допустима, потому что возвращает входной пиксель. Следовательно, feasible set никогда не пуст.

Предложенный risk-optimal gain можно спроецировать на этот многоугольник:

[
\min_{(g_V,g_C)\in\mathcal C_x}
w_V(g_V-g_V^*)^2+
w_C(g_C-g_C^*)^2.
]

Это значительно лучше `Clamp(0,1)`, потому что:

* ограничения учитываются до формирования результата;
* hue не меняется случайно из-за поканального clipping;
* можно измерить расстояние проекции;
* identity остаётся гарантированным fallback;
* возникает формально доказуемое свойство метода.

Именно эта часть выглядит одним из наиболее сильных кандидатов на научную новизну.

---

# Как переделать fusion CAP и DCP

Текущий вариант:

[
t=\min(t_{\text{CAP}},t_{\text{DCP}})
]

слишком агрессивен.

Лучше перевести каждую оценку в optical depth:

[
D_i=-\ln\max(t_i,\varepsilon).
]

Затем вычислить robust center:

[
\bar D=
\operatorname{weighted,median}_i D_i
]

и расхождение:

[
\sigma_D=
1.4826\operatorname{weighted,MAD}_i
(D_i-\bar D).
]

После этого:

[
\bar t=e^{-\bar D},
]

[
\sigma_t^2\approx\bar t^2\sigma_D^2.
]

В ensemble должны входить разные семейства:

* DCP;
* CAP-HSV;
* saturation/brightness или RSVT;
* Haze-Lines;
* optional local estimator.

Тогда конфликт между CAP и DCP становится не ошибкой выбора, а полезной картой uncertainty.

---

# Чем заменить Laplacian/Gaussian pyramid

Просто заменить её обычным DWT недостаточно: wavelet-based haze-lines and denoising уже опубликован. ([IEEE Xplore][8])

## Вариант 1 — UTAW на stationary à trous wavelets

Это лучший первый вариант по соотношению сложности, качества и новизны.

À trous не делает downsampling: все полосы сохраняются в полном разрешении. Это упрощает точное пиксельное согласование с (t(x)), uncertainty и цветовой confidence. Само полноразмерное устройство à trous является известным свойством transform. ([ScienceDirect][9])

Используем B3-spline filter:

[
h=\frac1{16}[1,4,6,4,1].
]

Разложение:

[
c_0=L,
]

[
c_{l+1}=h_l*c_l,
]

[
w_l=c_l-c_{l+1}.
]

Все (w_l) имеют исходное разрешение.

Для каждой полосы:

[
P_l=\max(E[w_l^2]-N_l,0)
]

— оценка полезной мощности.

Модель выходной неопределённости:

[
Q_l=
N_l^{out}
+\kappa_l\sigma_D^2
+\xi_lU_A.
]

Reliability:

[
\boxed{
r_l=
\frac{P_l}
{P_l+Q_l+\varepsilon}
}.
]

При максимальном допустимом усилении (b_l):

[
\boxed{
G_l=1+(b_l-1)r_l
}.
]

И:

[
\hat w_l=G_lw_l.
]

Таким образом:

* в плотной и неопределённой дымке (r_l\to0);
* на реальном надёжном контуре (r_l\to1);
* крупные полосы обычно усиливаются сильнее не потому, что так вручную задано, а потому что у них выше signal-to-noise ratio;
* параметры `tLo/tHi` исчезают;
* `gFine/gMid/gCoarse` можно заменить band caps или вывести из noise budget.

Название:

> **UTAW — Uncertainty- and Transmission-Aware Stationary Wavelet Recovery.**

---

## Вариант 2 — DTCWT или complex steerable pyramid

Dual-Tree Complex Wavelet Transform обеспечивает approximate shift invariance и directional selectivity, которых нет у обычного decimated DWT. ([ScienceDirect][10])

Complex steerable pyramid даёт multiscale и multi-orientation representation и лучше описывает наклонные края, чем обычные separable пирамиды. ([cns.nyu.edu][11])

Здесь можно оценивать не только magnitude, но и phase coherence:

[
r_{l,o}=
\frac{|w_{l,o}|^2q_{\text{phase}}}
{|w_{l,o}|^2+
N_{l,o}^{out}+
\kappa_{l,o}\sigma_D^2+
\varepsilon}.
]

Настоящий контур обычно сохраняет согласованную phase между соседними масштабами. Шум — нет.

Это потенциально сильнее à trous, но значительно сложнее реализовать в C#/EmguCV.

---

## Вариант 3 — joint graph wavelets

Наиболее исследовательский вариант: построить graph, в котором связи зависят от:

* linear luminance;
* optical depth;
* hue/chroma;
* uncertainty.

Например:

[
w_{xy}=
\exp\left(
-\frac{|Y_x-Y_y|^2}{\sigma_Y^2}
-\frac{|D_x-D_y|^2}{\sigma_D^2}
-\frac{d_H(H_x,H_y)^2}{\sigma_H^2}
\right)
q_xq_y.
]

Graph wavelet bands тогда не пересекают сильные depth discontinuities и не смешивают разные объекты. Graph spectral image processing уже является развитым направлением, но в целевом поиске я не обнаружил прямого аналога training-free dehazing, где graph строился бы совместно по airlight-normalized HCV, optical depth и uncertainty. Это вывод по результатам поиска, а не доказательство отсутствия такого метода. ([arXiv][12])

---

## Сравнение вариантов

| Вариант                      | Практичность | Потенциал качества | Потенциал новизны |
| ---------------------------- | -----------: | -----------------: | ----------------: |
| À trous + UTAW risk          |          5/5 |                4/5 |             3.5/5 |
| DTCWT + phase confidence     |          3/5 |              4.5/5 |               4/5 |
| Complex steerable pyramid    |          3/5 |              4.5/5 |               4/5 |
| Joint graph wavelets         |          2/5 |                4/5 |             4.5/5 |
| TGV/variational              |        2.5/5 |                4/5 |             2.5/5 |
| Обычный Haar/DWT             |          4/5 |                3/5 |               1/5 |
| Текущий Laplacian smoothstep |          5/5 |                3/5 |             1.5/5 |

Для первой публикационной версии оптимален **à trous UTAW**.

---

# Как интегрировать Lab

Lab можно нормально использовать, просто не стоит выдавать его за точную физическую систему.

CIELAB нелинеен:

* RGB → XYZ — линейная часть;
* XYZ → Lab использует нелинейную кусочно-кубическую функцию;
* поэтому в общем случае:

[
Lab(I)\neq tLab(J)+(1-t)Lab(A).
]

Lab-dehazing и Lab color correction уже существуют; в 2026 году опубликован метод, сочетающий gamma-based linearization и CIELAB channel transfer. Поэтому сам переход в Lab не станет достаточным новым вкладом. ([ScienceDirect][13])

## Правильная роль Lab

После физического HCV-A²CR recovery:

1. преобразовать результат в **float Lab**, не в 8-bit Lab;
2. отдельно разложить (L^*), (a^*), (b^*);
3. применять более сильное UTAW-восстановление к (L^*);
4. к chroma применять gain, умноженный на (q_H);
5. ограничивать изменение цвета через локальный (\Delta E_{00});
6. преобразовать обратно и ещё раз проверить RGB-gamut.

То есть:

> HCV отвечает за физически согласованное восстановление, Lab — за перцептивный контроль результата.

## Tangent-Lab как отдельная ветка

Можно линеаризовать Lab около atmospheric light:

[
z(I)-z(A)\approx M_A(I-A),
]

где (M_A) — Jacobian преобразования linear RGB → Lab в точке (A).

Тогда можно вводить отдельные gains для tangent-(L^*) и tangent-chroma. Но это аппроксимация, точность которой ухудшается при удалении от (A). Я бы оставил её отдельной ablation-веткой, а не основным методом.

---

# Итоговый алгоритм

```text
sRGB input
    ↓
exact sRGB → linear RGB
    ↓
global/local atmospheric light A(x)
+ bootstrap uncertainty ΣA
    ↓
DCP + CAP-HSV + RSVT/saturation + Haze-Lines
    ↓
optical-depth robust fusion
D̄, σD, t̄, σt²
    ↓
X = I / A
    ↓
custom extended HCV:
H, C=max-min, V=max
    ↓
signal/noise/uncertainty maps
    ↓
closed-form gV and gC
+ hue confidence qH
    ↓
per-pixel convex gamut projection
    ↓
physical HCV-A²CR result
    ↓
UTAW stationary-wavelet reconstruction
    ↓
optional float-Lab perceptual correction
    ↓
linear RGB → sRGB
```

---

# Что реализовать в репозитории

Новые компоненты:

```text
Methods/ColorSpaceLinear.cs
Methods/AirlightNormalizedHcv.cs
Methods/TransmissionEnsemble.cs
Methods/OpticalDepthFusion.cs
Methods/A2crHcvRisk.cs
Methods/A2crHcvProjector.cs
Methods/StationaryAtrous.cs
Methods/UtawRecovery.cs
Methods/HcvA2crUtawMethod.cs
Methods/HcvA2crDiagnostics.cs
```

Минимальная первая версия:

1. linear RGB;
2. один CAP и один DCP estimator;
3. median/MAD fusion;
4. custom HCV conversion;
5. (g_V/g_C) по аналитической формуле;
6. быстрая лучевая gamut projection от ((1,1));
7. à trous decomposition только яркости;
8. диагностические карты.

После подтверждения результата:

* полный polygon projection;
* local (A(x));
* chroma bands;
* DTCWT/steerable variant;
* self-calibration priors.

---

# Что именно можно заявлять как новый метод

Наиболее защищаемый claim:

> **Training-free single-image dehazing with an exact airlight-normalized HCV representation, uncertainty-aware dual recovery gains, per-pixel convex gamut feasibility, and stationary-wavelet detail reconstruction governed by transmission and model uncertainty.**

Оценка компонентов:

| Компонент                         |                 Оценка новизны |
| --------------------------------- | -----------------------------: |
| HSV/CAP как таковой               |                            0/5 |
| Transmission-aware Laplacian      |                          1–2/5 |
| Обычная wavelet-замена            |                            1/5 |
| Airlight-normalized HCV relations |                  4/5, кандидат |
| Аналитические (g_V/g_C)           |                  4/5, кандидат |
| Convex gamut polygon              |              4–4.5/5, кандидат |
| UTAW reliability                  |                        3.5–4/5 |
| Полный HCV-A²CR-UTAW              |                        4–4.5/5 |
| Graph-HCV-A²CR                    | до 4.5/5, но высокая сложность |

Главный риск новизны находится не в формулах HCV — они выводятся достаточно естественно, — а в возможности существования близкой работы под другим названием. Поэтому перед arXiv понадобится расширенный citation chaining. Но уже сейчас это существенно более узкий, математически определённый и проверяемый вклад, чем текущие названия `Fractal HSV` или `Transmission-Scale Laplacian`.

# Подготовленные материалы

[Полное исследование HSV/HCV, Lab, Transmission-Aware и A²CR](sandbox:/mnt/data/SimpleDeHaze_HSV_Lab_Transmission_A2CR_research_ru.md)

[Матрица новизны и вариантов алгоритма](sandbox:/mnt/data/SimpleDeHaze_HCV_A2CR_novelty_matrix.csv)

[Воспроизводимая проверка HCV-равенств на случайных данных](sandbox:/mnt/data/verify_airlight_normalized_hcv.py)

Рациональный основной путь разработки: **HCV-A²CR сначала доказать отдельно от enhancement, затем добавить UTAW, а текущие CAP, DCP, TransScale и Fractal-HSV использовать как baseline и ablation-компоненты**.

[1]: https://arxiv.org/abs/2111.05700 "https://arxiv.org/abs/2111.05700"
[2]: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-cvi.2015.0451 "https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-cvi.2015.0451"
[3]: https://ueaeprints.uea.ac.uk/id/eprint/62621/ "https://ueaeprints.uea.ac.uk/id/eprint/62621/"
[4]: https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12396 "https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12396"
[5]: https://arxiv.org/abs/2403.12054 "https://arxiv.org/abs/2403.12054"
[6]: https://openaccess.thecvf.com/content/CVPR2026/html/Lyu_Disentanglement-wise_Image_Dehazing_through_Cross-Domain_Manifold_Consensus_CVPR_2026_paper.html "https://openaccess.thecvf.com/content/CVPR2026/html/Lyu_Disentanglement-wise_Image_Dehazing_through_Cross-Domain_Manifold_Consensus_CVPR_2026_paper.html"
[7]: https://openaccess.thecvf.com/content_cvpr_2016/html/Berman_Non-Local_Image_Dehazing_CVPR_2016_paper.html "https://openaccess.thecvf.com/content_cvpr_2016/html/Berman_Non-Local_Image_Dehazing_CVPR_2016_paper.html"
[8]: https://ieeexplore.ieee.org/document/9493242/ "https://ieeexplore.ieee.org/document/9493242/"
[9]: https://www.sciencedirect.com/science/article/pii/S1077201499902000 "https://www.sciencedirect.com/science/article/pii/S1077201499902000"
[10]: https://www.sciencedirect.com/science/article/pii/S1063520300903439 "https://www.sciencedirect.com/science/article/pii/S1063520300903439"
[11]: https://www.cns.nyu.edu/~eero/steerpyr/ "https://www.cns.nyu.edu/~eero/steerpyr/"
[12]: https://arxiv.org/abs/1801.04749 "https://arxiv.org/abs/1801.04749"
[13]: https://www.sciencedirect.com/science/article/pii/S026288562600212X "https://www.sciencedirect.com/science/article/pii/S026288562600212X"


# Научно-техническое исследование SimpleDeHaze

## HSV/HCV, CIELAB, Transmission-Aware и A²CR: как получить новый, проверяемый и практически работающий метод

**Дата аудита:** 1 августа 2026 года  
**Объект:** текущий публичный репозиторий `yellow444/SimpleDehaze` и переданный расширенный архив SimpleDeHaze  
**Статус документа:** проект исследовательской спецификации; утверждения о мировой новизне требуют расширенного библиографического поиска и экспериментальной проверки

---

## 1. Итог исследования

HSV и Lab не запрещены физической моделью дымки. Неверно только напрямую считать, что исходное уравнение атмосферного рассеяния сохраняет линейный вид в обычных нелинейных координатах HSV или CIELAB.

Для HSV обнаруживается более сильный путь. После:

1. декодирования sRGB в линейный RGB;
2. поканальной нормализации на атмосферный свет;
3. замены стандартной насыщенности `S` на абсолютную хрому `C = S·V = max(R,G,B)-min(R,G,B)`;

атмосферная модель получает **точное кусочно-линейное представление в координатах HCV**:

- оттенок `H` сохраняется;
- хрома `C` умножается на transmission `t`;
- смещение value относительно нормализованного atmospheric-light level `1` также умножается на `t`.

Это позволяет построить новый метод:

> **HCV-A²CR — Airlight-Normalized Hue–Chroma–Value Recovery with Uncertainty-Aware Dual Gains and Convex Gamut Constraints.**

К нему целесообразно добавить новый вариант Transmission-Aware обработки деталей:

> **UTAW — Uncertainty- and Transmission-Aware Stationary Wavelet recovery**, основанный не на decimated Gaussian/Laplacian pyramid и ручном `smoothstep(t)`, а на полноразмерном à trous-разложении и аналитической оценке восстановимости каждой полосы через signal power, шум, transmission и его неопределённость.

Полный рабочий кандидат:

> **HCV-A²CR-UTAW**.

Он существенно сильнее формулировок «HSV вместо RGB» или «wavelet вместо Laplacian», потому что новизна находится в совместной конструкции:

1. точные airlight-normalized HCV relations;
2. два аналитических recovery gain — для value-offset и chroma;
3. uncertainty transmission/airlight;
4. точная выпуклая область допустимых gains без post-hoc clipping;
5. transmission-aware stationary-wavelet reliability, выведенная из noise/uncertainty model.

CIELAB рекомендуется использовать как второй, перцептивный слой после физического восстановления. Отдельная исследовательская ветка — tangent-Lab A²CR, но она будет аппроксимационной, а не точной.

---

## 2. Что обнаружено в исходниках

### 2.1. Публичный `master`

В текущем публичном `master` присутствует `HsvCapMethod`, но HSV используется только для оценки глубины и transmission:

\[
d=\theta_0+\theta_1V+\theta_2S,
\qquad t=e^{-\beta d}.
\]

После этого изображение восстанавливается стандартно в BGR/RGB:

\[
J_c=\frac{I_c-A_c}{t}+A_c.
\]

Следовательно, существующий CAP-HSV — это **HSV prior для карты t**, а не HSV/HCV recovery.

В `DehazeCore.Normalize` публичной версии выполняется только деление байтов на 255. sRGB transfer function не инвертируется. `DehazeCore.Recover` разделяет среднее RGB-отклонение и остаточную хрому и использует разные floors, но всё происходит в gamma-кодированном RGB.

### 2.2. Расширенный переданный архив

Архив содержит более новую ветку разработки, отсутствующую в публичном `master`:

- `TransScaleLaplacianMethod.cs`;
- `FractalHsvMethod.cs`;
- `RfepDcpMethod.cs`;
- `LafTvMethod.cs`;
- несколько локальных haze/visibility методов;
- расширенный реестр примерно из 48 методов.

`TransScaleLaplacianMethod` делает следующее:

1. получает `t_CAP` и `t_DCP`;
2. сливает их через `min(t_CAP,t_DCP)`;
3. уточняет guided filter;
4. восстанавливает RGB с локальным `A(x)` и chroma-floor;
5. переводит результат в 8-bit Lab;
6. строит decimated Gaussian/Laplacian pyramid яркости;
7. усиливает полосы через ручной gate

\[
gate_l(x)=smoothstep(t_l(x);t_{lo},t_{hi})(1-s_l)+s_l;
\]

8. делает Lab enhancement, tone и color limiting.

Это хорошая инженерная композиция, но её научная слабость состоит в том, что:

- Laplacian/Gaussian multiscale dehazing уже опубликован;
- `min` двух priors наследует наиболее агрессивную ошибку любого из них;
- `smoothstep`, `tLo/tHi`, `gFine/gMid/gCoarse` являются ручными эвристиками;
- `PyrDown/PyrUp` вносит decimation и shift sensitivity;
- физика заканчивается до 8-bit Lab round-trip;
- transmission uncertainty не оценивается;
- утверждение об измеренном SNR не сопровождается noise model и воспроизводимым estimator.

### 2.3. Почему Fractal-HSV пока не является главным новым методом

`FractalRichness` использует только два окна:

\[
H=\frac{\ln(\sigma_{large}/\sigma_{small})}
        {\ln(k_{large}/k_{small})},
\qquad R=1-H.
\]

Это двухмасштабный roughness index, а не устойчивый estimator локальной fractal dimension. Более того, при локально постоянных `t` и `A`:

\[
I=tJ+(1-t)A
\quad\Rightarrow\quad
\sigma_r(I)=t\sigma_r(J).
\]

Поэтому отношение двух стандартных отклонений в идеальной локальной модели сокращает `t`:

\[
\frac{\sigma_{large}(I)}{\sigma_{small}(I)}=
\frac{\sigma_{large}(J)}{\sigma_{small}(J)}.
\]

Карту можно сохранить как **two-scale structure confidence**, но не следует делать из неё основной haze-density claim. В HCV-A²CR-UTAW её роль лучше заменить оценкой signal power и phase/orientation coherence в multiscale bands.

---

## 3. Где проходит граница известного

### 3.1. HSV и saturation/value уже использовались

Ближайшие направления:

- Color Attenuation Prior: линейная модель глубины по `V` и `S`;
- saturation/brightness-based transmission после нормализации на atmospheric light;
- RSVT: малое изменение hue и перемещение hazy/clear точек в S–V plane;
- physics-guided HSV decomposition в CVPR 2026;
- методы коррекции airlight/color cast через saturation и Lab.

Поэтому незащищаемые claims:

- «первый HSV dehazing»;
- «первое сохранение hue»;
- «первая оценка transmission через saturation/value»;
- «первое объединение RGB и HSV».

### 3.2. Multiscale и wavelets уже использовались

Опубликованы:

- model-driven dehazing с Laplacian/Gaussian pyramids и различной обработкой haze/noise на уровнях;
- wavelet-based haze-lines and denoising;
- многочисленные DWT/wavelet neural methods;
- variational dehazing/denoising с TV/TGV.

Поэтому простая замена Laplacian pyramid на Haar/DWT не создаёт нового метода.

### 3.3. CIELAB сам по себе тоже не новый вклад

Lab применяется для статистической color correction, white balance, enhancement и no-reference dehazing. Свежая работа 2026 года уже сочетает gamma-based linearization и CIELAB channel transfer.

Следовательно, научный вклад должен быть не в названии пространства, а в **новой математике восстановления и её проверяемых свойствах**.

---

## 4. Почему CAP-HSV визуально выигрывает у DCP на простых изображениях

Это ожидаемо и не является ошибкой наблюдения.

DCP предполагает, что в локальном patch чистой сцены хотя бы один цветовой канал имеет почти нулевое значение. Предположение часто ломается на:

- небе;
- белых стенах и автомобилях;
- светлых малонасыщенных объектах;
- источниках света;
- равномерных поверхностях;
- синтетических или минималистичных изображениях.

CAP использует другую корреляцию: дымка обычно повышает `V` и снижает `S`. На простом изображении с ясным разделением объекта и фона эта связь может оказаться значительно ближе к сцене, чем dark-channel assumption. Карта глубины получается более гладкой и меньше путает светлое небо с локальными тёмными каналами.

Однако CAP также может ошибаться: естественный белый или серый объект имеет высокое `V` и низкое `S`, хотя может быть расположен близко. Поэтому правильный вывод не «HSV всегда лучше», а:

> DCP и CAP наблюдают разные признаки сцены; их расхождение следует превращать в uncertainty, а не выбирать `min` или один prior навсегда.

---

## 5. Точная интеграция HSV через airlight-normalized HCV

### 5.1. Начальная физическая модель

В линейном RGB:

\[
I_c(x)=t(x)J_c(x)+(1-t(x))A_c(x)+n_c(x).
\]

Сначала рассмотрим глобальный или уже оценённый локальный `A(x)`, положительные компоненты `A_c>0` и общий для каналов scalar transmission.

Определим поканально нормализованные координаты:

\[
X_c=\frac{I_c}{A_c},
\qquad
Y_c=\frac{J_c}{A_c}.
\]

Без шума:

\[
\boxed{X=\mathbf 1+t(Y-\mathbf 1)}.
\]

То есть ко всем трём каналам применяется одна и та же положительная affine transform:

\[
f_t(z)=1+t(z-1).
\]

Она сохраняет порядок каналов.

### 5.2. HCV вместо обычного HSV

Для произвольного RGB-вектора `z` определим:

\[
V(z)=\max_c z_c,
\qquad
m(z)=\min_c z_c,
\qquad
C(z)=V(z)-m(z).
\]

В обычном HSV:

\[
S=\frac{C}{V},
\qquad C=S\,V.
\]

Поскольку `max` и `min` коммутируют с общей положительной affine transform:

\[
V_X=1+t(V_Y-1),
\]

\[
m_X=1+t(m_Y-1),
\]

а значит:

\[
\boxed{V_X-1=t(V_Y-1)},
\]

\[
\boxed{C_X=tC_Y}.
\]

Hue определяется относительными разностями каналов, делёнными на `C`. Общий scale `t` сокращается, а общий offset `1-t` исчезает в разностях:

\[
\boxed{H_X=H_Y}, \qquad C_Y>0.
\]

Итак, после airlight normalization:

> **Hue инвариантен, chroma и value-offset ослабляются одним transmission.**

Это точное свойство исходной scalar atmospheric model, а не визуальная эвристика.

### 5.3. Почему нельзя просто делить обычную saturation на `t`

Обычная saturation:

\[
S_X=\frac{C_X}{V_X}
=
\frac{tC_Y}{1+t(V_Y-1)}.
\]

Поэтому в общем случае:

\[
S_X\ne tS_Y.
\]

Если `t` известно, точное восстановление стандартных HSV-координат имеет вид:

\[
V_Y=1+\frac{V_X-1}{t},
\]

\[
C_Y=\frac{S_XV_X}{t},
\]

\[
S_Y=\frac{S_XV_X}{t+V_X-1},
\]

\[
H_Y=H_X.
\]

Последняя формула для `S_Y` может быть неустойчива при малом знаменателе. Поэтому численно лучше хранить `(H,C,V)`, а saturation вычислять лишь при необходимости.

### 5.4. Численная проверка

На одном миллионе случайных троек `A,J,t` прямое вычисление дало ошибки порядка machine precision:

| Равенство | максимальная абсолютная ошибка |
|---|---:|
| `V_X - 1 = t(V_Y - 1)` | `1.33×10⁻¹⁵` |
| `C_X = t C_Y` | `1.78×10⁻¹⁵` |
| circular hue equality | `2.33×10⁻¹³` |
| обратное RGB-восстановление | `1.84×10⁻¹⁴` |

Это не доказывает качество dehazing, но подтверждает алгебраическую корректность новой координатной записи.

---

## 6. Новый метод HCV-A²CR

### 6.1. Двухкоэффициентное восстановление

Классическая инверсия использует один gain `1/t` для всего RGB-сигнала. Вместо этого вводятся два recovery gain:

\[
\boxed{
\hat V_Y=1+g_V(V_X-1)
}
\]

\[
\boxed{
\hat C_Y=g_C C_X
}
\]

\[
\boxed{
\hat H_Y=H_X.
}
\]

Затем из `(H, C, V)` восстанавливается нормализованный RGB-вектор `\hat Y`, а итог:

\[
\hat J=A\odot \hat Y.
\]

Классическая физическая инверсия — частный случай:

\[
g_V=g_C=\frac1t.
\]

Безопасный identity fallback:

\[
g_V=g_C=1
\quad\Rightarrow\quad
\hat J=I.
\]

Разделение gains имеет физический смысл на этапе решения обратной задачи: value-offset и chroma могут иметь разные signal-to-noise ratios и разную чувствительность к ошибке `A`.

### 6.2. Аналитический uncertainty-aware gain

Для координаты `k∈{V,C}` зададим локальный quadratic risk:

\[
R_k(g)=
S_k\left[(g\bar t-1)^2+g^2\sigma_t^2\right]
+N_kg^2
+U_k(g-1)^2.
\]

Здесь:

- `S_k` — оценка мощности полезного сигнала;
- `N_k` — шумовая мощность соответствующей координаты;
- `\bar t` — robust transmission estimate;
- `\sigma_t^2` — uncertainty transmission;
- `U_k` — штраф за неопределённость atmospheric light и coordinate model;
- `g=1` — оставить наблюдение без агрессивной инверсии.

Минимум находится аналитически:

\[
\boxed{
g_k^*=
\frac{S_k\bar t+U_k}
{S_k(\bar t^2+\sigma_t^2)+N_k+U_k}
}.
\]

Практически:

\[
g_k=\operatorname{clip}
\left(g_k^*,1,\frac1{t_{min}}\right).
\]

Поведение:

- при точном `t` и отсутствии шума получается `1/t`;
- при высоком noise или disagreement gain приближается к identity;
- chroma может усиливаться слабее value-offset;
- ручной `chromaFloor` заменяется измеряемым risk.

### 6.3. Hue confidence

Hue неустойчив возле серой оси, где `C≈0`. Поэтому вводится:

\[
q_H=
\frac{C_X^2}
{C_X^2+N_C+\lambda_H\sigma_t^2+\varepsilon}.
\]

И effective chroma gain:

\[
\boxed{
g_C^{eff}=1+q_H(g_C-1)}.
\]

В насыщенной надёжной области `q_H≈1`; в сером шумном тумане `q_H≈0`, и метод не создаёт цвет из шума.

---

## 7. Точная convex gamut feasibility

Один из самых сильных кандидатов на научную новизну — возможность обеспечить RGB-gamut без жёсткого clipping.

При фиксированном hue каждый нормализованный канал HCV можно записать:

\[
Y_c=V_Y-\kappa_c(H)C_Y,
\qquad 0\le\kappa_c\le1.
\]

После dual-gain recovery:

\[
\hat J_c=
A_c\left[
1+g_V(V_X-1)-\kappa_c(H)g_CC_X
\right].
\]

Требование:

\[
0\le\hat J_c\le1
\]

даёт по два линейных ограничения для каждого канала, то есть шесть half-planes в плоскости `(g_V,g_C)`:

\[
0\le
1+g_V(V_X-1)-\kappa_cg_CC_X
\le\frac1{A_c}.
\]

Их пересечение — выпуклый многоугольник `\mathcal C_x`.

Точка `(1,1)` всегда допустима:

\[
\hat J(1,1)=I.
\]

Следовательно:

> Для любого входного пикселя в RGB cube допустимое множество gains непусто.

Можно проецировать risk-optimal point `(g_V^*,g_C^*)` на `\mathcal C_x` по weighted Euclidean metric:

\[
\min_{(g_V,g_C)\in\mathcal C_x}
w_V(g_V-g_V^*)^2+w_C(g_C-g_C^*)^2.
\]

Это маленькая 2D convex QP на пиксель. Быстрая версия — движение от `(1,1)` к предложенной точке до первого пересечения с RGB boundary.

В отличие от `Clip(0,1)`, такой recovery:

- не уничтожает информацию после вычисления;
- не меняет hue непредсказуемо;
- даёт карту projection strength;
- позволяет формально измерять, насколько агрессивный proposal был несовместим с наблюдаемым gamut.

---

## 8. Как получать transmission и uncertainty

### 8.1. Почему `min(t_CAP,t_DCP)` не подходит

Минимум означает: если хотя бы один prior ошибочно считает объект очень задымлённым, вся система принимает агрессивную оценку. Это особенно опасно на белых объектах для CAP и на небе/ярких поверхностях для DCP.

### 8.2. Fusion в optical-depth space

Использовать независимые семейства:

- canonical DCP;
- CAP-HSV;
- saturation/brightness или RSVT-type estimator;
- Haze-Lines;
- при необходимости local-airlight estimator.

Перевести их в optical depth:

\[
D_i=-\ln\max(t_i,\varepsilon).
\]

Robust center:

\[
\bar D=\operatorname{weighted\ median}_iD_i.
\]

Disagreement:

\[
\sigma_D=
1.4826\operatorname{weighted\ MAD}_i(D_i-\bar D).
\]

Возврат:

\[
\bar t=e^{-\bar D},
\qquad
\sigma_t^2\approx\bar t^2\sigma_D^2.
\]

Чтобы несколько похожих DCP-вариантов не создавали ложную уверенность, ensemble должен быть иерархическим: сначала uncertainty внутри семейства, затем между семействами.

### 8.3. Self-calibration prior

Дополнительный кандидат:

\[
F_s(I;D,A)=A+(I-A)e^{sD}
\]

— частичное удаление доли `s` optical depth. Если estimator согласован с моделью, то после частичного dehazing должен оставаться depth:

\[
E(F_s(I;D,A))\approx(1-s)D.
\]

Residual:

\[
r_i=
\operatorname{median}_{s\in\mathcal S}
\left|E_i(F_s(I;D_i,A))-(1-s)D_i\right|
\]

можно использовать для веса:

\[
w_i=e^{-r_i/\tau}.
\]

Это превращает ensemble из обычного голосования в model-consistency calibration.

---

## 9. Чем заменить Gaussian/Laplacian pyramid

### 9.1. Вариант A — stationary à trous wavelets: рекомендованный первый метод

#### Почему лучше текущей пирамиды

В à trous decomposition нет downsampling. Все subbands остаются полного разрешения, поэтому:

- transmission map не нужно `PyrDown`;
- меньше shift artifacts;
- проще согласовать band coefficient с pixelwise uncertainty;
- нет `PyrUp` mismatch на нечётных размерах;
- легко реализовать separable dilated convolution;
- полностью подходит CPU/GPU.

B3-spline filter:

\[
h=\frac1{16}[1,4,6,4,1].
\]

На уровне `l` между коэффициентами вставляются `2^l-1` нулей:

\[
c_0=L,
\qquad c_{l+1}=h_l*c_l,
\qquad w_l=c_l-c_{l+1}.
\]

Reconstruction:

\[
L=c_L+\sum_l w_l.
\]

Сам à trous не является новым. Новым должен быть transmission/uncertainty-derived band recovery.

### 9.2. UTAW: аналитический gain полосы

Для каждого `w_l` оценим локальную band power:

\[
P_l=\max\left(E[w_l^2]-N_l,0\right).
\]

После dehazing ожидаемый noise budget зависит от transmission:

\[
N_l^{out}\approx
N_l^{in}\,g_V^2
+\rho_lN_C^{in}\,g_C^2.
\]

Добавим model uncertainty:

\[
Q_l=
N_l^{out}
+\kappa_l\sigma_D^2
+\xi_lU_A.
\]

Reliability:

\[
\boxed{
r_l=\frac{P_l}{P_l+Q_l+\varepsilon}}.
\]

Пусть желаемый максимальный detail boost уровня — `b_l≥1`. Тогда:

\[
\boxed{
G_l=1+(b_l-1)r_l
}.
\]

И:

\[
\hat w_l=G_lw_l.
\]

Это заменяет ручной `smoothstep(t)`:

- плотная дымка, большой noise/uncertainty → `r_l≈0`, boost не применяется;
- надёжный edge/texture → `r_l≈1`, разрешается полный boost;
- coarse bands обычно имеют большую signal power и меньший relative noise, поэтому усиливаются естественно, без отдельного правила «крупные всегда»;
- параметры `tLo/tHi` больше не нужны.

Для chroma bands дополнительно:

\[
r_l^{C}\leftarrow r_l^{C}q_H,
\]

чтобы не усиливать случайный цвет возле серой оси.

### 9.3. Вариант B — Dual-Tree Complex Wavelet / complex steerable pyramid

Плюсы:

- approximate shift invariance;
- directional selectivity;
- magnitude/phase разделение;
- можно отличать устойчивый edge от isotropic noise;
- лучше для тонких наклонных контуров и силуэтов.

Новый t-aware coefficient reliability:

\[
r_{l,o}=
\frac{|w_{l,o}|^2\,q_{phase}}
{|w_{l,o}|^2+N_{l,o}^{out}+\kappa_{l,o}\sigma_D^2+\varepsilon}.
\]

`q_phase` можно получить через согласованность phase между соседними scales. Реальные контуры сохраняют coherent phase; случайный шум — нет.

Недостатки:

- сложнее реализация в C#/EmguCV;
- больше памяти;
- wavelet/complex pyramid сами по себе известны;
- потребуется собственная библиотека filters и тщательная boundary handling.

Это хороший `v2`, если à trous уже показал улучшение.

### 9.4. Вариант C — transmission/airlight graph wavelets: наиболее уникальный, но дорогой

Строится graph pixels/superpixels с весами:

\[
w_{xy}=
\exp\left(
-\frac{|Y_x-Y_y|^2}{\sigma_Y^2}
-\frac{|D_x-D_y|^2}{\sigma_D^2}
-\frac{d_H(H_x,H_y)^2}{\sigma_H^2}
\right)
q_xq_y.
\]

Graph Laplacian задаёт частоты, согласованные не только с геометрией изображения, но и с transmission discontinuities. Graph-wavelet bands можно вычислять Chebyshev polynomials без полной eigendecomposition.

Потенциальный claim:

> Multiscale recovery on a graph jointly induced by airlight-normalized color geometry, optical depth and uncertainty.

Преимущества:

- края разных глубин не смешиваются;
- не нужна rectangular scale pyramid;
- естественно работает с non-homogeneous haze;
- высокая потенциальная новизна.

Недостатки:

- высокая инженерная и вычислительная стоимость;
- сложнее reproducibility;
- необходимо доказать преимущество над guided/domain-transform/à trous.

### 9.5. Вариант D — TGV/variational decomposition

Можно совместно оптимизировать value, chroma и gains с TGV. Но TV/TGV dehazing уже имеет богатую литературу. Вклад возможен только через HCV convex constraints и uncertainty risk, а не через сам TGV.

### 9.6. Рейтинг вариантов

| Вариант | Практичность | Потенциал качества | Потенциал новизны | Рекомендация |
|---|---:|---:|---:|---|
| Stationary à trous + UTAW risk | 5/5 | 4/5 | 3.5/5 | Реализовать первым |
| DTCWT/steerable + phase confidence | 3/5 | 4.5/5 | 4/5 | Вторая статья/версия |
| Joint graph wavelets | 2/5 | 4/5 | 4.5/5 | Продвинутая ветка |
| TGV/variational | 2.5/5 | 4/5 | 2.5/5 | Только как solver/regularizer |
| Guided/domain-transform hierarchy | 5/5 | 3/5 | 1.5/5 | Инженерный baseline |
| Обычный Haar/DWT вместо Laplacian | 4/5 | 3/5 | 1/5 | Недостаточно для claim |

---

## 10. Как корректно интегрировать CIELAB

### 10.1. Почему direct Lab inversion не точна

CIELAB включает:

- линейное RGB→XYZ преобразование;
- нормализацию на reference white;
- кусочно-нелинейный cube-root transform;
- нелинейные `L*`, `a*`, `b*`.

Поэтому:

\[
Lab(I)\ne tLab(J)+(1-t)Lab(A)
\]

в общем случае.

### 10.2. Рекомендуемая роль Lab

После HCV-A²CR физического recovery:

1. перейти в **float Lab**, без 8-bit round-trip;
2. разложить `L*`, `a*`, `b*` stationary wavelets;
3. применять более сильный recovery к `L*` и меньший к chroma;
4. использовать `q_H`, `σ_D`, `U_A` для chroma reliability;
5. ограничивать perceptual displacement через локальный `ΔE00` budget;
6. вернуть linear RGB и проверить gamut projection.

Это не «физика в Lab», а **perceptually controlled refinement**.

### 10.3. Tangent-Lab A²CR — отдельный эксперимент

Можно линеаризовать mapping `f:RGB_lin→Lab` около airlight `A`:

\[
z(I)-z(A)\approx M_A(I-A),
\qquad M_A=J_f(A).
\]

В tangent coordinates можно применять отдельные gains для `L*` и chroma plane, затем инвертировать Jacobian. Но approximation error второго порядка растёт с удалением от `A`.

Этот вариант наиболее обоснован в плотной дымке, где наблюдаемый `I` близок к `A`, однако чистое `J` может быть далеко. Поэтому tangent-Lab следует сравнивать с точным HCV и linear-opponent A²CR, а не делать основным методом заранее.

---

## 11. Рекомендуемый полный алгоритм HCV-A²CR-UTAW

### Stage 1. Linear radiance

```text
8-bit sRGB
  → exact sRGB EOTF / LUT
  → linear BGR/RGB float
```

### Stage 2. Atmospheric light and uncertainty

- global `A`;
- optional local `A(x)`;
- bootstrap across patch sizes/estimators;
- covariance or robust spread `Σ_A`.

### Stage 3. Transmission ensemble

```text
DCP-linear
CAP-HSV/HCV
Saturation-brightness / RSVT
Haze-lines
(optional local estimator)
```

### Stage 4. Optical-depth fusion

```text
t_i → D_i=-log(t_i)
robust family center
between-family weighted median
MAD/bootstrap uncertainty
D̄, σ_D, t̄, σ_t²
```

### Stage 5. Airlight-normalized HCV

```text
X = I / A
H, C=max(X)-min(X), V=max(X)
```

### Stage 6. Risk maps and gains

```text
S_V, S_C
N_V, N_C
U_V, U_C from A uncertainty
closed-form g_V, g_C
hue confidence q_H
```

### Stage 7. Convex feasible recovery

```text
proposal (g_V,g_C)
→ project into per-pixel RGB-feasible polygon
→ reconstruct normalized Y(H,C,V)
→ J0 = A ⊙ Y
```

### Stage 8. UTAW details

```text
float luminance/L* and chroma
→ à trous full-resolution bands
→ signal/noise/uncertainty reliability r_l
→ adaptive coefficient gains
→ reconstruction
```

### Stage 9. Output

```text
final gamut check
linear RGB → sRGB OETF
metrics + diagnostic maps
```

Diagnostic outputs должны включать:

- `t̄`;
- `σ_D`;
- `g_V`;
- `g_C`;
- `q_H`;
- projection factor/distance;
- band reliability `r_l`;
- clipping that would have occurred under scalar inversion.

---

## 12. План интеграции в SimpleDeHaze

### 12.1. Новые файлы

```text
Methods/ColorSpaceLinear.cs
Methods/AirlightNormalizedHcv.cs
Methods/TransmissionEnsemble.cs
Methods/OpticalDepthFusion.cs
Methods/A2crHcvRisk.cs
Methods/A2crHcvProjector.cs
Methods/StationaryAtrous.cs
Methods/UtawRecovery.cs
Methods/HcvA2crUtawMethod.cs
Methods/HcvA2crDiagnostics.cs
```

### 12.2. API-компоненты

```csharp
public readonly record struct HcvPixel(float H, float C, float V);

public readonly record struct GainProposal(
    float ValueGain,
    float ChromaGain,
    float HueConfidence);

public sealed record TransmissionEstimate(
    Mat MeanTransmission,
    Mat OpticalDepth,
    Mat OpticalDepthSigma,
    IReadOnlyDictionary<string, Mat> Members);
```

### 12.3. Псевдокод метода

```csharp
public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
{
    using Mat iLin = LinearColor.DecodeSrgb(input.Mat);

    AirlightEstimate air = AirlightEstimator.Bootstrap(iLin, p);
    using TransmissionEstimate tr = TransmissionEnsemble.Estimate(iLin, air, p);

    using HcvPlanes hcv = AirlightNormalizedHcv.Forward(iLin, air.Mean);
    using RiskMaps risk = A2crHcvRisk.Estimate(iLin, hcv, tr, air, p);
    using GainMaps proposal = A2crHcvRisk.ClosedForm(risk, tr, p);
    using GainMaps feasible = A2crHcvProjector.Project(hcv, air.Mean, proposal);

    using Mat jLin0 = AirlightNormalizedHcv.Recover(hcv, air.Mean, feasible);
    using Mat jLin = UtawRecovery.Process(jLin0, tr, air, feasible, p);

    return LinearColor.EncodeSrgb(jLin);
}
```

### 12.4. Что изменить в существующем TransScale

До появления нового метода текущий `TransScaleLaplacianMethod` стоит превратить в baseline:

- заменить `min(tCap,tDcp)` на robust fusion;
- добавить `σ_D`;
- убрать ручные `tLo/tHi` из основной конфигурации;
- не использовать 8-bit Lab внутри pyramid;
- переименовать `FractalRichness` в `TwoScaleRoughnessConfidence`;
- сохранять raw diagnostic maps;
- явно отделить physical recovery от perceptual enhancement.

---

## 13. Экспериментальный протокол

### 13.1. Основные гипотезы

**H1.** Airlight-normalized HCV recovery уменьшает hue/chroma error на bright/low-saturation regions по сравнению с scalar RGB inversion.

**H2.** Dual uncertainty-aware gains уменьшают noise amplification и clipping в dense haze без потери контраста в надёжных областях.

**H3.** Convex gamut projection сохраняет цвет лучше, чем independent RGB clipping.

**H4.** UTAW снижает halos/shift artifacts и повышает detail recovery по сравнению с decimated Laplacian/Gaussian pyramid.

**H5.** Robust optical-depth ensemble превосходит `min(CAP,DCP)` на белых объектах, небе и non-homogeneous haze.

### 13.2. Baselines

| ID | Метод |
|---|---|
| B0 | Canonical DCP |
| B1 | CAP-HSV |
| B2 | Current TransScale Laplacian |
| B3 | Current Fractal-HSV |
| B4 | RGB A²CR |
| B5 | HCV exact scalar `1/t` |
| B6 | HCV-A²CR without gamut projection |
| B7 | HCV-A²CR full |
| B8 | HCV-A²CR + current Laplacian |
| B9 | HCV-A²CR + UTAW |
| B10 | HCV-A²CR + DTCWT/steerable, optional |

### 13.3. Данные

- O-HAZE;
- I-HAZE;
- Dense-Haze;
- NH-HAZE;
- RESIDE SOTS;
- synthetic RGB-D generation из DIODE;
- RTTS/Foggy Driving для real unpaired cases.

### 13.4. Controlled synthetic stress test

Генерировать:

\[
I=tJ+(1-t)A+n
\]

по сетке:

- `t`: 0.03–1.0;
- neutral, blue, yellow airlight;
- Gaussian и Poisson-Gaussian noise;
- controlled bias в `A`;
- controlled bias/variance в `t`;
- uniform и non-uniform haze;
- bright-white, sky-like, near-gray, saturated и textured patches.

### 13.5. Метрики

Full-reference:

- PSNR;
- SSIM;
- CIEDE2000;
- LPIPS;
- hue circular error;
- chroma relative error;
- luminance error.

Method-specific:

- RGB cube violation before projection;
- clipping fraction и clipping energy baseline;
- projection distance;
- noise amplification in flat areas;
- edge/halo overshoot;
- `t` error и calibration of `σ_t`;
- uncertainty-error correlation;
- runtime, RAM, VRAM.

### 13.6. Ablation

```text
sRGB-domain vs linear RGB
single A vs local A(x)
DCP vs CAP vs ensemble
min fusion vs median/MAD fusion
scalar gain vs dual gains
without/with hue confidence
without/with convex projection
Laplacian vs à trous
manual smoothstep vs risk-derived reliability
Lab 8-bit vs Lab float
```

Все общие параметры фиксируются на development split до открытия final test results.

---

## 14. Матрица возможных claims

| Claim | Статус после поиска | Риск |
|---|---|---:|
| «HSV используется для dehazing» | давно известно | максимальный |
| «Hue почти сохраняется при haze» | близко к RSVT и HSV literature | высокий |
| «Transmission-aware Laplacian pyramid» | близко к multiscale Laplacian/Gaussian dehazing | высокий |
| «Wavelet вместо Laplacian» | wavelet dehazing существует | высокий |
| «Lab color correction after dehazing» | существует | высокий |
| Airlight-normalized HCV exact relations | прямого совпадения в целевом поиске не найдено; saturation/brightness prior близок | средний |
| Dual risk-optimal gains в HCV | прямого аналога не найдено | умеренно низкий |
| Exact per-pixel convex gamut polygon for dehazing gains | прямого аналога не найдено | умеренно низкий |
| HCV-A²CR + uncertainty-aware stationary bands | близкие части существуют раздельно | умеренный |
| Joint optical-depth/color/uncertainty graph wavelets | прямого dehazing-аналога в целевом поиске не найдено | умеренно низкий, но реализация сложна |

Отрицательный поиск не является доказательством мировой новизны. Перед формальным claim нужны IEEE Xplore, Scopus/Web of Science, Google Scholar, Semantic Scholar, OpenAlex, Springer, ScienceDirect, ACM и citation chaining.

---

## 15. Какую версию разрабатывать

### Версия 1 — публикационно реалистичная

> **HCV-A²CR-UTAW**

Состав:

- linear RGB;
- global A + bootstrap uncertainty;
- 3–4 независимых t priors;
- optical-depth median/MAD;
- exact HCV recovery;
- risk-derived `gV/gC`;
- hue confidence;
- convex gamut projection;
- stationary à trous detail recovery.

Это сочетает хорошую реализуемость и конкретную математическую новизну.

### Версия 2 — усиленная

> **HCV-A²CR-DTCWT**

Добавить orientation/phase coherence для различения структурных границ и шума.

### Версия 3 — наиболее исследовательская

> **Graph-HCV-A²CR**

Graph строится по linear luminance, hue/chroma, optical depth и uncertainty; multiscale coefficients регулируются recoverability risk.

---

## 16. Заключение

Главная идея не состоит в том, чтобы выбрать RGB, HSV или Lab как «лучшее пространство».

Правильное разделение ролей:

- **linear RGB** — физическая модель и atmospheric light;
- **airlight-normalized HCV** — точные hue/chroma/value relations и новый dual-gain recovery;
- **optical depth** — fusion и regularization transmission;
- **float Lab** — perceptual refinement и color-error control после физического recovery;
- **stationary/complex/graph wavelets** — multiscale recoverability, но с gains из noise/uncertainty, а не с ручным `smoothstep`.

Поэтому наиболее сильный новый метод для SimpleDeHaze:

\[
\boxed{
\text{HCV-A²CR-UTAW}
}
\]

Его защищаемая формулировка:

> Training-free single-image dehazing with an exact airlight-normalized HCV representation, uncertainty-aware dual recovery gains, per-pixel convex gamut feasibility, and stationary-wavelet detail reconstruction governed by transmission and model uncertainty.

---

## 17. Ближайшая литература

1. K. He, J. Sun, X. Tang. *Single Image Haze Removal Using Dark Channel Prior*. IEEE TPAMI, 2011.
2. Q. Zhu, J. Mai, L. Shao. *A Fast Single Image Haze Removal Algorithm Using Color Attenuation Prior*. IEEE TIP, 2015. DOI: 10.1109/TIP.2015.2446191.
3. D. Berman, T. Treibitz, S. Avidan. *Non-Local Image Dehazing*. CVPR, 2016.
4. D. Hu et al. *Fast Outdoor Hazy Image Dehazing Based on Saturation and Brightness*. IET Image Processing, 2022. DOI: 10.1049/ipr2.12396.
5. L.-A. Tran, D.-C. Park. *Haze Removal via Regional Saturation-Value Translation and Soft Segmentation*. arXiv:2403.12054.
6. T. Lyu, M. Ju, K.-K. Ma. *Disentanglement-wise Image Dehazing through Cross-Domain Manifold Consensus*. CVPR, 2026.
7. Z. Li, H. Shu, C. Zheng. *Multi-Scale Single Image Dehazing Using Laplacian and Gaussian Pyramids*. arXiv:2111.05700.
8. *Single Image Dehazing Using Wavelet-Based Haze-Lines and Denoising*. IEEE, document 9493242.
9. N. Kingsbury. *Complex Wavelets for Shift Invariant Analysis and Filtering of Signals*. Applied and Computational Harmonic Analysis, 2001.
10. F. Fang, F. Li, T. Zeng. *Single Image Dehazing and Denoising: A Fast Variational Approach*. SIAM Journal on Imaging Sciences, 2014.
11. *CIELAB-based Color Channel Transfer with Gamma Correction for No-Reference Image Dehazing*. Image and Vision Computing, 2026. DOI: 10.1016/j.imavis.2026.106105.
