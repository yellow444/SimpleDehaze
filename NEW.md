> **Статус реализации на 30.07.2026.** Проверяемое ядро реализовано:
> `A2crRisk`, `A2crFeasibleProjector`, `AirlightBootstrap`, `OpticalDepthFusion`,
> `A2crRecovery`, `A2crTvRefiner`, `A2crMethod` и `A2crDiagnostics`. Добавлены B0–B9
> synthetic stress benchmark, диагностические карты и численные тесты. B9 теперь решает совместную
> выпуклую задачу из §6 методом Condat/Vũ: два gain-поля, uncertainty-risk, coupling, edge-aware TV
> и точная евклидова проекция на per-pixel RGB-feasible polygon. Лучевая проекция сохранена как
> быстрая B8/финальная safety-проверка. Self-calibration из §11 остаётся отдельным опциональным вкладом,
> который намеренно не смешивается с первой версией метода.
> Систематический targeted literature review выполнен и зафиксирован в
> `SimpleDeHaze/docs/research/literature-review-2026-07.md`: прямой аналог полной комбинации не
> найден, но это не доказательство мирового приоритета. Научная новизна остаётся проверяемой
> гипотезой до внешних baseline и независимого blind test.

Да. Здесь есть возможность сделать **не очередную вариацию DCP**, а новый центральный метод, вокруг которого остальные 48 алгоритмов станут baseline, источниками prior и абляциями.

# Основной кандидат: A²CR-Dehaze

Рабочее название:

> **A²CR-Dehaze: Airlight-Aligned, Uncertainty-Aware Convex Recovery for Training-Free Single-Image Dehazing**

Ключевая идея: не пытаться в сотый раз немного иначе оценить карту трансмиссии (t), а заменить **сам оператор восстановления изображения**.

Это хорошо соответствует исходной идее SimpleDeHaze: интерпретируемый non-ML-метод, не зависящий от обучающего датасета и пригодный для реализации в .NET.

---

## 1. Где находится настоящая новая идея

Обычное восстановление использует:

[
J=A+\frac{I-A}{t}.
]

Один и тот же коэффициент (1/t) усиливает одновременно:

* полезный контраст;
* цветовые различия;
* шум;
* ошибку карты трансмиссии;
* ошибку атмосферного света;
* JPEG-артефакты.

Особенно плохо это работает при (t\rightarrow0): коэффициент становится огромным, а затем результат приходится клиппировать в ([0,1]).

В SimpleDeHaze уже есть важный зачаток новой идеи: в `DehazeCore.Recover` средняя RGB-компонента и цветовой остаток восстанавливаются через разные floor. Сейчас это эвристика `chromaFloor`. Вместо неё можно построить строгий метод.

---

# 2. Новый airlight-aligned оператор восстановления

Работаем в **линейном RGB**, а не непосредственно в gamma-кодированном sRGB.

Пусть:

[
I=tJ+(1-t)A+n,
]

где:

* (I) — входное изображение;
* (J) — чистое изображение;
* (A) — atmospheric light;
* (t) — transmission;
* (n) — шум.

Определяем цветовое направление атмосферного света:

[
u=\frac{A}{|A|_2}.
]

Разность относительно атмосферного света:

[
d=I-A.
]

Теперь разбиваем её не на условную «яркость и цветность по серой оси», а относительно **реального цвета атмосферной завесы**:

[
p=uu^Td,
]

[
q=(I_3-uu^T)d.
]

Здесь:

* (p) — компонента вдоль цвета atmospheric light;
* (q) — двумерная цветовая компонента, ортогональная atmospheric light.

Новый оператор:

[
\boxed{
J=A+g_{\parallel}p+g_{\perp}q
}
]

или в матричной форме:

[
G=g_{\perp}I_3+
(g_{\parallel}-g_{\perp})uu^T,
]

[
J=A+G(I-A).
]

Это осесимметричный обратный оператор с двумя собственными значениями:

* (g_{\parallel}) управляет восстановлением контраста вдоль atmospheric-light axis;
* (g_{\perp}) управляет восстановлением цветовых отклонений.

Классическая инверсия полностью входит в метод как частный случай:

[
g_{\parallel}=g_{\perp}=\frac1t.
]

То есть мы не выбрасываем физическую модель. Мы строим её **регуляризованное обобщение**.

Важно: это не «две физические трансмиссии». Атмосферная модель остаётся скалярной. Два коэффициента появляются только в обратной задаче, потому что разные компоненты сигнала имеют разный SNR и разную чувствительность к ошибкам.

---

# 3. Коэффициенты не подбираются вручную

Самое сильное место метода — (g_{\parallel}) и (g_{\perp}) можно вывести аналитически из ожидаемой ошибки.

Пусть оценка atmospheric light содержит ошибку:

[
\hat A=A+e.
]

Для одной из компонент (k\in{\parallel,\perp}):

[
\hat J_k=\hat A_k+g_k(I_k-\hat A_k).
]

Ошибка восстановления:

[
\hat J_k-J_k
============

(g_kt-1)(J_k-A_k)
+
g_kn_k
+
(1-g_k)e_k.
]

Введём:

* (S_k) — локальную энергию полезного сигнала;
* (N_k) — дисперсию шума;
* (U_k) — неопределённость atmospheric light;
* (\bar t) — среднюю оценку transmission;
* (\sigma_t^2) — неопределённость transmission.

Получаем ожидаемый риск:

[
R_k(g)=
S_k\left[(g\bar t-1)^2+g^2\sigma_t^2\right]
+
N_kg^2
+
U_k(1-g)^2.
]

Минимум находится в закрытой форме:

[
\boxed{
g_k^*=
\frac{S_k\bar t+U_k}
{S_k(\bar t^2+\sigma_t^2)+N_k+U_k}
}
]

После этого:

[
1\leq g_k\leq\frac1{t_{\min}}.
]

## Что делает эта формула

Когда оценки точны и шума нет:

[
N_k=U_k=\sigma_t^2=0
\quad\Longrightarrow\quad
g_k=\frac1t.
]

То есть метод автоматически превращается в стандартную физическую инверсию.

Когда участок почти не содержит полезной цветовой информации, но содержит шум:

[
S_{\perp}\ll N_{\perp},
]

получается:

[
g_{\perp}\rightarrow1.
]

Цветовой шум практически не усиливается.

Когда atmospheric light или transmission ненадёжны:

[
U_k,\sigma_t^2\rightarrow\infty,
]

коэффициент также стремится к безопасному значению (1). Алгоритм не пытается агрессивно «восстановить» то, о чём у него нет информации.

Получается автоматический переход:

[
\text{полная физическая инверсия}
\longleftrightarrow
\text{осторожное восстановление}
\longleftrightarrow
\text{оставить вход без усиления}.
]

Без нейросети, таблицы ручных порогов и post-hoc исправлений.

---

# 4. Откуда брать неопределённость transmission

Здесь текущая библиотека методов становится преимуществом.

Нужно выбрать не 20 близких DCP-вариаций, а 3–5 **разных семейств**:

* DCP;
* Color Attenuation/HSV prior;
* Haze-lines;
* boundary-constrained estimator;
* локальный atmospheric-light метод.

Каждый даёт карту (t_i). Переводим её в optical depth:

[
D_i=-\ln\max(t_i,\varepsilon).
]

Центральная оценка:

[
\bar D=
\operatorname{weighted\ median}_iD_i.
]

Оценка расхождения:

[
\sigma_D=
1.4826,
\operatorname{weighted\ MAD}_i(D_i-\bar D).
]

Возвращаемся к transmission:

[
\bar t=e^{-\bar D},
]

[
\sigma_t^2\approx \bar t^2\sigma_D^2.
]

Таким образом, методы не просто голосуют за среднее значение. Их несогласие становится **картой неопределённости**.

Например:

* DCP и haze-lines согласны — можно сильнее восстанавливать;
* DCP считает небо густым туманом, а HSV и haze-lines не согласны — усиление автоматически уменьшается;
* все prior сильно расходятся возле источника света — алгоритм сохраняет вход, а не создаёт чёрную или перенасыщенную область.

Чтобы похожие DCP-варианты не создавали ложную уверенность, разброс нужно оценивать ещё и через bootstrap:

* разные размеры окон;
* разные способы оценки (A);
* разные параметры guided filter;
* разные масштабы изображения.

---

# 5. Точное отсутствие clipping

Вместо восстановления с последующим:

```csharp
Clamp(0, 1)
```

можно заранее ограничить коэффициенты.

Для каждого RGB-канала:

[
0
\leq
A_c+p_cg_{\parallel}+q_cg_{\perp}
\leq
1.
]

Для трёх каналов получается шесть линейных неравенств относительно двух неизвестных:

[
(g_{\parallel},g_{\perp}).
]

Они образуют выпуклый многоугольник допустимых коэффициентов.

## Почему допустимая область всегда существует

Точка:

[
g_{\parallel}=g_{\perp}=1
]

даёт:

[
A+p+q=A+(I-A)=I.
]

А входное изображение уже находится в ([0,1]^3).

Следовательно:

[
(1,1)\in C_x,
]

где (C_x) — допустимая область конкретного пикселя.

Это даёт простую теорему:

> Для любого входного RGB-пикселя допустимое множество коэффициентов A²CR непусто.

Метод всегда может безопасно отступить к исходному изображению.

---

## Быстрая версия без итерационного solver

Пусть аналитическая формула предложила:

[
g^*=(g_{\parallel}^*,g_{\perp}^*).
]

Двигаемся от безопасной точки ((1,1)):

[
g(\alpha)=1+\alpha(g^*-1),
\qquad 0\leq\alpha\leq1.
]

Изображение вдоль этого луча:

[
J(\alpha)=I+\alpha\Delta,
]

где:

[
\Delta=
(g_{\parallel}^*-1)p+
(g_{\perp}^*-1)q.
]

Для каждого канала:

[
\alpha_c=
\begin{cases}
\dfrac{1-I_c}{\Delta_c},&\Delta_c>0,[6pt]
\dfrac{I_c}{-\Delta_c},&\Delta_c<0,[6pt]
1,&\Delta_c=0.
\end{cases}
]

Финально:

[
\boxed{
\alpha^*=
\min(1,\alpha_R,\alpha_G,\alpha_B)
}
]

и:

[
g_{\mathrm{safe}}=
1+\alpha^*(g^*-1).
]

Это:

* (O(1)) на пиксель;
* не требует итераций;
* гарантирует RGB-valid результат;
* не использует clipping;
* хорошо переносится на CPU и GPU.

Строго говоря, это максимальный допустимый шаг по выбранному лучу, а не евклидова проекция на весь многоугольник. Но для первой реализации этого достаточно.

---

# 6. Полная выпуклая версия для статьи

Для двух карт коэффициентов можно решать глобальную задачу:

[
\begin{aligned}
\min_{g_{\parallel},g_{\perp}}
\quad&
\sum_x
R_{\parallel,x}(g_{\parallel,x})
+
2R_{\perp,x}(g_{\perp,x})
\
&+
\lambda_{\parallel}
TV_w(g_{\parallel})
+
\lambda_{\perp}
TV_w(g_{\perp})
\
&+
\mu
(g_{\parallel}-g_{\perp})^2
\
\text{при условии}\quad&
(g_{\parallel,x},g_{\perp,x})\in C_x.
\end{aligned}
]

Здесь:

* (TV_w) сохраняет границы объектов;
* коэффициент (\mu) возвращает решение к обычной скалярной инверсии там, где нет оснований разделять gains;
* (C_x) обеспечивает RGB-feasibility.

При фиксированных (A), (t) и картах неопределённости задача выпуклая. При положительной кривизне локальных рисков решение будет единственным.

Подходящие solver:

* Chambolle–Pock;
* projected ADMM;
* primal-dual hybrid gradient;
* покоординатная TV-оптимизация с двумерной polygon projection.

Это уже полноценный математический contribution, а не комбинация фильтров.

---

# 7. Почему это похоже на действительно новый метод

В ближайшей литературе есть отдельные части идеи, но они решают другие задачи.

**Boundary-Constrained Contextual Regularization** ограничивает скалярную карту transmission, а не строит двухкоэффициентный обратный оператор восстановления. ([Open Access CVF][1])

**Haze-lines** использует геометрию RGB-линий относительно atmospheric light для оценки transmission и distance, но затем всё равно восстанавливает изображение через обычную скалярную atmospheric inversion. ([Open Access CVF][2])

**Bounded Channel Difference Prior** использует межканальные различия для оценки локального скалярного (t), а не для uncertainty-aware matrix recovery. ([Open Access CVF][3])

Разделение структуры, яркости и цвета само по себе уже нельзя объявлять новым: существуют YCbCr-guided dehazing-сети и Physics-Guided HSV decomposition. ([AAAI Publications][4])

Поэтому защищаемая новизна должна звучать точно:

> **Airlight-aligned axisymmetric inverse operator с двумя analytically derived uncertainty-aware gains и точной выпуклой RGB-feasible областью.**

В выполненном поиске я не нашёл прямого аналога, который одновременно содержал бы:

1. разложение относительно направления atmospheric light;
2. два коэффициента регуляризованной инверсии;
3. аналитический вывод gains из риска;
4. transmission disagreement;
5. atmospheric-light uncertainty;
6. точную выпуклую gamut-feasibility;
7. training-free реализацию.

Это не юридическая гарантия мировой новизны, но уже **конкретный и достаточно узкий кандидат на научный claim**, который можно полноценно проверять.

Моя оценка потенциала:

| Свойство                                    |          Оценка |
| ------------------------------------------- | --------------: |
| Математическая новизна                      |             4/5 |
| Связь с текущим репозиторием                |             5/5 |
| Реализуемость без ML                        |             5/5 |
| Возможность доказательств                   |           4.5/5 |
| Потенциал для arXiv                         |             4/5 |
| Риск полного совпадения с известным методом | умеренно низкий |

---

# 8. Как встроить метод в SimpleDeHaze

## Предлагаемая структура

```text
Methods/
    LinearRgb.cs
    OpticalDepthFusion.cs
    AirlightBootstrap.cs
    A2crRisk.cs
    A2crFeasibleProjector.cs
    A2crMethod.cs
    A2crDiagnostics.cs
```

## Полный pipeline

```text
sRGB input
    ↓
linear RGB
    ↓
несколько оценок atmospheric light
    ↓
robust A + covariance ΣA
    ↓
DCP / CAP / Haze-lines / Boundary t maps
    ↓
fusion in optical-depth space
    ↓
t̄ + transmission uncertainty σt²
    ↓
airlight-aligned decomposition p, q
    ↓
signal/noise/uncertainty estimation
    ↓
closed-form g_parallel, g_perp
    ↓
RGB-feasible projection
    ↓
optional convex TV refinement
    ↓
linear RGB result
    ↓
sRGB output
```

Упрощённый псевдокод:

```csharp
Mat inputLinear = LinearRgb.Decode(inputSrgb);

AirlightEstimate airlight = EstimateAirlightBootstrap(inputLinear);
TransmissionEnsemble ensemble = EstimateIndependentTransmissions(
    inputLinear,
    airlight.Value);

OpticalDepthEstimate depth = FuseInOpticalDepth(ensemble);

AirlightComponents components = DecomposeAroundAirlight(
    inputLinear,
    airlight.Value);

RiskMaps risk = EstimateRiskMaps(
    components,
    depth,
    airlight.Covariance,
    noiseModel);

GainMaps gains = ComputeClosedFormGains(risk);
gains = ProjectToRgbFeasibleSet(inputLinear, airlight.Value, components, gains);

Mat resultLinear = Recover(airlight.Value, components, gains);
Mat resultSrgb = LinearRgb.Encode(resultLinear);
```

---

# 9. Что делать с уже существующими «новыми» методами

Их не нужно удалять. Нужно дать им более сильные роли.

## RFEP

Становится baseline:

> scalar feasible transmission/recovery constraint.

Сравниваем:

* обычный `tmin`;
* RFEP scalar bound;
* A²CR two-gain feasible recovery.

Так RFEP помогает доказать, что преимущество возникает именно из двухмерной геометрии recovery, а не просто из дополнительного ограничения.

## Fractal-Guided HSV

Его roughness map можно использовать как дополнительную оценку:

[
S_{\perp}
]

или confidence структурной информации.

Но основной claim будет не «дымка имеет фрактальную размерность», а:

> roughness confidence помогает отличать полезную высокочастотную структуру от шума.

## Transmission-Scale Laplacian

Его лучше оставить как дополнительную noise-aware multiscale recovery-ветку. Сам принцип многомасштабного dehazing через Laplacian/Gaussian pyramids уже исследован. ([arXiv][5])

Но A²CR может выдавать карту:

[
g_{\perp}(x),\quad
g_{\parallel}(x),\quad
\sigma_t(x),
]

по которой затем вычисляются не ручные smoothstep-gates, а статистически обоснованные gain для каждой полосы.

## Остальные методы

Они становятся:

* independent prior estimators;
* baseline;
* ablation components;
* ensemble members;
* failure-case detectors.

То есть вместо заявления «в репозитории 48 новых методов» получится гораздо более сильная конструкция:

> один новый recovery framework, совместимый с 48 transmission estimators и refinement pipelines.

---

# 10. Обязательный эксперимент

Первый эксперимент должен быть не «посмотрели на красивую картинку», а controlled stress test.

Генерируем:

[
I=tJ+(1-t)A+n
]

с известными:

* (J);
* (A);
* (t);
* noise realization;
* ошибкой (\hat A);
* ошибкой (\hat t).

Проверяем по сетке:

[
t\in{0.05,0.1,0.2,\ldots,1},
]

разные цвета (A):

* нейтральный;
* голубой;
* жёлтый;
* серый;

и разные виды шума:

* Gaussian;
* Poisson–Gaussian;
* JPEG;
* цветовой шум;
* локальная ошибка transmission.

Сравниваем:

| ID | Вариант                              |
| -- | ------------------------------------ |
| B0 | scalar (1/t)                         |
| B1 | scalar (1/\max(t,t_{\min}))          |
| B2 | текущий `chromaFloor`                |
| B3 | RFEP scalar constraint               |
| B4 | A²CR без uncertainty                 |
| B5 | A²CR + noise                         |
| B6 | A²CR + transmission uncertainty      |
| B7 | A²CR + atmospheric-light uncertainty |
| B8 | A²CR + RGB-feasibility               |
| B9 | A²CR + spatial TV                    |

Ключевые метрики:

* PSNR;
* SSIM;
* CIEDE2000;
* hue-angle error;
* chroma error;
* доля каналов за пределами ([0,1]) до clamp;
* величина необходимого clipping;
* усиление шума на плоских участках;
* ошибка в sky/dense-haze regions;
* распределение (g_{\parallel}-g_{\perp});
* распределение projection factor (\alpha);
* runtime;
* RAM/VRAM.

На реальных данных разумный минимальный набор:

* O-HAZE — 45 реальных outdoor-пар; ([arXiv][6])
* I-HAZE — 35 реальных indoor-пар; ([arXiv][7])
* Dense-Haze — 33 пары с плотной реальной дымкой; ([arXiv][8])
* CARLA-Haze — большой контролируемый synthetic benchmark с различными сценами и распределениями дымки. ([Open Access CVF][9])

---

# 11. Второй возможный contribution: self-calibration optical depth

После основного метода можно добавить ещё одну сильную идею.

Пусть:

[
D=-\ln t.
]

Частично удалим долю (s) optical depth:

[
F_s(I;D,A)=A+(I-A)e^{sD}.
]

Если (D) оценена правильно, после частичного dehazing должна остаться:

[
D_{\mathrm{remaining}}=(1-s)D.
]

Следовательно, любой estimator (E) должен удовлетворять:

[
E(F_s(I;D,A))
\approx
(1-s)D.
]

Можно измерять self-consistency каждого prior:

[
r_i(x)=
\operatorname{median}_{s\in S}
\left|
E_i(F_s(I;D_i,A))
-----------------

(1-s)D_i
\right|.
]

И строить веса:

[
w_i(x)=
\exp\left(-\frac{r_i(x)}{\tau}\right).
]

То есть transmission estimators сами проверяют, сохраняется ли их оценка при контролируемом частичном удалении дымки.

Это можно назвать:

> **Optical-Depth Semigroup Self-Calibration**

Тогда A²CR получает не просто разброс prior, а **самокалиброванные веса их надёжности**.

Я бы не смешивал это с первой реализацией, но для полноценной статьи это отличный второй contribution.

---

# 12. Как формулировать статью

## Название arXiv

> **A²CR-Dehaze: Airlight-Aligned Convex Recovery with Uncertainty-Aware Dual Gains for Training-Free Single-Image Dehazing**

## Три основных вклада

1. **Новый inverse operator**

[
G=g_{\perp}I+
(g_{\parallel}-g_{\perp})uu^T,
]

который включает классическую atmospheric inversion как частный случай.

2. **Аналитические uncertainty-aware gains**

[
g_k^*=
\frac{S_k\bar t+U_k}
{S_k(\bar t^2+\sigma_t^2)+N_k+U_k}.
]

3. **Точная выпуклая RGB-feasible область**

без post-hoc clipping и с доказанной непустотой.

## Безопасная формулировка claim

> We introduce a training-free recovery operator that replaces scalar atmospheric inversion with two airlight-aligned gains. The gains are obtained by minimizing an uncertainty-aware reconstruction risk and are constrained by an exact per-pixel convex RGB-feasibility set. Standard atmospheric inversion is recovered as a special case, while the identity solution is always feasible, preventing post-hoc clipping by construction.

Это уже звучит как метод, а не как «ещё один набор коэффициентов для DCP».

---

# 13. Какой получится материал для Habr

Заголовок:

> **Почему удаление дымки ломается на этапе восстановления: строим новый convex recovery operator на .NET**

Структура:

1. Почему ошибка не только в transmission map.
2. Как (1/t) усиливает одновременно детали и шум.
3. Почему текущий `chromaFloor` оказался подсказкой.
4. Airlight-aligned разложение.
5. Вывод формулы двух gains.
6. Геометрия допустимого RGB-многоугольника.
7. Реализация projection на C#.
8. Карты (g_{\parallel}), (g_{\perp}), uncertainty и (\alpha).
9. Controlled benchmark.
10. Где метод выигрывает и где отступает к исходному изображению.

---

# Итог

Самая сильная стратегия для репозитория:

> **Не создавать ещё один prior для (t), а создать новый класс обратных операторов восстановления.**

В A²CR есть всё, что требуется для отдельного метода:

* новая математическая конструкция;
* понятная связь с atmospheric model;
* аналитический вывод;
* выпуклая оптимизация;
* доказуемые свойства;
* безопасный fallback;
* отсутствие необходимости в обучении;
* естественная интеграция в существующий .NET-код;
* хорошие абляции;
* отдельные диагностические карты;
* возможность CPU/GPU-реализации.

[Спецификация A²CR/AACR-Dehaze с формулами, алгоритмом и планом экспериментов](sandbox:/mnt/data/SimpleDeHaze_AACR_new_method_spec_ru.md)

Следующий рациональный шаг — внедрить минимальное ядро из `LinearRgb`, `A2crRisk`, `A2crFeasibleProjector` и `A2crMethod`, сначала используя одну DCP-карту и искусственно задаваемые uncertainty/noise maps, чтобы отдельно проверить именно новый recovery operator.

[1]: https://openaccess.thecvf.com/content_iccv_2013/html/Meng_Efficient_Image_Dehazing_2013_ICCV_paper.html "https://openaccess.thecvf.com/content_iccv_2013/html/Meng_Efficient_Image_Dehazing_2013_ICCV_paper.html"
[2]: https://openaccess.thecvf.com/content_cvpr_2016/html/Berman_Non-Local_Image_Dehazing_CVPR_2016_paper.html "https://openaccess.thecvf.com/content_cvpr_2016/html/Berman_Non-Local_Image_Dehazing_CVPR_2016_paper.html"
[3]: https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Zhao_Single_Image_Dehazing_Using_Bounded_Channel_Difference_Prior_CVPRW_2021_paper.html "https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Zhao_Single_Image_Dehazing_Using_Bounded_Channel_Difference_Prior_CVPRW_2021_paper.html"
[4]: https://ojs.aaai.org/index.php/AAAI/article/view/32297 "https://ojs.aaai.org/index.php/AAAI/article/view/32297"
[5]: https://arxiv.org/abs/2111.05700 "https://arxiv.org/abs/2111.05700"
[6]: https://arxiv.org/abs/1804.05101 "https://arxiv.org/abs/1804.05101"
[7]: https://arxiv.org/abs/1804.05091 "https://arxiv.org/abs/1804.05091"
[8]: https://arxiv.org/abs/1904.02904 "https://arxiv.org/abs/1904.02904"
[9]: https://openaccess.thecvf.com/content/WACV2026W/WVAQ/html/Velesaca_CARLA-Haze_A_Synthetic_Benchmark_for_Outdoor_Image_Dehazing_WACVW_2026_paper.html "https://openaccess.thecvf.com/content/WACV2026W/WVAQ/html/Velesaca_CARLA-Haze_A_Synthetic_Benchmark_for_Outdoor_Image_Dehazing_WACVW_2026_paper.html"
