# C³R-HSV: cylindrical confidence-constrained recovery

**Статус:** отрицательный эксперимент, зарегистрирован в GUI и `--selftest`, но удалён из
curated/recommended-набора. В статье используется только как отрицательная абляция, а не как
самостоятельная SOTA-заявка.

**Код:** [`Methods/HsvC3rMethod.cs`](../../Methods/HsvC3rMethod.cs)

C³R-HSV — HSV-аналог идеи A²CR. Он не заменяет существующий Color Attenuation Prior
и не объявляет HSV физической моделью рассеяния. CAP-HSV в проекте оценивает карту
пропускания в HSV, после чего использует обычное поканальное RGB-восстановление. C³R-HSV
меняет именно **оператор восстановления**.

## Зачем не делить H, S и V независимо

Hue является углом:

\[
0^\circ \equiv 360^\circ.
\]

Поэтому линейная фильтрация `H` создаёт ложный разрыв около красного цвета. Кроме того, при
\(S\rightarrow0\) оттенок становится неопределённым. C³R-HSV использует непрерывные
цилиндрические координаты:

\[
C=SV,
\qquad
c_x=C\cos H,
\qquad
c_y=C\sin H,
\]

\[
z=(V,c_x,c_y).
\]

Здесь \(C\) — абсолютная chroma. Допустимая область стандартного HSV имеет вид

\[
0\leq \sqrt{c_x^2+c_y^2}\leq V\leq1.
\]

Это усечённый second-order cone.

## 1. Две независимые оценки transmission

Метод строит:

- \(t_{\mathrm{CAP}}\) из Color Attenuation Prior;
- \(t_{\mathrm{DCP}}\) из dark channel.

Слияние выполняется не непосредственно в \(t\), а в optical depth:

\[
D_i=-\ln t_i,
\]

\[
\bar D=wD_{\mathrm{CAP}}+(1-w)D_{\mathrm{DCP}},
\qquad
\bar t=e^{-\bar D}.
\]

Расхождение двух prior становится локальной оценкой неопределённости. Среднее зависит от
`capWeight`, но disagreement намеренно считается с равными prior probabilities:

\[
\sigma_D^2=\frac14(D_{\mathrm{CAP}}-D_{\mathrm{DCP}})^2,
\]

\[
\sigma_t^2\approx \bar t^2\sigma_D^2.
\]

Если CAP и DCP расходятся на небе, источнике света или белом объекте, recovery становится
осторожнее вместо агрессивного усиления одного ошибочного prior. Эта форма также не позволяет
автоподбору поставить `capWeight` в 0 или 1 и тем самым искусственно обнулить uncertainty.

## 2. Linear-light recovery

Коэффициенты CAP оцениваются в обычном HSV/sRGB, поскольку именно для такого пространства
они были выведены. DCP, atmospheric light и физическая RGB-инверсия вычисляются в linear RGB.
Для цилиндрического оператора linear RGB переводится:

```text
sRGB -> linear RGB -> HSV
```

Оператор восстановления применяется к этим linear-light значениям. После восстановления:

```text
HSV -> linear RGB -> sRGB
```

## 3. Airlight-aligned разложение chroma

Atmospheric light переводится в те же координаты:

\[
A_z=(V_A,c_{Ax},c_{Ay}).
\]

Если chroma атмосферного света достаточна, вводятся радиальное и тангенциальное направления:

\[
e_r=\frac{c_A}{\lVert c_A\rVert},
\qquad
e_\tau=(-e_{ry},e_{rx}).
\]

Остаток пикселя относительно \(A_z\) раскладывается на три компоненты:

\[
d_V=V-V_A,
\]

\[
d_r=(c-c_A)^Te_r,
\qquad
d_\tau=(c-c_A)^Te_\tau.
\]

Их смысл:

- \(d_V\) — яркостная компонента;
- \(d_r\) — изменение chroma вдоль цвета atmospheric light;
- \(d_\tau\) — компонент, сильнее всего связанный с поворотом hue.

Когда \(A\) почти серый, направление его hue не определено. Тогда радиальный и
тангенциальный gain усредняются и превращаются в один изотропный chroma-gain.

## 4. Три uncertainty-aware gain

Для каждой компоненты \(k\in\{V,r,\tau\}\) используется локальный риск:

\[
R_k(g)=
S_k\left[(g\bar t-1)^2+g^2\sigma_t^2\right]
+N_kg^2
+Q_k(g-1)^2,
\]

где наблюдаемая локальная энергия предварительно очищается от оценки шума и пересчитывается в
latent signal energy:

\[
S_k=\frac{\max(E_{\mathrm{obs},k}-N_k,0)}
{\bar t^2+\sigma_t^2+\varepsilon}.
\]

Остальные обозначения:

- \(S_k\) — локальная энергия компоненты;
- \(N_k\) — энергия высокочастотного шума;
- \(\sigma_t^2\) — расхождение transmission-prior;
- \(Q_k\) — дополнительный штраф, притягивающий решение к identity \(g=1\).

Минимум находится аналитически:

\[
g_k^*=
\frac{S_k\bar t+Q_k}
{S_k(\bar t^2+\sigma_t^2)+N_k+Q_k}.
\]

Далее:

\[
1\leq g_k\leq
\min\left(g_{\max},\frac1{\bar t}\right).
\]

Предельные случаи:

\[
N_k,Q_k,\sigma_t^2\rightarrow0
\quad\Rightarrow\quad
g_k\rightarrow\frac1{\bar t},
\]

\[
Q_k\rightarrow\infty
\quad\Rightarrow\quad
g_k\rightarrow1.
\]

То есть метод переходит от физической инверсии к исходному пикселю в зависимости от
достоверности информации.

Тангенциальная компонента получает более сильный штраф при малой chroma, поскольку в этой
области даже небольшой шум может создать большой случайный скачок hue.

## 5. Кандидат восстановления

\[
V^*=V_A+g_Vd_V,
\]

\[
c^*=c_A+g_rd_re_r+g_\tau d_\tau e_\tau.
\]

Этот кандидат может быть слишком ярким или насыщенным. Независимое clipping \(V\) и \(S\)
исказило бы направление восстановления, поэтому используется общая геометрическая проекция.

## 6. Confidence-dependent HSV-коридор

Для входного пикселя задаётся максимально допустимая saturation:

\[
S_{\max}
=
S_{\mathrm{in}}
+
(1-S_{\mathrm{in}})\rho,
\]

\[
\rho=
\rho_{\max}
(1-\bar t)
q_H
(1-q_U),
\]

где:

- \(q_H\) — доверие к hue, зависящее от входной chroma;
- \(q_U\) — нормированная uncertainty;
- \(\rho_{\max}\) — параметр `satRoom`.

Допустимое множество:

\[
K_x=
\left\{
(V,c):
0\leq V\leq1,\;
\lVert c\rVert_2\leq S_{\max}V
\right\}.
\]

Поскольку \(S_{\max}\geq S_{\mathrm{in}}\), входной пиксель всегда принадлежит \(K_x\).

Вместо независимого clamp рассматривается отрезок:

\[
z(\alpha)=z_{\mathrm{in}}+\alpha(z^*-z_{\mathrm{in}}),
\qquad
0\leq\alpha\leq1.
\]

Алгоритм находит максимальное допустимое \(\alpha\). Ограничение по \(V\) вычисляется
аналитически, а пересечение с cone уточняется двоичным поиском.

### Свойство непустоты

Точка \(\alpha=0\) равна входному пикселю и допустима. Поэтому решение существует для любого
входного HSV-пикселя, даже если transmission и atmospheric light ошибочны.

## 7. Hue trust region

После cone projection радиус chroma не меняется, но угол дополнительно ограничивается:

\[
|\Delta H|
\leq
H_{\max}
(1-\bar t)
q_H
q_A
(1-q_U),
\]

где \(q_A\) — доверие к hue atmospheric light.

При сером atmospheric light, слабой chroma или сильном расхождении prior метод не создаёт
новый оттенок. При достаточном доверии допускается ограниченная коррекция hue.

## Параметры

| Параметр | Назначение |
|---|---|
| `beta` | коэффициент CAP |
| `omega` | сила DCP |
| `capWeight` | вес CAP в optical-depth fusion |
| `patch` | радиус dark-channel |
| `rmin` | радиус min-фильтра CAP |
| `rguide`, `eps` | guided filter |
| `min` | нижний предел transmission |
| `energy` | окно оценки локальной энергии |
| `noise` | вес high-pass noise penalty |
| `unc` | сила возврата к identity при расхождении prior |
| `hueGuard` | защита слабой chroma |
| `hueFloor` | масштаб confidence функции hue |
| `hueMax` | максимальный hue shift |
| `satRoom` | максимальное расширение saturation corridor |
| `maxGain` | общий потолок gain |

## Чем метод отличается от CAP-HSV

| CAP-HSV | C³R-HSV |
|---|---|
| HSV используется для оценки глубины | HSV используется и для нового recovery |
| один transmission prior | CAP+DCP и карта расхождения |
| стандартная RGB-инверсия | три risk-derived gain |
| H формально не моделируется | H представлен через `cos/sin` |
| clipping после RGB recovery | допустимый HSV-коридор содержит input fallback |

## Граница заявления о новизне

Использование HSV само по себе не является новым:

- Regional Saturation-Value Translation работает в плоскости S-V и наблюдает малое изменение H;
- современные работы используют специальные HSV/HVI-разложения для контроля цветовых
  артефактов.

Проверяемый contribution C³R-HSV сформулирован уже:

> training-free cylindrical HSV recovery with three uncertainty-aware gains, circular hue
> handling, and a guaranteed feasible saturation/value corridor.

Однако круговое представление hue уже используется в HVI и CIM-D, поэтому само отображение
`(S cos H, S sin H, V)` не является достаточной самостоятельной novelty-заявкой. Более узкая
проверяемая гипотеза C³R — совместное применение трёх risk-derived gain, disagreement-aware
fallback и гарантированно непустого saturation/value corridor.

## Результат честного пилота

На O-HAZE использованы отдельные validation/test splits, одинаковый предел размера 192 px и
фиксированные параметры на test. Из пяти заранее заданных C³R-конфигураций лучшая validation
конфигурация `v2` отключила `noise`, `unc` и `hueGuard`. Следовательно, O-HAZE не поддержал
гипотезу о пользе этих трёх штрафов.

| Метод | PSNR ↑ | SSIM ↑ | CIEDE2000 ↓ | Clip % ↓ | Color × |
|---|---:|---:|---:|---:|---:|
| A²CR | **16.90** | **0.81** | **14.04** | 8.52 | 1.92 |
| HSV²CR | 16.61 | 0.78 | 14.30 | 0.57 | 1.78 |
| C³R default | 15.08 | 0.64 | 15.48 | **0.00** | **1.11** |
| C³R-v2 frozen | 15.84 | 0.67 | 14.72 | **0.00** | 1.28 |

C³R-v2 уступает A²CR по PSNR на 20/22, по SSIM на 21/22 и по CIEDE2000 на 16/22 test-изображений.
Его подтверждённая сильная сторона в этом пилоте — консервативное, допустимое восстановление без
value clipping, а не более высокое качество удаления дымки. Поэтому A²CR остаётся единственным
основным recovery contribution статьи, а C³R сохраняется только как отрицательная абляция и не
выносится в заголовок.

Ближайшие работы:

- Tran, Park, *Haze Removal via Regional Saturation-Value Translation and Soft Segmentation*,
  arXiv:2403.12054.
- Yan et al., *HVI: A New Color Space for Low-light Image Enhancement*, CVPR 2025.
- Lyu et al., *Disentanglement-wise Image Dehazing through Cross-Domain Manifold Consensus*,
  CVPR 2026.

## Точность физической интерпретации

Atmospheric scattering model линейна в RGB, но преобразование RGB→HSV нелинейно и кусочно
зависит от максимального канала. Поэтому cylindrical recovery нельзя называть точной
репараметризацией физической инверсии. Это ограниченный геометрический оператор в linear-light
HSV; A²CR остаётся физически точным recovery core в linear RGB.

## Обязательные абляции

1. Только CAP + обычная RGB-инверсия.
2. CAP+DCP fusion без uncertainty penalty.
3. Один общий gain для \(V,c_x,c_y\).
4. Три gain без cone projection.
5. Cone projection без saturation corridor: \(S_{\max}=1\).
6. Полный C³R-HSV.
7. Полный C³R-HSV без hue trust region.
8. Recovery в sRGB против recovery в linear RGB.

## Метрики

Помимо PSNR/SSIM/CIEDE2000 нужны метрики, связанные с заявленным вкладом:

- circular hue error только на пикселях с надёжной GT chroma;
- saturation error;
- доля пикселей, где \(\alpha<1\);
- среднее ограничение \(1-\alpha\);
- шум на плоских областях;
- ошибка отдельно для sky/non-sky;
- disagreement CAP/DCP против фактической ошибки результата;
- runtime и peak memory.

## Ожидаемые ограничения

- Две карты transmission ещё не дают статистически калиброванную uncertainty.
- Глобальный atmospheric light недостаточен для неоднородной подсветки.
- HSV не является перцептивно равномерным пространством.
- CAP-коэффициенты зависят от исходного статистического предположения.
- Цикл по пикселям и преобразования цветового пространства сейчас являются CPU baseline, а не
  оптимизированной реализацией.
- Параметры нельзя подбирать на финальном test split.

Следующий этап — провести факторную абляцию на одинаковых transmission/A/noise условиях,
калибровать disagreement против фактической ошибки и проверять C³R прежде всего на сценах с
цветным atmospheric light и насыщенными объектами. Отдельная статья оправдана только если эти
тесты покажут повторяемое преимущество по hue/chroma без потери PSNR/SSIM; текущие данные этого
не показывают.
