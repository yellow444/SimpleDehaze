# PF-SFGF - Pyramid-Fused DCP with Spectral Gain and Fast Guided Filter

> Статус: **реализовано** - `PF-SFGF (pyramid + fast GF)`
> ([`PfSfgfMethod.cs`](../../Methods/PfSfgfMethod.cs)).

PF-SFGF - скоростной DCP-вариант из `TEMP.md`: несколько dark-channel масштабов,
доверительное слияние трансмиссии, fast guided refinement и ограниченная компенсация
средних частот после восстановления.

## Transmission Fusion

Считаем три DCP-карты:

$$
t_s(x)=1-\omega_s\,\operatorname{dark}_{r_s}(I/A),
\qquad r_s\in\{3,7,15\}.
$$

Малое окно лучше на границах и тонких деталях, большое - на гладких дальних областях.
В реализации вес строится из `flat = exp(-k|nabla Y|)` и `edge = 1-flat`:

$$
w_{small}=0.15+0.85\,edge,\qquad
w_{large}=0.15+0.85\,flat,
$$

$$
w_{mid}=0.20+2\,edge\,flat.
$$

Итоговая карта:

$$
\tilde t(x)=
\frac{w_1t_1+w_2t_2+w_3t_3}
{w_1+w_2+w_3+\varepsilon}.
$$

После этого bright/sky mask поднимает `t` к `t_sky`, чтобы DCP-пирамида не затемняла
небо и белые области.

## Fast Guided Filter

Финальное уточнение `t` использует общий helper:
[`Refiners.FastGuided`](../../Methods/Refiners.cs). Он запускает guided filter на уменьшенной
копии и апсемплит результат:

```text
guide_s = downsample(guide, s)
t_s     = downsample(t, s)
q_s     = GuidedFilter(guide_s, t_s, r/s, eps)
q       = upsample(q_s)
```

Это дешевле полного `GuidedFilter` на больших кадрах и удобно для preview/video режимов.

## Spectral Gain

После chroma-safe recovery получается `J0`. Дымка часто съедает полезные средние частоты,
но бездумный sharpening даёт ringing и шум. Поэтому в реализации используется ограниченный
band-pass через difference-of-Gaussians:

$$
B = G_{\sigma_1}(J_0)-G_{\sigma_2}(J_0),
\qquad
J = J_0 + \alpha(1-\bar t)B.
$$

По умолчанию `sigma1=1.2`, `sigma2=12`, а итоговый множитель ограничен сверху (`<=0.18`).
Это приближение к FFT mid-band gain из `TEMP.md`, но без тяжёлого частотного блока и с меньшим
риском ringing.

## Параметры по умолчанию

| Параметр | Значение | Смысл |
|---|---:|---|
| `omega` | `0.95` | базовая сила DCP |
| `r1/r2/r3` | `3/7/15` | радиусы DCP-пирамиды |
| `tsky` | `0.66` | мягкость bright/sky-зон |
| `min` | `0.08` | нижний порог яркостной трансмиссии |
| `chroma` | `0.35` | нижний порог хромы |
| `refine` | `48` | радиус fast guided refinement |
| `fast` | `4` | downsample-фактор fast guided filter |
| `gain` | `0.12` | сила mid-frequency compensation |
| `color` | `1.25` | потолок усиления цветности |

## Абляции

Для статьи удобно сравнивать:

- single DCP radius;
- `DCP - Multi-Scale Fusion` без fast GF и spectral gain;
- `PF-SFGF` с `gain=0`;
- полный `PF-SFGF`.
