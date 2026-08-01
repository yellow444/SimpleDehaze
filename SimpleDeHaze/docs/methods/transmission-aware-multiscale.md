# Transmission-aware multiscale: Laplacian, Edge и UTAW

Статус: реализованное экспериментальное семейство. В GUI сохранены отдельными методами:

- старый [`TransScaleLaplacianMethod.cs`](../../Methods/TransScaleLaplacianMethod.cs);
- HSV Edge абляция [`TransmissionAwareHsvEdgeMethod.cs`](../../Methods/TransmissionAwareHsvEdgeMethod.cs);
- validation-selected CPU UTAW
  [`TransmissionAwareHsvUtawMethod.cs`](../../Methods/TransmissionAwareHsvUtawMethod.cs);
- математически эквивалентный hybrid CUDA UTAW
  [`TransmissionAwareHsvUtawGpuMethod.cs`](../../Methods/TransmissionAwareHsvUtawGpuMethod.cs).

В curated-набор входят старый Laplacian и CPU UTAW. Edge сохранён для сравнения, GPU — как
инженерный prototype. Это не означает универсального превосходства рекомендованных методов.

## Общая физическая часть

Все варианты используют один pipeline:

1. CAP и DCP дают две карты transmission;
2. берётся их консервативный минимум и выполняется guided refinement;
3. recovery использует локальное поле atmospheric light и chroma protection;
4. меняется только multiscale stage;
5. применяются одинаковые white balance, tone, vibrance и optional denoise.

Поэтому сравнение basis не смешивается с другой оценкой `t` или `A(x)`.

## Пространства и basis

| Параметр | Значение | Реализация |
|---|---:|---|
| `space` | 0 | float Lab-L |
| `space` | 1 | float HSV-V; H/S не фильтруются как линейные сигналы |
| `basis` | 0 | decimated Gaussian/Laplacian pyramid |
| `basis` | 1 | full-resolution Domain Transform residual bands |
| `basis` | 2 | stationary B3-spline à trous, CPU |
| `basis` | 3 | тот же HSV-V à trous stage, hybrid CUDA; без CUDA CPU fallback |

Структурные `space/basis` исключены из AutoTuner. Специализированные Edge и UTAW wrappers
показывают только параметры своего basis; неактивные координаты не расходуют search budget.

## Legacy Laplacian gate

Для полосы `l` строится smoothstep transmission gate, который ослабляет мелкую деталь при малом
`t` и стремится к единице на крупных масштабах. При наличии richness-карты фактический gate:

```text
sf = l / (levels - 2)
s  = smoothstep((t_l - tLo) / (tHi - tLo))
gate_l = max(s * (1 - sf) + sf, 0.8 * richness_l * sf)
contribution_l = gain_l * gate_l * band_l
```

Предыдущая документация ошибочно описывала richness как отдельный множитель. Код всегда
использовал максимум двух gates; формула выше соответствует реализации.

## UTAW reliability

UTAW использует undecimated B3-spline à trous: все полосы остаются в полном разрешении и не
получают block/mosaic artefacts от down/up-sampling. Для полосы `w_l`:

```text
U_l = uNoise² / max(t², eps)
      + uUnc * (1 + 0.35*l) * sigmaDepth²

P_l = max(boxMean(w_l²) - U_l, 0)
r_l = P_l / (P_l + U_l + eps) * transmissionGate(t,l)
delta_l = clip((gain_l - 1) * r_l * w_l, -uLimit, uLimit)
```

`sigmaDepth=|log(t_CAP)-log(t_DCP)|` — proxy расхождения priors в optical-depth, а не
калиброванная статистическая дисперсия. `uLimit` — amplitude cap каждой добавки, не строгий
суммарный energy budget.

Validation-selected U1 defaults:

```text
gFine=0, gMid=1.3, gCoarse=1.1,
uNoise=0.006, uUnc=2, uRadius=3, uLimit=0.025
```

Они выбраны на 94 validation-парах четырёх наборов до открытия frozen test. Разница с исходным
U0 практически мала; параметры не следует выдавать за универсальный optimum.

## Frozen test

91 test-пара O-/I-/Dense-/NH-HAZE, full profile, `maxdim=192`, commit `bd4660a`, `dirty=false`.

| Метод | PSNR ↑ | SSIM ↑ | DE00 ↓ | Clip % ↓ | Flat-noise × ↓ |
|---|---:|---:|---:|---:|---:|
| Lab-L Laplacian | 12.0503 | 0.3984 | 22.7291 | 15.7387 | 1.7180 |
| HSV Edge | 12.1169 | 0.4381 | 22.6803 | **13.0613** | **1.4849** |
| HSV UTAW U1 | **12.1703** | **0.4735** | **22.5740** | 14.7793 | 1.8724 |

UTAW против Laplacian: PSNR 68/91, SSIM 91/91, DE00 61/91. Против Edge: PSNR 82/91,
SSIM 86/91, DE00 83/91. Но UTAW лучше Edge по clipping только на 24/91 и по flat-noise на
2/91. Поэтому UTAW — structural-quality candidate с явным noise/clipping trade-off.

На O-HAZE test отдельно: 14.293/0.641/18.458 против Edge 14.233/0.599/18.557 и старого
14.079/0.550/18.656 по PSNR/SSIM/DE00.

Полные таблицы, validation protocol и scene 08:
[hcv-a2cr-utaw-study-2026-08.md](../research/hcv-a2cr-utaw-study-2026-08.md).

## Scene 08

U1 получил full SSIM 0.484 и ROI SSIM 0.534 против 0.423/0.461 у Laplacian и 0.395/0.465 у
Edge. Визуально UTAW убирает значительную часть мозаики Edge, но сохраняет мелкую фактуру.
Сцена уже использовалась при разработке; это не blind evidence и не отменяет авторского
предпочтения старого Transmission-aware без нового просмотра.

## CPU/GPU

CUDA-вариант переносит HSV conversion, sparse separable B3 filtering, box-power и coefficient
arithmetic. CAP/DCP, guided filter, local airlight, recovery и postprocessing остаются CPU.

На RTX 3080 CPU↔GPU max abs error `4.77e-7`. End-to-end timing при 800 px, warmup=2,
repeat=5:

| Метод | Mean ms ↓ | ms/MP ↓ |
|---|---:|---:|
| Laplacian | **338.2** | **681.9** |
| HSV Edge | 387.8 | 778.9 |
| UTAW CPU | 420.6 | 841.2 |
| UTAW hybrid CUDA | 428.6 | 859.1 |

Hybrid GPU в этом режиме на 1.9% медленнее CPU. GPU-метод оставлен для дальнейшей разработки,
но speedup claim отсутствует. На машине без CUDA wrapper использует CPU stage; наличие слова GPU
в имени само по себе не доказывает использование CUDA.

## AutoTuner

Quick search реально меняет `uNoise`, `uUnc`, `uLimit`; `uRadius` участвует в thorough.
Зафиксированные аудиты:

- quick: 100 unique, 0 failures, полное покрытие 9 координат;
- thorough: 160 unique, 0 failures, полное покрытие 25 координат;
- full-resolution verification не ухудшила objective.

Это исключает механическое залипание поиска. Научный подбор всё равно обязан идти на validation,
а не на test или одном удобном кадре.

## Граница новизны

Gaussian/Laplacian pyramids, Domain Transform, stationary wavelets и wavelet dehazing известны.
Нельзя заявлять новыми сам basis, HSV dehazing или GPU filtering. Проверяемая гипотеза проекта —
совместная transmission/noise/prior-disagreement reliability и bounded coefficient amplitude при
общей физической части. Даже она остаётся кандидатом до независимого literature review и внешнего
сравнения.
