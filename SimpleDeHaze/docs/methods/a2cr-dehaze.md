# A²CR-Dehaze: airlight-aligned dual-gain recovery

Статус: **экспериментальный метод реализован**, claim мировой новизны не установлен.

## Оператор

В линейном RGB берём `u=A/||A||`, `d=I-A` и раскладываем

$$p=uu^Td,\qquad q=(I_3-uu^T)d.$$

Восстановление выполняется двумя gains:

$$J=A+g_\parallel p+g_\perp q.$$

Для каждой компоненты минимизируется квадратичный риск

$$R(g)=S[(g\bar t-1)^2+g^2\sigma_t^2]+Ng^2+U(1-g)^2,$$

откуда

$$g^*=\frac{S\bar t+U}{S(\bar t^2+\sigma_t^2)+N+U},\qquad
1\le g\le1/t_{min}.$$

При нулевых uncertainty/noise оба gain точно переходят в `1/t`; это закреплено unit-тестом.

## Feasibility без содержательного clamp

Предложенные gains движутся от безопасной точки `(1,1)`:

$$g(\alpha)=1+\alpha(g^*-1),\quad J(\alpha)=I+\alpha\Delta.$$

Максимальный `α∈[0,1]` вычисляется по трём RGB-каналам в закрытой форме. Точка `(1,1)`
возвращает вход, поэтому множество вдоль луча всегда непусто. На 50 000 случайных пикселях
projector не дал ни одного выхода за RGB-куб.

Для joint-TV используется полный выпуклый многоугольник: шесть RGB-неравенств плюс
`1≤g_parallel,g_perp≤1/t_min`. Евклидова проекция в 2D вычисляется точно перебором допустимых
проекций на рёбра и их пересечений; кэшированный polygon используется внутри итерационного solver.
На 20 000 случайных пикселях эта проекция всегда допустима и никогда не дальше от предложения,
чем прежняя лучевая точка; ещё 5 000 тестов сверяют кэшированную и прямую реализации.

## Реализованный pipeline

1. Точная sRGB → linear RGB кривая из `ColorSpace`.
2. Bootstrap `A` по 1/3/5 вариантам patch/top; медиана и поканальная variance.
3. Ensemble из DCP bootstrap, CAP и упрощённой haze-line карты.
4. Weighted median в `D=-ln(t)` и `1.4826·weighted MAD`; затем `σt²≈t²σD²`.
5. Локальная энергия `S_parallel/S_perp`, аналитические gains и опциональная связь `μ`.
6. При `tv>0` — совместный Condat/Vũ primal-dual solver: два квадратичных риска, coupling,
   edge-aware isotropic TV и точный per-pixel polygon constraint.
7. Финальная ray-feasibility как независимый численный safety-net и возврат в sRGB.

Шаги solver выбраны из оценки Lipschitz-константы локального квадратичного риска и границы
`||∇||²≤8`. Из всех итераций возвращается лучшее допустимое решение по исходной convex objective,
поэтому целевая функция не может стать хуже допустимой инициализации. Это закреплено unit-тестом.

## Controlled stress benchmark

```powershell
dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release -- --a2cr-stress `
  --out=benchmark_results/a2cr-stress-full-joint-tv.csv
```

Grid: 11 значений `t`, четыре цвета `A`, Gaussian/Poisson–Gaussian/JPEG/color/local-t noise,
10 вариантов B0–B9; всего 2200 строк, ошибок нет. Средние диагностические результаты:

| Вариант | PSNR ↑ | SSIM ↑ | CIEDE2000 ↓ | invalid после |
|---|---:|---:|---:|---:|
| B0 scalar `1/t` | 21.53 | 0.600 | 15.07 | 13% |
| B2 chromaFloor | 23.26 | 0.620 | 11.52 | 2% |
| B3 scalar boundary | 22.39 | 0.650 | 14.35 | 0% |
| B4 A²CR без uncertainty | 22.12 | 0.630 | 14.51 | 6% |
| B5 + noise | 22.54 | 0.640 | 13.80 | 5% |
| B6 + transmission uncertainty | 23.57 | 0.670 | 12.96 | 2% |
| B7 + airlight uncertainty | 24.63 | 0.690 | 11.27 | 1% |
| B8 + RGB feasibility | 24.74 | 0.700 | 11.25 | **0%** |
| B9 + joint TV | **26.03** | **0.830** | **10.96** | **0%** |

Это controlled diagnostic, не доказательство качества на реальных датасетах.

## DIODE RGB-D controlled benchmark

Реализован отдельный streaming benchmark из `NEW2.md`: 500 кадров DIODE validation,
scene-level split 295/104/101, пять уровней transmission, три airlight и три noise recipes —
22 500 воспроизводимых условий. Синтез идёт в linear RGB, известны `t`, `A` и noise; CSV содержит
ошибки `t/A/g`, clipping, projection fraction, chroma-weighted hue, flat-noise residual, CPU/RAM
и настоящий LPIPS 0.1.4/AlexNet. Generated изображения не сохраняются и не раздувают диск.

Validation pilot: 10 кадров, 450 recipes, B0–B9, `maxdim=192`, 4 500 строк, ошибок нет:

| Вариант | PSNR ↑ | SSIM ↑ | CIEDE2000 ↓ | LPIPS ↓ | invalid после |
|---|---:|---:|---:|---:|---:|
| B0 scalar | 18.28 | 0.360 | 14.36 | 0.860 | 27% |
| B3 boundary | 19.25 | 0.450 | 16.32 | 0.790 | 0% |
| B7 uncertainty | 18.81 | 0.360 | **12.31** | 0.850 | 24% |
| B8 feasible | 20.15 | 0.430 | 13.57 | 0.800 | **0%** |
| B9 feasible + joint TV | **21.26** | **0.595** | 13.13 | **0.624** | **0%** |

B9 лучше B0 по SSIM/LPIPS в 450/450 recipes, по PSNR в 441/450, но по CIEDE только в
327/450. Поэтому корректный вывод — улучшение feasibility, структуры и perceptual distance,
а не безусловное улучшение цвета. Подробный протокол и определения метрик:
`docs/research/a2cr-data-protocol.md`.

Полный joint-TV grid на 500 кадрах (`maxdim=96`, B0/B3/B7/B8/B9) дал 112 500/112 500
корректных строк. B9: PSNR 21,748, SSIM 0,707, CIEDE 11,543, chroma error 11,372,
flat-noise 0,0202 и invalid-after 0. B7 сохранил лучший средний CIEDE 11,303 и chroma 8,968.
B9 выиграл у B0 по PSNR в 22 201/22 500, SSIM в 22 500, CIEDE в 18 439 и chroma error в
17 082 recipes. По сравнению с B8 он добавляет примерно 30,18 ms/кадр при `maxdim=96`.

Frame-level stratified bootstrap (10 000 повторов; 45 recipes сначала усредняются внутри каждого
из 500 кадров) для B9-B0: `+2,95 dB` PSNR, 95% CI `[2,90; 3,00]`; `+0,191` SSIM
`[0,187; 0,196]`; `-0,978` CIEDE `[-1,074; -0,883]`; `-1,299` chroma error
`[-1,443; -1,151]`. Для B9-B8 интервалы также не пересекают ноль по этим метрикам, но runtime
существенно хуже. Это поддерживает feasibility/structure/noise claim, а не universal-color claim.

![DIODE bootstrap effects](../research/figures/diode-bootstrap-effects.png)

## O-HAZE smoke

На фиксированном test split из 22 пар, `maxdim=800`, `core`, `repeat=1`:

| Метод | PSNR ↑ | SSIM ↑ | CIEDE2000 ↓ | flat noise × ↓ | ms |
|---|---:|---:|---:|---:|---:|
| Canonical DCP | 16.27 | 0.750 | 16.78 | 2.80 | 105 |
| RFEP | **17.13** | 0.760 | 14.74 | **1.89** | 255 |
| A²CR | 17.05 | **0.800** | **14.34** | 2.78 | 533 |

A²CR улучшил SSIM и цветовую ошибку, но не PSNR, скорость или flat-noise amplification.
Это smoke из dirty worktree, а не публикационный результат; default noise/risk estimation требует
дальнейшей калибровки на validation split.

## Четыре real-paired test-split, fixed defaults

Без подбора после просмотра результатов, `core`, `maxdim=800`, `repeat=1`:

| Dataset | A²CR PSNR / SSIM / CIEDE / LPIPS | Лучший другой LPIPS | Наблюдение |
|---|---|---|---|
| I-HAZE, 15 | 16.62 / **0.82** / 13.66 / 0.280 | RFEP 0.239 | A²CR clip/noise и perceptual хуже RFEP |
| O-HAZE, 22 | 17.05 / **0.80** / **14.34** / 0.331 | canonical 0.298 | близкий PSNR, но LPIPS/скорость/noise хуже |
| Dense-Haze, 27 | **12.48 / 0.48 / 22.11 / 0.692** | canonical 0.716 | A²CR лучший из четырёх по всем метрикам |
| NH-HAZE, 27 | **13.23 / 0.60** / 20.90 / 0.462 | canonical 0.452 | Haze-Lines CIEDE 20.85 немного лучше |

Это не blind test: реальные наборы уже использовались во время разработки. LPIPS — настоящий
`lpips==0.1.4`, AlexNet v0.1, CUDA; 364/364 строк успешны. Таблица подтверждает, что метод не
доминирует и его clipping/noise/perceptual stability вне Dense-Haze остаются открытой проблемой.

## Диагностические карты

```powershell
dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release -- --a2cr-diag `
  --image=SimpleDeHaze/dataset/09_outdoor_hazy.jpg --out=benchmark_results/a2cr-diag
```

Сохраняются result, `t`, `σt²`, `σD`, `g_parallel`, `g_perp`, `α` и JSON с оценками `A`,
параметрами, долей projected pixels и нарушениями до/после projector.

## Не закрыто

- внешний полнофункциональный BCCR baseline (B3 проверяет только boundary-компонент);
- официальный CARLA-Haze adapter и действительно слепой внешний real-data benchmark;
- optical-depth semigroup self-calibration;
- GPU-реализация.

Поиск ближайших работ и границы claim зафиксированы в
`docs/research/literature-review-2026-07.md`. До слепого внешнего теста безопасная формулировка —
«экспериментальный airlight-aligned dual-gain recovery framework», а не «первый в мире метод».
