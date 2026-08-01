# NEW3 audit: HCV-A²CR, validity fusion и UTAW

Дата фиксации: 2026-08-01.

## Решение

`NEW3.md` проверен против текущего кода, формул, литературы и четырёх paired datasets.
Результат разделился на две части:

1. airlight-normalized HCV даёт точную и полезную систему координат, но HCV recovery,
   HCV↔RGB fusion и HCV-UTAW пока **не превосходят** основной RGB A²CR;
2. stationary HSV-V UTAW внутри старого Transmission-aware даёт устойчивый выигрыш структуры,
   но чаще усиливает flat-noise относительно HSV Edge и не даёт end-to-end GPU speedup.

Поэтому новые методы выведены в программу для сравнения, но отдельная arXiv-статья сейчас не
создаётся. Корректный формат — расширение/appendix основной A²CR работы и внутренний
экспериментальный отчёт.

## Что в NEW3 уже было реализовано

До этой работы репозиторий уже содержал:

- точную sRGB linearization;
- bootstrap-оценку `A`;
- DCP/CAP/Haze-Lines ensemble в optical depth;
- weighted median/MAD uncertainty;
- два A²CR gain с closed-form risk;
- точный RGB-feasible polygon и joint-TV;
- frozen validation/test protocol и аудит AutoTuner.

Поэтому эти элементы не были повторно объявлены новыми. Исправлены два реальных пробела:

- коррелированные DCP perturbations получили опциональный family-balanced fusion;
- для HCV добавлено согласование `Var(t)` с refined mean через
  `t_refined² sigma_D² + sigma_floor²`.

Family-balanced fusion и новое propagation не улучшили O-HAZE validation; они сохранены как
явные абляции (`family`, `refvar`), а не выданы за победившие решения.

## Реализованные методы

| Метод в программе | Смысл | Статус |
|---|---|---|
| HCV-A²CR | exact airlight-normalized HCV, risk gains `gV/gC`, feasible polygon | исследовательская абляция |
| HCV↔RGB A²CR validity fusion | confidence-gated convex fusion при общих `A,t` | смешанный результат |
| HCV-A²CR-UTAW | HCV recovery + stationary à trous value bands | отрицательная абляция |
| Transmission-aware HSV UTAW | HSV-V stationary bands + noise/disagreement reliability | practically positive structural candidate |
| Transmission-aware HSV UTAW GPU | тот же UTAW stage на CUDA, остальной pipeline CPU | hybrid prototype |

Подробная HCV-математика: [hcv-a2cr-utaw.md](../methods/hcv-a2cr-utaw.md). Формулы
Transmission UTAW: [transmission-aware-multiscale.md](../methods/transmission-aware-multiscale.md).

## Frozen HCV test

O-HAZE, 22 test-пары, `core`, `maxdim=192`, commit `0609747`, `dirty=false`.
Fusion `hcvPrior=0.25` был выбран только на validation.

| Метод | PSNR ↑ | SSIM ↑ | DE00 ↓ | Clip % ↓ |
|---|---:|---:|---:|---:|
| RGB A²CR | **16.9009** | 0.8085 | **14.0445** | **8.5235** |
| HCV-A²CR | 16.8568 | **0.8115** | 14.1283 | 8.9005 |
| HCV↔RGB fusion | 16.8888 | 0.8114 | 14.0960 | 8.8765 |
| HCV-A²CR-UTAW | 16.6761 | 0.7953 | 14.3088 | 12.7340 |

HCV-A²CR и fusion немного повышают средний SSIM, но ухудшают PSNR, DE00 и clipping. HCV-UTAW
ухудшает все четыре показателя. Это не основание заменять A²CR или выносить HCV в название статьи.

## Выбор Transmission UTAW только на validation

Шесть заранее заданных профилей прогнаны на 94 validation-парах O-/I-/Dense-/NH-HAZE:
564/564 успешных запусков, test не открывался. Единственным небазовым вариантом, у которого
одновременно не ухудшились combined clipping и flat-noise, стал U1:

```text
gFine=0, gMid=1.3, gCoarse=1.1,
uNoise=0.006, uUnc=2, uRadius=3, uLimit=0.025
```

| Validation, image-weighted | PSNR ↑ | SSIM ↑ | DE00 ↓ | Clip % ↓ | Flat-noise × ↓ |
|---|---:|---:|---:|---:|---:|
| U0 | 12.727576 | 0.497956 | 21.670014 | 14.351739 | 2.190075 |
| U1 | **12.728805** | **0.498032** | **21.669129** | **14.345345** | **2.189777** |

Разница очень мала. U1 называется conservative validation-selected profile, а не существенным
улучшением.

## Frozen U1 test на четырёх наборах

91 test-пара, 273/273 successful, full profile, `maxdim=192`, commit `bd4660a`, `dirty=false`.

| Метод | PSNR ↑ | SSIM ↑ | DE00 ↓ | Clip % ↓ | Flat-noise × ↓ |
|---|---:|---:|---:|---:|---:|
| старый Lab-L Laplacian | 12.0503 | 0.3984 | 22.7291 | 15.7387 | 1.7180 |
| HSV Edge | 12.1169 | 0.4381 | 22.6803 | **13.0613** | **1.4849** |
| HSV UTAW U1 | **12.1703** | **0.4735** | **22.5740** | 14.7793 | 1.8724 |

UTAW против старого Laplacian выигрывает PSNR на 68/91, SSIM на 91/91, DE00 на 61/91 и
clipping на 59/91. Против HSV Edge: PSNR 82/91, SSIM 86/91, DE00 83/91, но clipping только
24/91 и flat-noise только 2/91. Следовательно, это устойчивый structural-quality выигрыш,
а не универсальное улучшение изображения.

### По наборам

| Dataset | Метод | PSNR | SSIM | DE00 | Clip % |
|---|---|---:|---:|---:|---:|
| O-HAZE | Lap / Edge / UTAW | 14.079 / 14.233 / **14.293** | 0.550 / 0.599 / **0.641** | 18.656 / 18.557 / **18.458** | 9.931 / **7.737** / 7.981 |
| I-HAZE | Lap / Edge / UTAW | **12.173** / 12.061 / 12.077 | 0.393 / 0.418 / **0.430** | **20.640** / 20.844 / 20.825 | 35.190 / **31.361** / 35.930 |
| Dense-Haze | Lap / Edge / UTAW | 11.236 / 11.292 / **11.372** | 0.333 / 0.364 / **0.403** | 24.087 / 23.996 / **23.838** | 11.180 / **8.810** / 10.214 |
| NH-HAZE | Lap / Edge / UTAW | 11.144 / 11.249 / **11.292** | 0.343 / 0.392 / **0.432** | 25.850 / 25.744 / **25.636** | 14.224 / **11.484** / 13.133 |

I-HAZE остаётся режимом отказа по PSNR/DE00/clipping. Этот набор не позволяет писать, что UTAW
лучше старого метода по всем данным.

## Scene 08

Clean development case, `maxdim=800`, commit `bd4660a`; сцена и GT уже использовались при
разработке и не являются blind test.

| Метод | Full PSNR | Full SSIM | Full DE00 | ROI PSNR | ROI SSIM | ROI DE00 |
|---|---:|---:|---:|---:|---:|---:|
| Laplacian | 13.144 | 0.423 | **18.982** | 7.878 | 0.461 | 32.862 |
| HSV Edge | 13.038 | 0.395 | 19.156 | 7.857 | 0.465 | 32.925 |
| HSV UTAW U1 | **13.146** | **0.484** | 19.005 | **7.921** | **0.534** | **32.783** |

UTAW устраняет заметную часть мозаичной фактуры Edge и лучше сохраняет структуру, но исходное
авторское предпочтение старого Transmission-aware не отменяется автоматически. Для публикации
нужен blind pairwise study.

## CPU/GPU

Численный тест на RTX 3080 реально выполнил обе ветви: максимальное расхождение CPU↔CUDA
`4.77e-7` при допуске `7e-4`.

Clean end-to-end timing: первые 8 O-HAZE validation, `maxdim=800`, warmup=2, repeat=5.

| Метод | Mean ms ↓ | ms/MP ↓ |
|---|---:|---:|
| Laplacian | **338.2** | **681.9** |
| HSV Edge | 387.8 | 778.9 |
| HSV UTAW CPU | 420.6 | 841.2 |
| HSV UTAW hybrid CUDA | 428.6 | 859.1 |

Hybrid CUDA здесь на 1.9% медленнее CPU. На GPU перенесены HSV conversion, sparse separable B3
filters, box-power и coefficient arithmetic; transmission, local airlight, recovery и финальная
обработка остаются CPU. Speedup claim отсутствует.

## AutoTuner

После удаления basis-specific мёртвых координат:

- quick: 100 unique, 45 cache hits, 0 failures, 9/9 координат покрыты, 7/9 изменились;
- thorough: 160 unique, 11 cache hits, 0 failures, 25/25 координат покрыты, 13/25 изменились;
- full-resolution objective в обоих аудитах не ухудшился.

`uNoise`, `uUnc`, `uLimit` участвуют в quick search; `uRadius` — в thorough. Edge-параметры не
попадают в UTAW search, UTAW-параметры — в Edge search. Это проверяет механику поиска, но не
разрешает подгонять параметры на test.

## Граница новизны и литература

HSV dehazing, hue invariance и saturation-based transmission уже известны:

- [Wan & Chen, VCIP 2015](https://ieeexplore.ieee.org/document/7457892);
- [Kim et al., IEEE TIP 2020](https://ieeexplore.ieee.org/document/8882514);
- [Saturation Line Prior, IEEE TIP 2023](https://doi.org/10.1109/TIP.2023.3279980);
- [RSVT 2024](https://arxiv.org/abs/2403.12054);
- [CIM-D, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lyu_Disentanglement-wise_Image_Dehazing_through_Cross-Domain_Manifold_Consensus_CVPR_2026_paper.html).

Airlight-centered RGB geometry известна по
[Narasimhan–Nayar](https://publications.ri.cmu.edu/chromatic-framework-for-vision-in-bad-weather)
и [Haze-Lines](https://openaccess.thecvf.com/content_cvpr_2016/html/Berman_Non-Local_Image_Dehazing_CVPR_2016_paper.html).
Wavelet/à trous decomposition также не является новой сама по себе.

Защищаемые **кандидаты**, требующие независимой проверки:

- regularized `gV/gC` с точным HCV feasible polygon;
- validity gate двух RGB-feasible recovery systems;
- совместная transmission/noise/prior-disagreement reliability полос.

Нельзя заявлять новыми: HSV dehazing, сохранение Hue, saturation prior, обработку только Lab-L,
wavelet basis или сам факт переноса фильтра на GPU.

## Публикационное решение

Сейчас:

- в существующую A²CR статью можно добавить точную HCV-репараметризацию и честную отрицательную
  абляцию;
- UTAW можно описать как ongoing perceptual branch в appendix/technical report;
- отдельные arXiv/Habr статьи не создаются, потому что нет достаточного нового результата.

Для второй статьи нужны native/800 multi-dataset результаты с LPIPS, confidence intervals,
blind human study, внешние saturation/wavelet baselines, уменьшение flat-noise/clipping и
самостоятельный GPU speedup. До этого сохраняется одна основная статья A²CR.
