# Transmission-aware RGB/HSV multiscale study

Дата фиксации: 2026-08-01.

## Вопрос

Можно ли сохранить визуально сильное удаление пелены исходного Transmission-aware, но улучшить
качество заменой анализируемого цвета и/или Laplacian pyramid? Исследование меняет ровно два
фактора при общей оценке `t`, локального `A(x)`, recovery и постобработке:

1. Lab-L против HSV-V;
2. downsampled Gaussian/Laplacian pyramid против полноразмерных Domain Transform residual bands.

## Что уже известно

- Li et al., *Multi-Scale Single Image Dehazing Using Laplacian and Gaussian Pyramids*,
  [arXiv:2111.05700](https://arxiv.org/abs/2111.05700), уже используют Gaussian/Laplacian
  decomposition и различную обработку уровней.
- *Fast Single Image Dehazing via Multilevel Wavelet Transform based Optimization*,
  [arXiv:1904.08573](https://arxiv.org/abs/1904.08573), уже использует multilevel Haar для
  ускорения dehazing optimization.
- WaveDH, [arXiv:2404.01604](https://arxiv.org/abs/2404.01604), и
  [DW-GAN, CVPRW 2021](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Fu_DW-GAN_A_Discrete_Wavelet_Transform_GAN_for_NonHomogeneous_Dehazing_CVPRW_2021_paper.html)
  показывают, что wavelet subbands в dehazing сами по себе не новы.
- Edge-aware bilateral processing для dehazing также известно, например
  [Ultra-High-Definition Image Dehazing via Multi-Guided Bilateral Learning, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zheng_Ultra-High-Definition_Image_Dehazing_via_Multi-Guided_Bilateral_Learning_CVPR_2021_paper.html).

Поэтому допустимый исследовательский claim узкий: детерминированная композиция HSV-V или
linear-luma residual bands с transmission/scale gate. Поиск не нашёл точного аналога такой
композиции, но отрицательный поиск не является доказательством мировой новизны.

## 2×2 validation, O-HAZE, 23 изображения

Full profile, `--maxdim=192 --evalfull`, одинаковые defaults кроме двух структурных факторов.

| Пространство | Базис | PSNR ↑ | SSIM ↑ | DE00 ↓ | Clip % ↓ | Runtime, ms |
|---|---|---:|---:|---:|---:|---:|
| Lab-L | Laplacian | 14.38 | 0.58 | 18.61 | **9.28** | 33.35 |
| HSV-V | Laplacian | 14.50 | 0.61 | 18.48 | 9.61 | **28.66** |
| linear luma → RGB | edge residual | 14.74 | 0.61 | **18.28** | 9.30 | 43.43 |
| HSV-V | edge residual | **14.76** | **0.63** | 18.29 | 9.84 | 30.68 |

Однопроходные времена при 192 px оказались шумными и не используются для speed claim.

## Выбор gain только на validation

Заранее заданы четыре конфигурации. Выбран E3: при практически равном качестве с E0 он немного
снижает clipping и визуально мягче.

| ID | `gFine/gMid/gCoarse` | PSNR ↑ | SSIM ↑ | DE00 ↓ | Clip % ↓ |
|---|---|---:|---:|---:|---:|
| E0 | 0.5 / 1.9 / 1.5 | 14.76 | 0.63 | 18.29 | 9.84 |
| E1 | 0.4 / 1.5 / 1.3 | 14.75 | 0.62 | 18.30 | 9.58 |
| E2 | 0.3 / 1.3 / 1.2 | 14.73 | 0.61 | 18.32 | **9.40** |
| E3 | 0.5 / 1.6 / 1.2 | **14.76** | **0.63** | **18.29** | 9.69 |

## Frozen test, O-HAZE, 22 изображения

| Метод | PSNR ↑ | SSIM ↑ | DE00 ↓ | Clip % ↓ |
|---|---:|---:|---:|---:|
| прежний Lab-L Laplacian | 14.08 | 0.55 | 18.66 | 9.93 |
| HSV-V Laplacian | 14.13 | 0.59 | 18.56 | 7.81 |
| HSV-V edge E3 | **14.23** | **0.60** | **18.56** | **7.74** |

HSV-V edge E3 выигрывает у прежнего baseline на 14/22 изображениях по PSNR, 21/22 по SSIM,
12/22 по DE00 и 15/22 по clipping. Это небольшой, но широко распределённый выигрыш, а не эффект
одного кадра.

## Честный timing

Первые восемь validation-изображений, full profile, `--maxdim=800`, один warm-up и три повтора.

| Метод | Mean, ms | Median, ms | ms/MP |
|---|---:|---:|---:|
| Lab-L Laplacian | **352.3** | **346.3** | **712.9** |
| HSV-V Laplacian | 361.0 | 349.5 | 727.7 |
| HSV-V edge E3 | 398.0 | 378.3 | 804.8 |

Новый метод не быстрее: edge-вариант примерно на 13% медленнее baseline. Основная стоимость
остаётся в общей оценке transmission/локального airlight; замена basis не устраняет её.

## Scene 08 и риск артефактов

По авторскому просмотру прежний Transmission-aware остаётся очень сильным по удалению пелены.
HSV edge E3 делает фактуру и локальный контраст резче, но в плотной дымке может превращать слабую
структуру в блоковую/мозаичную микротекстуру. Это не видно полностью из средних PSNR/SSIM и требует
слепого попарного human study. Метод нельзя объявлять визуально лучшим только по scene 08.

## Проверка автоматического подбора

Quick audit на отдельном development-изображении: 80 уникальных evaluation, 56 cache hits,
0 failures, изменились 5 из 6 параметров, покрытие всех координат полное, full objective
`-2.233 → 43.073`, reference component `9.134 → 50.485`. Поиск не залипает, но этот аудит не
заменяет validation/test protocol и не разрешает подбирать параметры на test.

## Решение и следующая работа

1. Зарегистрированный `Transmission-aware HSV Edge Bands` оставить quality-кандидатом с E3.
2. HSV-V Laplacian оставить более консервативной абляцией: почти то же время и меньше clipping.
3. Не добавлять wavelet только ради заявления новизны: литература это направление уже покрывает.
4. Следующий содержательный шаг — texture-confidence/energy budget для подавления мозаики и
   RGB-feasible ограничитель после полос, затем frozen test на I-/NH-/Dense-Haze.
5. Скорость улучшать в общей оценке `t` и `A(x)` (downsample/fast guided/local-airlight reuse),
   а не приписывать её новому basis.
6. После внешней проверки провести blind pairwise study: старый Transmission-aware, HSV-Laplacian,
   HSV-edge E3 и A²CR.
