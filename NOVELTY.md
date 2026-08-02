# NOVELTY.md — что в этом репозитории новое, а что нет

Единственный источник правды по формулировкам новизны. Если другой документ репозитория
противоречит этой таблице — прав этот файл, а тот документ надо править.

Правило: **уникальное название метода не равно новому математическому результату**. Отсутствие
точного совпадения по названию в поиске — отрицательный результат поиска, а не доказательство
новизны.

## Матрица claim'ов

| Claim | Ближайшая известная работа | В чём отличие | Чем проверяется | Разрешённая формулировка |
|---|---|---|---|---|
| Нижняя граница `t` из `0 ≤ J ≤ 1` | **Meng et al., ICCV 2013** (boundary constraint) | отличия нет: та же формула при `C0=0, C1=1` | `--mathtest`, проверка 3 | **Не заявлять новым.** Ссылаться на Meng et al. |
| Граница `t` для *chroma-safe* восстановления `J=A+d̄/max(t,m)+δ/max(t,q)` | не найдено | классический boundary constraint выведен для деления всего `(I−A)` на один `t` и к этой формуле не применим; здесь допустимое множество — отрезок, а не луч | `--mathtest`, проверки 4–5 (20 000 случайных наборов + сверка с независимой реализацией) | Кандидат в вклад. Нужны абляция и сравнение с проекцией на `t_box` |
| Проекция до и после edge-aware уточнения | вариации известны | своя композиция, не более | замер доли нарушений в `--mathtest` | Инженерное решение, не научная новизна |
| «Фрактальная» карта насыщенности | Xu et al., CVPR 2017 и др. работы по фрактальным признакам | два масштаба без регрессии — это не оценка фрактальной размерности | нет калибровки на fBm | **Не называть фрактальной.** Только «двухмасштабная шероховатость» |
| Многомасштабная шероховатость с МНК-наклоном и R² | стандартный приём оценки показателя степени | своя реализация, но не новый метод; включается как `rough=1`, прежний путь оставлен для контроля | `rough=0/1` как абляция | Инструмент, не вклад |
| Transmission-aware Laplacian | **arXiv:2111.05700**, Multi-Scale Single Image Dehazing Using Laplacian and Gaussian Pyramids | там уже есть разложение и кадра, и трансмиссии по масштабам, и разное шумоподавление по уровням | сравнение с базовой пирамидой | **Не заявлять первым.** Отличие — конкретная функция гейта |
| Transmission-aware HSV edge bands | Laplacian/Gaussian dehazing, wavelet dehazing, Domain Transform и bilateral dehazing уже известны | одна физическая часть и transmission/scale gate, residual bands по HSV-V | 2×2 val; frozen O-HAZE test; scene-08 review | Историческая quality-абляция. Не заявлять новым basis или speedup; после UTAW исключён из curated-набора, но оставлен как более гладкий контроль |
| Transmission-aware HSV UTAW | stationary/à trous wavelets и wavelet dehazing известны | local-power reliability с `noise²/t²`, CAP↔DCP optical-depth disagreement, transmission gate и bounded band amplitude | 94-pair val; frozen 91-pair O/I/Dense/NH test; scene 08; 64 tests | **Structural-quality кандидат**, не доказанная мировая новизна: SSIM лучше Laplacian 91/91 и Edge 86/91, но flat-noise хуже Edge 89/91 |
| Hybrid CUDA UTAW | CUDA color/filter primitives и GPU wavelet processing известны | математически эквивалентный sparse B3 stage, общий pipeline пока CPU | CPU↔GPU max abs `4.77e-7`; clean 800 px timing | Инженерный prototype, не вклад и не speedup: end-to-end на RTX 3080 на 1.9% медленнее CPU |
| Гейт полос из модели шума `S/(S+σ²/t²)` | винеровская фильтрация; оценка σ по Донохо | float linear-luminance реализация вместо подобранного smoothstep | `wiener=0/1` как абляция | Кандидат в вклад после измерений |
| A²CR airlight-aligned dual-gain recovery | boundary constraint, haze-lines, uncertainty networks и color-constrained recovery содержат отдельные близкие элементы | два gain вдоль/поперёк `A`, closed-form uncertainty risk, exact RGB-feasible polygon и joint primal-dual TV | 43 tests, B0–B9 stress, DIODE 112 500 rows, bootstrap, real paired | **Кандидат**: targeted review не нашёл прямого аналога, но не заявлять мировой новизной без независимой проверки |
| Exact airlight-normalized HCV | HSV-модель Wan–Chen; saturation priors; RSVT; CIM-D; airlight geometry и Haze-Lines | для `X=I/A` точная репараметризация `X=1+t(Y-1)`, откуда `C_X=tC_Y`, `V_X-1=t(V_Y-1)` | synthetic inverse test; frozen O-HAZE test | Точная алгебра, **не новый физический закон/HSV method** и не основание заявлять «впервые» |
| HCV `gV/gC` и feasible polygon | близкие dual-gain/risk/constraint элементы есть в A²CR и color-constrained recovery | два regularized gain в HCV basis; пересечение линейных RGB half-planes и евклидова проекция | 20 000 случайных проекций; finite pipeline; frozen test | Технический кандидат. Гарантия только при `feasible=1`, фиксированных `p,q`, `A_c>=0.02`; defensive clamp остаётся |
| HCV↔RGB validity fusion | multi-color-space fusion и self-consistency gates известны | общие `A,t`; hue confidence, projection distance и forward re-hazing residual управляют выпуклым смешиванием двух feasible recoveries | O-HAZE val + clean frozen test | Эвристика, не вероятность и не аналитический optimum; SSIM слегка выше A²CR, но PSNR/DE00/clipping хуже |
| HCV-A²CR-UTAW | HCV dehazing и wavelet dehazing известны по отдельности | uncertainty-controlled normalized-Value bands поверх HCV feasible recovery | identity/constant tests; O-HAZE frozen test | **Отрицательная абляция:** уступил A²CR по средним PSNR/SSIM/DE00/clipping; не отдельный claim |
| C³R cylindrical HSV recovery | RSVT (HSV S--V), HVI (polarized HS), CIM-D (`S cos H`, `S sin H`, `V`) | три risk-derived gain, prior-disagreement fallback и input-containing saturation/value corridor; это приближённая геометрия, не точная ASM-инверсия | 4 специальных теста; O-HAZE 23 val / 22 test negative pilot | **Только дополняющая гипотеза/абляция.** Не отдельный claim: frozen v2 уступил A²CR на 20/22 PSNR и 21/22 SSIM изображениях |
| CAR chromatic airlight residual | CAP, local airlight field и Color-Constrained Dehazing Model | global chromatic anchor, coarse chroma donor и gamut-confidence fusion | scene-08 ROI, global/ROI objective conflict, autotuner audit | Отдельная проверяемая гипотеза/case study, не общий claim и не blind result |
| Локальное поле атмосферного света | пространственно-переменный airlight известен | быстрая реализация; в коде WLS, а не TV | — | Инженерная реализация |
| Спектральная модель `t_c = t^{k_c}` | wavelength-dependent transmission публиковалась | своё — оценка η по декоррелации остаточного каста с глубиной | не реализовано | Направление, не результат |
| .NET-лаборатория из 55 методов с единым API | аналогов немного | воспроизводимый стенд и контролируемое сравнение; CPU/CUDA — backend одного метода, а не дубли строк | бенчмарк и 64 численных/регрессионных теста | **Самый защищаемый вклад репозитория** |

## Переименования после аудита

| Было | Стало | Почему |
|---|---|---|
| `RFEP-DCP (radiance-feasible)` | `Boundary-Constrained Prior Fusion (быв. RFEP)` | название заявляло новизну границы, которой нет |
| `Dark Channel Prior (CPU/GPU)` | `Legacy поканальный (не канонический DCP)` + постоянный переключатель CPU/CUDA | метод считает поканальный минимум и экспоненциальную `t` — это не DCP; backend не является отдельным алгоритмом |
| `NiqeProxy` / `BrisqueProxy` | `NaturalnessDev` / `ArtifactDev` | это собственные эвристики, а не NIQE и BRISQUE |
| `FractalRichness` (смысл) | «двухмасштабная шероховатость» | не является оценкой фрактальной размерности |

Эталонный baseline — отдельная реализация **`DCP канонический (He 2009, baseline)`**: минимум по каналам,
затем по окну, стандартная инверсия, никакой косметики. Сравнивать новые приоры нужно с ним.

## Что нельзя утверждать без новых экспериментов

* «превосходит классические baseline» — внутренние DCP/RFEP/Haze-Lines и четыре real-paired
  набора дают смешанный результат; для общего claim всё ещё нужны внешние полные BCCR/PF-DCP
  реализации и независимый blind test;
* «убирает ореолы» — нужна отдельная метрика ореолов или слепое сравнение людьми;
* «быстрее» — только с указанием железа, разрешения и режима (`core` / `full`);
* «без артефактов» — неверно для любого детерминированного алгоритма;
* NIQE и BRISQUE — в репозитории их нет, есть собственные эвристики с другими именами.

## Известные режимы отказа

Тёмный канал: небо, белые стены, снег, блики, ночные сцены с цветными источниками.
Локальное поле `A(x)`: переобучение под текстуру при малом числе ярких малонасыщенных зон.
Chroma-safe восстановление: при ослабленной проекции (`strict=0`) гарантия допустимости не выполняется.
Постобработка (тон, вибранс, ограничитель цветности): улучшает вид, но искажает сравнение приоров —
для измерений используйте явный `--profile=core` (`--nopost` оставлен только как alias).

## Проверка утверждений

```bash
dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release -- --mathtest
```

Проверяются: обратимость sRGB-кривой, замкнутость модели по яркости в линейном RGB, корректность
boundary constraint, корректность выведенной chroma-safe границы, совпадение матричной реализации
со скалярным эталоном, отсутствие нарушений допустимости при `strict=1`, вырожденные случаи метрик.
