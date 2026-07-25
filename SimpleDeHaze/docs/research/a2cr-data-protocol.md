# A²CR data and controlled-evaluation protocol

Статус: практический минимум из `NEW2.md` реализован 29 июля 2026 года. Этот файл фиксирует
гипотезу, данные, split, синтез и ограничения до дальнейшей настройки метода.

## Зафиксированная гипотеза

При ошибках `t_hat` и `A_hat` и низком цветном SNR uncertainty-aware airlight-aligned recovery
должен уменьшать выход за RGB-куб, clipping и усиление шума относительно scalar `1/t`, сохраняя
качество на участках с высокой transmission. Проверяются и отрицательные результаты: улучшение
одной метрики не считается доказательством улучшения всех аспектов изображения.

## Локальные данные и целостность

| Набор | Фактическая локальная поставка | Назначение |
|---|---:|---|
| I-HAZE | 30 пар в текущем официальном ZIP | real paired indoor |
| O-HAZE | 45 пар в репозитории | real paired outdoor |
| Dense-Haze | 55 пар в текущем ZIP официальной ссылки | real paired dense |
| NH-HAZE | 55 пар | real paired non-homogeneous |
| DIODE validation | 771 RGB/depth/mask triples | controlled RGB-D |

Расхождения I-HAZE (описано 35) и Dense-Haze (описано 33) не скрываются: manifest записывает
фактическое число файлов. Источники и размеры находятся в `datasets/data-sources.json`.
`tools/prepare-research-data.ps1` проверяет свободное место, точный размер DIODE archive и
опубликованный MD5 `5c895d09201b88973c8fe4552a67dd85`; `benchdata/files_sha256.json`
содержит SHA-256 2625 локальных файлов. Неполная распаковка без completion marker не
переиспользуется.

До загрузки DIODE на F: было 247,30 GiB свободно; после распаковки, verified LPIPS/AlexNet cache
и всех результатов осталось 234,80 GiB. HazeSpace2M, DIODE train и RESIDE ITS/OTS не
загружались. Generated hazy/transmission frames не материализуются, поэтому сетка не раздувает диск.

Важно: реальные наборы уже просматривались и использовались в инженерных smoke-тестах. Называть
их ретроспективно «слепым закрытым тестом» нельзя. Конфигурация теперь фиксируется; по-настоящему
слепой внешний тест потребует ещё не просмотренный набор или сторонний evaluator.

## DIODE selection и split

`tools/prepare-diode.ps1` находит 771 полный triple и детерминированно выбирает 500 кадров:
250 indoor и 250 outdoor. DIODE val содержит лишь шесть верхнеуровневых физических сцен с
неравным числом кадров. Скрипт перебирает все `3^6` назначения целых scene groups и минимизирует
квадратичное отклонение числа кадров от 60/20/20:

| Split | Кадров | Доля | Scene leakage |
|---|---:|---:|---:|
| development | 295 | 59,0% | 0 |
| validation | 104 | 20,8% | 0 |
| internal_test | 101 | 20,2% | 0 |

Manifest: `benchdata/manifests/diode-val-500.json`. Все кадры одной `scene_XXXXX` находятся
только в одном split; деление по отдельным изображениям запрещено.

## Streaming synthesis

Синтез выполняется в linear RGB:

`I = J*t + A*(1-t) + n`, `t = exp(-beta*d)`.

Для каждого кадра используются пять target transmission на 90-м percentile валидной глубины
`[0.80, 0.60, 0.40, 0.20, 0.10]`, три linear-RGB airlight (`neutral`, `cool`, `warm`) и три noise
recipe (`clean`, Gaussian, high-count Poisson-Gaussian approximation). `beta` вычисляется как
`-ln(target_t)/p90(depth)`. Итого manifest задаёт ровно `500*5*3*3 = 22 500` рецептов.

Каждая CSV-строка содержит source paths, stable random seed, `beta`, точный `A_rgb`, noise
parameters и виртуальные URI `stream://.../hazy` и `stream://.../transmission`. Эти artifacts
однозначно воспроизводятся, но не сохраняются. Seed — signed Int32 из первых четырёх little-endian
байтов SHA-256 от global seed, frame id, target transmission, airlight id и noise id.

## Recovery ablations

| ID | Смысл |
|---|---|
| G0 | scalar recovery в gamma/sRGB, только для изолированной color-space абляции |
| B0 | scalar `1/t_hat`, почти без floor |
| B1 | scalar + `t_min=0.08` |
| B2 | B1 + `chromaFloor=0.35` |
| B3 | scalar + известный RGB boundary constraint |
| B4 | airlight-aligned dual gain без uncertainty/noise |
| B5 | B4 + известная noise variance |
| B6 | B5 + transmission uncertainty |
| B7 | B6 + atmospheric-light uncertainty |
| B8 | B7 + exact RGB-feasible ray projection |
| B9 | B8 + joint edge-aware TV двух gain с точной polygon projection |

B3 проверяет известную RGB boundary идею, но не выдаётся за полную reference-реализацию BCCR с
contextual regularization. Внешний BCCR по-прежнему должен импортироваться через `--evaluate`.

## Метрики

Считаются PSNR, SSIM, CIEDE2000, настоящий LPIPS 0.1.4/AlexNet v0.1, hue/chroma errors,
invalid fraction до/после projection, required clipping, flat-region noise, ошибки `t`, `A`,
`g_parallel`, `g_perp`, projection fraction, CPU time и RAM. Метод CPU-only, поэтому его VRAM
пуст; LPIPS выполняется отдельным streaming evaluator на CUDA и не включается в CPU timing.

Hue — circular Lab error, взвешенный reference chroma, только при `C*_ab(gt)>=2`; доля таких
пикселей записывается рядом. Это исключает математически неопределённый hue серых пикселей.
Flat-noise — high-pass luminance residual на 20% самых плоских valid-depth пикселей относительно
известного noiseless synthesis. Для `clean` amplification оставляется `N/A`, а не делится на ноль.

## Validation pilot

На первых 10 validation frames, всех 450 recipes, `maxdim=192`, B0–B9 и LPIPS получено 4 500
строк без ошибок и без non-finite обязательных значений:

| Variant | PSNR ↑ | SSIM ↑ | CIEDE ↓ | LPIPS ↓ | invalid after |
|---|---:|---:|---:|---:|---:|
| B0 | 18,28 | 0,360 | 14,36 | 0,860 | 0,270 |
| B3 | 19,25 | 0,450 | 16,32 | 0,790 | 0 |
| B7 | 18,81 | 0,360 | **12,31** | 0,850 | 0,240 |
| B8 | 20,15 | 0,430 | 13,57 | 0,800 | **0** |
| B9 | **21,264** | **0,595** | 13,129 | **0,624** | **0** |

B9 лучше B0 по SSIM и LPIPS в 450/450 recipes, по PSNR в 441/450 и по CIEDE в 327/450.
Следовательно, pilot поддерживает гипотезу о feasibility/perceptual stability, но не доказывает
безусловное улучшение цвета. B7 имеет лучшую среднюю CIEDE, а B9 выигрывает по структуре/LPIPS.

Изолированная G0/B0 абляция также неоднозначна: linear B0 лучше по PSNR в 363/450 и CIEDE в
378/450, но gamma G0 лучше по SSIM и LPIPS во всех 450. Linear RGB требуется физической моделью;
его нельзя рекламировать как универсальный perceptual win.

## Полный controlled DIODE grid

Полный joint-TV запуск на всех 500 кадрах, 22 500 recipes и вариантах B0/B3/B7/B8/B9 (`maxdim=96`)
создал 112 500 строк за 2287,2 с, без единого сбоя. Потоковый валидатор проверил 53 столбца,
22 500 уникальных seed, ровно 45 recipes на кадр, пять вариантов на recipe, отсутствие дублей,
scene leakage, пустых обязательных значений и non-finite чисел. Полный грид не пересчитывал
дорогой LPIPS: настоящий LPIPS покрыт pilot выше; `lpips_status=not_requested` проверяется явно.

| Variant | PSNR ↑ | SSIM ↑ | CIEDE ↓ | hue° ↓ | chroma ↓ | invalid after | flat residual σ ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 18,797 | 0,515 | 12,521 | 32,786 | 12,672 | 0,259 | 0,0816 |
| B3 | 19,965 | 0,611 | 13,758 | 31,145 | 16,556 | **0** | 0,0303 |
| B7 | 19,257 | 0,514 | **11,303** | 32,457 | **8,968** | 0,227 | 0,0538 |
| B8 | 20,723 | 0,602 | 11,951 | **31,029** | 11,894 | **0** | 0,0294 |
| B9 | **21,748** | **0,707** | 11,543 | 31,038 | 11,372 | **0** | **0,0202** |

B9 лучше B0 по PSNR в 22 201/22 500 recipes, SSIM во всех 22 500, CIEDE в 18 439, hue в 16 199
и chroma error в 17 082. B8/B9 устраняют все выходы за RGB-куб после projection, но B7 остаётся
лучшим по среднему CIEDE/chroma. Следовательно, данные подтверждают устойчивость/feasibility,
но опровергают формулировку об универсальном цветовом превосходстве B9. B9 требует в среднем
31,18 ms против 1,01 ms у B8, то есть joint solver примерно в 31 раз дороже.

## Статистическая неопределённость

`tools/analyze_publication_results.py` проверяет pairing/finiteness, сначала усредняет 45 recipes
внутри кадра, а затем делает 10 000 stratified bootstrap повторов отдельно для indoor/outdoor.
Для B9-B0: PSNR `+2,951 dB [2,900; 3,001]`, SSIM `+0,191 [0,187; 0,196]`, CIEDE
`-0,978 [-1,074; -0,883]`, chroma `-1,299 [-1,443; -1,151]`, flat-noise
`-0,0615 [-0,0632; -0,0597]`. Leave-one-physical-scene-group-out сохраняет знаки эффектов,
но DIODE validation содержит лишь шесть физических scene groups, поэтому обычный frame bootstrap
не следует трактовать как шесть независимых миров.

На real-paired наборах используется paired bootstrap по image id. Интервалы смешанные: например,
на O-HAZE SSIM A²CR-RFEP положителен, но LPIPS хуже; на Dense-Haze A²CR лучше RFEP/Haze-Lines
по всем четырём метрикам; на I-HAZE различия PSNR/SSIM/CIEDE с RFEP включают ноль, а LPIPS хуже.
Это descriptive inference на уже просмотренных наборах, не подтверждающий blind generalization.
Полные 120 записей: `benchmark_results/publication-statistics.json` и `.csv`.

CSV: `benchmark_results/diode-controlled-full-500-96.csv`; проверенное summary:
`benchmark_results/diode-controlled-full-500-96-summary.json`.

## Real-paired test с LPIPS

После фиксации defaults четыре метода прогнаны в одинаковом `core`-профиле, `maxdim=800`,
`evalfull`, по одному repeat. Получено 364 уникальные строки, 364 `ok`, 0 failures; во всех
строках LPIPS конечен и имеет статус `alex_v0.1_cuda`. LPIPS не входит во время CPU-метода,
а GPU-memory поля оставлены пустыми, чтобы не приписывать evaluator алгоритму.

| Dataset | Method | n | PSNR ↑ | SSIM ↑ | CIEDE ↓ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|
| I-HAZE | canonical DCP | 15 | 13,171 | 0,677 | 18,775 | 0,276 |
| I-HAZE | RFEP | 15 | **16,670** | 0,805 | **12,564** | **0,239** |
| I-HAZE | Haze-Lines | 15 | 16,329 | 0,773 | 12,849 | 0,268 |
| I-HAZE | A²CR | 15 | 16,616 | **0,822** | 13,659 | 0,280 |
| O-HAZE | canonical DCP | 22 | 16,270 | 0,754 | 16,784 | **0,298** |
| O-HAZE | RFEP | 22 | **17,126** | 0,762 | 14,742 | 0,301 |
| O-HAZE | Haze-Lines | 22 | 15,972 | 0,719 | 15,122 | 0,333 |
| O-HAZE | A²CR | 22 | 17,046 | **0,798** | **14,342** | 0,331 |
| Dense-Haze | canonical DCP | 27 | 12,122 | 0,452 | 23,940 | 0,716 |
| Dense-Haze | RFEP | 27 | 10,968 | 0,429 | 24,713 | 0,750 |
| Dense-Haze | Haze-Lines | 27 | 10,307 | 0,409 | 26,394 | 0,772 |
| Dense-Haze | A²CR | 27 | **12,480** | **0,478** | **22,110** | **0,692** |
| NH-HAZE | canonical DCP | 27 | 12,319 | 0,546 | 24,345 | **0,452** |
| NH-HAZE | RFEP | 27 | 13,031 | 0,548 | 21,027 | 0,469 |
| NH-HAZE | Haze-Lines | 27 | 12,546 | 0,521 | **20,851** | 0,501 |
| NH-HAZE | A²CR | 27 | **13,228** | **0,602** | 20,901 | 0,462 |

A²CR доминирует по всем четырём метрикам только на Dense-Haze. На I-HAZE LPIPS/PSNR/color
лучше у RFEP; на O-HAZE A²CR выигрывает SSIM/CIEDE, но не LPIPS/PSNR; на NH-HAZE выигрывает
PSNR/SSIM, но минимальный LPIPS у canonical DCP, а CIEDE у Haze-Lines. Наборы уже использовались
при разработке, поэтому это фиксированный regression test, а не blind publication test.

Сводка: `benchmark_results/core-lpips-real-paired-test-800-summary.csv`; рядом с каждым исходным
CSV находится `.meta.json` с manifest, параметрами, commit/dirty state и LPIPS environment.

## Команды

```powershell
pwsh tools/prepare-research-data.ps1 -Name DIODE-Val -Download -Extract
pwsh tools/prepare-diode.ps1
pwsh tools/setup-lpips.ps1

dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release -- --diode-benchmark `
  --split=validation --limit=10 --maxdim=192 --lpips `
  --out=benchmark_results/diode-controlled-validation-10.csv

dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release -- --diode-benchmark `
  --split=validation --limit=10 --maxdim=192 --variants=G0,B0 --lpips `
  --out=benchmark_results/diode-gamma-vs-linear-validation-10.csv

dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release -- --diode-benchmark `
  --split=all --limit=500 --maxdim=96 --variants=B0,B3,B7,B8,B9 `
  --out=benchmark_results/diode-controlled-full-500-96.csv

python tools/analyze_diode_csv.py benchmark_results/diode-controlled-full-500-96.csv `
  --json-out benchmark_results/diode-controlled-full-500-96-summary.json
```

`--lpips` никогда не должен тихо скачать модель: bridge требует checkpoint размером 244 408 911
байт и SHA-256 `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`.
Подготовку и контролируемую загрузку выполняет только `tools/setup-lpips.ps1` с 10 GiB reserve.
