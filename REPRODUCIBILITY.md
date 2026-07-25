# Воспроизводимость

Что нужно, чтобы числа из этого репозитория можно было пересчитать и сравнить.

## Окружение

| Компонент | Требование |
|---|---|
| .NET SDK | 8.0 или новее (проект — `net8.0-windows`, `win-x64`) |
| ОС | Windows 10/11 x64 (WinForms + WPF + CUDA-нативы Emgu) |
| Emgu.CV | `Emgu.CV.runtime.windows.cuda` 4.8.0.5324 из локального фида |
| GPU | опционально; без CUDA GPU-методы падают, остальные работают |

Пакет Emgu с CUDA не публикуется на nuget.org (лимит размера), поэтому один раз выполните:

```bash
pwsh tools/fetch-emgu-packages.ps1
```

Затем:

```bash
dotnet build SimpleDeHaze/SimpleDeHaze.csproj -c Release
```

## Данные

В репозитории лежат 45 пар `dataset/NN_outdoor_hazy.*` и `hazefree/NN_outdoor_GT.*` — это O-HAZE
(Ancuti et al., NTIRE 2018). Пары сопоставляются заменой `hazy` → `GT` и `dataset` → `hazefree`.

Готовый фиксированный manifest: `datasets/manifests/o-haze-in-repo.json` (23 val / 22 test).
Для I-HAZE, NH-HAZE и Dense-Haze используйте официальные страницы и скрипт, который строит
manifest с явными split и SHA-256 каждого файла:

```powershell
pwsh tools/prepare-dehaze-dataset.ps1 -Name I-HAZE -Download
pwsh tools/prepare-dehaze-dataset.ps1 -Name Dense-Haze -Download
pwsh tools/prepare-dehaze-dataset.ps1 -Name NH-HAZE -Download
```

Скрипт прекращает работу при неизвестном числе пар. Он отдельно записывает `publishedPairs` и
`archivePairs`: текущий официальный I-HAZE ZIP содержит 30 пар при 35 описанных сценах, а ZIP по
официальной ссылке Dense-Haze — 55 пронумерованных пар при 33 на странице. Эти расхождения не
скрываются и описаны в `datasets/README.md`. Данные кладутся в игнорируемый `benchdata/`.

## Прогон бенчмарка

```powershell
dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release -- --benchmark `
  --manifest=datasets/manifests/o-haze-in-repo.json --split=test --profile=core --evalfull `
  --warmup=1 --repeat=3 --methods="^DCP канонический|^Boundary-Constrained" `
  --out=benchmark_results/o-haze-core-test.csv
```

Ключи:

| Ключ | Смысл |
|---|---|
| `--profile=core` | применить явный core-профиль. Неподдержанные методы пропускаются; `--include-unprofiled` разрешает их только с `profile_supported=0`. `--nopost` оставлен как совместимый alias |
| `--profile=full` | полный конвейер метода |
| `--linear` | физика в линейном радиансе там, где метод поддерживает параметр `linear` |
| `--evalfull` | считать метрики без внутреннего уменьшения до 1024 |
| `--native` | обрабатывать нативное разрешение; это уже default |
| `--maxdim=N` | явный downscale для smoke/итераций; исходный и обработанный размеры пишутся в CSV |
| `--warmup=N`, `--repeat=N` | число прогревов и измерений; в CSV пишутся median/min/p95 |
| `--manifest=path` | явный список пар, split и опциональные SHA-256 |
| `--input-dir`, `--gt-dir` | fallback discovery без manifest; для статьи manifest предпочтителен |
| `--split=val` / `--split=test` | выбрать зафиксированный в manifest split |
| `--params=k=v,...` | точная конфигурация ablation; итоговые параметры каждого метода сохраняются в metadata |
| `--limit=N`, `--images=regex`, `--methods=regex` | ограничить набор |

Рядом с CSV пишется `*.meta.json`: дата, режим, разрешения, машина, число ядер, версия рантайма,
модель CPU/GPU, память GPU, драйвер, OpenCV, **хеш/dirty-state Git**, список изображений и точные
параметры методов. CSV содержит peak working set и наблюдаемую GPU memory; memory-pass выполняется
отдельно от timing, чтобы опрос памяти не искажал время.

Полная матрица обязательных ablation запускается командой:

```powershell
pwsh tools/run-dehaze-ablation.ps1 -Split test
```

Для быстрого smoke: `-Quick`.

## A²CR controlled stress test

Полная матрица B0–B9 (11 уровней `t`, четыре цвета `A`, пять типов шума):

```powershell
dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release -- --a2cr-stress `
  --out=benchmark_results/a2cr-stress-full-joint-tv.csv
```

Быстрая проверка — добавить `--quick`. Для сохранения result и карт `t`, `σt²`, `σD`,
`g_parallel`, `g_perp`, `alpha`:

```powershell
dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release -- --a2cr-diag `
  --image=SimpleDeHaze/dataset/09_outdoor_hazy.jpg --out=benchmark_results/a2cr-diag
```

Подробные формулы, варианты и ограничения: `SimpleDeHaze/docs/methods/a2cr-dehaze.md`.

## DIODE RGB-D controlled benchmark

```powershell
pwsh tools/prepare-research-data.ps1 -Name DIODE-Val -Download -Extract
pwsh tools/prepare-diode.ps1
pwsh tools/setup-lpips.ps1

dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release -- --diode-benchmark `
  --manifest=benchdata/manifests/diode-val-500.json --split=validation `
  --limit=10 --maxdim=192 --lpips `
  --out=benchmark_results/diode-controlled-validation-10-joint-tv-lpips.csv

dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release -- --diode-benchmark `
  --manifest=benchdata/manifests/diode-val-500.json --split=all `
  --limit=500 --maxdim=96 --variants=B0,B3,B7,B8,B9 `
  --out=benchmark_results/diode-controlled-full-500-96-joint-tv.csv

python tools/analyze_diode_csv.py benchmark_results/diode-controlled-full-500-96-joint-tv.csv `
  --json-out benchmark_results/diode-controlled-full-500-96-joint-tv-summary.json

python tools/analyze_publication_results.py `
  --diode benchmark_results/diode-controlled-full-500-96-joint-tv.csv `
  --real benchmark_results/core-lpips-i-haze-test-800.csv `
         benchmark_results/core-lpips-o-haze-test-800.csv `
         benchmark_results/core-lpips-dense-haze-test-800.csv `
         benchmark_results/core-lpips-nh-haze-test-800.csv `
  --json-out benchmark_results/publication-statistics.json `
  --csv-out benchmark_results/publication-statistics.csv
```

`--variants` принимает `G0` и подмножество `B0..B9`; `G0` — намеренная gamma-domain абляция.
Без `--lpips` соответствующая колонка помечена `not_requested`. С `--lpips` используется настоящий
LPIPS 0.1.4/AlexNet v0.1 через один долгоживущий Python-процесс: PNG передаются в памяти, а не
пишутся на диск. `tools/setup-lpips.ps1` сохраняет не менее 10 GiB, держит package/model cache в
игнорируемом `benchdata/` и проверяет SHA-256 checkpoint. Сам benchmark не скачивает модель тихо.

Manifest содержит 500 кадров и 22 500 recipes. Зафиксированный joint-TV streaming run выше дал
112 500 строк и 0 failures; валидатор проверяет всю сетку, seed, scene split, обязательные числа
и нулевой invalid-after у B3/B8/B9. Подробные результаты, определения seed/split/метрик и честные
отрицательные выводы находятся в `SimpleDeHaze/docs/research/a2cr-data-protocol.md`.

Настоящий LPIPS доступен и обычному real-paired benchmark. Пример фиксированного I-HAZE test:

```powershell
dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release -- --benchmark `
  --manifest=benchdata/I-HAZE/simpledehaze-manifest.json --split=test `
  --profile=core --maxdim=800 --evalfull --warmup=0 --repeat=1 --no-memory `
  --methods="^(DCP канонический|Boundary-Constrained|A²CR-Dehaze|Haze-Lines)" --lpips `
  --out=benchmark_results/core-lpips-i-haze-test-800.csv
```

Аналогичные результаты для O-/Dense-/NH-HAZE и общая таблица лежат в
`benchmark_results/core-lpips-*-haze-test-800.csv` и
`benchmark_results/core-lpips-real-paired-test-800-summary.csv`: 364/364 строк успешны.

## Внешние baseline

Предварительно рассчитанные результаты BCCR/PF-DCP/non-local/Tarel и других реализаций оцениваются
тем же модулем метрик. Resize запрещён; имя результата должно совпадать с hazy-файлом либо `id` manifest:

```powershell
dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release -- --evaluate `
  --manifest=datasets/manifests/o-haze-in-repo.json --split=test `
  --predictions-dir=D:\results\BCCR --method-name="BCCR reference" `
  --out=benchmark_results/bccr-o-haze.csv
```

## Какие метрики считать основными

**Основные** (без подгонки под эталон): `psnr`, `ssim`, `ciede2000`, `lpips`, `mse`, `clip_pct`,
`flat_noise_x`, `ms`, `ms_per_mp`.

**Диагностические**: `*_aligned_diag` — считаются ПОСЛЕ поканального аффинного совмещения результата
с эталоном. Совмещение использует эталон для правки результата, поэтому эти числа завышены и не
могут быть основным результатом.

**Собственные эвристики**: `natur_dev_own`, `artifact_dev_own` — это НЕ NIQE и НЕ BRISQUE,
сравнивать их с публикуемыми значениями этих метрик нельзя.

## Численные проверки

```bash
dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release -- --mathtest
dotnet run --project SimpleDeHaze.Tests/SimpleDeHaze.Tests.csproj -c Release
```

Обе команды возвращают 0 только при успехе. Отдельный test executable не зависит от VSTest/testhost,
которому некоторые Windows sandbox запрещают следить за родительским процессом. Проверяются:
обратимость sRGB-кривой, точность модели по яркости в линейном RGB, корректность границ
допустимости, identity-метрики, manifest split и запрет неявного core-профиля. Для A²CR
дополнительно проверяются closed-form minimum риска, частный случай `1/t`, 50 000 случайных
ray-feasible projections, точная polygon projection (20 000 случаев), cached/direct equivalence
(5 000 случаев), weighted optical-depth median/MAD, уменьшение joint-TV objective и feasibility
каждого пикселя, DIODE `.npy` float/mask
и корректное исключение ахроматических пикселей из hue error. Дополнительно проверяются
эталонные пары CIEDE2000 Sharma, формулы `beta=-ln(t90)/d90` и Poisson-Gaussian variance,
а также отсутствие повторного `*255` для уже 8-битного LPIPS-входа.

## Что фиксировать при публикации чисел

1. Хеш коммита (есть в `*.meta.json`).
2. Режим: `core` или `full`, с `--linear` или без.
3. `--maxdim` и `--evalfull` — метрики зависят от разрешения; для основного результата используйте native.
4. Железо, версии и память берутся из metadata, но также сохраните температуру/режим питания отдельно.
5. Параметры методов записываются в metadata; если параметры
   подбирались, подбор должен идти на отдельном validation-подмножестве, а не на тестовом.
6. Публикуемый прогон обязан иметь `dirty=false`; иначе commit не описывает фактический код.

## Публикационные материалы

```powershell
python tools/build_publication_figures.py
pwsh paper/a2cr-dehaze/build.ps1
```

LaTeX source: `paper/a2cr-dehaze/`. Проверенный PDF:
`output/pdf/a2cr-dehaze-preprint.pdf`. Habr draft:
`SimpleDeHaze/docs/articles/habr-a2cr-car-hsv.md`. Literature search protocol:
`SimpleDeHaze/docs/research/literature-review-2026-07.md`.

## Известные источники расхождений

* автоподбор параметров по одному изображению переобучается под него;
* без `--evalfull` `Metrics.EvalMaxSide` уменьшает копию для метрик до 1024;
* JPEG-вход не является строгим sRGB (у камеры своя тональная кривая), поэтому линеаризация — модель;
* GPU-методы зависят от версии драйвера и CUDA-рантайма.
