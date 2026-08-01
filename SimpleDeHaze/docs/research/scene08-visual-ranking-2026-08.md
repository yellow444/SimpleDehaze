# Scene 08: визуальное ранжирование и конфликт целей

Дата фиксации: 2026-08-01.

## Наблюдение

При просмотре default-результатов O-HAZE scene 08 авторское визуальное ранжирование:

1. Transmission-aware Laplacian — лучше всего снимает видимую пелену и даёт наиболее сильный
   локальный контраст.
2. A²CR — второй: лучше сохраняет мелкие детали, но остаётся холодный/синий оттенок.
3. C³R-HSV — визуально неприемлем: недовосстановление верхней части кадра и грязные неоднородные
   участки.

Это полезное qualitative observation, но не blind user study и не основание объявлять метод
универсально лучшим.

## Одинаковый прогон scene 08

Все методы запущены с defaults при `--maxdim=800` через `--scene8-study`.

```powershell
dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release --no-build -- `
  --scene8-study --maxdim=800 "--methods=^(Transmission-aware|A²CR|C³R)" `
  --out=benchmark_results/scene8-trans-a2cr-c3r
```

| Метод | Full PSNR ↑ | Full SSIM ↑ | Full DE00 ↓ | ROI PSNR ↑ | ROI SSIM ↑ | ROI DE00 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| A²CR | **15.74** | **0.664** | **14.74** | **10.24** | **0.557** | **26.40** |
| C³R-HSV | 12.64 | 0.476 | 19.29 | 8.58 | 0.408 | 30.79 |
| Transmission-aware | 13.14 | 0.423 | 18.98 | 7.88 | 0.461 | 32.86 |

Метрики предпочитают A²CR, хотя субъективно Transmission-aware выглядит убедительнее. Это не
ошибка наблюдателя: PSNR/SSIM штрафуют сильный контраст, цветовой сдвиг и локальную обработку,
которые одновременно могут усиливать ощущение удаления дымки.

## Проверка на всём O-HAZE test

Одинаковый core benchmark, 22 test-изображения, `--maxdim=192 --evalfull`:

```powershell
dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release --no-build -- `
  --benchmark --manifest=datasets/manifests/o-haze-in-repo.json --split=test `
  --profile=core --maxdim=192 --evalfull --warmup=0 --repeat=1 --no-memory `
  "--methods=^(A²CR-Dehaze|C³R-HSV|Transmission-aware)" `
  --out=benchmark_results/trans-a2cr-c3r-ohaze-test-192.csv
```

| Метод | PSNR ↑ | SSIM ↑ | DE00 ↓ | Clip % ↓ | Contrast × | Color × |
|---|---:|---:|---:|---:|---:|---:|
| A²CR | **16.90** | **0.81** | **14.04** | 8.52 | 1.52 | 1.92 |
| C³R-HSV | 15.08 | 0.64 | 15.48 | **0.00** | 1.11 | 1.11 |
| Transmission-aware | 14.25 | 0.55 | 18.84 | 14.20 | **1.79** | 1.85 |

Transmission-aware действительно имеет самый сильный contrast expansion, но это сопровождается
худшими средними reference-метриками и максимальным clipping. Следовательно, его текущую версию
нельзя заменить на A²CR как основной количественный метод.

## Публикационное решение

- A²CR остаётся основным recovery contribution.
- Transmission-aware становится приоритетной perceptual-ветвью для дальнейшей работы.
- C³R остаётся только отрицательной абляцией и не упоминается в названии статьи.

Следующая содержательная работа для Transmission-aware: отделить полезное удаление low-frequency
veil от чрезмерного clipping/цветового усиления, добавить bounded RGB-feasible output, затем
провести парное слепое сравнение A²CR против Transmission-aware и их каскада. До human study
визуальное ранжирование scene 08 следует называть авторским qualitative observation.
