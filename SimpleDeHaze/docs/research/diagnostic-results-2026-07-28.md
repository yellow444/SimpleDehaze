# Диагностические результаты 2026-07-28

Это проверка работоспособности протокола после аудита `TEMP.md`, **не публикационная таблица**.
Запуск сделан из dirty worktree, с `warmup=0`, `repeat=1`, без memory-pass; на машине был открыт
другой экземпляр GUI. Поэтому quality-метрики пригодны как диагностика, а времена — только как
ориентир. CSV и metadata находятся в игнорируемом каталоге `benchmark_results/native/`.

## Протокол

- fixed `test` split из manifest;
- нативное разрешение и `--evalfull`;
- `--profile=core`;
- два внутренних метода: canonical DCP и Boundary-Constrained Prior Fusion (RFEP);
- основные метрики без GT-alignment.

```powershell
dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release --no-build -- `
  --benchmark --manifest=<manifest.json> --split=test --profile=core --evalfull --native `
  --warmup=0 --repeat=1 --no-memory `
  --methods="^DCP канонический|^Boundary-Constrained" --out=<result.csv>
```

## Средние по изображениям

| Dataset | Метод | N | PSNR ↑ | SSIM ↑ | CIEDE2000 ↓ | clip, % ↓ |
|---|---|---:|---:|---:|---:|---:|
| O-HAZE | Canonical DCP | 22 | 14.98 | 0.600 | 19.13 | 3.95 |
| O-HAZE | RFEP | 22 | 16.43 | 0.650 | 15.97 | 1.39 |
| I-HAZE | Canonical DCP | 15 | 11.72 | 0.610 | 21.70 | 3.90 |
| I-HAZE | RFEP | 15 | 15.47 | 0.770 | 14.15 | 0.75 |
| Dense-Haze | Canonical DCP | 27 | 11.93 | 0.440 | 24.44 | 0.19 |
| Dense-Haze | RFEP | 27 | 10.93 | 0.450 | 24.84 | 0.03 |
| NH-HAZE | Canonical DCP | 27 | 12.02 | 0.520 | 25.31 | 4.23 |
| NH-HAZE | RFEP | 27 | 12.87 | 0.530 | 21.54 | 1.00 |

Все 182 строки рассчитаны успешно. RFEP лучше canonical DCP по PSNR/CIEDE2000 на O-, I- и
NH-HAZE, но хуже на Dense-Haze (SSIM там почти равен). Это **не** подтверждает универсальное
превосходство и подчёркивает необходимость внешних baseline и failure-case анализа.

## Ограничения данных

- текущий официальный I-HAZE ZIP содержит 30 пар при 35 сценах на странице;
- текущий ZIP по официальной ссылке Dense-Haze содержит 55 пар при 33 на странице;
- manifest сохраняет оба числа (`publishedPairs` и `archivePairs`), test split строится
  детерминированным чередованием и не выдаётся за официальный challenge split.

## Что нужно для публикационной таблицы

1. Чистый tagged commit (`dirty=false`).
2. Заморозить параметры по `val`, затем не менять их на `test`.
3. `warmup>=1`, `repeat>=3`, отдельный memory-pass на свободной машине.
4. Импортировать результаты внешних BCCR/PF-DCP/non-local/Tarel через `--evaluate`.
5. Публиковать raw CSV/metadata и визуальные failure cases.
