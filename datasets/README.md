# Benchmark datasets

Данные не дублируются в Git. Официальные страницы и обязательные ссылки:

| Dataset | Пар | Официальная страница |
|---|---:|---|
| O-HAZE | 45 | https://data.vision.ee.ethz.ch/cvl/ntire18/o-haze/ |
| I-HAZE | 35 описано; 30 в текущем официальном архиве | https://data.vision.ee.ethz.ch/cvl/ntire18/i-haze/ |
| Dense-Haze | 33 описано; 55 в текущем официальном архиве | https://data.vision.ee.ethz.ch/cvl/ntire19/dense-haze/ |
| NH-HAZE | 55 | https://data.vision.ee.ethz.ch/cvl/ntire20/nh-haze/ |

Загрузка и построение фиксированного manifest с SHA-256:

```powershell
pwsh tools/prepare-dehaze-dataset.ps1 -Name I-HAZE -Download
```

Если архив уже скачан и распакован:

```powershell
pwsh tools/prepare-dehaze-dataset.ps1 -Name I-HAZE -Source D:\datasets\I-HAZE
```

Для I-HAZE допускаются обе известные официальные поставки: полные 35 пар и текущий публичный
архив из 30 пар (сцены 01–25 и 31–35). У Dense-Haze официальная страница описывает 33 пары,
но файл по её ссылке сейчас содержит 55 пронумерованных GT/hazy пар; адаптер принимает обе
поставки и явно записывает фактическое число. Для остальных наборов число пар должно точно
совпасть с опубликованным. Полученный
`simpledehaze-manifest.json` содержит явные пути, split и хеш каждого файла. Перед загрузкой
проверьте условия использования на официальной странице и цитируйте соответствующую работу.

Включённый в репозиторий O-HAZE описан готовым manifest `manifests/o-haze-in-repo.json`.

## DIODE validation и controlled synthesis

Каталог источников и storage policy: `data-sources.json`. DIODE validation готовится двумя
отдельными шагами, чтобы повреждённый архив или неполная распаковка не могли попасть в эксперимент:

```powershell
pwsh tools/prepare-research-data.ps1 -Name DIODE-Val -Download -Extract
pwsh tools/prepare-diode.ps1
```

Первый скрипт перед каждой фазой сохраняет не менее 10 GiB свободного места, проверяет размер
`2 774 625 282` байта и официальный MD5 `5c895d09201b88973c8fe4552a67dd85`, затем строит общий
SHA-256 inventory (2625 файлов в проверенном состоянии от 29 июля 2026). Extraction принимается
только после точной проверки 771 RGB/depth/mask triple и completion marker. Второй скрипт выбирает
500 кадров с балансом 250 indoor / 250 outdoor и создаёт scene-level split 295/104/101 без утечки сцен.

Manifest `benchdata/manifests/diode-val-500.json` задаёт 22 500 linear-RGB recipes как декартово
произведение 500 кадров, пяти уровней дымки, трёх airlight и трёх noise recipes. Hazy/transmission
генерируются потоково и не сохраняются; это намеренная защита диска, а не отсутствие metadata.
Каждая строка benchmark содержит исходные пути, seed, `beta`, `A`, noise parameters и виртуальные
`stream://` URI. Проверенный полный прогон — 112 500 строк, 0 failures; его CSV и JSON-summary
находятся в `benchmark_results/`. Полный протокол: `SimpleDeHaze/docs/research/a2cr-data-protocol.md`.
