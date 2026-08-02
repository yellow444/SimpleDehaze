# SimpleDeHaze

Экспериментальный стенд классического, non-ML удаления дымки на **.NET 8** через **Emgu.CV**.
В проекте есть legacy DCP-реализации для **CPU** (`DeHazeCPU`) и **GPU/CUDA** (`DeHazeGPU`),
а также модульный `Methods/` framework из 55 методов: DCP/CAP/RFEP-DCP/BRACE/PF-SFGF/LAF/GDR-SP,
A²CR, exact HCV-A²CR/fusion, Transmission-aware Laplacian/Edge/UTAW CPU+CUDA и enhancement-baselines.

**Документация по алгоритму:** [docs/README.md](docs/README.md)
> '4.8' в зависимостях - это версия Emgu CV / OpenCV, **не** .NET Framework. Проект на `net8.0`.

## Быстрый старт

Все команды - из **корня репозитория**. Нужен **.NET 8 SDK+** и **PowerShell 7+** (`pwsh`).

```powershell
# одной командой: скачать GPU-зависимости (~864 МБ) и собрать
pwsh build.ps1
```

Запуск (по умолчанию - **графический интерфейс**):

```powershell
dotnet run --project .\SimpleDeHaze\SimpleDeHaze.csproj -c Release                # GUI (новый, WPF)
dotnet run --project .\SimpleDeHaze\SimpleDeHaze.csproj -c Release -- foto.jpg     # GUI с уже открытым файлом
dotnet run --project .\SimpleDeHaze\SimpleDeHaze.csproj -c Release -- --old-ui     # прежний GUI на WinForms
dotnet run --project .\SimpleDeHaze\SimpleDeHaze.csproj -c Release -- --batch      # прежний прогон по dataset\ (окна OpenCV)
dotnet run --project .\SimpleDeHaze\SimpleDeHaze.csproj -c Release -- --selftest   # headless-проверка всех методов
dotnet run --project .\SimpleDeHaze\SimpleDeHaze.csproj -c Release -- --mathtest   # численные проверки физического ядра
dotnet run --project .\SimpleDeHaze\SimpleDeHaze.csproj -c Release -- --benchmark --limit=3 --maxdim=800 --out=bench.csv
dotnet run --project .\SimpleDeHaze\SimpleDeHaze.csproj -c Release -- --benchmark --manifest=datasets/manifests/o-haze-in-repo.json --split=test --profile=core --evalfull --out=benchmark_results/bench_core.csv
dotnet run --project .\SimpleDeHaze.Tests\SimpleDeHaze.Tests.csproj -c Release
```

Для **сравнения приоров** используйте `--profile=core`: он допускает только методы с явно заданным
core-профилем, поэтому косметика не отключается эвристически по имени параметра. Эталонный baseline -
метод `DCP канонический (He 2009, baseline)`; методы с пометкой *Legacy* - историческая ветка
проекта, это **не** канонический DCP. Подробности: [NOVELTY.md](../NOVELTY.md),
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

## Интерфейс (GUI)

Окно: выбор **метода** (A²CR/HCV dual-gain recovery, Transmission-aware Laplacian/UTAW CPU/GPU,
legacy DCP CPU/GPU, CAP HSV, RFEP-DCP, BRACE-DCP, PF-SFGF, LAF-TV/WLS, GDR-SP,
Gradient Domain, CLAHE/Retinex и др.), **ползунки параметров** под выбранный метод, кнопка
**'Вычислить'** (считает в фоне, показывает время), панели **вход | результат** и
**'Сохранить...'**. Каждый метод сам объявляет свои параметры, поэтому ползунки генерируются
автоматически.

У семи реально поддерживаемых CUDA-конвейеров backend выбирается постоянным переключателем
**«Вычисление: CPU/CUDA»** внутри параметров метода: A²CR, HSV²CR, legacy поканальный,
Beltrami, Matting WLS, Transmission-aware Laplacian и Transmission-aware HSV UTAW. Кнопка
**GPU** слева только фильтрует этот список и сама backend не переключает.

Benchmark в GUI и `--selftest` считают PSNR/SSIM, совмещённые PSNR/SSIM, CIEDE2000,
no-reference score, runtime и две явно собственные диагностические эвристики
`NaturalnessDev`/`ArtifactDev`, не совместимые с официальными NIQE/BRISQUE.
Для воспроизводимых прогонов по `dataset/` есть headless-режим `--benchmark`; он пишет CSV
без сохранения изображений и поддерживает manifests, фиксированные split, warm-up/repeat, измерение
памяти и точную фиксацию параметров. Полный протокол: [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).
Отдельный executable сейчас выполняет 64 численных и регрессионных теста, включая HCV feasible
projection, stationary à trous identity и условную CPU↔CUDA эквивалентность.

Добавить новый метод (из [docs/methods](docs/methods/README.md)): реализуйте интерфейс
`SimpleDeHaze.Methods.IDeHazeMethod` (имя, список `ParamDef`, метод `Process`) и впишите класс в
`SimpleDeHaze.Methods.MethodRegistry` - он сам появится в выпадающем списке и в `--selftest`.

<details>
<summary>То же вручную, по шагам</summary>

```powershell
pwsh tools\fetch-emgu-packages.ps1                          # 1) GPU-пакеты Emgu -> localpackages\
dotnet restore .\SimpleDeHaze\SimpleDeHaze.csproj           # 2) restore (+ CUDA-библиотеки с nuget.org)
dotnet build   .\SimpleDeHaze\SimpleDeHaze.csproj -c Release # 3) build
```
</details>

## Требования

- **Windows x64**, **.NET 8 SDK** (или новее), **PowerShell 7+**.
- Для GPU: видеокарта **NVIDIA** + актуальный драйвер. **CUDA Toolkit ставить НЕ нужно** -
  рантайм идёт внутри пакетов Emgu. Проверка карты: `nvidia-smi`.

## Зависимости: почему отдельный шаг

GPU-рантайм `Emgu.CV.runtime.windows.cuda 4.8.0.5324` и две его под-зависимости
(`blas.lt 12.0.104`, `dnn.cnn.infer 8.8.0`) убраны с nuget.org - их `.nupkg` превышают
лимит nuget.org в 250 МБ. Скрипт `tools\fetch-emgu-packages.ps1` качает их из официального
[GitHub-релиза Emgu](https://github.com/emgucv/emgucv/releases/tag/4.8.0) в `localpackages\`
(папка в `.gitignore`). Остальные CUDA-библиотеки тянутся с nuget.org при restore.

**В git - только текст** (исходники, `nuget.config`, скрипты). Гигабайты `nupkg`/`dll`
не хранятся: `localpackages\`, `bin\`, `obj\` - в `.gitignore`.

## Обновление версии

GPU-стек зафиксирован на **4.8.0.5324** (управляемый `Emgu.CV` обязан совпадать с версией
рантайма). Версии **4.9-4.13 с GPU** - это коммерческая поставка Emgu или самостоятельная
сборка OpenCV+CUDA; поднятие только `Emgu.CV` уберёт GPU.
