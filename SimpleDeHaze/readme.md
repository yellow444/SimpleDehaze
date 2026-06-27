# SimpleDeHaze

Удаление дымки методом **Dark Channel Prior** на **.NET 8** через **Emgu.CV** -
реализации для **CPU** (`DeHazeCPU`) и **GPU/CUDA** (`DeHazeGPU`).

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
dotnet run --project .\SimpleDeHaze\SimpleDeHaze.csproj -c Release                # GUI
dotnet run --project .\SimpleDeHaze\SimpleDeHaze.csproj -c Release -- foto.jpg     # GUI с уже открытым файлом
dotnet run --project .\SimpleDeHaze\SimpleDeHaze.csproj -c Release -- --batch      # прежний прогон по dataset\ (окна OpenCV)
dotnet run --project .\SimpleDeHaze\SimpleDeHaze.csproj -c Release -- --selftest   # headless-проверка всех методов
```

## Интерфейс (GUI)

Окно: выбор **метода** (DCP CPU, DCP GPU/CUDA, CAP HSV), **ползунки параметров** под выбранный
метод, кнопка **'Вычислить'** (считает в фоне, показывает время), панели **вход | результат** и
**'Сохранить...'**. Каждый метод сам объявляет свои параметры, поэтому ползунки генерируются
автоматически.

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
