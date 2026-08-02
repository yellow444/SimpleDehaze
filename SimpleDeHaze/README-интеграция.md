# Как новый UI попадает в приложение

Макет в браузере — это только дизайн. Здесь код WPF-окна, который делает то же поверх ваших
существующих `Methods/` (реестр, ParamDef, AutoTuner, Metrics не меняются).

## 1. Скопировать файлы

```
dotnet/Gui/Infra.cs            -> SimpleDeHaze/Gui/Modern/Infra.cs
dotnet/Gui/ParamCatalog.cs     -> SimpleDeHaze/Gui/Modern/ParamCatalog.cs
dotnet/Gui/MainViewModel.cs    -> SimpleDeHaze/Gui/Modern/MainViewModel.cs
dotnet/Gui/NewMainWindow.xaml  -> SimpleDeHaze/Gui/Modern/NewMainWindow.xaml
dotnet/Gui/NewMainWindow.xaml.cs -> SimpleDeHaze/Gui/Modern/NewMainWindow.xaml.cs
```

## 2. Включить WPF в csproj

В `SimpleDeHaze.csproj`, в тот же `PropertyGroup`, где уже есть `UseWindowsForms`:

```xml
<UseWPF>true</UseWPF>
```

WinForms и WPF в одном `net8.0-windows` проекте живут вместе — старый `MainForm` остаётся рабочим,
пока новый UI не догонит его по функциям.

## 3. Точка входа

**Сделано.** В `Program.Main` новое окно - поведение по умолчанию, старое - по флагу:

```csharp
string? file = args.FirstOrDefault(a => !a.StartsWith("--") && File.Exists(a));
if (args.Contains("--old-ui") || args.Contains("--winforms"))
{
    Application.Run(new MainForm(file));
    return;
}

var wpf = new System.Windows.Application();
wpf.Run(new Gui.Modern.NewMainWindow(file));
```

Запуск:

```powershell
dotnet run --project .\SimpleDeHaze\SimpleDeHaze.csproj                 # новый WPF-интерфейс
dotnet run --project .\SimpleDeHaze\SimpleDeHaze.csproj -- --old-ui     # прежний WinForms
```

### Что потребовалось в `SimpleDeHaze.csproj`

Проект был чисто WinForms-овским, поэтому XAML не компилировался
(`System.Windows.Media`/`Controls` не существовали). Добавлено:

* `<UseWPF>true</UseWPF>` - рядом с `<UseWindowsForms>true</UseWindowsForms>`;
* удаление WPF-овских implicit usings (`<Using Remove="System.Windows*" />`), иначе в старых
  WinForms-файлах `Application`, `Label`, `Button`, `Cursors` становятся неоднозначными;
* возврат `<Using Include="System.IO" />` - `UseWPF` убирает его из implicit usings
  (конфликт `System.IO.Path` и `System.Windows.Shapes.Path`).

В WPF-файлах одноимённые с WinForms типы разведены алиасами: `Brush`/`Color` в
`MainViewModel.cs`, `Button`/`MessageBox`/`OpenFileDialog`/`SaveFileDialog`/`DataFormats`/
`DragEventArgs`/`KeyEventArgs` в `NewMainWindow.xaml.cs`.

## Что именно решает этот код (и где это в файлах)

| Проблема в старом UI | Решение | Где |
|---|---|---|
| 47 методов в одном combo | список с поиском, тегами (★/GPU/быстрые/небо) и группами по семействам | `MainViewModel.RebuildFilter`, `MethodItem.Classify` |
| 5 кнопок подбора | одна кнопка + меню из 3 режимов с ценой в секундах | `NewMainWindow.xaml` (ContextMenu `AutoMenu`) |
| ползунки с формулами | человеческие названия + подписи «меньше/больше» + числовое поле | `ParamCatalog`, `ParamRow`, `ParamTemplate` |
| 8–20 параметров сразу | 3 главных (те, что участвуют в авто-подборе) + Expander «Продвинутые» | `MainViewModel.RebuildParams` |
| моноблок из 14 метрик | оценка крупно + вердикт словами + 5 строк с полосками и «норма/перебор» | `MainViewModel.ShowMetrics`, `MetricRow` |
| таблица на 14 колонок | выдвижная панель по F9, колонки с префиксом группы, экспорт CSV | `NewMainWindow.xaml` DataGrid, `Csv_Click` |
| непонятный прогресс | текст «попыток N, лучший скор S» + процент + «Стоп» | `AutoThoroughAsync`, `AutoBestAsync` |
| нет живого превью | дебаунс 120 мс, прогон на 480 px при перетаскивании | `MainViewModel.OnParamChanged` |
| детали не видны | лупа 100 % поверх результата (Z) | `NewMainWindow.xaml`, `OnKey` |

## Чего в этом скелете сознательно нет

- **Синхронный зум/панорама трёх панелей** (в WinForms это `ZoomImageView` + `ViewState`).
  В WPF это отдельный контрол на `ScrollViewer` + `ScaleTransform`; сейчас панели просто
  вписывают изображение (`Stretch=Uniform`), лупа показывает 1:1.
- **Шторка до/после** и режимы «только результат» — есть в прототипе, в коде пока нет.
- **Песочница** (`PlaygroundForm`) остаётся на WinForms; её можно открывать из нового окна как есть.
- **RU/EN переключатель** — строки пока зашиты по-русски; под локализацию нужен `.resx`.
- Группировка колонок таблицы сделана префиксом в заголовке («К эталону · PSNR»), а не двухуровневой
  шапкой: у WPF `DataGrid` нет column groups из коробки.

Код не компилировался у меня — я не запускаю .NET. Ожидаемые правки при первой сборке: пространства
имён `using` под ваш `ImplicitUsings`, точные имена полей `Metrics.Report` (я использую
`Score, HasRef, Psnr, PsnrAligned, SsimAligned, Ciede2000Aligned, HazeRemoved, ContrastGain,
ColorRatio, ClipPct, NaturalnessDev, ArtifactDev`) и сигнатуры `AutoTuner.Optimize / OptimizeThorough /
PickBest` (сверял по вашему `AutoTuner.cs`).
