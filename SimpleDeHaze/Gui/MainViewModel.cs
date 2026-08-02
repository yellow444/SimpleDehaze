using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows.Media;
using System.Windows.Media.Imaging;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using SimpleDeHaze.Methods;

// В проекте включены и WinForms, и WPF, поэтому System.Drawing и System.Windows.Media
// конфликтуют по именам. В этом файле кисти/цвета - всегда WPF-овские.
using Brush = System.Windows.Media.Brush;
using Color = System.Windows.Media.Color;

namespace SimpleDeHaze.Gui.Modern
{
    // ---------- строки списка методов ----------

    public sealed class MethodItem : Obs
    {
        public IDeHazeMethod Method { get; }
        public string Name { get; }
        public string Family { get; }
        public bool Recommended { get; }
        public bool IsGpu { get; }
        public bool? IsCudaAvailable { get; }
        public bool IsSkyAware { get; }

        private double? _score;
        public double? Score { get => _score; set { Set(ref _score, value); Raise(nameof(ScoreText)); } }
        public string ScoreText => _score is null ? "—" : _score.Value.ToString("0");

        private long _ms;
        public long Ms { get => _ms; set { Set(ref _ms, value); Raise(nameof(Meta)); } }

        public string Meta => (_ms > 0 ? _ms + " мс" : "не запускался")
                            + (IsGpu ? IsCudaAvailable == true ? " · CUDA готова" : " · CUDA недоступна" : "")
                            + (IsSkyAware ? " · бережёт небо" : "");

        public MethodItem(IDeHazeMethod m)
        {
            Method = m;
            Name = (MethodRegistry.Recommended.Contains(m.Name) ? "★ " : "") + m.Name;
            Recommended = MethodRegistry.Recommended.Contains(m.Name);
            IsGpu = m.Name.Contains("GPU", StringComparison.OrdinalIgnoreCase)
                 || m.Name.Contains("CUDA", StringComparison.OrdinalIgnoreCase);
            IsCudaAvailable = IsGpu ? TransmissionAwareHsvUtawGpuMethod.IsCudaAvailable : null;
            IsSkyAware = m.Parameters.Any(p => p.Key is "tsky" or "tauV" or "tauS")
                      || m.Name.Contains("sky", StringComparison.OrdinalIgnoreCase)
                      || m.Name.Contains("небо", StringComparison.OrdinalIgnoreCase);
            Family = Classify(m);
        }

        /// <summary>Семейство для группировки в списке — по имени/параметрам, без правки реестра.</summary>
        private static string Classify(IDeHazeMethod m)
        {
            string n = m.Name.ToLowerInvariant();
            if (MethodRegistry.Recommended.Contains(m.Name)) return "Рекомендованные";
            if (n.Contains("локальная дымка")) return "Локальная дымка";
            if (n.Contains("color attenuation") || n.Contains("cap")) return "Color Attenuation";
            if (n.Contains("retinex") || n.Contains("clahe") || n.Contains("msrcr")) return "Enhancement (не физика)";
            if (n.Contains("видимост") || n.Contains("вуаль") || n.Contains("силуэт")) return "Видимость и вуаль";
            if (n.Contains("лапласиан") || n.Contains("пирамид") || n.Contains("fusion") || n.Contains("цепочка")) return "Пирамиды, контуры, цепочки";
            if (n.Contains("dcp") || n.Contains("dark channel")) return "Тёмный канал (DCP)";
            return "Прочие";
        }
    }

    // ---------- строка параметра ----------

    public sealed class ParamRow : Obs
    {
        private readonly Action _changed;
        public ParamDef Def { get; }
        public string Name { get; }
        public string Low { get; }
        public string High { get; }
        public string Hint { get; }
        public string Badge => Def.Search ? "авто" : Def.Tunable ? "подбор" : "режим";
        public bool IsPrimary { get; }

        public double Min => Def.Min;
        public double Max => Def.Max;
        public double Step => Def.Step > 0 ? Def.Step : (Def.IsInt ? 1 : (Def.Max - Def.Min) / 200.0);

        private double _value;
        public double Value
        {
            get => _value;
            set
            {
                double v = Def.Coerce(value);
                if (Set(ref _value, v)) { Raise(nameof(Display)); _changed(); }
            }
        }

        public string Display => Def.IsInt ? _value.ToString("0") : _value.ToString("0.####");

        public ParamRow(ParamDef d, double value, bool primary, Action changed)
        {
            Def = d; _value = d.Coerce(value); IsPrimary = primary; _changed = changed;
            var human = ParamCatalog.For(d.Key);
            Name = human?.Name ?? d.Label;
            Low = human?.Low ?? "меньше";
            High = human?.High ?? "больше";
            Hint = ParamHelp.For(d);
        }
    }

    // ---------- строка «понятных» метрик ----------

    public sealed class MetricRow
    {
        public string Label { get; init; } = "";
        public string Value { get; init; } = "";
        public string Verdict { get; init; } = "";
        public double Fraction { get; init; }      // 0..1 — заполнение полоски
        public bool IsWarn { get; init; }
        public Brush Brush => IsWarn ? Palette.Warn : Palette.Accent;
    }

    public static class Palette
    {
        public static readonly Brush Accent = new SolidColorBrush(Color.FromRgb(0x5A, 0xC2, 0x8E));
        public static readonly Brush Warn = new SolidColorBrush(Color.FromRgb(0xD2, 0x9B, 0x5E));
        public static readonly Brush Text = new SolidColorBrush(Color.FromRgb(0xE8, 0xE9, 0xEA));
    }

    // ---------- строка таблицы сравнения ----------

    public sealed class BenchRow
    {
        public string Method { get; init; } = "";
        public string Mode { get; init; } = "";
        public double? Score { get; init; }
        public double? Psnr { get; init; }
        public double? PsnrAligned { get; init; }
        public double? Ssim { get; init; }
        public double? Ciede { get; init; }
        public double HazePct { get; init; }
        public double Contrast { get; init; }
        public double ColorX { get; init; }
        public double NaturalnessDev { get; init; }
        public long Ms { get; init; }
        public double MsPerMp { get; init; }
        public string? Error { get; init; }
    }

    // ---------- главная ViewModel ----------

    public sealed class MainViewModel : Obs, IDisposable
    {
        private Image<Bgr, byte>? _input;
        private Image<Bgr, byte>? _gt;
        private Mat? _lastResult;
        private CancellationTokenSource? _cts;
        private readonly Dictionary<string, Dictionary<string, double>> _memory = new();  // метод -> параметры

        public ObservableCollection<MethodItem> AllMethods { get; } = new();
        public ObservableCollection<MethodGroup> Groups { get; } = new();
        public ObservableCollection<ParamRow> PrimaryParams { get; } = new();
        public ObservableCollection<ParamRow> AdvancedParams { get; } = new();
        public ObservableCollection<MetricRow> Metrics { get; } = new();
        public ObservableCollection<BenchRow> Bench { get; } = new();
        public ObservableCollection<FrameItem> Frames { get; } = new();

        public string[] Goals { get; } = { "Объекты и контуры", "Сочность", "По эталону" };
        public string[] Presets { get; } = { "Мягко", "Норма", "Сильно" };

        public MainViewModel()
        {
            foreach (var m in MethodRegistry.All) AllMethods.Add(new MethodItem(m));
            ProcessCommand = new AsyncCmd(RunAsync, () => _input != null);
            AutoQuickCommand = new AsyncCmd(AutoQuickAsync, () => _input != null);
            AutoThoroughCommand = new AsyncCmd(AutoThoroughAsync, () => _input != null);
            AutoBestCommand = new AsyncCmd(AutoBestAsync, () => _input != null);
            RunAllCommand = new AsyncCmd(RunAllAsync, () => _input != null);
            StopCommand = new Cmd(() => _cts?.Cancel(), () => IsBusy);
            ResetParamsCommand = new Cmd(ResetParams);
            PresetCommand = new Cmd(p => ApplyPreset(p as string ?? "Норма"));
            NextFrameCommand = new Cmd(() => StepFrame(+1));
            PrevFrameCommand = new Cmd(() => StepFrame(-1));

            LoadDataset();
            Selected = AllMethods.FirstOrDefault(x => x.Recommended) ?? AllMethods.First();
            RebuildFilter();
        }

        // --- команды ---
        public AsyncCmd ProcessCommand { get; }
        public AsyncCmd AutoQuickCommand { get; }
        public AsyncCmd AutoThoroughCommand { get; }
        public AsyncCmd AutoBestCommand { get; }
        public AsyncCmd RunAllCommand { get; }
        public Cmd StopCommand { get; }
        public Cmd ResetParamsCommand { get; }
        public Cmd PresetCommand { get; }
        public Cmd NextFrameCommand { get; }
        public Cmd PrevFrameCommand { get; }

        // --- фильтры и список ---
        private string _query = "";
        public string Query { get => _query; set { if (Set(ref _query, value)) RebuildFilter(); } }

        private bool _onlyRecommended, _onlyFast, _onlyGpu, _onlySky;
        public bool OnlyRecommended { get => _onlyRecommended; set { if (Set(ref _onlyRecommended, value)) RebuildFilter(); } }
        public bool OnlyFast { get => _onlyFast; set { if (Set(ref _onlyFast, value)) RebuildFilter(); } }
        public bool OnlyGpu { get => _onlyGpu; set { if (Set(ref _onlyGpu, value)) RebuildFilter(); } }
        public bool OnlySkyAware { get => _onlySky; set { if (Set(ref _onlySky, value)) RebuildFilter(); } }

        public string ShownCount => Groups.Sum(g => g.Items.Count) + " / " + AllMethods.Count;

        private void RebuildFilter()
        {
            var visible = AllMethods.Where(m =>
                (string.IsNullOrWhiteSpace(_query) || m.Name.Contains(_query, StringComparison.OrdinalIgnoreCase)) &&
                (!_onlyRecommended || m.Recommended) &&
                (!_onlyGpu || m.IsGpu) &&
                (!_onlySky || m.IsSkyAware) &&
                (!_onlyFast || (m.Ms > 0 && m.Ms <= 200) || m.Method.Parameters.Count <= 4));

            Groups.Clear();
            foreach (var g in visible.GroupBy(m => m.Family).OrderBy(g => g.Key == "Рекомендованные" ? 0 : 1))
                Groups.Add(new MethodGroup(g.Key, g.ToList()));
            Raise(nameof(ShownCount));
            Raise(nameof(NothingFound));
        }

        public bool NothingFound => Groups.Count == 0;

        private MethodItem _selected = null!;
        public MethodItem Selected
        {
            get => _selected;
            set
            {
                if (value == null || !Set(ref _selected, value)) return;
                Raise(nameof(Description));
                RebuildParams();
            }
        }

        public string Description => _selected?.Method.Description ?? "";

        // --- параметры ---
        private void RebuildParams()
        {
            PrimaryParams.Clear(); AdvancedParams.Clear();
            var m = Selected.Method;
            if (!_memory.TryGetValue(m.Name, out var store)) { store = new(); _memory[m.Name] = store; }

            // «главными» считаем участвующие в авто-подборе (не более 3), остальное — в «Продвинутые»
            var ordered = m.Parameters.OrderByDescending(p => p.Search).ToList();
            var primaryKeys = ordered.Where(p => p.Search).Take(3).Select(p => p.Key).ToHashSet();
            if (primaryKeys.Count == 0) primaryKeys = ordered.Take(3).Select(p => p.Key).ToHashSet();

            foreach (var d in m.Parameters)
            {
                double v = store.TryGetValue(d.Key, out var sv) ? Math.Clamp(sv, d.Min, d.Max) : d.Default;
                store[d.Key] = v;
                var row = new ParamRow(d, v, primaryKeys.Contains(d.Key), OnParamChanged);
                (row.IsPrimary ? PrimaryParams : AdvancedParams).Add(row);
            }
            Raise(nameof(AdvancedCount));
        }

        public string AdvancedCount => AdvancedParams.Count.ToString();

        private IReadOnlyDictionary<string, double> CurrentValues()
        {
            var d = new Dictionary<string, double>();
            foreach (var r in PrimaryParams.Concat(AdvancedParams)) d[r.Def.Key] = r.Value;
            return d;
        }

        private void ApplyValues(IReadOnlyDictionary<string, double> vals)
        {
            _suppressLive = true;
            foreach (var r in PrimaryParams.Concat(AdvancedParams))
                if (vals.TryGetValue(r.Def.Key, out var v)) r.Value = v;
            _suppressLive = false;
            foreach (var kv in vals) _memory[Selected.Method.Name][kv.Key] = kv.Value;
        }

        private void ResetParams()
        {
            _memory.Remove(Selected.Method.Name);
            RebuildParams();
        }

        /// <summary>Пресеты «Мягко/Норма/Сильно» — сдвиг силы и защиты относительно дефолтов метода.</summary>
        private void ApplyPreset(string name)
        {
            double k = name switch { "Мягко" => 0.6, "Сильно" => 1.5, _ => 1.0 };
            foreach (var r in PrimaryParams.Concat(AdvancedParams))
            {
                if (r.Def.Key is "beta" or "omega" or "p" or "restore" or "clahe")
                    r.Value = Math.Clamp(r.Def.Default * k, r.Def.Min, r.Def.Max);
                else if (r.Def.Key is "min" or "tsky")
                    r.Value = Math.Clamp(r.Def.Default / k, r.Def.Min, r.Def.Max);
            }
        }

        // --- живое превью ---
        private bool _live = true, _suppressLive;
        public bool LivePreview { get => _live; set => Set(ref _live, value); }

        private CancellationTokenSource? _liveCts;

        private void OnParamChanged()
        {
            if (_suppressLive || !_live || _input == null || IsBusy) return;
            _liveCts?.Cancel();
            _liveCts = new CancellationTokenSource();
            var token = _liveCts.Token;
            var args = new Dictionary<string, double>(CurrentValues());
            var method = Selected.Method;
            _ = Task.Run(async () =>
            {
                try
                {
                    await Task.Delay(120, token);              // дебаунс перетаскивания
                    if (token.IsCancellationRequested) return;
                    using var preview = Downscale(_input!, 480);
                    using var res = method.Process(preview, args);
                    using var disp = new Mat();
                    res.ConvertTo(disp, DepthType.Cv8U, 255.0);
                    var bmp = Img.ToBitmap(disp);
                    var rep = Methods.Metrics.Evaluate(res, null, preview.Mat);
                    if (token.IsCancellationRequested) return;
                    App.Post(() => { ResultImage = bmp; ShowMetrics(rep, preview: true); });
                }
                catch (OperationCanceledException) { }
                catch (Exception ex)
                {
                    App.Post(() => JobLabel = "ошибка · " + ex.Message);
                }
            }, token);
        }

        // --- запуск ---
        private bool _busy;
        public bool IsBusy
        {
            get => _busy;
            private set
            {
                if (!Set(ref _busy, value)) return;
                ProcessCommand.Refresh(); AutoQuickCommand.Refresh(); AutoThoroughCommand.Refresh();
                AutoBestCommand.Refresh(); RunAllCommand.Refresh(); StopCommand.Refresh();
            }
        }

        private string _jobLabel = "готово";
        public string JobLabel { get => _jobLabel; private set => Set(ref _jobLabel, value); }

        private double _jobPercent;
        public double JobPercent { get => _jobPercent; private set => Set(ref _jobPercent, value); }

        public int Goal { get; set; } = 0;

        private AutoTuneGoal TuneGoal() => Goal switch
        {
            2 => _gt != null ? AutoTuneGoal.Reference : AutoTuneGoal.ObjectVisibility,
            1 => AutoTuneGoal.Vivid,
            _ => AutoTuneGoal.ObjectVisibility
        };

        private async Task RunAsync()
        {
            IsBusy = true; JobLabel = "Обработка полного кадра"; JobPercent = 0;
            try
            {
                var args = new Dictionary<string, double>(CurrentValues());
                var method = Selected.Method;
                var sw = Stopwatch.StartNew();
                var (bmp, rep, result) = await Task.Run(() =>
                {
                    var res = method.Process(_input!, args);
                    using var disp = new Mat();
                    res.ConvertTo(disp, DepthType.Cv8U, 255.0);
                    return (Img.ToBitmap(disp), Methods.Metrics.Evaluate(res, _gt?.Mat, _input!.Mat), res);
                });
                sw.Stop();
                _lastResult?.Dispose(); _lastResult = result;
                ResultImage = bmp;
                Selected.Ms = sw.ElapsedMilliseconds;
                Selected.Score = rep.Score;
                ShowMetrics(rep, preview: false);
                JobLabel = $"готово · {method.Name}, {sw.ElapsedMilliseconds} мс";
                JobPercent = 100;
            }
            finally { IsBusy = false; }
        }

        private async Task AutoQuickAsync()
        {
            IsBusy = true; JobLabel = "Быстрый подбор параметров"; JobPercent = 30;
            try
            {
                var method = Selected.Method;
                var cur = new Dictionary<string, double>(CurrentValues());
                var best = await Task.Run(() => AutoTuner.Optimize(method, _input!, cur, _gt?.Mat, TuneGoal()));
                ApplyValues(best);
                await RunAsync();
            }
            finally { IsBusy = false; }
        }

        private async Task AutoThoroughAsync()
        {
            IsBusy = true; _cts = new CancellationTokenSource();
            var token = _cts.Token;
            try
            {
                var method = Selected.Method;
                var cur = new Dictionary<string, double>(CurrentValues());
                var best = await Task.Run(() => AutoTuner.OptimizeThorough(
                    method, _input!, cur, minColor: KeepColor ? 1.2 : 0.0,
                    progress: (evals, score) => App.Post(() =>
                    {
                        JobLabel = $"Тщательный подбор · попыток {evals}, лучший скор {score:F0}";
                        JobPercent = Math.Min(99, evals / 3.2);
                    }),
                    cancelled: () => token.IsCancellationRequested,
                    gt: _gt?.Mat, goal: TuneGoal(), evalMaxDim: FullEval ? Math.Max(_input!.Width, _input.Height) : 340));
                ApplyValues(best);
                await RunAsync();
            }
            finally { _cts?.Dispose(); _cts = null; IsBusy = false; }
        }

        private async Task AutoBestAsync()
        {
            IsBusy = true; _cts = new CancellationTokenSource();
            var token = _cts.Token;
            try
            {
                var (best, tuned, _) = await Task.Run(() => AutoTuner.PickBest(
                    MethodRegistry.All, _input!,
                    progress: text => App.Post(() => { JobLabel = text; JobPercent = Math.Min(99, JobPercent + 2); }),
                    cancelled: () => token.IsCancellationRequested,
                    gt: _gt?.Mat, goal: TuneGoal()));
                var item = AllMethods.First(x => x.Method.Name == best.Name);
                Selected = item;
                ApplyValues(tuned);
                await RunAsync();
            }
            finally { _cts?.Dispose(); _cts = null; IsBusy = false; }
        }

        private async Task RunAllAsync()
        {
            IsBusy = true; JobLabel = "Прогон методов"; JobPercent = 0;
            Bench.Clear();
            try
            {
                using var img = Downscale(_input!, 800);
                Mat? gt = null;
                if (_gt != null) { gt = new Mat(); CvInvoke.Resize(_gt.Mat, gt, img.Size, 0, 0, Inter.Area); }
                var list = MethodRegistry.All;
                double mp = img.Width * img.Height / 1_000_000.0;
                for (int i = 0; i < list.Count; i++)
                {
                    var m = list[i];
                    var def = m.Parameters.ToDictionary(p => p.Key, p => p.Default);
                    var row = await Task.Run(() =>
                    {
                        var sw = Stopwatch.StartNew();
                        try
                        {
                            using var res = m.Process(img, def);
                            sw.Stop();
                            var r = Methods.Metrics.Evaluate(res, gt, img.Mat);
                            return new BenchRow
                            {
                                Method = m.Name, Mode = "умолч.", Score = r.Score,
                                Psnr = r.HasRef ? r.Psnr : null, PsnrAligned = r.HasRef ? r.PsnrAligned : null,
                                Ssim = r.HasRef ? r.Ssim : null, Ciede = r.HasRef ? r.Ciede2000 : null,
                                HazePct = r.HazeRemoved * 100, Contrast = r.ContrastGain, ColorX = r.ColorRatio,
                                NaturalnessDev = r.NaturalnessDev, Ms = sw.ElapsedMilliseconds, MsPerMp = sw.ElapsedMilliseconds / Math.Max(1e-6, mp)
                            };
                        }
                        catch (Exception ex)
                        {
                            sw.Stop();
                            return new BenchRow { Method = m.Name, Mode = "умолч.", Ms = sw.ElapsedMilliseconds, Error = ex.Message };
                        }
                    });
                    Bench.Add(row);
                    var item = AllMethods.First(x => x.Method.Name == m.Name);
                    item.Ms = row.Ms; item.Score = row.Score;
                    JobLabel = $"Прогон методов · {i + 1}/{list.Count}";
                    JobPercent = (i + 1) * 100.0 / list.Count;
                }
                gt?.Dispose();
                MetricsPanelOpen = true;
            }
            finally { IsBusy = false; }
        }

        public bool KeepColor { get; set; }
        public bool FullEval { get; set; }

        // --- метрики в человеческом виде ---
        private int _score;
        public int Score { get => _score; private set { Set(ref _score, value); Raise(nameof(ScoreBrush)); } }
        public Brush ScoreBrush => _score >= 60 ? Palette.Accent : _score >= 40 ? Palette.Text : Palette.Warn;

        private string _verdict = "Нажмите «Обработать»";
        public string Verdict { get => _verdict; private set => Set(ref _verdict, value); }

        private string _gtMode = "без эталона";
        public string GtMode { get => _gtMode; private set => Set(ref _gtMode, value); }

        private void ShowMetrics(Methods.Metrics.Report r, bool preview)
        {
            Score = (int)Math.Round(r.Score);
            GtMode = r.HasRef ? "с эталоном" : "без эталона";
            Verdict = Score >= 65 ? "Дымка снята, цвет и света в норме"
                    : Score >= 40 ? "Приемлемо: попробуйте усилить дехейз"
                    : "Слабо: цвет или шум вышли за норму";

            Metrics.Clear();
            Metrics.Add(new MetricRow { Label = "Дымка убрана", Value = $"{r.HazeRemoved * 100:F0} %",
                Verdict = r.HazeRemoved < 0.25 ? "мало" : r.HazeRemoved > 0.8 ? "перебор" : "норма",
                Fraction = r.HazeRemoved, IsWarn = r.HazeRemoved < 0.25 || r.HazeRemoved > 0.8 });
            Metrics.Add(new MetricRow { Label = "Контраст", Value = $"×{r.ContrastGain:F2}",
                Verdict = r.ContrastGain > 2.1 ? "жёстко" : "хорошо",
                Fraction = Math.Min(1, r.ContrastGain / 2.5), IsWarn = r.ContrastGain > 2.1 });
            Metrics.Add(new MetricRow { Label = "Насыщенность цвета", Value = $"×{r.ColorRatio:F2}",
                Verdict = r.ColorRatio > 1.5 ? "перенасыщено" : "в норме",
                Fraction = Math.Min(1, r.ColorRatio / 3.0), IsWarn = r.ColorRatio > 1.5 });
            Metrics.Add(new MetricRow { Label = "Пересветы / завалы", Value = $"{r.ClipPct:F1} %",
                Verdict = r.ClipPct > 1.5 ? "много" : "почти нет",
                Fraction = Math.Min(1, r.ClipPct / 8.0), IsWarn = r.ClipPct > 1.5 });
            Metrics.Add(new MetricRow { Label = "Артефакты (собств. оценка)", Value = $"{r.ArtifactDev:F1}",
                Verdict = r.ArtifactDev > 24 ? "заметен" : "средне",
                Fraction = Math.Min(1, r.ArtifactDev / 40.0), IsWarn = r.ArtifactDev > 24 });

            if (r.HasRef)
            {
                Metrics.Add(new MetricRow { Label = "PSNR к эталону (основной)", Value = $"{r.Psnr:F2} дБ",
                    Verdict = r.Psnr >= 18 ? "хорошо" : "средне",
                    Fraction = Math.Min(1, r.Psnr / 30.0), IsWarn = r.Psnr < 14 });
                Metrics.Add(new MetricRow { Label = "PSNR после GT-alignment (диагн.)", Value = $"{r.PsnrAligned:F2} дБ",
                    Verdict = "не для ranking", Fraction = Math.Min(1, r.PsnrAligned / 30.0), IsWarn = false });
            }

            JobLabel = preview ? "превью 480 px" : JobLabel;
        }

        // --- изображения и датасет ---
        private BitmapSource? _inputImage, _resultImage, _gtImage;
        public BitmapSource? InputImage { get => _inputImage; private set => Set(ref _inputImage, value); }
        public BitmapSource? ResultImage { get => _resultImage; private set => Set(ref _resultImage, value); }
        public BitmapSource? GtImage { get => _gtImage; private set { Set(ref _gtImage, value); Raise(nameof(HasGt)); } }
        public bool HasGt => _gtImage != null;

        private string _fileInfo = "кадр не выбран";
        public string FileInfo { get => _fileInfo; private set => Set(ref _fileInfo, value); }

        private bool _metricsOpen;
        public bool MetricsPanelOpen { get => _metricsOpen; set => Set(ref _metricsOpen, value); }

        private bool _loupe = true;
        public bool LoupeOn { get => _loupe; set => Set(ref _loupe, value); }

        private static string DatasetDir() => Path.Combine(AppContext.BaseDirectory, "dataset");
        private static string HazefreeDir() => Path.Combine(AppContext.BaseDirectory, "hazefree");

        private void LoadDataset()
        {
            if (!Directory.Exists(DatasetDir())) return;
            var gtFiles = Directory.Exists(HazefreeDir())
                ? Directory.GetFiles(HazefreeDir(), "*.*").Where(IsImage).ToList()
                : new List<string>();

            foreach (var hazy in Directory.GetFiles(DatasetDir(), "*.*").Where(IsImage).OrderBy(x => x))
            {
                string key = Key(Path.GetFileNameWithoutExtension(hazy));
                string? gt = gtFiles.FirstOrDefault(g => Key(Path.GetFileNameWithoutExtension(g)) == key);
                Frames.Add(new FrameItem(hazy, gt));
            }
            if (Frames.Count > 0) OpenFrame(Frames[0]);
        }

        private static bool IsImage(string p)
        {
            var e = Path.GetExtension(p).ToLowerInvariant();
            return e is ".jpg" or ".jpeg" or ".png" or ".bmp" or ".tif" or ".tiff";
        }

        private static string Key(string nameNoExt)
        {
            var s = nameNoExt.ToLowerInvariant();
            foreach (var tok in new[] { "hazy", "haze", "groundtruth", "ground_truth", "gt", "clean", "clear", "original", "input" })
                s = s.Replace(tok, "");
            return s.Trim('_', '-', ' ', '.');
        }

        private FrameItem? _frame;
        public FrameItem? CurrentFrame
        {
            get => _frame;
            set { if (value != null && Set(ref _frame, value)) OpenFrame(value); }
        }

        public void OpenFrame(FrameItem f)
        {
            _input?.Dispose();
            _input = new Image<Bgr, byte>(f.HazyPath);
            InputImage = Img.ToBitmap(_input.Mat);
            ResultImage = null;
            _lastResult?.Dispose(); _lastResult = null;

            _gt?.Dispose(); _gt = null; GtImage = null;
            if (f.GtPath != null) { _gt = new Image<Bgr, byte>(f.GtPath); GtImage = Img.ToBitmap(_gt.Mat); }

            FileInfo = $"{Path.GetFileName(f.HazyPath)}  {_input.Width}×{_input.Height}";
            Verdict = "Нажмите «Обработать»";
            Metrics.Clear();
            ProcessCommand.Refresh(); AutoQuickCommand.Refresh(); AutoThoroughCommand.Refresh();
            AutoBestCommand.Refresh(); RunAllCommand.Refresh();
        }

        private void StepFrame(int delta)
        {
            if (Frames.Count == 0 || _frame == null) return;
            int i = (Frames.IndexOf(_frame) + delta + Frames.Count) % Frames.Count;
            CurrentFrame = Frames[i];
        }

        public void LoadHazy(string path) => OpenFrame(new FrameItem(path, null));

        public void LoadGt(string path)
        {
            _gt?.Dispose();
            _gt = new Image<Bgr, byte>(path);
            GtImage = Img.ToBitmap(_gt.Mat);
        }

        public bool SaveResult(string path)
        {
            if (_lastResult == null) return false;
            using var disp = new Mat();
            _lastResult.ConvertTo(disp, DepthType.Cv8U, 255.0);
            CvInvoke.Imwrite(path, disp);
            return true;
        }

        private static Image<Bgr, byte> Downscale(Image<Bgr, byte> img, int maxDim)
        {
            double s = Math.Min(1.0, (double)maxDim / Math.Max(img.Width, img.Height));
            return s >= 1.0 ? img.Clone() : img.Resize((int)(img.Width * s), (int)(img.Height * s), Inter.Area);
        }

        public void Dispose()
        {
            _liveCts?.Cancel();
            _cts?.Cancel();
            _input?.Dispose(); _gt?.Dispose(); _lastResult?.Dispose();
        }
    }

    public sealed class MethodGroup
    {
        public string Title { get; }
        public IReadOnlyList<MethodItem> Items { get; }
        public string Count => Items.Count.ToString();
        public MethodGroup(string title, IReadOnlyList<MethodItem> items) { Title = title; Items = items; }
    }

    public sealed class FrameItem
    {
        public string HazyPath { get; }
        public string? GtPath { get; }
        public string Label => Path.GetFileNameWithoutExtension(HazyPath)[..Math.Min(2, Path.GetFileNameWithoutExtension(HazyPath).Length)];
        public bool HasGt => GtPath != null;
        public BitmapSource Thumb { get; }

        public FrameItem(string hazy, string? gt)
        {
            HazyPath = hazy; GtPath = gt;
            using var im = new Image<Bgr, byte>(hazy);
            double s = 96.0 / Math.Max(im.Width, im.Height);
            using var sm = im.Resize(Math.Max(1, (int)(im.Width * s)), Math.Max(1, (int)(im.Height * s)), Inter.Area);
            Thumb = Img.ToBitmap(sm.Mat);
        }
    }
}
