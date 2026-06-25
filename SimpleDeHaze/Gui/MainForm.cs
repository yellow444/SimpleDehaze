using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Windows.Forms;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

using SimpleDeHaze.Methods;

namespace SimpleDeHaze.Gui
{
    /// <summary>
    /// Окно дехейзинга: карусель датасета, три синхронных вида (вход с дымкой | результат |
    /// эталон без дымки), загрузка по клику на поля, выбор метода, параметры, авто-подбор,
    /// PSNR результата против эталона.
    /// </summary>
    public sealed class MainForm : Form
    {
        private Image<Bgr, byte>? _input;     // с дымкой
        private Image<Bgr, byte>? _gt;        // эталон (без дымки), опц.
        private Mat? _lastResult;

        private readonly ViewState _view = new();
        private readonly ZoomImageView _hazyView, _resultView, _gtView;

        private readonly ComboBox _methodCombo = new() { DropDownStyle = ComboBoxStyle.DropDownList, Width = 230 };
        private readonly TextBox _descBox = new()
        {
            Dock = DockStyle.Top, Height = 150, Multiline = true, ReadOnly = true, WordWrap = true,
            ScrollBars = ScrollBars.Vertical, BackColor = Color.FromArgb(248, 248, 248), BorderStyle = BorderStyle.None
        };
        private readonly TextBox _metricsBox = new()
        {
            Dock = DockStyle.Top, Height = 168, Multiline = true, ReadOnly = true, WordWrap = false,
            ScrollBars = ScrollBars.Vertical, BackColor = Color.FromArgb(245, 248, 245),
            BorderStyle = BorderStyle.None, Font = new Font(FontFamily.GenericMonospace, 8.25f),
            Text = "Метрики появятся после 'Вычислить'."
        };
        private readonly FlowLayoutPanel _paramsPanel = new() { Dock = DockStyle.Fill, FlowDirection = FlowDirection.TopDown, WrapContents = false, AutoScroll = true };
        private readonly FlowLayoutPanel _carousel = new() { Dock = DockStyle.Fill, FlowDirection = FlowDirection.LeftToRight, WrapContents = false, AutoScroll = true };
        private readonly Button _runBtn = new() { Text = "Вычислить", Width = 100 };
        private readonly Button _autoBestBtn = new() { Text = "Авто-лучший", Width = 118 };
        private readonly CheckBox _autoLoadCheck = new() { Text = "авто-лучший при загрузке", AutoSize = true, Padding = new Padding(4, 11, 0, 0) };
        private readonly Button _autoBtn = new() { Text = "Авто-параметры", Width = 120 };
        private readonly Button _tuneBtn = new() { Text = "Тщательный подбор", Width = 140 };
        private readonly CheckBox _keepColorCheck = new() { Text = "цвет >= 1.2", AutoSize = true, Padding = new Padding(4, 11, 0, 0) };
        private readonly Button _benchBtn = new() { Text = "Прогнать все", Width = 102 };
        private readonly Button _defBtn = new() { Text = "По умолчанию", Width = 110 };
        private readonly Button _fitBtn = new() { Text = "Вписать", Width = 78 };
        private readonly Button _saveBtn = new() { Text = "Сохранить...", Width = 96, Enabled = false };
        private readonly Label _status = new() { AutoSize = true, Padding = new Padding(10, 9, 0, 0), Text = "Выберите изображение в карусели или кликните по полю" };

        private readonly DataGridView _benchGrid = new()
        {
            Dock = DockStyle.Fill, ReadOnly = true, AllowUserToAddRows = false, AllowUserToDeleteRows = false,
            AllowUserToResizeRows = false, RowHeadersVisible = false, MultiSelect = false, AutoGenerateColumns = false,
            SelectionMode = DataGridViewSelectionMode.FullRowSelect, BackgroundColor = Color.White,
            BorderStyle = BorderStyle.None, Font = new Font(FontFamily.GenericSansSerif, 8f),
            ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.DisableResizing
        };
        private readonly Button _csvBtn = new() { Text = "Экспорт CSV...", Dock = DockStyle.Bottom, Height = 28, Enabled = false };

        private readonly Font _gridBold = new(FontFamily.GenericSansSerif, 8f, FontStyle.Bold);
        private List<BenchRow>? _lastBench;
        private System.Threading.CancellationTokenSource? _tuneCts;
        private bool _busy;

        private readonly Dictionary<string, double> _values = new();
        private readonly Dictionary<string, TrackBar> _bars = new();
        private readonly Dictionary<string, Action> _updaters = new();
        private readonly Dictionary<string, ParamDef> _defs = new();
        private readonly List<(string hazy, string? gt)> _entries = new();

        public MainForm(string? initialFile = null)
        {
            Text = "SimpleDeHaze - дехейзинг";
            Width = 1720; Height = 880; StartPosition = FormStartPosition.CenterScreen;

            _hazyView = new ZoomImageView(_view) { Dock = DockStyle.Fill, EmptyHint = "Кликните, чтобы загрузить изображение С ДЫМКОЙ" };
            _resultView = new ZoomImageView(_view) { Dock = DockStyle.Fill, EmptyHint = "Результат появится здесь" };
            _gtView = new ZoomImageView(_view) { Dock = DockStyle.Fill, EmptyHint = "Кликните, чтобы загрузить ЭТАЛОН (без дымки)" };
            _hazyView.Clicked += () => { if (_busy) return; if (LoadDialog(DatasetDir(), out var f)) LoadHazy(f); FitAll(); AutoIfEnabled(); };
            _gtView.Clicked += () => { if (_busy) return; if (LoadDialog(HazefreeDir(), out var f)) LoadGt(f); FitAll(); };

            foreach (var m in MethodRegistry.All) _methodCombo.Items.Add(m.Name);
            _methodCombo.SelectedIndexChanged += (_, _) => RebuildParams();
            _runBtn.Click += async (_, _) => await RunAsync();
            _autoBestBtn.Click += async (_, _) =>
            {
                if (_tuneCts != null) { _tuneCts.Cancel(); _autoBestBtn.Text = "Останавливаю..."; _autoBestBtn.Enabled = false; }
                else await AutoBestAsync();
            };
            _autoBtn.Click += async (_, _) => await AutoAsync();
            _tuneBtn.Click += async (_, _) =>
            {
                if (_tuneCts != null) { _tuneCts.Cancel(); _tuneBtn.Text = "Останавливаю..."; _tuneBtn.Enabled = false; }
                else await TuneThoroughAsync();
            };
            _benchBtn.Click += async (_, _) => await RunAllAsync();
            _defBtn.Click += (_, _) => RebuildParams();
            _fitBtn.Click += (_, _) => FitAll();
            _saveBtn.Click += (_, _) => SaveResult();
            _csvBtn.Click += (_, _) => ExportCsv();
            _benchGrid.CellDoubleClick += (_, e) => { if (e.RowIndex >= 0) SelectMethodFromGrid(e.RowIndex); };
            BuildBenchColumns();

            var top = new FlowLayoutPanel { Dock = DockStyle.Top, Height = 44, Padding = new Padding(6, 7, 0, 0) };
            top.Controls.Add(new Label { Text = "Метод:", AutoSize = true, Padding = new Padding(6, 9, 4, 0) });
            top.Controls.Add(_methodCombo);
            top.Controls.Add(_runBtn);
            top.Controls.Add(_autoBestBtn);
            top.Controls.Add(_autoLoadCheck);
            top.Controls.Add(_autoBtn);
            top.Controls.Add(_tuneBtn);
            top.Controls.Add(_keepColorCheck);
            top.Controls.Add(_benchBtn);
            top.Controls.Add(_defBtn);
            top.Controls.Add(_fitBtn);
            top.Controls.Add(_saveBtn);
            top.Controls.Add(_status);

            var left = new Panel { Dock = DockStyle.Left, Width = 366 };
            left.Controls.Add(_paramsPanel);
            left.Controls.Add(new Label { Text = "Параметры", Dock = DockStyle.Top, Height = 24, Font = new Font(Font, FontStyle.Bold), Padding = new Padding(6, 4, 0, 0) });
            left.Controls.Add(_metricsBox);
            left.Controls.Add(new Label { Text = "Метрики качества", Dock = DockStyle.Top, Height = 22, Font = new Font(Font, FontStyle.Bold), Padding = new Padding(6, 3, 0, 0) });
            left.Controls.Add(_descBox);
            left.Controls.Add(new Label { Text = "Описание метода", Dock = DockStyle.Top, Height = 22, Font = new Font(Font, FontStyle.Bold), Padding = new Padding(6, 3, 0, 0) });

            var carouselPanel = new Panel { Dock = DockStyle.Bottom, Height = 132 };
            carouselPanel.Controls.Add(_carousel);
            carouselPanel.Controls.Add(new Label { Text = "Датасет (клик - выбрать пару 'с дымкой / эталон'):", Dock = DockStyle.Top, Height = 18, ForeColor = Color.Gray, Padding = new Padding(4, 2, 0, 0) });

            var tablePanel = new Panel { Dock = DockStyle.Left, Width = 548 };
            tablePanel.Controls.Add(_benchGrid);
            tablePanel.Controls.Add(_csvBtn);
            tablePanel.Controls.Add(new Label
            {
                Text = "Сравнение методов - кнопка 'Прогнать все' (умолч. + авто).\nКлик по заголовку - сортировка. Двойной клик по строке - выбрать метод.",
                Dock = DockStyle.Top, Height = 34, ForeColor = Color.Gray, Padding = new Padding(4, 2, 0, 0)
            });

            var center = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 3, RowCount = 1 };
            for (int i = 0; i < 3; i++) center.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100f / 3));
            center.Controls.Add(Cell("Вход (с дымкой) - клик, чтобы загрузить", _hazyView), 0, 0);
            center.Controls.Add(Cell("Результат  (колесо - зум, перетаскивание - панорама)", _resultView), 1, 0);
            center.Controls.Add(Cell("Эталон (без дымки) - клик, чтобы загрузить", _gtView), 2, 0);

            // порядок важен: фон (center, Fill) первым -> края резервируются в обратном z-порядке.
            // Слева направо получаем: параметры (left) - таблица (tablePanel) - изображения (center).
            Controls.Add(center);
            Controls.Add(tablePanel);
            Controls.Add(left);
            Controls.Add(carouselPanel);
            Controls.Add(top);

            if (_methodCombo.Items.Count > 0) _methodCombo.SelectedIndex = 0;
            BuildCarousel();
            if (initialFile != null && File.Exists(initialFile)) { LoadHazy(initialFile); TryAutoGt(initialFile); }
            else if (_entries.Count > 0) { LoadHazy(_entries[0].hazy); if (_entries[0].gt != null) LoadGt(_entries[0].gt!); }

            Load += (_, _) => { FitAll(); LoadThumbsAsync(); };
            FormClosed += (_, _) => { _input?.Dispose(); _gt?.Dispose(); _lastResult?.Dispose(); };
        }

        private static Panel Cell(string title, ZoomImageView v)
        {
            var pnl = new Panel { Dock = DockStyle.Fill };
            pnl.Controls.Add(v);
            pnl.Controls.Add(new Label { Text = title, Dock = DockStyle.Top, Height = 20, ForeColor = Color.Gray, Padding = new Padding(4, 2, 0, 0) });
            return pnl;
        }

        private IDeHazeMethod Current => MethodRegistry.All[_methodCombo.SelectedIndex];

        // ---------- датасет / пары ----------

        private static string DatasetDir() => Path.Combine(AppContext.BaseDirectory, "dataset");
        private static string HazefreeDir() => Path.Combine(AppContext.BaseDirectory, "hazefree");

        private static bool IsImage(string p)
        {
            var e = Path.GetExtension(p).ToLowerInvariant();
            return e is ".jpg" or ".jpeg" or ".png" or ".bmp" or ".tif" or ".tiff";
        }

        /// <summary>Нормализованный ключ имени: убираем маркеры hazy/GT и разделители -> 'начальная часть'.</summary>
        private static string Key(string nameNoExt)
        {
            var s = nameNoExt.ToLowerInvariant();
            foreach (var tok in new[] { "hazy", "haze", "groundtruth", "ground_truth", "gt", "clean", "clear", "original", "input" })
                s = s.Replace(tok, "");
            return s.Trim('_', '-', ' ', '.');
        }

        private void BuildCarousel()
        {
            _entries.Clear();
            _carousel.Controls.Clear();
            if (!Directory.Exists(DatasetDir())) return;

            var gtFiles = Directory.Exists(HazefreeDir()) ? Directory.GetFiles(HazefreeDir(), "*.*").Where(IsImage).ToList() : new List<string>();
            foreach (var hazy in Directory.GetFiles(DatasetDir(), "*.*").Where(IsImage).OrderBy(x => x))
            {
                string key = Key(Path.GetFileNameWithoutExtension(hazy));
                string? gt = gtFiles.FirstOrDefault(g => Key(Path.GetFileNameWithoutExtension(g)) == key);
                _entries.Add((hazy, gt));

                var pb = new PictureBox
                {
                    Width = 84, Height = 84, SizeMode = PictureBoxSizeMode.Zoom, Cursor = Cursors.Hand,
                    BorderStyle = BorderStyle.FixedSingle, Margin = new Padding(3), Tag = (hazy, gt),
                    BackColor = gt != null ? Color.FromArgb(225, 240, 225) : Color.FromArgb(245, 235, 225)
                };
                int idx = _entries.Count - 1;
                pb.Click += (_, _) => { if (_busy) return; var e = _entries[idx]; LoadHazy(e.hazy); if (e.gt != null) LoadGt(e.gt); else ClearGt(); FitAll(); AutoIfEnabled(); };
                var tip = new ToolTip();
                tip.SetToolTip(pb, Path.GetFileName(hazy) + (gt != null ? "\n+ эталон: " + Path.GetFileName(gt) : "\n(без эталона)"));
                _carousel.Controls.Add(pb);
            }
        }

        private void LoadThumbsAsync()
        {
            var jobs = _carousel.Controls.OfType<PictureBox>()
                .Select(pb => (box: pb, hazy: (((string, string?))pb.Tag!).Item1)).ToArray();
            System.Threading.Tasks.Task.Run(() =>
            {
                foreach (var (box, hazy) in jobs)
                {
                    if (IsDisposed) return;
                    try
                    {
                        using var im = new Image<Bgr, byte>(hazy);
                        double s = 80.0 / Math.Max(im.Width, im.Height);
                        using var sm = im.Resize(Math.Max(1, (int)(im.Width * s)), Math.Max(1, (int)(im.Height * s)), Inter.Area);
                        var bmp = MatToBitmap(sm.Mat);
                        if (!IsDisposed) BeginInvoke(() => { box.Image?.Dispose(); box.Image = bmp; });
                    }
                    catch { /* пропускаем нечитаемые */ }
                }
            });
        }

        // ---------- загрузка ----------

        private static bool LoadDialog(string initialDir, out string file)
        {
            file = "";
            using var ofd = new OpenFileDialog
            {
                Filter = "Изображения|*.jpg;*.jpeg;*.png;*.bmp;*.tif|Все файлы|*.*",
                InitialDirectory = Directory.Exists(initialDir) ? initialDir : AppContext.BaseDirectory
            };
            if (ofd.ShowDialog() != DialogResult.OK) return false;
            file = ofd.FileName; return true;
        }

        private void LoadHazy(string path)
        {
            try
            {
                _input?.Dispose();
                _input = new Image<Bgr, byte>(path);
                _hazyView.Image = MatToBitmap(_input.Mat);
                _resultView.Image = null;
                _lastResult?.Dispose(); _lastResult = null; _saveBtn.Enabled = false;
                _metricsBox.Text = "Метрики появятся после 'Вычислить'.";
                _status.Text = $"{Path.GetFileName(path)}  {_input.Width}x{_input.Height}";
            }
            catch (Exception ex) { MessageBox.Show(this, ex.Message, "Не удалось открыть"); }
        }

        private void LoadGt(string path)
        {
            try { _gt?.Dispose(); _gt = new Image<Bgr, byte>(path); _gtView.Image = MatToBitmap(_gt.Mat); }
            catch (Exception ex) { MessageBox.Show(this, ex.Message, "Не удалось открыть эталон"); }
        }

        private void ClearGt() { _gt?.Dispose(); _gt = null; _gtView.Image = null; }

        private void TryAutoGt(string hazyPath)
        {
            if (!Directory.Exists(HazefreeDir())) return;
            string key = Key(Path.GetFileNameWithoutExtension(hazyPath));
            var gt = Directory.GetFiles(HazefreeDir(), "*.*").Where(IsImage).FirstOrDefault(g => Key(Path.GetFileNameWithoutExtension(g)) == key);
            if (gt != null) LoadGt(gt); else ClearGt();
        }

        private void FitAll() => _hazyView.Fit();   // общий ViewState -> подстраиваются все три

        /// <summary>Если включено 'авто-лучший при загрузке' - на новом кадре сразу запустить авто-подбор лучшего метода.</summary>
        private void AutoIfEnabled() { if (_autoLoadCheck.Checked && !_busy && _input != null) _ = AutoBestAsync(); }

        // ---------- параметры (как раньше) ----------

        private void RebuildParams()
        {
            _descBox.Text = Current.Description;
            _paramsPanel.Controls.Clear();
            _values.Clear(); _bars.Clear(); _updaters.Clear(); _defs.Clear();
            foreach (var d in Current.Parameters)
            {
                _values[d.Key] = d.Default; _defs[d.Key] = d;
                var box = new Panel { Width = 310, Height = 48, Margin = new Padding(4, 4, 4, 0) };
                var label = new Label { Dock = DockStyle.Top, Height = 18, AutoEllipsis = true };
                var bar = new TrackBar { Dock = DockStyle.Top, Minimum = 0, Maximum = 1000, TickStyle = TickStyle.None, Height = 28 };
                void Update()
                {
                    double v = PosToValue(d, bar.Value);
                    _values[d.Key] = v;
                    label.Text = $"{d.Label} = {(d.IsInt ? v.ToString("0") : v.ToString("0.#####"))}" + (d.Search ? "   [авто]" : "");
                }
                bar.Value = ValueToPos(d, d.Default);
                bar.Scroll += (_, _) => Update();
                Update();
                box.Controls.Add(bar); box.Controls.Add(label);
                _paramsPanel.Controls.Add(box);
                _bars[d.Key] = bar; _updaters[d.Key] = Update;
            }
        }

        private void ApplyValues(IReadOnlyDictionary<string, double> vals)
        {
            foreach (var kv in vals)
                if (_bars.TryGetValue(kv.Key, out var bar) && _defs.TryGetValue(kv.Key, out var d)) { bar.Value = ValueToPos(d, kv.Value); _updaters[kv.Key](); }
        }

        private static double PosToValue(ParamDef d, int pos)
        {
            double frac = pos / 1000.0;
            double v = d.Log ? d.Min * Math.Pow(d.Max / d.Min, frac) : d.Min + frac * (d.Max - d.Min);
            if (d.Step > 0 && !d.Log) v = Math.Round(v / d.Step) * d.Step;
            if (d.IsInt) v = Math.Round(v);
            return Math.Clamp(v, d.Min, d.Max);
        }

        private static int ValueToPos(ParamDef d, double v)
        {
            v = Math.Clamp(v, d.Min, d.Max);
            double frac = d.Log ? Math.Log(v / d.Min) / Math.Log(d.Max / d.Min) : (v - d.Min) / (d.Max - d.Min);
            return (int)Math.Round(Math.Clamp(frac, 0, 1) * 1000);
        }

        // ---------- запуск ----------

        private async Task RunAsync()
        {
            if (_input is null) { MessageBox.Show(this, "Сначала выберите изображение с дымкой."); return; }
            SetBusy(true, $"Вычисление: {Current.Name}...");
            try { await ComputeAndShow(Current, new Dictionary<string, double>(_values)); }
            catch (Exception ex) { MessageBox.Show(this, ex.ToString(), "Ошибка вычисления"); _status.Text = "Ошибка"; }
            finally { SetBusy(false); }
        }

        /// <summary>Прогнать метод на полном кадре, показать результат + метрики. Не управляет SetBusy (это делает вызывающий).</summary>
        private async Task ComputeAndShow(IDeHazeMethod method, Dictionary<string, double> args, string prefix = "")
        {
            var input = _input!.Clone();
            try
            {
                var sw = Stopwatch.StartNew();
                Mat result = await Task.Run(() => method.Process(input, args));
                sw.Stop();
                using (var disp = new Mat()) { result.ConvertTo(disp, DepthType.Cv8U, 255.0); _resultView.Image = MatToBitmap(disp); }
                _lastResult?.Dispose(); _lastResult = result; _saveBtn.Enabled = true;
                var rep = Metrics.Evaluate(result, _gt?.Mat, _input!.Mat);
                _metricsBox.Text = rep.Format();
                _status.Text = $"{prefix}{method.Name}: {sw.ElapsedMilliseconds} мс  ({result.Width}x{result.Height}){rep.StatusSuffix()}";
            }
            finally { input.Dispose(); }
        }

        private async Task AutoBestAsync()
        {
            if (_input is null) { MessageBox.Show(this, "Сначала выберите изображение."); return; }
            var input = _input.Clone();
            _tuneCts = new System.Threading.CancellationTokenSource();
            var token = _tuneCts.Token;
            SetBusy(true, "Авто-лучший: сканирую методы...");
            _autoBestBtn.Enabled = true; _autoBestBtn.Text = "Стоп";
            try
            {
                var (best, tuned, _) = await Task.Run(() => AutoTuner.PickBest(MethodRegistry.All, input,
                    msg => { if (!IsDisposed) BeginInvoke(() => _status.Text = msg); },
                    () => token.IsCancellationRequested));
                if (token.IsCancellationRequested) { _status.Text = "Авто-лучший: прервано"; return; }

                int mi = MethodRegistry.All.ToList().FindIndex(m => m.Name == best.Name);
                if (mi >= 0) _methodCombo.SelectedIndex = mi;   // RebuildParams -> дефолты
                ApplyValues(tuned);                              // проставить подобранные
                await ComputeAndShow(best, tuned, "Авто-лучший -> ");
            }
            catch (Exception ex) { MessageBox.Show(this, ex.ToString(), "Ошибка авто-подбора"); }
            finally
            {
                _tuneCts.Dispose(); _tuneCts = null;
                _autoBestBtn.Text = "Авто-лучший";
                input.Dispose(); SetBusy(false);
            }
        }

        private async Task AutoAsync()
        {
            if (_input is null) { MessageBox.Show(this, "Сначала выберите изображение."); return; }
            var method = Current;
            if (!method.Parameters.Any(p => p.Search)) { _status.Text = "У метода нет параметров для авто-подбора"; return; }
            var cur = new Dictionary<string, double>(_values);
            var input = _input.Clone();
            SetBusy(true, "Подбор параметров по метрике...");
            try
            {
                var best = await Task.Run(() => AutoTuner.Optimize(method, input, cur));
                ApplyValues(best);
                await ComputeAndShow(method, best, "Авто-параметры -> ");
            }
            catch (Exception ex) { MessageBox.Show(this, ex.ToString(), "Ошибка подбора"); }
            finally { input.Dispose(); SetBusy(false); }
        }

        private async Task TuneThoroughAsync()
        {
            if (_input is null) { MessageBox.Show(this, "Сначала выберите изображение."); return; }
            var method = Current;
            if (method.Parameters.Count == 0) { _status.Text = "У метода нет параметров для подбора"; return; }
            var cur = new Dictionary<string, double>(_values);
            var input = _input.Clone();
            double minColor = _keepColorCheck.Checked ? 1.2 : 0.0;   // 'не гасить цвет' -> пол насыщенности
            _tuneCts = new System.Threading.CancellationTokenSource();
            var token = _tuneCts.Token;
            SetBusy(true, "Тщательный подбор" + (minColor > 0 ? " (цель: не гасить цвет)" : "") + "... нажмите 'Стоп', чтобы прервать");
            _tuneBtn.Enabled = true; _tuneBtn.Text = "Стоп";          // оставляем активной - для отмены
            try
            {
                int last = 0;
                var best = await Task.Run(() => AutoTuner.OptimizeThorough(method, input, cur, minColor,
                    (e, s) => { if (e - last < 3 || IsDisposed) return; last = e; BeginInvoke(() => _status.Text = $"Тщательный подбор... попыток {e}, лучший скор {s:F1}"); },
                    () => token.IsCancellationRequested));
                ApplyValues(best);
                await ComputeAndShow(method, best, token.IsCancellationRequested ? "Прервано -> " : "Тщательный подбор -> ");
            }
            catch (Exception ex) { MessageBox.Show(this, ex.ToString(), "Ошибка подбора"); }
            finally
            {
                _tuneCts.Dispose(); _tuneCts = null;
                _tuneBtn.Text = "Тщательный подбор";
                input.Dispose(); SetBusy(false);
            }
        }

        // ---------- прогон всех методов (бенчмарк) ----------

        internal readonly record struct BenchRow(string Name, string Mode, bool Ok, Metrics.Report Rep, long Ms, string? Error);

        private void BuildBenchColumns()
        {
            _benchGrid.Columns.Clear();
            AddBenchCol("name", "Метод", 134, false, null, typeof(string));
            AddBenchCol("mode", "Парам.", 50, false, null, typeof(string));
            AddBenchCol("score", "Оценка", 50, true, "0", typeof(double));
            AddBenchCol("psnrRaw", "сырой", 52, true, "0.00", typeof(double));
            AddBenchCol("psnr", "совмещ", 58, true, "0.00", typeof(double));
            AddBenchCol("ssim", "SSIM", 46, true, "0.000", typeof(double));
            AddBenchCol("haze", "дымка", 50, true, "0", typeof(double));
            AddBenchCol("color", "цветx", 42, true, "0.00", typeof(double));
            AddBenchCol("ms", "мс", 40, true, "0", typeof(double));
        }

        private void AddBenchCol(string name, string header, int width, bool rightAlign, string? format, Type valueType)
        {
            var c = new DataGridViewTextBoxColumn
            {
                Name = name, HeaderText = header, Width = width, ValueType = valueType,
                SortMode = DataGridViewColumnSortMode.Automatic
            };
            if (format != null) c.DefaultCellStyle.Format = format;
            if (rightAlign) c.DefaultCellStyle.Alignment = DataGridViewContentAlignment.MiddleRight;
            _benchGrid.Columns.Add(c);
        }

        private async Task RunAllAsync()
        {
            if (_input is null) { MessageBox.Show(this, "Сначала выберите изображение с дымкой."); return; }

            // прогон на уменьшенной копии - иначе 25x2 тяжёлых методов на полном кадре это минуты
            var img = Downscale(_input, 800);
            Mat? gt = null;
            if (_gt != null) { gt = new Mat(); CvInvoke.Resize(_gt.Mat, gt, img.Size, 0, 0, Inter.Area); }
            var methods = MethodRegistry.All;
            int total = methods.Count;

            _benchGrid.Rows.Clear();
            SetBusy(true, "Прогон всех методов...");
            try
            {
                var rows = await Task.Run(() => RunBenchCore(methods, img, gt,
                    d => { if (!IsDisposed) BeginInvoke(() => _status.Text = $"Прогон методов... {d}/{total}"); }));

                foreach (var r in rows.OrderByDescending(x => x.Ok ? x.Rep.Score : double.NegativeInfinity))
                    AddBenchRow(r);

                _lastBench = rows;
                _csvBtn.Enabled = rows.Count > 0;
                int okCount = rows.Count(x => x.Ok);
                _status.Text = $"Готово: {rows.Count} строк ({okCount} ок), кадр {img.Width}x{img.Height}" +
                               (gt == null ? ", без эталона" : "") + ". Клик по заголовку - сортировка, двойной клик - выбрать метод.";
            }
            catch (Exception ex) { MessageBox.Show(this, ex.ToString(), "Ошибка прогона"); _status.Text = "Ошибка прогона"; }
            finally { img.Dispose(); gt?.Dispose(); SetBusy(false); }
        }

        /// <summary>Ядро прогона: каждый метод с параметрами по умолчанию и (если есть искомые) с авто-подбором.</summary>
        private static List<BenchRow> RunBenchCore(IReadOnlyList<IDeHazeMethod> methods, Image<Bgr, byte> img, Mat? gt, Action<int>? progress)
        {
            // прогрев: первый вызов несёт JIT всего конвейера + инициализацию CUDA-контекста
            // (в сыром прогоне это давало первому GPU-методу ~2 с вместо ~0.2 с) - иначе тайминги искажены.
            foreach (var w in methods.Take(2))
                try { using (w.Process(img, w.Parameters.ToDictionary(x => x.Key, x => x.Default))) { } } catch { }

            var rows = new List<BenchRow>();
            int done = 0;
            foreach (var m in methods)
            {
                var def = m.Parameters.ToDictionary(x => x.Key, x => x.Default);
                rows.Add(Bench(m, img, def, gt, "умолч."));
                if (m.Parameters.Any(p => p.Search))
                {
                    Dictionary<string, double> tuned;
                    try { tuned = AutoTuner.Optimize(m, img, def); } catch { tuned = def; }
                    rows.Add(Bench(m, img, tuned, gt, "авто"));
                }
                progress?.Invoke(++done);
            }
            return rows;
        }

        /// <summary>Headless-проверка бенчмарка: синхронно прогнать всё на текущем кадре, заполнить таблицу, вернуть сводку.</summary>
        internal string BenchmarkSelfTest()
        {
            if (_input is null) return "BENCH-NOINPUT";
            using var img = Downscale(_input, 480);
            Mat? gt = null;
            if (_gt != null) { gt = new Mat(); CvInvoke.Resize(_gt.Mat, gt, img.Size, 0, 0, Inter.Area); }
            try
            {
                var rows = RunBenchCore(MethodRegistry.All, img, gt, null);
                _lastBench = rows;
                _benchGrid.Rows.Clear();
                foreach (var r in rows.OrderByDescending(x => x.Ok ? x.Rep.Score : double.NegativeInfinity)) AddBenchRow(r);
                int ok = rows.Count(x => x.Ok);
                var r0 = _benchGrid.Rows.Count > 0 ? _benchGrid.Rows[0] : null;

                var csv = BuildBenchCsv(rows);
                int csvLines = csv.TrimEnd('\n', '\r').Split('\n').Length;
                string csvPath;
                try
                {
                    csvPath = Path.Combine(Environment.CurrentDirectory, "benchmark_test.csv");
                    File.WriteAllText(csvPath, csv, new System.Text.UTF8Encoding(true));
                }
                catch (Exception ex) { csvPath = "(запись не удалась: " + ex.GetType().Name + ")"; }

                return $"BENCH-OK rows={_benchGrid.Rows.Count} ok={ok} cols={_benchGrid.Columns.Count} csvLines={csvLines} " +
                       $"top='{r0?.Cells[0].Value}' score={r0?.Cells[2].Value} rawPSNR={r0?.Cells[3].Value} alnPSNR={r0?.Cells[4].Value} csv={csvPath}";
            }
            finally { gt?.Dispose(); }
        }

        private static BenchRow Bench(IDeHazeMethod m, Image<Bgr, byte> img, IReadOnlyDictionary<string, double> p, Mat? gt, string mode)
        {
            var sw = Stopwatch.StartNew();
            try
            {
                using var res = m.Process(img, p);
                sw.Stop();
                var rep = Metrics.Evaluate(res, gt, img.Mat);
                return new BenchRow(m.Name, mode, true, rep, sw.ElapsedMilliseconds, null);
            }
            catch (Exception ex)
            {
                sw.Stop();
                return new BenchRow(m.Name, mode, false, default, sw.ElapsedMilliseconds,
                    (ex.GetType().Name + ": " + ex.Message).Replace(';', ',').Replace('\r', ' ').Replace('\n', ' '));
            }
        }

        private void AddBenchRow(BenchRow r)
        {
            object?[] vals = r.Ok
                ? new object?[]
                {
                    r.Name, r.Mode, Math.Round(r.Rep.Score),
                    r.Rep.HasRef ? r.Rep.Psnr : null,
                    r.Rep.HasRef ? r.Rep.PsnrAligned : null,
                    r.Rep.HasRef ? r.Rep.SsimAligned : null,
                    Math.Round(r.Rep.HazeRemoved * 100), r.Rep.ColorRatio, (double)r.Ms
                }
                : new object?[] { $"{r.Name} - FAIL", r.Mode, null, null, null, null, null, null, (double)r.Ms };

            int i = _benchGrid.Rows.Add(vals!);
            var row = _benchGrid.Rows[i];
            row.Tag = r.Name;
            if (!r.Ok) { row.DefaultCellStyle.ForeColor = Color.Gray; row.Cells[0].ToolTipText = r.Error ?? ""; return; }
            if (r.Mode == "авто") row.Cells[1].Style.BackColor = Color.FromArgb(225, 235, 250);
            row.Cells[2].Style.BackColor = Lerp(Color.White, Color.FromArgb(150, 215, 150), Math.Clamp(r.Rep.Score / 70.0, 0, 1));
            row.Cells[2].Style.Font = _gridBold;
            if (r.Rep.ColorRatio > 1.5) row.Cells[7].Style.ForeColor = Color.Red;
        }

        private void SelectMethodFromGrid(int rowIndex)
        {
            if (_busy) return;
            if (_benchGrid.Rows[rowIndex].Tag is string name)
            {
                int mi = MethodRegistry.All.ToList().FindIndex(m => m.Name == name);
                if (mi >= 0) _methodCombo.SelectedIndex = mi;
            }
        }

        private static Image<Bgr, byte> Downscale(Image<Bgr, byte> img, int maxDim)
        {
            double s = Math.Min(1.0, (double)maxDim / Math.Max(img.Width, img.Height));
            return s >= 1.0 ? img.Clone() : img.Resize((int)(img.Width * s), (int)(img.Height * s), Inter.Area);
        }

        private static Color Lerp(Color a, Color b, double t) => Color.FromArgb(
            (int)(a.R + (b.R - a.R) * t), (int)(a.G + (b.G - a.G) * t), (int)(a.B + (b.B - a.B) * t));

        private void ExportCsv()
        {
            if (_lastBench is null || _lastBench.Count == 0) { _status.Text = "Сначала 'Прогнать все'."; return; }
            using var sfd = new SaveFileDialog { Filter = "CSV|*.csv", FileName = "dehaze_benchmark.csv" };
            if (sfd.ShowDialog(this) != DialogResult.OK) return;
            try
            {
                File.WriteAllText(sfd.FileName, BuildBenchCsv(_lastBench), new System.Text.UTF8Encoding(true)); // BOM -> Excel читает кириллицу
                _status.Text = "Сохранено: " + sfd.FileName;
            }
            catch (Exception ex) { MessageBox.Show(this, ex.Message, "Не удалось сохранить CSV"); }
        }

        /// <summary>Сборка CSV прогона (разделитель ';', числа в инвариантной культуре). Полный набор метрик.</summary>
        internal static string BuildBenchCsv(IEnumerable<BenchRow> rows)
        {
            var ci = System.Globalization.CultureInfo.InvariantCulture;
            static string Num(double v, System.Globalization.CultureInfo c) => double.IsNaN(v) ? "" : v.ToString("0.######", c);

            var sb = new System.Text.StringBuilder();
            sb.AppendLine("Метод;Режим;Оценка;PSNR_сырой;PSNR_совмещ;SSIM_сырой;SSIM_совмещ;Дымка_убрана_%;Контраст_x;Грани_x;Пересвет_%;Цвет_x;мс;OK;Ошибка");
            foreach (var r in rows.OrderByDescending(x => x.Ok ? x.Rep.Score : double.NegativeInfinity))
            {
                var p = r.Rep;
                string Ref(double v) => r.Ok && p.HasRef ? Num(v, ci) : "";
                string[] f = r.Ok
                    ? new[]
                    {
                        r.Name, r.Mode, Num(p.Score, ci), Ref(p.Psnr), Ref(p.PsnrAligned), Ref(p.Ssim), Ref(p.SsimAligned),
                        Num(p.HazeRemoved * 100, ci), Num(p.ContrastGain, ci), Num(p.EdgeGain, ci), Num(p.ClipPct, ci), Num(p.ColorRatio, ci),
                        r.Ms.ToString(ci), "1", ""
                    }
                    : new[] { r.Name, r.Mode, "", "", "", "", "", "", "", "", "", "", r.Ms.ToString(ci), "0", r.Error ?? "" };
                sb.AppendLine(string.Join(";", f));
            }
            return sb.ToString();
        }

        private void SetBusy(bool busy, string? status = null)
        {
            _busy = busy;
            _runBtn.Enabled = _autoBestBtn.Enabled = _autoBtn.Enabled = _tuneBtn.Enabled = _benchBtn.Enabled = _defBtn.Enabled = _methodCombo.Enabled = !busy;
            if (status != null) _status.Text = status;
        }

        private void SaveResult()
        {
            if (_lastResult is null) return;
            using var sfd = new SaveFileDialog { Filter = "PNG|*.png|JPEG|*.jpg|BMP|*.bmp", FileName = "dehazed.png" };
            if (sfd.ShowDialog(this) != DialogResult.OK) return;
            using var disp = new Mat(); _lastResult.ConvertTo(disp, DepthType.Cv8U, 255.0);
            CvInvoke.Imwrite(sfd.FileName, disp);
        }

        private static Bitmap MatToBitmap(Mat mat)
        {
            using var buf = new VectorOfByte();
            CvInvoke.Imencode(".png", mat, buf);
            using var ms = new MemoryStream(buf.ToArray());
            using var tmp = new Bitmap(ms);
            return new Bitmap(tmp);
        }
    }
}
