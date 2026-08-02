using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Forms;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

using SimpleDeHaze.Methods;

namespace SimpleDeHaze.Gui
{
    /// <summary>
    /// «Песочница»: интерактивная игра с изображением - живое превью конвейера фильтров.
    /// Поканальные усиления, яркость/контраст/гамма, насыщенность/оттенок, постеризация (округление),
    /// инверсия, ч/б, порог, размытие/резкость, контуры (Canny), изоляция канала. Превью считается на
    /// уменьшенной копии (быстро), «Применить»/«Сохранить» - на полном размере.
    /// </summary>
    public sealed class PlaygroundForm : Form
    {
        private readonly record struct Spec(double Min, double Max, bool IsInt);

        private readonly Mat _src;           // 8U BGR, владеем
        private readonly Mat _previewSrc;    // уменьшенная копия под живое превью
        private readonly Action<Mat>? _apply;

        private readonly PictureBox _preview = new() { Dock = DockStyle.Fill, SizeMode = PictureBoxSizeMode.Zoom, BackColor = Color.FromArgb(32, 32, 32) };
        private readonly FlowLayoutPanel _controls = new() { Dock = DockStyle.Fill, FlowDirection = FlowDirection.TopDown, WrapContents = false, AutoScroll = true, Padding = new Padding(6, 4, 6, 4) };
        private readonly ComboBox _chCombo = new() { DropDownStyle = ComboBoxStyle.DropDownList, Width = 150 };
        private readonly ComboBox _dehazeCombo = new() { DropDownStyle = ComboBoxStyle.DropDownList, Width = 292 };
        private readonly FlowLayoutPanel _methodParamsPanel = new() { FlowDirection = FlowDirection.TopDown, WrapContents = false, AutoSize = true, AutoSizeMode = AutoSizeMode.GrowAndShrink, Width = 292, Margin = new Padding(0, 2, 0, 4) };
        private readonly ToolTip _tip = new() { AutoPopDelay = 30000, InitialDelay = 250, ReshowDelay = 80, ShowAlways = true };
        private readonly Dictionary<string, TrackBar> _mBars = new();
        private readonly Dictionary<string, ParamDef> _mDefs = new();
        private readonly Dictionary<string, double> _mValues = new();
        private IDeHazeMethod? _dehazeMethod;

        private readonly Dictionary<string, TrackBar> _bars = new();
        private readonly Dictionary<string, Spec> _specs = new();
        private readonly Dictionary<string, Label> _barLabels = new();
        private readonly Dictionary<string, CheckBox> _checks = new();
        private readonly Dictionary<string, string> _labelFmt = new();
        private bool _suppress;

        public PlaygroundForm(Mat src8, Action<Mat>? apply = null)
        {
            _src = src8.Clone();
            _apply = apply;
            _previewSrc = Downscale(_src, 1000);

            Text = "Песочница - фильтры и цвет";
            Width = 1180; Height = 820; StartPosition = FormStartPosition.CenterParent;

            // --- дымка и маски ---
            Header("Дымка и маски");
            _dehazeCombo.Items.Add("Без дехейза");
            for (int i = 0; i < MethodRegistry.All.Count; i++)
            {
                string star = MethodRegistry.Recommended.Contains(MethodRegistry.All[i].Name) ? "★ " : "";
                _dehazeCombo.Items.Add($"{star}{i + 1:00}. {MethodRegistry.All[i].Name}");
            }
            _dehazeCombo.SelectedIndex = 0;
            _dehazeCombo.SelectedIndexChanged += (_, _) => OnDehazeChanged();
            _controls.Controls.Add(new Label { Text = "Убрать дымку (метод + настройки):", AutoSize = true, Margin = new Padding(0, 6, 0, 0) });
            _controls.Controls.Add(_dehazeCombo);
            _controls.Controls.Add(_methodParamsPanel);
            AddSlider("veil", "Вычесть вуаль/фон", 0, 1, 0);
            AddSlider("clahe", "Локальный контраст (CLAHE)", 0, 6, 0);
            AddCheck("autolevels", "Авто-уровни");

            // --- цвет / тон ---
            Header("Цвет и тон");
            AddSlider("gainR", "Канал R (усиление)", 0, 2, 1);
            AddSlider("gainG", "Канал G (усиление)", 0, 2, 1);
            AddSlider("gainB", "Канал B (усиление)", 0, 2, 1);
            AddSlider("bright", "Яркость", -120, 120, 0, true);
            AddSlider("contrast", "Контраст", 0, 2.5, 1);
            AddSlider("gamma", "Гамма", 0.2, 3, 1);
            AddSlider("sat", "Насыщенность", 0, 2.5, 1);
            AddSlider("hue", "Оттенок (сдвиг)", -90, 90, 0, true);
            AddCombo();

            // --- стилизация ---
            Header("Стилизация");
            AddCheck("gray", "Ч/б (оттенки серого)");
            AddCheck("invert", "Инверсия (негатив)");
            AddCheck("poster", "Округление уровней (постеризация)");
            AddSlider("levels", "  уровней", 2, 64, 8, true);

            // --- детали ---
            Header("Детали и контуры");
            AddSlider("blur", "Размытие (σ)", 0, 15, 0);
            AddSlider("sharpen", "Резкость (unsharp)", 0, 3, 0);
            AddCheck("thresh", "Порог (бинаризация)");
            AddSlider("threshLv", "  уровень порога", 0, 255, 128, true);
            AddCheck("edges", "Контуры (Canny, поверх)");
            AddSlider("edgeLo", "  порог Canny нижний", 0, 255, 50, true);
            AddSlider("edgeHi", "  порог Canny верхний", 0, 255, 150, true);

            // --- кнопки ---
            var btns = new FlowLayoutPanel { Dock = DockStyle.Bottom, Height = 40, FlowDirection = FlowDirection.LeftToRight, Padding = new Padding(6, 6, 0, 0) };
            var reset = new Button { Text = "Сброс", Width = 70 };
            var applyBtn = new Button { Text = "Применить к результату", Width = 170, Enabled = _apply != null };
            var save = new Button { Text = "Сохранить...", Width = 100 };
            var close = new Button { Text = "Закрыть", Width = 80 };
            reset.Click += (_, _) => ResetAll();
            applyBtn.Click += (_, _) => { if (_apply != null) { using var full = ApplyPipeline(_src); _apply(full); } };
            save.Click += (_, _) => SaveFull();
            close.Click += (_, _) => Close();
            btns.Controls.AddRange(new Control[] { reset, applyBtn, save, close });

            var leftHost = new Panel { Dock = DockStyle.Left, Width = 312 };
            leftHost.Controls.Add(_controls);
            leftHost.Controls.Add(btns);

            Controls.Add(_preview);
            Controls.Add(leftHost);

            FormClosed += (_, _) => { _preview.Image?.Dispose(); _src.Dispose(); _previewSrc.Dispose(); };
            Recompute();
        }

        // ---------- построение контролов ----------

        private void Header(string text) =>
            _controls.Controls.Add(new Label { Text = text, AutoSize = true, Font = new Font(Font, FontStyle.Bold), Margin = new Padding(0, 8, 0, 2), ForeColor = Color.FromArgb(70, 90, 120) });

        private void AddSlider(string key, string label, double min, double max, double def, bool isInt = false)
        {
            _specs[key] = new Spec(min, max, isInt);
            _labelFmt[key] = label;
            var lbl = new Label { AutoSize = true, Margin = new Padding(0, 4, 0, 0) };
            var bar = new TrackBar { Width = 290, Minimum = 0, Maximum = 1000, TickStyle = TickStyle.None, Height = 30 };
            bar.Value = ValueToPos(key, def);
            bar.Scroll += (_, _) => { UpdateLabel(key); if (!_suppress) Recompute(); };
            _bars[key] = bar; _barLabels[key] = lbl;
            string tip = FilterHelp(key, label);
            _tip.SetToolTip(bar, tip);
            _tip.SetToolTip(lbl, tip);
            _controls.Controls.Add(lbl);
            _controls.Controls.Add(bar);
            UpdateLabel(key);
        }

        /// <summary>Подсказки к собственным фильтрам песочницы (не параметры метода).</summary>
        private static string FilterHelp(string key, string label) => key switch
        {
            "gainR" or "gainG" or "gainB" => label + "\nМножитель яркости канала. 1 = без изменений, <1 приглушить, >1 усилить.",
            "bright" => label + "\nСдвиг яркости всех пикселей (+/−).",
            "contrast" => label + "\nРастяжение контраста вокруг средне-серого. >1 контрастнее, <1 мягче.",
            "gamma" => label + "\nГамма-кривая: <1 темнит тени/высветляет, >1 наоборот. 1 = без изменений.",
            "sat" => label + "\nНасыщенность (HSV·S). 1 = как есть, >1 сочнее, 0 = чб.",
            "hue" => label + "\nСдвиг оттенка (H) по кругу, в градусах.",
            "levels" => label + "\nЧисло уровней при постеризации (округлении). Меньше = грубее ступени.",
            "threshLv" => label + "\nПорог бинаризации (0..255).",
            "blur" => label + "\nσ гауссова размытия.",
            "sharpen" => label + "\nСила нерезкого маскирования (unsharp).",
            "edgeLo" or "edgeHi" => label + "\nНижний/верхний порог Canny для контуров.",
            "veil" => label + "\nВычесть крупномасштабный фон (пелену): кадр − k·размытие + средняя яркость. Снимает вуаль.",
            "clahe" => label + "\nЛокальный контраст (CLAHE) по яркости. 0 = выкл.",
            _ => label,
        };

        private void AddCheck(string key, string label)
        {
            var c = new CheckBox { Text = label, AutoSize = true, Margin = new Padding(0, 4, 0, 0) };
            c.CheckedChanged += (_, _) => { if (!_suppress) Recompute(); };
            _tip.SetToolTip(c, key switch
            {
                "gray" => "Перевести в оттенки серого.",
                "invert" => "Инверсия цветов (негатив).",
                "poster" => "Округлить яркости до N уровней (постеризация).",
                "thresh" => "Бинаризация по порогу.",
                "edges" => "Наложить контуры Canny поверх.",
                "autolevels" => "Поканальное растяжение по перцентилям [1%,99%].",
                _ => label,
            });
            _checks[key] = c;
            _controls.Controls.Add(c);
        }

        private void AddCombo()
        {
            _chCombo.Items.AddRange(new object[] { "Все каналы", "Только R (серым)", "Только G (серым)", "Только B (серым)" });
            _chCombo.SelectedIndex = 0;
            _chCombo.SelectedIndexChanged += (_, _) => { if (!_suppress) Recompute(); };
            _controls.Controls.Add(new Label { Text = "Показ канала", AutoSize = true, Margin = new Padding(0, 6, 0, 0) });
            _controls.Controls.Add(_chCombo);
        }

        /// <summary>Смена метода дехейза: строим панель его параметров и пересчитываем.</summary>
        private void OnDehazeChanged()
        {
            int idx = _dehazeCombo.SelectedIndex;
            _dehazeMethod = idx <= 0 ? null : Methods.MethodRegistry.All[idx - 1];
            RebuildMethodParams();
            if (!_suppress) Recompute();
        }

        /// <summary>Динамические ползунки параметров выбранного метода дехейза (с их дефолтами).</summary>
        private void RebuildMethodParams()
        {
            _methodParamsPanel.Controls.Clear();
            _mBars.Clear(); _mDefs.Clear(); _mValues.Clear();
            if (_dehazeMethod == null) return;
            foreach (var d in _dehazeMethod.Parameters)
            {
                _mValues[d.Key] = d.Default; _mDefs[d.Key] = d;
                var lbl = new Label { AutoSize = true, Margin = new Padding(0, 2, 0, 0), ForeColor = Color.FromArgb(60, 60, 60) };
                var bar = new TrackBar { Width = 286, Minimum = 0, Maximum = 1000, TickStyle = TickStyle.None, Height = 28 };
                bar.Enabled = d.IsEnabled && d.Max > d.Min;
                void Upd()
                {
                    double v = MethodPosToVal(d, bar.Value);
                    _mValues[d.Key] = v;
                    string shown = d.Key == CudaBackend.ParameterKey ? (v >= 0.5 ? "CUDA" : "CPU")
                        : d.IsInt ? v.ToString("0") : v.ToString("0.#####");
                    lbl.Text = $"{d.Label} = {shown}";
                }
                bar.Value = MethodValToPos(d, d.Default);
                bar.Scroll += (_, _) => { Upd(); if (!_suppress) Recompute(); };
                Upd();
                string tip = ParamHelp.For(d);
                _tip.SetToolTip(bar, tip);
                _tip.SetToolTip(lbl, tip);
                _methodParamsPanel.Controls.Add(lbl);
                _methodParamsPanel.Controls.Add(bar);
                _mBars[d.Key] = bar;
            }
        }

        private static double MethodPosToVal(ParamDef d, int pos)
        {
            return d.FromFraction(pos / 1000.0);
        }

        private static int MethodValToPos(ParamDef d, double v)
        {
            return (int)Math.Round(d.ToFraction(v) * 1000.0);
        }

        private void UpdateLabel(string key) => _barLabels[key].Text = $"{_labelFmt[key]} = {FormatVal(key)}";

        private string FormatVal(string key)
        {
            double v = Val(key);
            return _specs[key].IsInt ? v.ToString("0") : v.ToString("0.##");
        }

        private double Val(string key)
        {
            var s = _specs[key];
            double v = s.Min + _bars[key].Value / 1000.0 * (s.Max - s.Min);
            return s.IsInt ? Math.Round(v) : v;
        }

        private int ValueToPos(string key, double v)
        {
            var s = _specs[key];
            if (s.Max <= s.Min) return 0;
            return (int)Math.Round(Math.Clamp((v - s.Min) / (s.Max - s.Min), 0, 1) * 1000);
        }

        private void ResetAll()
        {
            _suppress = true;
            foreach (var (key, bar) in _bars)
            {
                double def = key switch
                {
                    "gainR" or "gainG" or "gainB" or "contrast" or "gamma" or "sat" => 1,
                    "levels" => 8, "threshLv" => 128, "edgeLo" => 50, "edgeHi" => 150,
                    _ => 0
                };
                bar.Value = ValueToPos(key, def);
                UpdateLabel(key);
            }
            foreach (var c in _checks.Values) c.Checked = false;
            _chCombo.SelectedIndex = 0;
            _dehazeCombo.SelectedIndex = 0;
            _suppress = false;
            Recompute();
        }

        /// <summary>Headless-проверка конвейера: прогнать на полном источнике (опц. включив часть фильтров).</summary>
        internal Mat RunPipelineForTest(bool toggle)
        {
            if (toggle)
            {
                _dehazeCombo.SelectedIndex = 1;   // первый метод из реестра (проверяем путь дехейза + его параметры)
                _bars["gainR"].Value = ValueToPos("gainR", 1.5);
                _bars["hue"].Value = ValueToPos("hue", 40);
                _bars["sat"].Value = ValueToPos("sat", 1.6);
                _bars["veil"].Value = ValueToPos("veil", 0.5);
                _bars["clahe"].Value = ValueToPos("clahe", 3.0);
                _checks["autolevels"].Checked = true;
                _checks["invert"].Checked = true;
                _checks["poster"].Checked = true;
                _checks["edges"].Checked = true;
            }
            return ApplyPipeline(_src);
        }

        // ---------- конвейер фильтров ----------

        private void Recompute()
        {
            try
            {
                using var outp = ApplyPipeline(_previewSrc);
                var bmp = ToBitmap(outp);
                _preview.Image?.Dispose();
                _preview.Image = bmp;
            }
            catch { /* во время перетаскивания ползунка возможны промежуточные состояния */ }
        }

        /// <summary>
        /// Применить весь конвейер к 8U BGR; вернуть новый 8U BGR. Полностью исключение-безопасен: при
        /// любом сбое временные Mat (включая 'cur' и массивы из Split) освобождаются - утечек нет даже
        /// если ползунок даёт промежуточное состояние, на котором OpenCV кидает исключение.
        /// </summary>
        private Mat ApplyPipeline(Mat src8)
        {
            // выделить Mat, заполнить его через fill; при исключении в fill - освободить и пробросить.
            static Mat Op(Action<Mat> fill) { var m = new Mat(); try { fill(m); return m; } catch { m.Dispose(); throw; } }

            Mat cur = src8.Clone();
            try
            {
                void Replace(Mat next) { cur.Dispose(); cur = next; }

                // 0) дымка и маски - выбранный метод со СВОИМИ настройками
                if (_dehazeMethod != null)
                {
                    using var img = cur.ToImage<Bgr, byte>();
                    using var f = _dehazeMethod.Process(img, new Dictionary<string, double>(_mValues));
                    Replace(Op(m => f.ConvertTo(m, DepthType.Cv8U, 255.0)));
                }
                double veil = Val("veil");
                if (veil > 0.01)
                {
                    using var bg = new Mat(); CvInvoke.GaussianBlur(cur, bg, new Size(0, 0), 30.0);
                    var mean = CvInvoke.Mean(bg);
                    var m = new Mat();
                    try
                    {
                        CvInvoke.AddWeighted(cur, 1.0, bg, -veil, 0.0, m);   // cur - veil·фон
                        using var sc = new ScalarArray(new MCvScalar(veil * mean.V0, veil * mean.V1, veil * mean.V2));
                        CvInvoke.Add(m, sc, m);                              // вернуть среднюю яркость
                    }
                    catch { m.Dispose(); throw; }
                    Replace(m);
                }
                double clahe = Val("clahe");
                if (clahe > 0.05)
                {
                    using var lab = new Mat(); CvInvoke.CvtColor(cur, lab, ColorConversion.Bgr2Lab);
                    var ch = lab.Split();
                    try { CvInvoke.CLAHE(ch[0], clahe, new Size(8, 8), ch[0]); using (var v = new VectorOfMat(ch)) CvInvoke.Merge(v, lab); }
                    finally { foreach (var c in ch) c.Dispose(); }
                    Replace(Op(m => CvInvoke.CvtColor(lab, m, ColorConversion.Lab2Bgr)));
                }
                if (_checks["autolevels"].Checked)
                    Replace(AutoLevels(cur));

                // 1) поканальные усиления
                double gr = Val("gainR"), gg = Val("gainG"), gb = Val("gainB");
                if (gr != 1 || gg != 1 || gb != 1)
                {
                    var ch = cur.Split();
                    try
                    {
                        ch[0].ConvertTo(ch[0], DepthType.Cv8U, gb, 0);
                        ch[1].ConvertTo(ch[1], DepthType.Cv8U, gg, 0);
                        ch[2].ConvertTo(ch[2], DepthType.Cv8U, gr, 0);
                        Replace(Op(m => { using var v = new VectorOfMat(ch); CvInvoke.Merge(v, m); }));
                    }
                    finally { foreach (var c in ch) c.Dispose(); }
                }

                // 2) яркость/контраст: out = contrast*in + (128*(1-contrast) + bright)
                double con = Val("contrast"), br = Val("bright");
                if (con != 1 || br != 0)
                    Replace(Op(m => cur.ConvertTo(m, DepthType.Cv8U, con, 128.0 * (1.0 - con) + br)));

                // 3) гамма (LUT)
                double ga = Val("gamma");
                if (Math.Abs(ga - 1.0) > 1e-3)
                    Replace(ApplyLut(cur, i => 255.0 * Math.Pow(i / 255.0, 1.0 / ga)));

                // 4) насыщенность / оттенок (HSV)
                double sat = Val("sat"), hue = Val("hue");
                if (sat != 1 || hue != 0)
                {
                    using var hsv = new Mat();
                    CvInvoke.CvtColor(cur, hsv, ColorConversion.Bgr2Hsv);
                    var ch = hsv.Split();
                    try
                    {
                        if (hue != 0)
                        {
                            int h = ((int)Math.Round(hue)) % 180; if (h < 0) h += 180;
                            var d = new byte[256];
                            for (int i = 0; i < 256; i++) d[i] = i < 180 ? (byte)((i + h) % 180) : (byte)i;
                            using var lut = new Mat(1, 256, DepthType.Cv8U, 1);
                            Marshal.Copy(d, 0, lut.DataPointer, 256);
                            CvInvoke.LUT(ch[0], lut, ch[0]);
                        }
                        if (sat != 1) ch[1].ConvertTo(ch[1], DepthType.Cv8U, sat, 0);
                        using (var v = new VectorOfMat(ch)) CvInvoke.Merge(v, hsv);
                    }
                    finally { foreach (var c in ch) c.Dispose(); }
                    Replace(Op(m => CvInvoke.CvtColor(hsv, m, ColorConversion.Hsv2Bgr)));
                }

                // 5) показ одного канала серым
                int chSel = _chCombo.SelectedIndex;
                if (chSel > 0)
                {
                    var ch = cur.Split();
                    try
                    {
                        int idx = chSel == 1 ? 2 : (chSel == 2 ? 1 : 0);   // R=2,G=1,B=0
                        Replace(Op(m => { using var v = new VectorOfMat(); v.Push(ch[idx]); v.Push(ch[idx]); v.Push(ch[idx]); CvInvoke.Merge(v, m); }));
                    }
                    finally { foreach (var c in ch) c.Dispose(); }
                }

                // 6) ч/б
                if (_checks["gray"].Checked)
                {
                    using var g = new Mat(); CvInvoke.CvtColor(cur, g, ColorConversion.Bgr2Gray);
                    Replace(Op(m => CvInvoke.CvtColor(g, m, ColorConversion.Gray2Bgr)));
                }

                // 7) инверсия
                if (_checks["invert"].Checked)
                    Replace(Op(m => CvInvoke.BitwiseNot(cur, m)));

                // 8) округление уровней (постеризация)
                if (_checks["poster"].Checked)
                {
                    int lv = Math.Max(2, (int)Val("levels"));
                    double step = 255.0 / (lv - 1);
                    Replace(ApplyLut(cur, i => Math.Round(Math.Round(i / step) * step)));
                }

                // 9) размытие
                double bl = Val("blur");
                if (bl >= 0.5)
                    Replace(Op(m => CvInvoke.GaussianBlur(cur, m, new Size(0, 0), bl)));

                // 10) резкость (unsharp)
                double sh = Val("sharpen");
                if (sh > 0.01)
                {
                    using var blur = new Mat(); CvInvoke.GaussianBlur(cur, blur, new Size(0, 0), 2.0);
                    Replace(Op(m => CvInvoke.AddWeighted(cur, 1.0 + sh, blur, -sh, 0, m)));
                }

                // 11) порог
                if (_checks["thresh"].Checked)
                {
                    using var g = new Mat(); CvInvoke.CvtColor(cur, g, ColorConversion.Bgr2Gray);
                    using var b = new Mat(); CvInvoke.Threshold(g, b, Val("threshLv"), 255, ThresholdType.Binary);
                    Replace(Op(m => CvInvoke.CvtColor(b, m, ColorConversion.Gray2Bgr)));
                }

                // 12) контуры (Canny поверх)
                if (_checks["edges"].Checked)
                {
                    using var g = new Mat(); CvInvoke.CvtColor(cur, g, ColorConversion.Bgr2Gray);
                    using var e = new Mat(); CvInvoke.Canny(g, e, Val("edgeLo"), Val("edgeHi"));
                    using var e3 = new Mat(); CvInvoke.CvtColor(e, e3, ColorConversion.Gray2Bgr);
                    Replace(Op(m => CvInvoke.Max(cur, e3, m)));
                }

                return cur;
            }
            catch { cur.Dispose(); throw; }
        }

        /// <summary>Авто-уровни: поканальное растяжение по перцентилям [1%, 99%] до [0,255].</summary>
        private static Mat AutoLevels(Mat bgr8)
        {
            var ch = bgr8.Split();
            try
            {
                for (int c = 0; c < 3; c++)
                {
                    int n = ch[c].Rows * ch[c].Cols;
                    var data = new byte[n]; ch[c].CopyTo(data);
                    var hist = new int[256]; foreach (var b in data) hist[b]++;
                    int need = Math.Max(1, (int)(n * 0.01));
                    int lo = 0, hi = 255, acc = 0;
                    for (int b = 0; b < 256; b++) { acc += hist[b]; if (acc >= need) { lo = b; break; } }
                    acc = 0;
                    for (int b = 255; b >= 0; b--) { acc += hist[b]; if (acc >= need) { hi = b; break; } }
                    if (hi - lo < 8) { lo = 0; hi = 255; }
                    double scale = 255.0 / (hi - lo);
                    ch[c].ConvertTo(ch[c], DepthType.Cv8U, scale, -lo * scale);
                }
                var m = new Mat();
                using (var v = new VectorOfMat(ch)) CvInvoke.Merge(v, m);
                return m;
            }
            finally { foreach (var c in ch) c.Dispose(); }
        }

        private static Mat ApplyLut(Mat src, Func<int, double> f)
        {
            var data = new byte[256];
            for (int i = 0; i < 256; i++) data[i] = (byte)Math.Clamp(Math.Round(f(i)), 0, 255);
            using var lut = new Mat(1, 256, DepthType.Cv8U, 1);
            Marshal.Copy(data, 0, lut.DataPointer, 256);
            var dst = new Mat();
            try { CvInvoke.LUT(src, lut, dst); return dst; }
            catch { dst.Dispose(); throw; }
        }

        // ---------- сохранение / утилиты ----------

        private void SaveFull()
        {
            using var sfd = new SaveFileDialog { Filter = "PNG|*.png|JPEG|*.jpg|BMP|*.bmp", FileName = "playground.png" };
            if (sfd.ShowDialog(this) != DialogResult.OK) return;
            using var full = ApplyPipeline(_src);
            CvInvoke.Imwrite(sfd.FileName, full);
        }

        private static Mat Downscale(Mat src, int maxDim)
        {
            int w = src.Cols, h = src.Rows;
            double s = Math.Min(1.0, (double)maxDim / Math.Max(w, h));
            var o = new Mat();
            if (s >= 1.0) src.CopyTo(o);
            else CvInvoke.Resize(src, o, new Size(Math.Max(1, (int)(w * s)), Math.Max(1, (int)(h * s))), 0, 0, Inter.Area);
            return o;
        }

        private static Bitmap ToBitmap(Mat mat)
        {
            using var buf = new VectorOfByte();
            CvInvoke.Imencode(".png", mat, buf);
            using var ms = new MemoryStream(buf.ToArray());
            using var tmp = new Bitmap(ms);
            return new Bitmap(tmp);
        }
    }
}
