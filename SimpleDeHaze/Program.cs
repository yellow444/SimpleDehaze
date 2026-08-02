using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using SimpleDeHaze.Benchmarking;
using SimpleDeHaze.Gui;
using SimpleDeHaze.Methods;

namespace SimpleDeHaze
{
    internal static class Program
    {
        [DllImport("kernel32.dll")]
        private static extern bool AttachConsole(int dwProcessId);

        [STAThread]
        public static void Main(string[] args)
        {
            CvInvoke.UseOptimized = true;

            // консольные режимы: печатать в родительский терминал
            if (args.Contains("--selftest") || args.Contains("--batch") || args.Contains("--benchmark") || args.Contains("--evaluate") || args.Contains("--a2cr-stress") || args.Contains("--diode-benchmark") || args.Contains("--a2cr-diag") || args.Contains("--scene8-study") || args.Contains("--autotune-audit") || args.Contains("--spot-tune") || args.Contains("--render") || args.Contains("--renderall") || args.Contains("--tunetest") || args.Contains("--noisetest") || args.Contains("--mathtest"))
            {
                AttachConsole(-1);
                if (args.Contains("--mathtest")) Environment.ExitCode = MathTest();
                else if (args.Contains("--selftest")) SelfTest();
                else if (args.Contains("--benchmark")) BenchmarkCsv(args);
                else if (args.Contains("--evaluate")) EvaluatePredictions(args);
                else if (args.Contains("--a2cr-stress")) Environment.ExitCode = A2crStressBenchmark.Run(
                    GetStringArg(args, "--out") ?? Path.Combine(Environment.CurrentDirectory, "benchmark_results", "a2cr-stress.csv"),
                    args.Contains("--quick"));
                else if (args.Contains("--diode-benchmark")) Environment.ExitCode = DiodeControlledBenchmark.Run(
                    GetStringArg(args, "--manifest") ?? Path.Combine(Environment.CurrentDirectory, "benchdata", "manifests", "diode-val-500.json"),
                    GetStringArg(args, "--out") ?? Path.Combine(Environment.CurrentDirectory, "benchmark_results", "diode-controlled.csv"),
                    GetStringArg(args, "--split") ?? "validation",
                    GetIntArg(args, "--limit", int.MaxValue),
                    GetIntArg(args, "--maxdim", 512),
                    GetStringArg(args, "--variants"),
                    args.Contains("--quick"),
                    args.Contains("--lpips"));
                else if (args.Contains("--scene8-study")) Environment.ExitCode = Scene8WallStudy.Run(
                    GetStringArg(args, "--image") ?? Path.Combine(Environment.CurrentDirectory, "SimpleDeHaze", "dataset", "08_outdoor_hazy.jpg"),
                    GetStringArg(args, "--gt") ?? Path.Combine(Environment.CurrentDirectory, "SimpleDeHaze", "hazefree", "08_outdoor_GT.jpg"),
                    GetStringArg(args, "--out") ?? Path.Combine(Environment.CurrentDirectory, "benchmark_results", "scene8-wall-study"),
                    GetIntArg(args, "--maxdim", 800),
                    GetStringArg(args, "--methods"),
                    GetStringArg(args, "--roi"),
                    GetStringArg(args, "--params"));
                else if (args.Contains("--autotune-audit")) Environment.ExitCode = AutoTuneAudit.Run(
                    GetStringArg(args, "--image") ?? Path.Combine(Environment.CurrentDirectory, "SimpleDeHaze", "dataset", "08_outdoor_hazy.jpg"),
                    GetStringArg(args, "--gt") ?? Path.Combine(Environment.CurrentDirectory, "SimpleDeHaze", "hazefree", "08_outdoor_GT.jpg"),
                    GetStringArg(args, "--out") ?? Path.Combine(Environment.CurrentDirectory, "benchmark_results", "autotune-audit"),
                    GetStringArg(args, "--methods") ?? "^Chromatic Airlight",
                    args.Contains("--thorough"),
                    GetIntArg(args, "--maxeval", args.Contains("--thorough") ? 120 : 60),
                    GetIntArg(args, "--evalmaxdim", args.Contains("--thorough") ? 256 : 320),
                    GetIntArg(args, "--maxdim", 800),
                    ParseAutoTuneGoal(GetStringArg(args, "--goal"), AutoTuneGoal.Reference));
                else if (args.Contains("--a2cr-diag")) A2crDiagnosticRun(args);
                else if (args.Contains("--spot-tune")) SpotTune(args);
                else if (args.Contains("--renderall")) RenderAll(args);
                else if (args.Contains("--render")) RenderOne(args);
                else if (args.Contains("--noisetest")) NoiseTest(args);
                else if (args.Contains("--tunetest")) TuneTest();
                else Batch();
                return;
            }

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            // smoke-проверка GUI: построить окно, прокрутить один цикл отрисовки, закрыться
            if (args.Contains("--guismoke")) { AttachConsole(-1); GuiSmoke(); return; }

            // headless-проверка песочницы: построить форму, прогнать конвейер фильтров, сохранить
            if (args.Contains("--playtest"))
            {
                AttachConsole(-1);
                try
                {
                    var p = Path.Combine(AppContext.BaseDirectory, "dataset", "09_outdoor_hazy.jpg");
                    using var full = new Image<Bgr, byte>(p);
                    using var img = full.Resize(700, (int)(700.0 * full.Height / full.Width), Inter.Area);
                    using var src8 = img.Mat.Clone();
                    using var pf = new Gui.PlaygroundForm(src8);
                    using var a = pf.RunPipelineForTest(false);
                    using var b = pf.RunPipelineForTest(true);
                    var outDir = Path.Combine(Environment.CurrentDirectory, "playtest_out");
                    Directory.CreateDirectory(outDir);
                    CvInvoke.Imwrite(Path.Combine(outDir, "default.png"), a);
                    CvInvoke.Imwrite(Path.Combine(outDir, "edited.png"), b);
                    Console.WriteLine($"PLAYTEST-OK default={a.Size} edited={b.Size} same={(a.Size == src8.Size && b.Size == src8.Size)} out={outDir}");
                }
                catch (Exception ex) { Console.WriteLine("PLAYTEST-FAIL " + ex); }
                return;
            }

            // headless-проверка бенчмарка 'Прогнать все'
            if (args.Contains("--benchtest"))
            {
                AttachConsole(-1);
                try { using var f = new MainForm(); Console.WriteLine(f.BenchmarkSelfTest()); }
                catch (Exception ex) { Console.WriteLine("BENCH-FAIL " + ex); }
                return;
            }

            // headless-проверка тщательного подбора: дефолт-скор -> подобранный скор
            if (args.Contains("--tunetest")) { AttachConsole(-1); TuneTest(); return; }

            // обычный режим - GUI (опц. путь к файлу первым аргументом).
            // По умолчанию открывается новый WPF-интерфейс; старый WinForms - по --old-ui.
            string? file = args.FirstOrDefault(a => !a.StartsWith("--") && File.Exists(a));
            if (args.Contains("--old-ui") || args.Contains("--winforms"))
            {
                Application.Run(new MainForm(file));
                return;
            }

            var wpf = new System.Windows.Application();
            wpf.Run(new Gui.Modern.NewMainWindow(file));
        }

        /// <summary>
        /// Численные проверки физического ядра (--mathtest). Проверяются УТВЕРЖДЕНИЯ, на которых
        /// строятся выводы в документации: обратимость sRGB-кривой, замкнутость модели по яркости
        /// в линейном пространстве, корректность границ допустимости (в т.ч. выведенной для
        /// chroma-safe восстановления) и совпадение матричной реализации со скалярным эталоном.
        /// Возвращает код выхода: 0 - все проверки пройдены.
        /// </summary>
        private static int MathTest()
        {
            int failed = 0;
            var rnd = new Random(20260727);

            void Check(string name, bool ok, string detail)
            {
                Console.WriteLine($"  [{(ok ? "OK  " : "FAIL")}] {name}: {detail}");
                if (!ok) failed++;
            }

            Console.WriteLine("MATHTEST");

            // 1. sRGB <-> линейный радианс: обратимость и совпадение LUT с аналитической кривой
            {
                double maxRound = 0, maxLut = 0;
                using var src = new Mat(1, 256, DepthType.Cv32F, 1);
                var vals = new float[256];
                for (int i = 0; i < 256; i++) vals[i] = i / 255f;
                System.Runtime.InteropServices.Marshal.Copy(vals, 0, src.DataPointer, 256);

                using var lin = ColorSpace.ToLinear(src);
                using var back = ColorSpace.ToSrgb(lin);
                var got = new float[256]; back.CopyTo(got);
                var linv = new float[256]; lin.CopyTo(linv);
                for (int i = 0; i < 256; i++)
                {
                    maxRound = Math.Max(maxRound, Math.Abs(got[i] - vals[i]));
                    double c = i / 255.0;
                    double expect = c <= 0.04045 ? c / 12.92 : Math.Pow((c + 0.055) / 1.055, 2.4);
                    maxLut = Math.Max(maxLut, Math.Abs(linv[i] - expect));
                }
                Check("sRGB roundtrip", maxRound < 1e-5, $"max|x - toSrgb(toLinear(x))| = {maxRound:E2}");
                Check("sRGB кривая vs аналитика", maxLut < 1e-6, $"max откл. = {maxLut:E2}");
            }

            // 2. Замкнутость модели по ЯРКОСТИ в линейном пространстве: Y_I = t*Y_J + (1-t)*Y_A точно.
            //    Это обоснование того, что коррекцию можно вести в яркости (и что Lab для этого не годится).
            {
                double maxErr = 0;
                for (int k = 0; k < 2000; k++)
                {
                    double[] J = { rnd.NextDouble(), rnd.NextDouble(), rnd.NextDouble() };
                    double[] A = { rnd.NextDouble(), rnd.NextDouble(), rnd.NextDouble() };
                    double t = 0.05 + 0.9 * rnd.NextDouble();
                    double[] w = { 0.0722, 0.7152, 0.2126 };   // B,G,R - линейные веса яркости sRGB
                    double yI = 0, yJ = 0, yA = 0;
                    for (int c = 0; c < 3; c++)
                    {
                        double I = J[c] * t + A[c] * (1 - t);
                        yI += w[c] * I; yJ += w[c] * J[c]; yA += w[c] * A[c];
                    }
                    maxErr = Math.Max(maxErr, Math.Abs(yI - (t * yJ + (1 - t) * yA)));
                }
                Check("яркость замкнута по модели (линейный RGB)", maxErr < 1e-12, $"max|невязка| = {maxErr:E2}");
            }

            // 3. Boundary constraint (Meng 2013): при t = t_box стандартная инверсия не выходит из [0,1]
            {
                double worst = 0;
                for (int k = 0; k < 20000; k++)
                {
                    double[] A = { 0.2 + 0.75 * rnd.NextDouble(), 0.2 + 0.75 * rnd.NextDouble(), 0.2 + 0.75 * rnd.NextDouble() };
                    double[] I = { rnd.NextDouble(), rnd.NextDouble(), rnd.NextDouble() };
                    double t = Math.Max(1e-3, BoxScalar(I, A));
                    for (int c = 0; c < 3; c++)
                    {
                        double J = (I[c] - A[c]) / t + A[c];
                        worst = Math.Max(worst, Math.Max(-J, J - 1));
                    }
                }
                Check("t_box гарантирует 0<=J<=1 (станд. инверсия)", worst < 1e-9, $"max выход за куб = {worst:E2}");
            }

            // 4. Выведенная граница для ФАКТИЧЕСКОГО chroma-safe восстановления
            //    J_c = A_c + dbar/max(t,m) + delta_c/max(t,q)
            {
                double worst = 0; int checkedN = 0;
                for (int k = 0; k < 20000; k++)
                {
                    double[] A = { 0.2 + 0.75 * rnd.NextDouble(), 0.2 + 0.75 * rnd.NextDouble(), 0.2 + 0.75 * rnd.NextDouble() };
                    double[] I = { rnd.NextDouble(), rnd.NextDouble(), rnd.NextDouble() };
                    double m = 0.02 + 0.1 * rnd.NextDouble();
                    double q = m + 0.5 * rnd.NextDouble();
                    double t = ChromaSafeScalar(I, A, m, q);
                    double dbar = ((I[0] - A[0]) + (I[1] - A[1]) + (I[2] - A[2])) / 3.0;
                    double u = Math.Max(t, m), v = Math.Max(t, q);
                    for (int c = 0; c < 3; c++)
                    {
                        double J = A[c] + dbar / u + ((I[c] - A[c]) - dbar) / v;
                        worst = Math.Max(worst, Math.Max(-J, J - 1));
                    }
                    checkedN++;
                }
                Check("chroma-safe граница гарантирует 0<=J<=1", worst < 1e-6, $"max выход за куб = {worst:E2} на {checkedN} наборах");
            }

            // 5. Матричная реализация границы совпадает со скалярным эталоном
            {
                const int n = 64;
                using var img = new Mat(n, n, DepthType.Cv32F, 3);
                var data = new float[n * n * 3];
                for (int i = 0; i < data.Length; i++) data[i] = (float)rnd.NextDouble();
                System.Runtime.InteropServices.Marshal.Copy(data, 0, img.DataPointer, data.Length);
                var A = new MCvScalar(0.72, 0.78, 0.83);
                double m = 0.08, q = 0.35;

                using var boundMat = DehazeCore.ChromaSafeLowerBound(img, A, m, q);
                var got = new float[n * n];
                boundMat.CopyTo(got);

                double maxDiff = 0;
                double[] av = { A.V0, A.V1, A.V2 };
                for (int i = 0; i < n * n; i++)
                {
                    double[] I = { data[i * 3], data[i * 3 + 1], data[i * 3 + 2] };
                    maxDiff = Math.Max(maxDiff, Math.Abs(got[i] - ChromaSafeScalar(I, av, m, q)));
                }
                Check("ChromaSafeLowerBound: Mat == скалярный эталон", maxDiff < 2e-6, $"max разница = {maxDiff:E2}");
            }

            // 6. Строгий режим проекции действительно убирает нарушения допустимости
            {
                var dir = Path.Combine(AppContext.BaseDirectory, "dataset");
                var file = Directory.Exists(dir) ? Directory.GetFiles(dir, "*.jpg").FirstOrDefault() : null;
                if (file == null) Console.WriteLine("  [SKIP] нет dataset/ - проверка проекции пропущена");
                else
                {
                    using var full = new Image<Bgr, byte>(file);
                    using var img = full.Resize(480, (int)(480.0 * full.Height / full.Width), Inter.Area);
                    var method = new RfepDcpMethod();

                    var strict = method.Parameters.ToDictionary(x => x.Key, x => x.Default);
                    strict["strict"] = 1;
                    var (beforeS, afterS) = RfepDcpMethod.MeasureViolation(img, strict);

                    var soft = method.Parameters.ToDictionary(x => x.Key, x => x.Default);
                    var (beforeL, afterL) = RfepDcpMethod.MeasureViolation(img, soft);

                    Check("strict=1: нарушений допустимости нет", afterS < 1e-6,
                        $"до проекции {beforeS * 100:F1} % -> после {afterS * 100:F3} %");
                    Console.WriteLine($"  [INFO] strict=0 (дефолт): до {beforeL * 100:F1} % -> после {afterL * 100:F1} % " +
                                      "- ослабленная граница гарантий НЕ даёт, это осознанный размен");
                }
            }

            // 7. Метрики: на совпадающих изображениях SSIM = 1, PSNR максимален
            {
                var dir = Path.Combine(AppContext.BaseDirectory, "dataset");
                var file = Directory.Exists(dir) ? Directory.GetFiles(dir, "*.jpg").FirstOrDefault() : null;
                if (file == null) Console.WriteLine("  [SKIP] нет dataset/ - проверка метрик пропущена");
                else
                {
                    using var img = new Image<Bgr, byte>(file);
                    using var small = img.Resize(320, (int)(320.0 * img.Height / img.Width), Inter.Area);
                    using var f01 = new Mat();
                    small.Mat.ConvertTo(f01, DepthType.Cv32F, 1.0 / 255.0);
                    var rep = Metrics.Evaluate(f01, small.Mat, small.Mat);
                    Check("SSIM(x,x) = 1", Math.Abs(rep.Ssim - 1.0) < 1e-3, $"SSIM = {rep.Ssim:F6}");
                    Check("PSNR(x,x) велик", rep.Psnr > 45, $"PSNR = {rep.Psnr:F1} дБ");
                    Check("шум на плоских зонах не изменён", Math.Abs(rep.FlatNoiseRatio - 1.0) < 0.05, $"x{rep.FlatNoiseRatio:F3}");
                }
            }

            Console.WriteLine(failed == 0 ? "MATHTEST-OK все проверки пройдены" : $"MATHTEST-FAIL провалено проверок: {failed}");
            return failed == 0 ? 0 : 1;
        }

        /// <summary>Скалярный эталон boundary constraint (Meng et al., ICCV 2013) при C0=0, C1=1.</summary>
        private static double BoxScalar(double[] I, double[] A)
        {
            double b = 0;
            for (int c = 0; c < 3; c++)
            {
                b = Math.Max(b, (I[c] - A[c]) / Math.Max(1e-4, 1.0 - A[c]));
                b = Math.Max(b, (A[c] - I[c]) / Math.Max(1e-4, A[c]));
            }
            return Math.Clamp(b, 0, 1);
        }

        /// <summary>
        /// НЕЗАВИСИМАЯ (написанная отдельно от рабочего кода) реализация вывода границы для
        /// chroma-safe восстановления: перебирает допустимый отрезок по каждому каналу.
        /// Используется как эталон для перекрёстной проверки <see cref="DehazeCore.ChromaSafeAt"/>.
        /// </summary>
        private static double ChromaSafeScalar(double[] I, double[] A, double m, double q)
        {
            q = Math.Max(m, q);
            if (q <= m + 1e-9) return BoxScalar(I, A);

            double dbar = ((I[0] - A[0]) + (I[1] - A[1]) + (I[2] - A[2])) / 3.0;
            const double big = 1e6, eps = 1e-6;
            double lo = 0, hi = big;
            bool feasible = true;

            for (int c = 0; c < 3 && feasible; c++)
            {
                double e = A[c] + ((I[c] - A[c]) - dbar) / q;
                if (dbar > eps)
                {
                    if (e < 1.0 - eps) lo = Math.Max(lo, dbar / (1.0 - e)); else feasible = false;
                    if (feasible && e < -eps) hi = Math.Min(hi, dbar / -e);
                }
                else if (dbar < -eps)
                {
                    if (e > eps) lo = Math.Max(lo, -dbar / e); else feasible = false;
                    if (feasible && e > 1.0 + eps) hi = Math.Min(hi, -dbar / (e - 1.0));
                }
                else if (e < -eps || e > 1.0 + eps) feasible = false;
            }

            // в отрезок должно попадать u = max(t, m), а не сам t
            if (feasible && Math.Max(lo, m) <= hi && lo < q) return Math.Clamp(lo, 0, 1);
            return Math.Clamp(Math.Max(BoxScalar(I, A), q), 0, 1);
        }

        /// <summary>
        /// Проверка анти-шумового подбора на 09_outdoor: тщательный подбор метода 'объекты' с оценкой
        /// на ~полном кадре, сохраняет результат и печатает подобранные параметры. Опции: --maxdim, --evals.
        /// </summary>
        private static void NoiseTest(string[] args)
        {
            int maxDim = GetIntArg(args, "--maxdim", 1300);
            int evals = GetIntArg(args, "--evals", 150);
            string imagePath = Path.Combine(AppContext.BaseDirectory, "dataset", "09_outdoor_hazy.jpg");
            if (!File.Exists(imagePath)) { Console.WriteLine("NOISETEST-FAIL no 09 image"); return; }
            string outDir = GetStringArg(args, "--out") ?? Path.Combine(Environment.CurrentDirectory, "noisetest_out");
            Directory.CreateDirectory(outDir);

            using var full = new Image<Bgr, byte>(imagePath);
            double sc = Math.Min(1.0, (double)maxDim / Math.Max(full.Width, full.Height));
            using var img = sc >= 1.0 ? full.Clone() : full.Resize((int)(full.Width * sc), (int)(full.Height * sc), Inter.Area);

            string methodFilter = GetStringArg(args, "--method") ?? "объекты";
            var method = MethodRegistry.All.FirstOrDefault(x => MatchesFilter(x.Name, methodFilter)) ?? new LocalHazeObjectsMethod();
            string goalArg = GetStringArg(args, "--goal") ?? "obj";
            var goal = goalArg.StartsWith("ref") ? AutoTuneGoal.Reference : goalArg.StartsWith("viv") ? AutoTuneGoal.Vivid : AutoTuneGoal.ObjectVisibility;
            int evalDim = GetIntArg(args, "--evaldim", Math.Max(img.Width, img.Height));   // по умолч. оценка на оригинале

            Mat? gt = null;
            if (goal == AutoTuneGoal.Reference)
            {
                var gtFile = imagePath.Replace("hazy", "GT").Replace("dataset", "hazefree");
                if (File.Exists(gtFile))
                {
                    using var gtImg = new Image<Bgr, byte>(gtFile);
                    using var gtR = gtImg.Resize(img.Width, img.Height, Inter.Area);
                    gt = gtR.Mat.Clone();
                }
            }

            var def = method.Parameters.ToDictionary(p => p.Key, p => p.Default);
            double defaultObjective;
            using (var rDef = method.Process(img, def))
            {
                Save(rDef, Path.Combine(outDir, "objects_default.png"));
                defaultObjective = AutoTuner.Score(goal, rDef, img.Mat, gt, 0.0);
            }

            var sw = Stopwatch.StartNew();
            var tuned = AutoTuner.OptimizeThorough(method, img, def, maxEvals: evals,
                goal: goal, evalMaxDim: evalDim, gt: gt);
            sw.Stop();
            double tunedObjective;
            using (var rTuned = method.Process(img, tuned))
            {
                Save(rTuned, Path.Combine(outDir, "objects_tuned.png"));
                tunedObjective = AutoTuner.Score(goal, rTuned, img.Mat, gt, 0.0);
            }
            gt?.Dispose();

            // При оценке на исходном размере это ровно та же objective, которую видел оптимизатор.
            if (evalDim >= Math.Max(img.Width, img.Height) && tunedObjective + 1e-9 < defaultObjective)
                throw new InvalidOperationException($"Автоподбор ухудшил objective: {defaultObjective:F6} -> {tunedObjective:F6}.");

            Console.WriteLine($"NOISETEST method={method.Name} goal={goal} size={img.Width}x{img.Height} evalDim={evalDim} evals={evals} за {sw.ElapsedMilliseconds}мс");
            Console.WriteLine($"objective: {defaultObjective:F6} -> {tunedObjective:F6}");
            Console.WriteLine("подобрано: " + SpotParams(tuned));
            Console.WriteLine("результаты: " + outDir);

            static void Save(Mat r, string path) { using var m = new Mat(); r.ConvertTo(m, DepthType.Cv8U, 255.0); CvInvoke.Imwrite(path, m); }
        }

        /// <summary>Headless-проверка тщательного подбора: для нескольких методов печатает скор дефолтов -> скор после OptimizeThorough.</summary>
        private static void TuneTest()
        {
            var dir = Path.Combine(AppContext.BaseDirectory, "dataset");
            var file = Directory.Exists(dir) ? Directory.GetFiles(dir, "*.*").FirstOrDefault() : null;
            if (file is null) { Console.WriteLine("dataset пуст"); return; }
            using var full = new Image<Bgr, byte>(file);
            double sc = Math.Min(1.0, 800.0 / Math.Max(full.Width, full.Height));
            using var img = sc >= 1.0 ? full.Clone() : full.Resize((int)(full.Width * sc), (int)(full.Height * sc), Inter.Area);

            Mat? gt = null;
            var gtFile = file.Replace("hazy", "GT").Replace("dataset", "hazefree");
            if (File.Exists(gtFile))
            {
                using var gtFull = new Image<Bgr, byte>(gtFile);
                using var gtR = gtFull.Resize(img.Width, img.Height, Inter.Area);
                gt = gtR.Mat.Clone();
            }

            foreach (var m in MethodRegistry.All.Where(x =>
                x.Name.Contains("Локальная") || x.Name.Contains("Восстановление") || x.Name.Contains("Усиление")))
            {
                var def = m.Parameters.ToDictionary(p => p.Key, p => p.Default);
                using var r0 = m.Process(img, def); var d = Metrics.Evaluate(r0, null, img.Mat);
                var tuned = AutoTuner.OptimizeThorough(m, img, def, goal: AutoTuneGoal.ObjectVisibility);
                using var r1 = m.Process(img, tuned); var t = Metrics.Evaluate(r1, null, img.Mat);
                var tunedC = AutoTuner.OptimizeThorough(m, img, def, 1.2, goal: AutoTuneGoal.Vivid);   // цель: сочность + не гасить цвет
                using var r2 = m.Process(img, tunedC); var tc = Metrics.Evaluate(r2, null, img.Mat);
                string refPart = "";
                if (gt != null)
                {
                    var tunedR = AutoTuner.OptimizeThorough(m, img, def, gt: gt, goal: AutoTuneGoal.Reference);
                    using var rr = m.Process(img, tunedR);
                    var tr = Metrics.Evaluate(rr, gt, img.Mat);
                    refPart = $"  GT(raw PSNR{tr.Psnr:F1} MSE{tr.Mse:F0} SSIM{tr.Ssim:F3})";
                }
                int tunable = m.Parameters.Count(pp => pp.Tunable);
                int moved = m.Parameters.Count(pp => pp.Tunable && tuned.TryGetValue(pp.Key, out var tv2) && Math.Abs(tv2 - def[pp.Key]) > 1e-9);
                var movedKeys = string.Join(",", m.Parameters.Where(pp => pp.Tunable && tuned.TryGetValue(pp.Key, out var tv3) && Math.Abs(tv3 - def[pp.Key]) > 1e-9).Select(pp => pp.Key));
                Console.WriteLine($"{m.Name,-42} дефолт(оц{d.Score,3:F0} цвет{d.ColorRatio:F2})  объекты(оц{t.Score,3:F0} цвет{t.ColorRatio:F2})  сочн(оц{tc.Score,3:F0} цвет{tc.ColorRatio:F2})  подбор крутил {moved}/{tunable} парам [{movedKeys}]{refPart}");
            }
            gt?.Dispose();

            // 'Авто-лучший': скан всех + настройка победителя
            var sw = Stopwatch.StartNew();
            var (bm, bp, tunedObjective) = AutoTuner.PickBest(MethodRegistry.All, img, null, goal: AutoTuneGoal.ObjectVisibility);
            sw.Stop();
            using var rb = bm.Process(img, bp);
            double finalScore = Metrics.NoRefScore(rb, img.Mat);
            Console.WriteLine($"\nАВТО-ЛУЧШИЙ -> '{bm.Name}'  (objective {tunedObjective:F0}; NoRef {finalScore:F0})  за {sw.ElapsedMilliseconds}мс");
        }

        /// <summary>Headless-smoke: создать главное окно, дать ему отрисоваться и закрыть. Проверяет, что конструктор/Load не падают.</summary>
        private static void GuiSmoke()
        {
            try
            {
                using var f = new MainForm();
                using var t = new System.Windows.Forms.Timer { Interval = 1500 };
                t.Tick += (_, _) => { t.Stop(); f.Close(); };
                f.Shown += (_, _) => t.Start();
                Application.Run(f);
                Console.WriteLine("GUISMOKE-OK");
            }
            catch (Exception ex)
            {
                Console.WriteLine("GUISMOKE-FAIL " + ex.GetType().Name + ": " + ex.Message);
                Console.WriteLine(ex.StackTrace);
            }
        }

        /// <summary>Headless-проверка: прогнать все методы на одном изображении из dataset/.</summary>
        private static void SelfTest()
        {
            var dir = Path.Combine(AppContext.BaseDirectory, "dataset");
            var file = Directory.Exists(dir) ? Directory.GetFiles(dir, "*.*").FirstOrDefault() : null;
            if (file is null) { Console.WriteLine("dataset/ пуст или отсутствует"); return; }

            using var full = new Image<Bgr, byte>(file);
            double sc = Math.Min(1.0, 800.0 / Math.Max(full.Width, full.Height));   // прогон на уменьшенной копии
            using var img = sc >= 1.0 ? full.Clone() : full.Resize((int)(full.Width * sc), (int)(full.Height * sc), Inter.Area);
            var outDir = Path.Combine(Environment.CurrentDirectory, "selftest_out");
            Directory.CreateDirectory(outDir);
            CvInvoke.Imwrite(Path.Combine(outDir, "00_input.png"), img.Mat);
            Console.WriteLine($"selftest на {Path.GetFileName(file)} {img.Size} -> {outDir}");
            Mat? gt = null;
            var gtFile = file.Replace("hazy", "GT").Replace("dataset", "hazefree");
            if (File.Exists(gtFile))
            {
                using var gtImg = new Image<Bgr, byte>(gtFile);
                using var gtR = gtImg.Resize(img.Width, img.Height, Inter.Area);
                gt = gtR.Mat.Clone();
                using var inF = new Mat(); img.Mat.ConvertTo(inF, DepthType.Cv32F, 1.0 / 255.0);
                Console.WriteLine($"  (эталон hazefree: PSNR входа = {Metrics.Psnr(inF, gt):F2} дБ - базовая линия 'без обработки')");
            }
            var rows = new List<(string name, Metrics.Report rep, long ms)>();
            double megaPixels = img.Width * img.Height / 1_000_000.0;
            int idx = 0;
            foreach (var m in MethodRegistry.All)
            {
                idx++;
                var p = m.Parameters.ToDictionary(x => x.Key, x => x.Default);
                var sw = Stopwatch.StartNew();
                try
                {
                    using var res = m.Process(img, p);
                    sw.Stop();
                    var rep = Metrics.Evaluate(res, gt, img.Mat);
                    using var disp = new Mat(); res.ConvertTo(disp, DepthType.Cv8U, 255.0);
                    var safe = m.Name.Replace(' ', '_').Replace('*', '-').Replace('/', '-').Replace("(", "").Replace(")", "");
                    CvInvoke.Imwrite(Path.Combine(outDir, $"{idx:00}_{safe}.png"), disp);
                    rows.Add((m.Name, rep, sw.ElapsedMilliseconds));
                    string de = rep.HasRef ? $" DE00={rep.Ciede2000,5:F2}" : "";
                    double msMp = sw.ElapsedMilliseconds / Math.Max(1e-6, megaPixels);
                    Console.WriteLine($"  OK   {m.Name,-30} PSNR={rep.Psnr,5:F2} совмещ(диагн.)={rep.PsnrAligned,5:F2} SSIM={rep.Ssim:F3}{de} оценка={rep.Score,3:F0} дымка{rep.HazeRemoved * 100,3:F0}% пересв{rep.ClipPct,4:F1}% цветx{rep.ColorRatio:F2} нат~{rep.NaturalnessDev:F0} арт~{rep.ArtifactDev:F0} {sw.ElapsedMilliseconds}мс ({msMp:F0}мс/Мп)");
                }
                catch (Exception ex)
                {
                    sw.Stop();
                    Console.WriteLine($"  FAIL {m.Name,-14} {ex.GetType().Name}: {ex.Message}");
                }
            }

            if (gt != null && rows.Count > 0)
            {
                Console.WriteLine("\n  === рейтинг по ОСНОВНОМУ PSNR (без подгонки к GT) ===");
                foreach (var r in rows.OrderByDescending(x => x.rep.Psnr).Take(12))
                    Console.WriteLine($"    {r.rep.Psnr,5:F2} дБ  SSIM {r.rep.Ssim:F3}  DE00 {r.rep.Ciede2000,5:F2}  (aligned PSNR {r.rep.PsnrAligned,5:F2}, только диагностика)  {r.name}");
                Console.WriteLine("\n  === рейтинг по БЕЗ-ЭТАЛОННОЙ оценке (дымка/детали - пересвет/перенасыщение) ===");
                foreach (var r in rows.OrderByDescending(x => x.rep.Score).Take(12))
                    Console.WriteLine($"    {r.rep.Score,3:F0}/100  дымка{r.rep.HazeRemoved * 100,3:F0}%  пересвет {r.rep.ClipPct,4:F1}%  цвет x{r.rep.ColorRatio:F2}  {r.name}");
            }
            gt?.Dispose();
        }

        /// <summary>
        /// Headless benchmark по manifest либо паре input-dir/gt-dir. Вход по умолчанию остаётся
        /// в нативном разрешении; --maxdim задаёт явный быстрый режим. Профиль core применяется
        /// только к методам с явной конфигурацией в BenchmarkProfiles.
        /// </summary>
        private static void BenchmarkCsv(string[] args)
        {
            int limit = GetIntArg(args, "--limit", int.MaxValue);
            int maxDim = args.Contains("--native") ? int.MaxValue : GetIntArg(args, "--maxdim", int.MaxValue);
            int warmup = GetIntArg(args, "--warmup", 1);
            int repeat = GetIntArg(args, "--repeat", 3);
            string outPath = GetStringArg(args, "--out") ?? Path.Combine(Environment.CurrentDirectory, "dehaze_benchmark_full.csv");
            string? manifest = GetStringArg(args, "--manifest");
            string inputDir = GetStringArg(args, "--input-dir") ?? Path.Combine(AppContext.BaseDirectory, "dataset");
            string gtDir = GetStringArg(args, "--gt-dir") ?? Path.Combine(AppContext.BaseDirectory, "hazefree");
            string? imageFilter = GetStringArg(args, "--images");
            string? methodFilter = GetStringArg(args, "--methods");
            string? paramOverrides = GetStringArg(args, "--params");
            string profile = (GetStringArg(args, "--profile") ?? (args.Contains("--nopost") ? "core" : "full")).ToLowerInvariant();
            if (profile is not ("core" or "full")) throw new ArgumentException("--profile должен быть core или full");
            bool linear = args.Contains("--linear");
            bool includeUnprofiled = args.Contains("--include-unprofiled");
            bool measureMemory = !args.Contains("--no-memory");
            bool computeLpips = args.Contains("--lpips");
            string split = (GetStringArg(args, "--split") ?? "all").ToLowerInvariant();
            if (args.Contains("--evalfull")) Metrics.EvalMaxSide = int.MaxValue;

            var dataset = DatasetCatalog.Load(manifest, inputDir, gtDir, split, imageFilter, limit);

            var methods = MethodRegistry.All
                .Where(m => MatchesFilter(m.Name, methodFilter))
                .ToArray();
            if (methods.Length == 0) { Console.WriteLine("Нет методов под фильтр --methods"); return; }

            var runs = new List<(IDeHazeMethod Method, Dictionary<string, double> Params, bool ProfileSupported)>();
            foreach (var method in methods)
            {
                var configured = BenchParams(method, profile, linear, paramOverrides);
                if (profile == "core" && !configured.ProfileSupported && !includeUnprofiled)
                {
                    Console.WriteLine($"SKIP core: {method.Name} — явный core-профиль не определён");
                    continue;
                }
                runs.Add((method, configured.Params, configured.ProfileSupported));
            }
            if (runs.Count == 0) throw new InvalidOperationException("Нет методов с поддержкой выбранного профиля");

            string mode = profile + (linear ? "+linear" : "") + (split != "all" ? "+" + split : "");
            outPath = Path.GetFullPath(outPath);
            Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);
            using var lpips = computeLpips ? new LpipsBridge(Environment.CurrentDirectory) : null;
            using var writer = new StreamWriter(outPath, false, new System.Text.UTF8Encoding(true));
            var hardware = HardwareProbe.Capture(OpenCvVersion());
            foreach (var line in RunMetadata(dataset.Name, dataset.Version, mode, maxDim, dataset.Cases.Length,
                         runs.Count, warmup, repeat, measureMemory, hardware))
                writer.WriteLine("# " + line);
            // ПЕРВИЧНЫЕ метрики - psnr/ssim/ciede2000 без подгонки; *_aligned считаются ПОСЛЕ поканального
            // аффинного совмещения с эталоном (эталон используется для правки результата) и годятся
            // только как диагностика. natur_dev_own/artifact_dev_own - собственные эвристики, НЕ NIQE/BRISQUE.
            writer.WriteLine("dataset;image;method;mode;profile_supported;ok;error;source_width;source_height;processed_width;processed_height;repeats;" +
                             "ms_median;ms_min;ms_p95;ms_per_mp;peak_working_set_mb;peak_working_set_delta_mb;peak_gpu_used_mb;peak_gpu_delta_mb;score;psnr;ssim;ciede2000;lpips;lpips_status;mse;" +
                             "psnr_aligned_diag;ssim_aligned_diag;ciede2000_aligned_diag;mse_aligned_diag;" +
                             "natur_dev_own;artifact_dev_own;flat_noise_x;haze_removed_pct;contrast_x;edge_x;clip_pct;color_x");

            int ok = 0, fail = 0, rows = 0;
            foreach (var item in dataset.Cases)
            {
                using var full = new Image<Bgr, byte>(item.HazyPath);
                double sc = Math.Min(1.0, (double)maxDim / Math.Max(full.Width, full.Height));
                using var img = sc >= 1.0 ? full.Clone() : full.Resize((int)(full.Width * sc), (int)(full.Height * sc), Inter.Area);
                double mp = img.Width * img.Height / 1_000_000.0;

                Mat? gt = null;
                if (item.ClearPath != null && File.Exists(item.ClearPath))
                {
                    using var gtImg = new Image<Bgr, byte>(item.ClearPath);
                    using var gtR = gtImg.Resize(img.Width, img.Height, Inter.Area);
                    gt = gtR.Mat.Clone();
                }

                foreach (var run in runs)
                {
                    rows++;
                    try
                    {
                        var measured = BenchmarkMeasurement.Run(() => run.Method.Process(img, run.Params), warmup, repeat, measureMemory);
                        using var res = measured.Result;
                        var rep = Metrics.Evaluate(res, gt, img.Mat);
                        double lpipsValue = double.NaN;
                        string lpipsStatus = computeLpips ? "no_ground_truth" : "not_requested";
                        if (lpips != null && gt != null)
                        {
                            using var gtFloat = new Mat(); gt.ConvertTo(gtFloat, DepthType.Cv32F, 1.0 / 255.0);
                            lpipsValue = lpips.Evaluate(res, gtFloat);
                            lpipsStatus = $"alex_v0.1_{lpips.Device}";
                        }
                        if (computeLpips)
                            measured = measured with { PeakGpuUsedMb = null, PeakGpuDeltaMb = null };
                        WriteBenchRow(writer, dataset.Name, item.Id, run.Method.Name, mode, run.ProfileSupported,
                            full.Width, full.Height, img.Width, img.Height, repeat, measured,
                            measured.MedianMs / Math.Max(1e-6, mp), rep, lpipsValue, lpipsStatus);
                        ok++;
                    }
                    catch (Exception ex)
                    {
                        WriteBenchFail(writer, dataset.Name, item.Id, run.Method.Name, mode, run.ProfileSupported,
                            full.Width, full.Height, img.Width, img.Height, repeat, ex);
                        fail++;
                    }
                }

                gt?.Dispose();
                Console.WriteLine($"{item.Id}: готово, строк {rows}, ok {ok}, fail {fail}");
            }

            writer.Flush();
            string metaPath = Path.ChangeExtension(outPath, ".meta.json");
            File.WriteAllText(metaPath, MetadataJson(dataset.Name, dataset.Version, manifest, mode, maxDim,
                    warmup, repeat, measureMemory, dataset.Cases, runs, rows, ok, fail, hardware,
                    computeLpips, lpips?.Device),
                new System.Text.UTF8Encoding(false));

            Console.WriteLine($"BENCHMARK-CSV dataset={dataset.Name} mode={mode} images={dataset.Cases.Length} methods={runs.Count} rows={rows} ok={ok} fail={fail}");
            Console.WriteLine($"  csv:  {outPath}");
            Console.WriteLine($"  meta: {metaPath}");
        }

        /// <summary>
        /// Оценка заранее рассчитанного внешнего baseline. Файл результата обязан иметь размер входа;
        /// автоматический resize запрещён, чтобы случайно не улучшать метрики интерполяцией.
        /// </summary>
        private static void EvaluatePredictions(string[] args)
        {
            string? manifest = GetStringArg(args, "--manifest");
            string inputDir = GetStringArg(args, "--input-dir") ?? Path.Combine(AppContext.BaseDirectory, "dataset");
            string gtDir = GetStringArg(args, "--gt-dir") ?? Path.Combine(AppContext.BaseDirectory, "hazefree");
            string split = (GetStringArg(args, "--split") ?? "all").ToLowerInvariant();
            string predictionsDir = Path.GetFullPath(GetStringArg(args, "--predictions-dir")
                ?? throw new ArgumentException("--predictions-dir обязателен"));
            string methodName = GetStringArg(args, "--method-name") ?? Path.GetFileName(predictionsDir.TrimEnd(Path.DirectorySeparatorChar));
            string outPath = Path.GetFullPath(GetStringArg(args, "--out") ?? Path.Combine(Environment.CurrentDirectory, "external_evaluation.csv"));
            int limit = GetIntArg(args, "--limit", int.MaxValue);
            string? imageFilter = GetStringArg(args, "--images");
            Metrics.EvalMaxSide = int.MaxValue;

            if (!Directory.Exists(predictionsDir)) throw new DirectoryNotFoundException(predictionsDir);
            var dataset = DatasetCatalog.Load(manifest, inputDir, gtDir, split, imageFilter, limit);
            var predictionFiles = Directory.GetFiles(predictionsDir, "*.*", SearchOption.AllDirectories)
                .Where(IsImageFile).ToArray();
            Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);

            using var writer = new StreamWriter(outPath, false, new System.Text.UTF8Encoding(true));
            writer.WriteLine("dataset;image;method;ok;error;width;height;prediction;psnr;ssim;ciede2000;mse;clip_pct;flat_noise_x;psnr_aligned_diag;ssim_aligned_diag;ciede2000_aligned_diag");
            int ok = 0, fail = 0;
            var used = new List<string>();
            foreach (var item in dataset.Cases)
            {
                try
                {
                    if (item.ClearPath == null) throw new InvalidOperationException("GT не указан в manifest");
                    string prediction = FindPrediction(predictionFiles, item);
                    using var input = new Image<Bgr, byte>(item.HazyPath);
                    using var gt = new Image<Bgr, byte>(item.ClearPath);
                    using var pred = new Image<Bgr, byte>(prediction);
                    if (pred.Size != input.Size)
                        throw new InvalidDataException($"размер результата {pred.Width}x{pred.Height}, ожидается {input.Width}x{input.Height}");
                    using var f = new Mat(); pred.Mat.ConvertTo(f, DepthType.Cv32F, 1.0 / 255.0);
                    var report = Metrics.Evaluate(f, gt.Mat, input.Mat);
                    static string N(double v) => v.ToString("0.######", System.Globalization.CultureInfo.InvariantCulture);
                    writer.WriteLine(string.Join(";", new[]
                    {
                        dataset.Name, item.Id, methodName, "1", "", input.Width.ToString(), input.Height.ToString(), prediction,
                        N(report.Psnr), N(report.Ssim), N(report.Ciede2000), N(report.Mse), N(report.ClipPct), N(report.FlatNoiseRatio),
                        N(report.PsnrAligned), N(report.SsimAligned), N(report.Ciede2000Aligned)
                    }.Select(Csv)));
                    used.Add(prediction); ok++;
                }
                catch (Exception ex)
                {
                    string error = (ex.GetType().Name + ": " + ex.Message).Replace('\r', ' ').Replace('\n', ' ');
                    writer.WriteLine(string.Join(";", new[] { dataset.Name, item.Id, methodName, "0", error, "", "", "", "", "", "", "", "", "", "", "", "" }.Select(Csv)));
                    fail++;
                }
            }
            writer.Flush();

            string metaPath = Path.ChangeExtension(outPath, ".meta.json");
            var metadata = new
            {
                generated = DateTime.Now,
                dataset = new { dataset.Name, dataset.Version, manifest = manifest == null ? null : Path.GetFullPath(manifest) },
                split, method = methodName, predictions_dir = predictionsDir, rows = dataset.Cases.Length, ok, fail,
                commit = GitCommit(), dirty = GitDirty(), predictions = used
            };
            File.WriteAllText(metaPath, JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }), new System.Text.UTF8Encoding(false));
            Console.WriteLine($"EVALUATE dataset={dataset.Name} method={methodName} rows={dataset.Cases.Length} ok={ok} fail={fail}");
            Console.WriteLine($"  csv:  {outPath}");
            Console.WriteLine($"  meta: {metaPath}");
        }

        private static bool IsImageFile(string path)
            => new[] { ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff" }.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase);

        private static string FindPrediction(string[] files, BenchmarkCase item)
        {
            string hazyName = Path.GetFileName(item.HazyPath);
            var exact = files.FirstOrDefault(f => Path.GetFileName(f).Equals(hazyName, StringComparison.OrdinalIgnoreCase));
            if (exact != null) return exact;
            var byId = files.Where(f => Path.GetFileNameWithoutExtension(f).Equals(item.Id, StringComparison.OrdinalIgnoreCase)).ToArray();
            return byId.Length switch
            {
                1 => byId[0],
                0 => throw new FileNotFoundException($"результат для {item.Id} не найден"),
                _ => throw new InvalidDataException($"для {item.Id} найдено несколько результатов")
            };
        }

        /// <summary>
        /// Параметры метода для прогона. Core применяется только через явный профиль метода;
        /// неизвестные методы не маскируются под физическое ядро.
        /// <paramref name="linear"/> включает линейный радианс там, где метод это поддерживает.
        /// </summary>
        private static (Dictionary<string, double> Params, bool ProfileSupported) BenchParams(
            IDeHazeMethod m, string profile, bool linear, string? overrides)
        {
            var p = m.Parameters.ToDictionary(x => x.Key, x => x.Default);
            bool supported = profile == "full" || BenchmarkProfiles.TryApplyCore(m, p);
            if (linear && p.ContainsKey("linear")) p["linear"] = 1.0;
            ApplyParamOverrides(p, overrides);
            return (p, supported);
        }

        /// <summary>Строки метаданных прогона для шапки CSV (окружение + настройки).</summary>
        private static string[] RunMetadata(string dataset, string? datasetVersion, string mode, int maxDim,
            int images, int methods, int warmup, int repeat, bool memory, HardwareSnapshot hw) => new[]
        {
            $"generated={DateTime.Now:yyyy-MM-dd HH:mm:ss zzz}",
            $"dataset={dataset} version={datasetVersion ?? "unspecified"}",
            $"mode={mode} (core = только явно размеченный core-профиль, full = полный конвейер)",
            $"maxdim={(maxDim == int.MaxValue ? "native" : maxDim)} eval_max_side={(Metrics.EvalMaxSide == int.MaxValue ? "full" : Metrics.EvalMaxSide.ToString())}",
            $"warmup={warmup} repeat={repeat} memory_pass={memory}",
            $"images={images}  methods={methods}",
            $"machine={hw.Machine} os={hw.Os} cpu={hw.Cpu} logical_cores={hw.LogicalCores}",
            $"runtime={hw.Runtime} arch={hw.Architecture}",
            $"opencv={hw.OpenCv} cuda_devices={hw.CudaDevices} gpu={hw.Gpu ?? "none"} gpu_total_mb={hw.GpuTotalMb?.ToString("F0") ?? "n/a"} driver={hw.GpuDriver ?? "n/a"}",
            $"commit={GitCommit()}",
        };

        private static string MetadataJson(string dataset, string? datasetVersion, string? manifest, string mode,
            int maxDim, int warmup, int repeat, bool memory, BenchmarkCase[] cases,
            List<(IDeHazeMethod Method, Dictionary<string, double> Params, bool ProfileSupported)> runs,
            int rows, int ok, int fail, HardwareSnapshot hardware, bool computeLpips, string? lpipsDevice)
        {
            var payload = new
            {
                generated = DateTime.Now,
                dataset = new { name = dataset, version = datasetVersion, manifest = manifest == null ? null : Path.GetFullPath(manifest) },
                mode,
                max_dim = maxDim == int.MaxValue ? (int?)null : maxDim,
                eval_max_side = Metrics.EvalMaxSide == int.MaxValue ? (int?)null : Metrics.EvalMaxSide,
                timing = new { warmup, repeat, statistic = "median", memory_pass = memory },
                lpips = computeLpips
                    ? new { enabled = true, package = (string?)"lpips 0.1.4", network = (string?)"AlexNet v0.1/ImageNet", device = lpipsDevice, included_in_method_timing = false, gpu_memory_columns = "blank to avoid attributing evaluator VRAM to CPU method" }
                    : new { enabled = false, package = (string?)null, network = (string?)null, device = (string?)null, included_in_method_timing = false, gpu_memory_columns = "algorithm measurement" },
                rows, ok, fail,
                environment = hardware,
                commit = GitCommit(),
                dirty = GitDirty(),
                images = cases.Select(c => new { c.Id, hazy = c.HazyPath, clear = c.ClearPath, c.Split }),
                methods = runs.Select(r => new { name = r.Method.Name, profile_supported = r.ProfileSupported, parameters = r.Params })
            };
            return JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true });
        }

        /// <summary>Первая строка BuildInformation OpenCV - это «General configuration for OpenCV x.y.z».</summary>
        private static string OpenCvVersion()
        {
            try
            {
                var first = CvInvoke.BuildInformation.Split('\n').FirstOrDefault()?.Trim();
                if (!string.IsNullOrEmpty(first)) return first;
            }
            catch { /* в некоторых сборках Emgu BuildInformation недоступен */ }
            try
            {
                var v = typeof(CvInvoke).Assembly.GetName().Version;
                return v != null ? $"Emgu.CV {v}" : "unknown";
            }
            catch { return "unknown"; }
        }

        private static int CudaDeviceCount()
        {
            try { return Emgu.CV.Cuda.CudaInvoke.HasCuda ? Emgu.CV.Cuda.CudaInvoke.GetCudaEnabledDeviceCount() : 0; }
            catch { return 0; }
        }

        /// <summary>Хеш текущего коммита - чтобы строку CSV можно было привязать к состоянию кода.</summary>
        private static string GitCommit()
        {
            try
            {
                var psi = new ProcessStartInfo("git", "rev-parse --short HEAD")
                {
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    WorkingDirectory = AppContext.BaseDirectory,
                };
                using var proc = Process.Start(psi);
                if (proc == null) return "unknown";
                string outp = proc.StandardOutput.ReadToEnd().Trim();
                proc.WaitForExit(3000);
                return string.IsNullOrEmpty(outp) ? "unknown" : outp;
            }
            catch { return "unknown"; }
        }

        private static bool? GitDirty()
        {
            try
            {
                var psi = new ProcessStartInfo("git", "status --porcelain")
                {
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    WorkingDirectory = AppContext.BaseDirectory,
                };
                using var proc = Process.Start(psi);
                if (proc == null) return null;
                string output = proc.StandardOutput.ReadToEnd();
                if (!proc.WaitForExit(3000)) return null;
                return output.Length > 0;
            }
            catch { return null; }
        }

        private static int GetIntArg(string[] args, string key, int fallback)
        {
            string prefix = key + "=";
            var s = args.FirstOrDefault(a => a.StartsWith(prefix, StringComparison.OrdinalIgnoreCase));
            return s != null && int.TryParse(s[prefix.Length..], out int v) ? v : fallback;
        }

        private static string? GetStringArg(string[] args, string key)
        {
            string prefix = key + "=";
            return args.FirstOrDefault(a => a.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))?[prefix.Length..];
        }

        private static AutoTuneGoal ParseAutoTuneGoal(string? value, AutoTuneGoal fallback)
        {
            if (string.IsNullOrWhiteSpace(value)) return fallback;
            return value.Trim().ToLowerInvariant() switch
            {
                "ref" or "reference" or "gt" => AutoTuneGoal.Reference,
                "vivid" or "color" or "colour" => AutoTuneGoal.Vivid,
                "obj" or "object" or "objects" or "visibility" => AutoTuneGoal.ObjectVisibility,
                _ => throw new ArgumentException($"Неизвестная цель автоподбора '{value}'. Ожидалось ref, obj или vivid."),
            };
        }

        private static bool MatchesFilter(string value, string? regex)
        {
            if (string.IsNullOrWhiteSpace(regex))
                return true;
            return System.Text.RegularExpressions.Regex.IsMatch(value, regex, System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        }

        /// <summary>Отрендерить ВСЕ методы (параметры по умолчанию) в файлы NN.png + names.txt. Опции: --image, --maxdim, --out.</summary>
        private static void RenderAll(string[] args)
        {
            string imagePath = GetStringArg(args, "--image")
                ?? Path.Combine(AppContext.BaseDirectory, "dataset", "09_outdoor_hazy.jpg");
            if (!File.Exists(imagePath)) { Console.WriteLine("RENDERALL-FAIL no image: " + imagePath); return; }
            int maxDim = GetIntArg(args, "--maxdim", 1400);
            string outDir = GetStringArg(args, "--out") ?? Path.Combine(Environment.CurrentDirectory, "renderall_out");
            Directory.CreateDirectory(outDir);

            using var full = new Image<Bgr, byte>(imagePath);
            double sc = Math.Min(1.0, (double)maxDim / Math.Max(full.Width, full.Height));
            using var img = sc >= 1.0 ? full.Clone() : full.Resize((int)(full.Width * sc), (int)(full.Height * sc), Inter.Area);

            using var names = new StreamWriter(Path.Combine(outDir, "names.txt"), false, new System.Text.UTF8Encoding(true));
            var methods = MethodRegistry.All;
            for (int i = 0; i < methods.Count; i++)
            {
                var m = methods[i];
                var p = m.Parameters.ToDictionary(x => x.Key, x => x.Default);
                try
                {
                    using var res = m.Process(img, p);
                    using var r8 = new Mat(); res.ConvertTo(r8, DepthType.Cv8U, 255.0);
                    CvInvoke.Imwrite(Path.Combine(outDir, $"{i:00}.png"), r8);
                    names.WriteLine($"{i:00}\t{m.Name}");
                    Console.WriteLine($"RENDERALL {i:00} OK {m.Name}");
                }
                catch (Exception ex) { Console.WriteLine($"RENDERALL {i:00} FAIL {m.Name}: {ex.Message}"); }
            }
            Console.WriteLine($"RENDERALL-DONE size={img.Width}x{img.Height} out={outDir}");
        }

        /// <summary>Сохранить A²CR result и диагностические карты t, uncertainty, gains и alpha.</summary>
        private static void A2crDiagnosticRun(string[] args)
        {
            string imagePath = GetStringArg(args, "--image")
                ?? Path.Combine(Environment.CurrentDirectory, "SimpleDeHaze", "dataset", "09_outdoor_hazy.jpg");
            if (!File.Exists(imagePath)) imagePath = Path.Combine(AppContext.BaseDirectory, "dataset", "09_outdoor_hazy.jpg");
            if (!File.Exists(imagePath)) throw new FileNotFoundException("A²CR diagnostic input not found", imagePath);
            int maxDim = GetIntArg(args, "--maxdim", 1200);
            string outDir = GetStringArg(args, "--out") ?? Path.Combine(Environment.CurrentDirectory, "a2cr_diagnostics");
            using var full = new Image<Bgr, byte>(imagePath);
            double scale = Math.Min(1, maxDim / (double)Math.Max(full.Width, full.Height));
            using var input = scale >= 1 ? full.Clone() : full.Resize((int)(full.Width * scale), (int)(full.Height * scale), Inter.Area);
            var method = new A2crMethod();
            var parameters = method.Parameters.ToDictionary(x => x.Key, x => x.Default);
            ApplyParamOverrides(parameters, GetStringArg(args, "--params"));
            using var execution = A2crMethod.Execute(input, parameters);
            A2crDiagnostics.Save(execution, outDir, parameters);
            var d = execution.Recovery.Diagnostics;
            Console.WriteLine($"A2CR-DIAG-OK size={input.Width}x{input.Height} g_parallel={d.MeanGainParallel:F3} g_perp={d.MeanGainPerpendicular:F3} alpha={d.MeanAlpha:F4} projected={d.ProjectedPixelFraction:P2} invalid_after={d.InvalidChannelFractionAfter:P4} out={Path.GetFullPath(outDir)}");
        }

        /// <summary>Сохранить результат одного метода без GUI. Опции: --image, --method, --params=k=v,..., --maxdim, --roi, --out.</summary>
        private static void RenderOne(string[] args)
        {
            string imagePath = GetStringArg(args, "--image")
                ?? Path.Combine(Environment.CurrentDirectory, "SimpleDeHaze", "dataset", "09_outdoor_hazy.jpg");
            if (!File.Exists(imagePath))
                imagePath = Path.Combine(AppContext.BaseDirectory, "dataset", "09_outdoor_hazy.jpg");
            if (!File.Exists(imagePath)) { Console.WriteLine("RENDER-FAIL image not found: " + imagePath); return; }

            string methodFilter = GetStringArg(args, "--method") ?? "Силуэт";
            var method = MethodRegistry.All.FirstOrDefault(m => MatchesFilter(m.Name, methodFilter));
            if (method == null) { Console.WriteLine("RENDER-FAIL method not found: " + methodFilter); return; }

            int maxDim = GetIntArg(args, "--maxdim", 1100);
            string outDir = GetStringArg(args, "--out") ?? Path.Combine(Environment.CurrentDirectory, "render_out");
            Directory.CreateDirectory(outDir);

            using var full = new Image<Bgr, byte>(imagePath);
            double sc = Math.Min(1.0, (double)maxDim / Math.Max(full.Width, full.Height));
            using var img = sc >= 1.0 ? full.Clone() : full.Resize((int)(full.Width * sc), (int)(full.Height * sc), Inter.Area);
            var p = method.Parameters.ToDictionary(x => x.Key, x => x.Default);
            ApplyParamOverrides(p, GetStringArg(args, "--params"));

            var sw = Stopwatch.StartNew();
            using var result = method.Process(img, p);
            sw.Stop();
            using var result8 = new Mat();
            result.ConvertTo(result8, DepthType.Cv8U, 255.0);

            string safe = SafeName(method.Name);
            string fullPath = Path.Combine(outDir, safe + "_full.png");
            CvInvoke.Imwrite(fullPath, result8);

            var roi = ParseRoi(GetStringArg(args, "--roi"), img.Width, img.Height);
            using var crop = new Mat(result8, roi);
            string cropPath = Path.Combine(outDir, safe + "_crop.png");
            CvInvoke.Imwrite(cropPath, crop);

            Console.WriteLine($"RENDER-DONE method={method.Name} size={img.Width}x{img.Height} ms={sw.ElapsedMilliseconds} full={fullPath} crop={cropPath}");
            Console.WriteLine(SpotParams(p));
        }

        private static void ApplyParamOverrides(Dictionary<string, double> p, string? raw)
        {
            if (string.IsNullOrWhiteSpace(raw))
                return;
            foreach (var part in raw.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                var kv = part.Split('=', 2, StringSplitOptions.TrimEntries);
                if (kv.Length == 2 && p.ContainsKey(kv[0]) &&
                    double.TryParse(kv[1], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double value))
                    p[kv[0]] = value;
            }
        }

        private static string SafeName(string name)
        {
            var invalid = Path.GetInvalidFileNameChars();
            return new string(name.Select(ch => invalid.Contains(ch) || char.IsWhiteSpace(ch) ? '_' : ch).ToArray());
        }

        /// <summary>
        /// Локальный подбор VisibilityBoost под конкретную область кадра.
        /// По умолчанию: 09_outdoor, правый верхний угол с деревом.
        /// Опции: --image=path, --gt=path, --maxdim=N, --roi=x,y,w,h (доли кадра или пиксели), --out=dir.
        /// </summary>
        private static void SpotTune(string[] args)
        {
            string imagePath = GetStringArg(args, "--image")
                ?? Path.Combine(Environment.CurrentDirectory, "SimpleDeHaze", "dataset", "09_outdoor_hazy.jpg");
            string? gtPath = GetStringArg(args, "--gt")
                ?? Path.Combine(Environment.CurrentDirectory, "SimpleDeHaze", "hazefree", "09_outdoor_GT.jpg");
            int maxDim = GetIntArg(args, "--maxdim", 1000);
            string outDir = GetStringArg(args, "--out") ?? Path.Combine(Environment.CurrentDirectory, "spot_tune_09_tree");

            if (!File.Exists(imagePath))
                imagePath = Path.Combine(AppContext.BaseDirectory, "dataset", "09_outdoor_hazy.jpg");
            if (gtPath != null && !File.Exists(gtPath))
                gtPath = Path.Combine(AppContext.BaseDirectory, "hazefree", "09_outdoor_GT.jpg");
            if (!File.Exists(imagePath)) { Console.WriteLine("SPOT-TUNE-FAIL image not found: " + imagePath); return; }
            if (gtPath != null && !File.Exists(gtPath)) gtPath = null;

            using var full = new Image<Bgr, byte>(imagePath);
            double sc = Math.Min(1.0, (double)maxDim / Math.Max(full.Width, full.Height));
            using var img = sc >= 1.0 ? full.Clone() : full.Resize((int)(full.Width * sc), (int)(full.Height * sc), Inter.Area);

            Mat? gt = null;
            if (gtPath != null)
            {
                using var gtFull = new Image<Bgr, byte>(gtPath);
                using var gtResized = gtFull.Resize(img.Width, img.Height, Inter.Area);
                gt = gtResized.Mat.Clone();
            }

            var roi = ParseRoi(GetStringArg(args, "--roi"), img.Width, img.Height);
            Directory.CreateDirectory(outDir);
            using (var inputCrop = new Mat(img.Mat, roi))
                CvInvoke.Imwrite(Path.Combine(outDir, "00_input_crop.png"), inputCrop);
            if (gt != null)
            {
                using var gtCrop = new Mat(gt, roi);
                CvInvoke.Imwrite(Path.Combine(outDir, "00_gt_crop.png"), gtCrop);
            }

            var method = new VisibilityBoostMethod();
            var baseParams = method.Parameters.ToDictionary(x => x.Key, x => x.Default);
            var candidates = new List<(double score, Metrics.Report roi, Metrics.Report full, long ms, Dictionary<string, double> p)>();

            double[] omegas = { 0.82, 0.90 };
            double[] tmins = { 0.06, 0.10 };
            double[] chromas = { 0.32, 0.44 };
            double[] refines = { 22 };
            double[] tskys = { 0.62, 0.72 };
            double[] clips = { 4.2, 5.8 };
            double[] sats = { 0.50, 0.68 };
            double[] details = { 0.26, 0.40 };
            double[] colors = { 1.55 };
            double[] smooths = { 0.0, 2.0, 4.0 };

            int total = omegas.Length * tmins.Length * chromas.Length * refines.Length * tskys.Length *
                clips.Length * sats.Length * details.Length * colors.Length * smooths.Length;
            int tested = 0;
            using var inputRoi = new Mat(img.Mat, roi);
            Mat? gtRoi = gt != null ? new Mat(gt, roi) : null;
            try
            {
                foreach (double omega in omegas)
                    foreach (double tmin in tmins)
                        foreach (double chroma in chromas)
                            foreach (double refine in refines)
                                foreach (double tsky in tskys)
                                    foreach (double clip in clips)
                                        foreach (double sat in sats)
                                            foreach (double detail in details)
                                                foreach (double color in colors)
                                                    foreach (double smooth in smooths)
                                                    {
                                                        var p = new Dictionary<string, double>(baseParams)
                                                        {
                                                            ["omega"] = omega,
                                                            ["min"] = tmin,
                                                            ["chroma"] = chroma,
                                                            ["refine"] = refine,
                                                            ["tsky"] = tsky,
                                                            ["clip"] = clip,
                                                            ["sat"] = sat,
                                                            ["detail"] = detail,
                                                            ["color"] = color,
                                                            ["smooth"] = smooth,
                                                            ["patch"] = 5,
                                                            ["tiles"] = 8
                                                        };

                                                        var sw = Stopwatch.StartNew();
                                                        using var res = method.Process(img, p);
                                                        sw.Stop();
                                                        using var resRoi = new Mat(res, roi);
                                                        var roiRep = Metrics.Evaluate(resRoi, gtRoi, inputRoi);
                                                        var fullRep = Metrics.Evaluate(res, gt, img.Mat);
                                                        double score = TreeVisibilityScore(roiRep);
                                                        candidates.Add((score, roiRep, fullRep, sw.ElapsedMilliseconds, p));

                                                        tested++;
                                                        if (tested % 100 == 0)
                                                            Console.WriteLine($"SPOT-TUNE {tested}/{total} best={candidates.Max(x => x.score):F1}");
                                                    }
            }
            finally
            {
                gtRoi?.Dispose();
                gt?.Dispose();
            }

            var rankedAll = candidates
                .OrderByDescending(x => x.score)
                .ThenByDescending(x => x.roi.EdgeGain)
                .ThenByDescending(x => x.roi.ContrastGain)
                .ToArray();
            var ranked = rankedAll.Take(12).ToArray();

            using (var writer = new StreamWriter(Path.Combine(outDir, "spot_tune.csv"), false, new System.Text.UTF8Encoding(true)))
            {
                writer.WriteLine("rank;score;roi_ciedeA;roi_ssimA;roi_haze;roi_contrast;roi_edge;roi_clip;roi_color;full_score;full_ciedeA;full_ssimA;ms;params");
                for (int i = 0; i < rankedAll.Length; i++)
                {
                    var c = rankedAll[i];
                    writer.WriteLine(string.Join(";",
                        (i + 1).ToString(System.Globalization.CultureInfo.InvariantCulture),
                        N(c.score), N(c.roi.Ciede2000Aligned), N(c.roi.SsimAligned), N(c.roi.HazeRemoved * 100.0),
                        N(c.roi.ContrastGain), N(c.roi.EdgeGain), N(c.roi.ClipPct), N(c.roi.ColorRatio),
                        N(c.full.Score), N(c.full.Ciede2000Aligned), N(c.full.SsimAligned),
                        c.ms.ToString(System.Globalization.CultureInfo.InvariantCulture), Csv(SpotParams(c.p))));
                }
            }

            for (int i = 0; i < Math.Min(8, ranked.Length); i++)
            {
                var c = ranked[i];
                using var res = method.Process(img, c.p);
                using var r8 = new Mat();
                res.ConvertTo(r8, DepthType.Cv8U, 255.0);
                string stem = $"{i + 1:00}_score{c.score:F1}_edge{c.roi.EdgeGain:F1}_contrast{c.roi.ContrastGain:F1}".Replace(',', '.');
                CvInvoke.Imwrite(Path.Combine(outDir, stem + "_full.png"), r8);
                using var crop = new Mat(r8, roi);
                CvInvoke.Imwrite(Path.Combine(outDir, stem + "_crop.png"), crop);
            }

            Console.WriteLine($"SPOT-TUNE-DONE image={Path.GetFileName(imagePath)} size={img.Width}x{img.Height} roi={roi.X},{roi.Y},{roi.Width},{roi.Height} total={tested} out={outDir}");
            for (int i = 0; i < Math.Min(8, ranked.Length); i++)
            {
                var c = ranked[i];
                Console.WriteLine($"{i + 1,2}. score={c.score,5:F1} roi: DEa={c.roi.Ciede2000Aligned,5:F2} SSIMa={c.roi.SsimAligned:F3} haze={c.roi.HazeRemoved * 100,5:F1}% contrast={c.roi.ContrastGain:F2} edge={c.roi.EdgeGain:F2} clip={c.roi.ClipPct:F2}% color={c.roi.ColorRatio:F2} fullScore={c.full.Score:F1} :: {SpotParams(c.p)}");
            }

            static string N(double v) => double.IsNaN(v) ? "" : v.ToString("0.######", System.Globalization.CultureInfo.InvariantCulture);
        }

        private static System.Drawing.Rectangle ParseRoi(string? raw, int width, int height)
        {
            double[] v = raw?.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(s => double.Parse(s, System.Globalization.CultureInfo.InvariantCulture))
                .ToArray()
                ?? new[] { 0.70, 0.00, 0.30, 0.28 };
            if (v.Length != 4) throw new ArgumentException("--roi должен быть x,y,w,h");

            bool fractional = v.All(x => x >= 0.0 && x <= 1.0);
            int x = fractional ? (int)Math.Round(v[0] * width) : (int)Math.Round(v[0]);
            int y = fractional ? (int)Math.Round(v[1] * height) : (int)Math.Round(v[1]);
            int w = fractional ? (int)Math.Round(v[2] * width) : (int)Math.Round(v[2]);
            int h = fractional ? (int)Math.Round(v[3] * height) : (int)Math.Round(v[3]);
            x = Math.Clamp(x, 0, width - 2);
            y = Math.Clamp(y, 0, height - 2);
            w = Math.Clamp(w, 2, width - x);
            h = Math.Clamp(h, 2, height - y);
            return new System.Drawing.Rectangle(x, y, w, h);
        }

        private static double TreeVisibilityScore(Metrics.Report r)
        {
            double edge = Math.Clamp((r.EdgeGain - 1.0) / 5.5, 0.0, 1.0);
            double contrast = Math.Clamp((r.ContrastGain - 1.0) / 3.2, 0.0, 1.0);
            double haze = Math.Clamp(r.HazeRemoved, 0.0, 1.0);
            double colorOk = Math.Clamp((1.85 - Math.Abs(r.ColorRatio - 1.45)) / 1.85, 0.0, 1.0);
            double refShape = r.HasRef ? Math.Clamp((18.0 - r.Ciede2000Aligned) / 18.0, 0.0, 1.0) : 0.5;
            double ssim = r.HasRef ? Math.Clamp((r.SsimAligned - 0.35) / 0.45, 0.0, 1.0) : 0.5;
            double clipPenalty = Math.Clamp(r.ClipPct / 2.0, 0.0, 1.0);
            double noisyEdge = Math.Clamp((r.EdgeGain - 18.0) / 18.0, 0.0, 1.0);
            double noisyContrast = Math.Clamp((r.ContrastGain - 12.0) / 12.0, 0.0, 1.0);

            double good = 0.32 * edge + 0.25 * contrast + 0.18 * haze + 0.10 * colorOk + 0.10 * refShape + 0.05 * ssim;
            return 100.0 * Math.Clamp(good - 0.18 * clipPenalty - 0.12 * noisyEdge - 0.08 * noisyContrast, 0.0, 1.0);
        }

        private static string SpotParams(IReadOnlyDictionary<string, double> p)
        {
            static string V(double x) => x.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture);
            string[] preferred = { "omega", "min", "tmin", "chroma", "refine", "tsky", "patch", "clip", "sat", "detail", "color", "alpha", "thr", "small", "large", "gain", "center", "depth", "bg", "base", "eps", "post", "smooth", "context", "overlay" };
            var ordered = preferred
                .Where(p.ContainsKey)
                .Concat(p.Keys.OrderBy(k => k, StringComparer.OrdinalIgnoreCase).Where(k => !preferred.Contains(k)))
                .Select(k => $"{k}={V(p[k])}");
            return string.Join(" ", ordered);
        }

        private static void WriteBenchRow(StreamWriter writer, string dataset, string image, string method,
            string mode, bool profileSupported, int sourceWidth, int sourceHeight, int processedWidth,
            int processedHeight, int repeats, MeasurementResult measurement, double msPerMp, Metrics.Report r,
            double lpips, string lpipsStatus)
        {
            static string N(double v) => double.IsFinite(v) ? v.ToString("0.######", System.Globalization.CultureInfo.InvariantCulture) : "";
            static string O(double? v) => v.HasValue ? N(v.Value) : "";
            var fields = new[]
            {
                dataset, image, method, mode, profileSupported ? "1" : "0", "1", "",
                sourceWidth.ToString(), sourceHeight.ToString(), processedWidth.ToString(), processedHeight.ToString(), repeats.ToString(),
                N(measurement.MedianMs), N(measurement.MinMs), N(measurement.P95Ms), N(msPerMp),
                N(measurement.PeakWorkingSetMb), N(measurement.PeakWorkingSetDeltaMb), O(measurement.PeakGpuUsedMb), O(measurement.PeakGpuDeltaMb),
                N(r.Score),
                N(r.Psnr), N(r.Ssim), N(r.Ciede2000), N(lpips), lpipsStatus, N(r.Mse),             // первичные
                N(r.PsnrAligned), N(r.SsimAligned), N(r.Ciede2000Aligned), N(r.MseAligned),       // диагностические
                N(r.NaturalnessDev), N(r.ArtifactDev), N(r.FlatNoiseRatio),
                N(r.HazeRemoved * 100.0), N(r.ContrastGain), N(r.EdgeGain), N(r.ClipPct), N(r.ColorRatio)
            };
            writer.WriteLine(string.Join(";", fields.Select(Csv)));
        }

        private static void WriteBenchFail(StreamWriter writer, string dataset, string image, string method,
            string mode, bool profileSupported, int sourceWidth, int sourceHeight, int processedWidth,
            int processedHeight, int repeats, Exception ex)
        {
            string err = (ex.GetType().Name + ": " + ex.Message).Replace('\r', ' ').Replace('\n', ' ');
            var fields = new List<string>
            {
                dataset, image, method, mode, profileSupported ? "1" : "0", "0", err,
                sourceWidth.ToString(), sourceHeight.ToString(), processedWidth.ToString(), processedHeight.ToString(), repeats.ToString()
            };
            while (fields.Count < 39) fields.Add("");
            writer.WriteLine(string.Join(";", fields.Select(Csv)));
        }

        private static string Csv(string s)
        {
            if (s.Contains(';') || s.Contains('"') || s.Contains('\n'))
                return "\"" + s.Replace("\"", "\"\"") + "\"";
            return s;
        }

        /// <summary>Прежний пакетный прогон по dataset/ с показом окон OpenCV и записью в result/.</summary>
        private static void Batch()
        {
            var debug = false;
            var files = Directory.GetFiles(Path.Combine(AppContext.BaseDirectory, "dataset"), "*.*");
            var filesClear = Directory.GetFiles(Path.Combine(AppContext.BaseDirectory, "hazefree"), "*.*");
            var path = Environment.CurrentDirectory;
            Directory.CreateDirectory(Path.Combine(path, "result"));
            foreach (var file in files)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                var fileName = Path.GetFileNameWithoutExtension(file);
                var fileCpu = Path.Combine(path, "result", $"{fileName}_Cpu.png");
                var fileGpu = Path.Combine(path, "result", $"{fileName}_Gpu.png");
                fileName = Path.Combine(path, "result", $"{fileName}");
                using var inputImage = new Image<Bgr, byte>(file);
                using var clearImage = new Image<Bgr, byte>(filesClear.FirstOrDefault(x =>
                    file.Replace("hazy", "GT").Replace("dataset", "hazefree").Equals(x, StringComparison.InvariantCultureIgnoreCase)));
                CvInvoke.NamedWindow("Haze Image", WindowFlags.AutoSize);
                CvInvoke.Imshow("Haze Image", inputImage.Convert<Bgr, byte>().Resize((int)(900f / inputImage.Height * inputImage.Width), 800, Inter.Lanczos4));
                CvInvoke.NamedWindow("Clear Image", WindowFlags.AutoSize);
                CvInvoke.Imshow("Clear Image", clearImage.Convert<Bgr, byte>().Resize((int)(900f / inputImage.Height * inputImage.Width), 800, Inter.Lanczos4));
                var patch = (int)(inputImage.Size.Height > inputImage.Size.Width ? inputImage.Size.Width * 0.01 + 1 : inputImage.Size.Height * 0.001 + 1);
                if (debug) CvInvoke.WaitKey();

                DeHazeCPU deHazeCPU = new();
                var _deHazeCPU = deHazeCPU.RemoveHaze(inputImage.Clone(), debug: debug, beta: 0.5f, patchDarkChannel: (int)(patch * 0.5), decompositionSize: (int)(patch * 0.5), min: 2 / 255f, percen: 0.5f, refineSize: (int)(patch * 2), eps: 0.001d / patch, fileName);
                CvInvoke.NamedWindow("DeHazeCPU Image", WindowFlags.AutoSize);
                CvInvoke.Imshow("DeHazeCPU Image", (_deHazeCPU * 255).ToImage<Bgr, byte>().Resize((int)(900f / inputImage.Height * inputImage.Width), 800, Inter.Lanczos4));
                CvInvoke.Imwrite(fileCpu, _deHazeCPU * 255, new KeyValuePair<ImwriteFlags, int>(ImwriteFlags.PngCompression, 0));
                if (debug) CvInvoke.WaitKey();

                using DeHazeGPU deHazeGPU = new();
                using var _deHazeGPU = deHazeGPU.RemoveHaze(inputImage.Clone(), debug: debug, beta: 0.5f, patchDarkChannel: (int)(patch * 0.5), decompositionSize: (int)(patch * 0.5), min: 2 / 255f, percen: 0.5f, refineSize: (int)(patch * 2), eps: 0.001d / patch, fileName);
                CvInvoke.NamedWindow("DeHazeGPU Image", WindowFlags.AutoSize);
                CvInvoke.Imshow("DeHazeGPU Image", (_deHazeGPU * 255).ToImage<Bgr, byte>().Resize((int)(900f / inputImage.Height * inputImage.Width), 800, Inter.Lanczos4));
                CvInvoke.Imwrite(fileGpu, _deHazeGPU * 255, new KeyValuePair<ImwriteFlags, int>(ImwriteFlags.PngCompression, 0));
                if (debug) CvInvoke.WaitKey();
            }
        }
    }
}
