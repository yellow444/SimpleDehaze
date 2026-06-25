using System.Diagnostics;
using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

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
            if (args.Contains("--selftest") || args.Contains("--batch"))
            {
                AttachConsole(-1);
                if (args.Contains("--selftest")) SelfTest();
                else Batch();
                return;
            }

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            // smoke-проверка GUI: построить окно, прокрутить один цикл отрисовки, закрыться
            if (args.Contains("--guismoke")) { AttachConsole(-1); GuiSmoke(); return; }

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

            // обычный режим - GUI (опц. путь к файлу первым аргументом)
            string? file = args.FirstOrDefault(a => !a.StartsWith("--") && File.Exists(a));
            Application.Run(new MainForm(file));
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

            foreach (var m in MethodRegistry.All.Where(x =>
                x.Name.Contains("Спектрально") || x.Name.Contains("Быстрая") || x.Name.Contains("Retinex +")))
            {
                var def = m.Parameters.ToDictionary(p => p.Key, p => p.Default);
                using var r0 = m.Process(img, def); var d = Metrics.Evaluate(r0, null, img.Mat);
                var tuned = AutoTuner.OptimizeThorough(m, img, def);
                using var r1 = m.Process(img, tuned); var t = Metrics.Evaluate(r1, null, img.Mat);
                var tunedC = AutoTuner.OptimizeThorough(m, img, def, 1.2);   // цель: не гасить цвет (>=1.2)
                using var r2 = m.Process(img, tunedC); var tc = Metrics.Evaluate(r2, null, img.Mat);
                Console.WriteLine($"{m.Name,-32} дефолт(оц{d.Score,3:F0} цвет{d.ColorRatio:F2})  тщат(оц{t.Score,3:F0} цвет{t.ColorRatio:F2})  тщат+цвет>=1.2(оц{tc.Score,3:F0} цвет{tc.ColorRatio:F2})");
            }

            // 'Авто-лучший': скан всех + настройка победителя
            var sw = Stopwatch.StartNew();
            var (bm, bp, scanScore) = AutoTuner.PickBest(MethodRegistry.All, img, null);
            sw.Stop();
            using var rb = bm.Process(img, bp);
            double finalScore = Metrics.NoRefScore(rb, img.Mat);
            Console.WriteLine($"\nАВТО-ЛУЧШИЙ -> '{bm.Name}'  (скан {scanScore:F0} -> настроено {finalScore:F0})  за {sw.ElapsedMilliseconds}мс");
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
                    Console.WriteLine($"  OK   {m.Name,-30} PSNR={rep.Psnr,5:F2} совмещ={rep.PsnrAligned,5:F2} SSIM={rep.SsimAligned:F3} оценка={rep.Score,3:F0} дымка{rep.HazeRemoved * 100,3:F0}% пересв{rep.ClipPct,4:F1}% цветx{rep.ColorRatio:F2} {sw.ElapsedMilliseconds}мс");
                }
                catch (Exception ex)
                {
                    sw.Stop();
                    Console.WriteLine($"  FAIL {m.Name,-14} {ex.GetType().Name}: {ex.Message}");
                }
            }

            if (gt != null && rows.Count > 0)
            {
                Console.WriteLine("\n  === рейтинг по СОВМЕЩЁННОМУ PSNR (выровнена экспозиция/ББ - честнее 'сырого') ===");
                foreach (var r in rows.OrderByDescending(x => x.rep.PsnrAligned).Take(12))
                    Console.WriteLine($"    {r.rep.PsnrAligned,5:F2} дБ  SSIM {r.rep.SsimAligned:F3}  (сырой PSNR {r.rep.Psnr,5:F2})  {r.name}");
                Console.WriteLine("\n  === рейтинг по БЕЗ-ЭТАЛОННОЙ оценке (дымка/детали - пересвет/перенасыщение) ===");
                foreach (var r in rows.OrderByDescending(x => x.rep.Score).Take(12))
                    Console.WriteLine($"    {r.rep.Score,3:F0}/100  дымка{r.rep.HazeRemoved * 100,3:F0}%  пересвет {r.rep.ClipPct,4:F1}%  цвет x{r.rep.ColorRatio:F2}  {r.name}");
            }
            gt?.Dispose();
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
