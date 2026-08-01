using System.Diagnostics;
using System.Text.Json;

using Emgu.CV;
using Emgu.CV.Cuda;

using Microsoft.Win32;

using SimpleDeHaze.Methods;

namespace SimpleDeHaze.Benchmarking;

internal sealed record BenchmarkCase(string Id, string HazyPath, string? ClearPath, string Split);

internal sealed class DatasetManifest
{
    public string Name { get; set; } = "dataset";
    public string? Version { get; set; }
    public string? Root { get; set; }
    public List<DatasetPair> Pairs { get; set; } = new();

    internal sealed class DatasetPair
    {
        public string Id { get; set; } = "";
        public string Hazy { get; set; } = "";
        public string? Clear { get; set; }
        public string Split { get; set; } = "all";
        public string? Sha256Hazy { get; set; }
        public string? Sha256Clear { get; set; }
    }
}

internal static class DatasetCatalog
{
    private static readonly string[] ImageExtensions = { ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff" };

    public static (string Name, string? Version, BenchmarkCase[] Cases) Load(
        string? manifestPath, string inputDir, string gtDir, string split, string? imageFilter, int limit)
    {
        IEnumerable<BenchmarkCase> cases;
        string name;
        string? version = null;

        if (!string.IsNullOrWhiteSpace(manifestPath))
        {
            string fullManifest = Path.GetFullPath(manifestPath);
            var model = JsonSerializer.Deserialize<DatasetManifest>(File.ReadAllText(fullManifest),
                new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
                ?? throw new InvalidDataException($"Пустой manifest: {fullManifest}");
            string manifestDir = Path.GetDirectoryName(fullManifest)!;
            string root = Path.GetFullPath(Path.Combine(manifestDir, model.Root ?? "."));
            name = string.IsNullOrWhiteSpace(model.Name) ? Path.GetFileNameWithoutExtension(fullManifest) : model.Name;
            version = model.Version;
            cases = model.Pairs.Select((p, i) =>
            {
                if (string.IsNullOrWhiteSpace(p.Hazy)) throw new InvalidDataException($"pairs[{i}].hazy отсутствует");
                string id = string.IsNullOrWhiteSpace(p.Id) ? Path.GetFileNameWithoutExtension(p.Hazy) : p.Id;
                string hazy = Path.GetFullPath(Path.Combine(root, p.Hazy));
                string? clear = string.IsNullOrWhiteSpace(p.Clear) ? null : Path.GetFullPath(Path.Combine(root, p.Clear));
                ValidateFile(hazy, p.Sha256Hazy, $"{id}: hazy");
                if (clear != null) ValidateFile(clear, p.Sha256Clear, $"{id}: clear");
                return new BenchmarkCase(id, hazy, clear, string.IsNullOrWhiteSpace(p.Split) ? "all" : p.Split.ToLowerInvariant());
            });
        }
        else
        {
            inputDir = Path.GetFullPath(inputDir);
            gtDir = Path.GetFullPath(gtDir);
            if (!Directory.Exists(inputDir)) throw new DirectoryNotFoundException($"input-dir отсутствует: {inputDir}");
            name = Path.GetFileName(inputDir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
            var files = Directory.GetFiles(inputDir)
                .Where(IsImage)
                .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
                .ToArray();
            cases = files.Select((f, i) => new BenchmarkCase(
                Path.GetFileNameWithoutExtension(f), f, FindClear(f, gtDir), i % 2 == 0 ? "val" : "test"));
        }

        cases = cases.Where(c => MatchesSplit(c.Split, split))
            .Where(c => MatchesFilter(c.Id + " " + Path.GetFileName(c.HazyPath), imageFilter))
            .Take(limit);
        var result = cases.ToArray();
        if (result.Length == 0) throw new InvalidOperationException("Dataset пуст после --split/--images/--limit");
        return (name, version, result);
    }

    private static bool MatchesSplit(string pairSplit, string requested)
        => requested == "all" || pairSplit == "all" || pairSplit.Equals(requested, StringComparison.OrdinalIgnoreCase);

    private static bool MatchesFilter(string value, string? pattern)
        => string.IsNullOrWhiteSpace(pattern) || System.Text.RegularExpressions.Regex.IsMatch(value, pattern, System.Text.RegularExpressions.RegexOptions.IgnoreCase);

    private static bool IsImage(string path) => ImageExtensions.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase);

    private static string? FindClear(string hazyPath, string gtDir)
    {
        if (!Directory.Exists(gtDir)) return null;
        string file = Path.GetFileName(hazyPath);
        var candidates = new[]
        {
            file.Replace("hazy", "GT", StringComparison.OrdinalIgnoreCase),
            file.Replace("hazy", "clear", StringComparison.OrdinalIgnoreCase),
            file.Replace("haze", "GT", StringComparison.OrdinalIgnoreCase),
            file,
        };
        foreach (string candidate in candidates)
        {
            string exact = Path.Combine(gtDir, candidate);
            if (File.Exists(exact)) return exact;
            string stem = Path.GetFileNameWithoutExtension(candidate);
            var found = Directory.GetFiles(gtDir).FirstOrDefault(f => IsImage(f) && Path.GetFileNameWithoutExtension(f).Equals(stem, StringComparison.OrdinalIgnoreCase));
            if (found != null) return found;
        }
        return null;
    }

    private static void ValidateFile(string path, string? expectedSha256, string label)
    {
        if (!File.Exists(path)) throw new FileNotFoundException($"{label}: файл отсутствует", path);
        if (string.IsNullOrWhiteSpace(expectedSha256)) return;
        using var stream = File.OpenRead(path);
        string actual = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream));
        if (!actual.Equals(expectedSha256, StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"{label}: SHA-256 {actual}, ожидался {expectedSha256}");
    }
}

internal static class BenchmarkProfiles
{
    /// <summary>
    /// Явные профили: если метод здесь не перечислен, benchmark не имеет права называть его результат
    /// «core». Это предотвращает прежнюю ошибку, когда параметры обнулялись только по похожим именам.
    /// </summary>
    public static bool TryApplyCore(IDeHazeMethod method, Dictionary<string, double> p)
    {
        switch (method)
        {
            case CanonicalDcpMethod:
                return true;
            case A2crMethod:
                Set(p, "tv", 0);
                return true;
            case HcvA2crMethod:
            case HcvRgbA2crFusionMethod:
            case HcvA2crUtawMethod:
                return true;
            case HsvA2crMethod:
                Set(p, "tv", 0);
                return true;
            case HsvC3rMethod:
                return true;
            case ColorCubeMethod:
                return true;
            case RfepDcpMethod:
                Set(p, "tone", 0); SetMax(method, p, "color");
                return true;
            case LafTvMethod:
                Set(p, "tone", 0); SetMax(method, p, "color");
                return true;
            case FractalHsvMethod:
                Set(p, "clahe", 0); Set(p, "wb", 0); Set(p, "sat", 0); Set(p, "tone", 0);
                Set(p, "colorRestore", 0); Set(p, "smooth", 0); SetMax(method, p, "color");
                return true;
            case TransScaleLaplacianMethod:
            case TransmissionAwareHsvEdgeMethod:
            case TransmissionAwareHsvUtawMethod:
            case TransmissionAwareHsvUtawGpuMethod:
                Set(p, "wb", 0); Set(p, "sat", 0); Set(p, "tone", 0); Set(p, "smooth", 0); SetMax(method, p, "color");
                return true;
            default:
                return false;
        }
    }

    private static void Set(Dictionary<string, double> p, string key, double value)
    {
        if (p.ContainsKey(key)) p[key] = value;
    }

    private static void SetMax(IDeHazeMethod method, Dictionary<string, double> p, string key)
    {
        var def = method.Parameters.FirstOrDefault(x => x.Key == key);
        if (def != null) p[key] = def.Max;
    }
}

internal sealed record MeasurementResult(Mat Result, double MedianMs, double MinMs, double P95Ms,
    double PeakWorkingSetMb, double PeakWorkingSetDeltaMb, double? PeakGpuUsedMb, double? PeakGpuDeltaMb);

internal static class BenchmarkMeasurement
{
    public static MeasurementResult Run(Func<Mat> action, int warmup, int repeat, bool measureMemory)
    {
        warmup = Math.Max(0, warmup);
        repeat = Math.Max(1, repeat);
        for (int i = 0; i < warmup; i++) using (action()) { }

        var times = new double[repeat];
        Mat? result = null;
        for (int i = 0; i < repeat; i++)
        {
            var sw = Stopwatch.StartNew();
            var next = action();
            sw.Stop();
            result?.Dispose();
            result = next;
            times[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(times);

        double peakWs = CurrentWorkingSetMb(), baseWs = peakWs;
        double? baseGpu = CurrentGpuUsedMb(), peakGpu = baseGpu;
        if (measureMemory)
        {
            using var done = new ManualResetEventSlim(false);
            var sampler = Task.Run(() =>
            {
                while (!done.IsSet)
                {
                    peakWs = Math.Max(peakWs, CurrentWorkingSetMb());
                    double? gpu = CurrentGpuUsedMb();
                    if (gpu.HasValue) peakGpu = Math.Max(peakGpu ?? gpu.Value, gpu.Value);
                    done.Wait(10);
                }
            });
            try { using (action()) { } }
            finally
            {
                done.Set();
                sampler.Wait();
            }
            peakWs = Math.Max(peakWs, CurrentWorkingSetMb());
            double? gpu = CurrentGpuUsedMb();
            if (gpu.HasValue) peakGpu = Math.Max(peakGpu ?? gpu.Value, gpu.Value);
        }

        int p95Index = Math.Clamp((int)Math.Ceiling(times.Length * 0.95) - 1, 0, times.Length - 1);
        double median = times.Length % 2 == 1 ? times[times.Length / 2] : (times[times.Length / 2 - 1] + times[times.Length / 2]) / 2;
        return new MeasurementResult(result!, median, times[0], times[p95Index], peakWs,
            Math.Max(0, peakWs - baseWs), peakGpu, peakGpu.HasValue && baseGpu.HasValue ? Math.Max(0, peakGpu.Value - baseGpu.Value) : null);
    }

    private static double CurrentWorkingSetMb()
    {
        using var process = Process.GetCurrentProcess();
        return process.WorkingSet64 / 1048576.0;
    }

    private static double? CurrentGpuUsedMb()
    {
        try
        {
            if (!CudaInvoke.HasCuda || CudaInvoke.GetCudaEnabledDeviceCount() == 0) return null;
            using var info = new CudaDeviceInfo(CudaInvoke.GetDevice());
            return (info.TotalMemory - info.FreeMemory) / 1048576.0;
        }
        catch { return null; }
    }
}

internal sealed record HardwareSnapshot(string Machine, string Os, string Cpu, int LogicalCores,
    string Runtime, string Architecture, string OpenCv, int CudaDevices, string? Gpu, double? GpuTotalMb, string? GpuDriver);

internal static class HardwareProbe
{
    public static HardwareSnapshot Capture(string openCv)
    {
        int cudaDevices = 0;
        string? gpu = null, driver = null;
        double? total = null;
        try
        {
            cudaDevices = CudaInvoke.HasCuda ? CudaInvoke.GetCudaEnabledDeviceCount() : 0;
            if (cudaDevices > 0)
            {
                using var info = new CudaDeviceInfo(CudaInvoke.GetDevice());
                gpu = info.Name;
                total = info.TotalMemory / 1048576.0;
                driver = RunAndRead("nvidia-smi", "--query-gpu=driver_version --format=csv,noheader").Split('\n', StringSplitOptions.RemoveEmptyEntries).FirstOrDefault()?.Trim();
            }
        }
        catch { }

        string cpu = Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER") ?? "unknown";
        try
        {
            using var key = Registry.LocalMachine.OpenSubKey(@"HARDWARE\DESCRIPTION\System\CentralProcessor\0");
            cpu = key?.GetValue("ProcessorNameString")?.ToString()?.Trim() ?? cpu;
        }
        catch { }

        return new HardwareSnapshot(Environment.MachineName, Environment.OSVersion.VersionString, cpu,
            Environment.ProcessorCount, Environment.Version.ToString(),
            System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture.ToString(),
            openCv, cudaDevices, gpu, total, driver);
    }

    private static string RunAndRead(string file, string arguments)
    {
        try
        {
            var psi = new ProcessStartInfo(file, arguments) { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true };
            using var p = Process.Start(psi);
            if (p == null) return "";
            string text = p.StandardOutput.ReadToEnd();
            p.WaitForExit(3000);
            return text;
        }
        catch { return ""; }
    }
}
