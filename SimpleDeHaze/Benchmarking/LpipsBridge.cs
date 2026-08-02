using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

using Emgu.CV;
using Emgu.CV.CvEnum;

namespace SimpleDeHaze.Benchmarking;

/// <summary>One persistent Python process; images are streamed in memory and never materialized.</summary>
internal sealed class LpipsBridge : IDisposable
{
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern uint GetDllDirectory(uint bufferLength, StringBuilder buffer);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SetDllDirectory(string? pathName);

    private readonly Process _process;
    private readonly Stream _input;
    private readonly List<string> _errors = new();
    private bool _disposed;
    public string Device { get; }

    public LpipsBridge(string repositoryRoot)
    {
        repositoryRoot = Path.GetFullPath(repositoryRoot);
        string script = Path.Combine(repositoryRoot, "tools", "lpips_stream.py");
        string packages = Path.Combine(repositoryRoot, "benchdata", "python_packages");
        string modelCache = Path.Combine(repositoryRoot, "benchdata", "model_cache");
        string checkpoint = Path.Combine(modelCache, "hub", "checkpoints", "alexnet-owt-7be5be79.pth");
        if (!File.Exists(script)) throw new FileNotFoundException("LPIPS stream evaluator is absent", script);
        if (!Directory.Exists(packages)) throw new DirectoryNotFoundException($"LPIPS package directory is absent: {packages}");
        if (!Directory.Exists(modelCache)) throw new DirectoryNotFoundException($"LPIPS model cache is absent: {modelCache}");
        if (!File.Exists(checkpoint)) throw new FileNotFoundException("Verified AlexNet checkpoint is absent; run tools/setup-lpips.ps1", checkpoint);
        var checkpointInfo = new FileInfo(checkpoint);
        if (checkpointInfo.Length != 244_408_911)
            throw new InvalidDataException($"AlexNet checkpoint size mismatch: {checkpointInfo.Length} != 244408911");
        using (var stream = File.OpenRead(checkpoint))
        {
            string hash = Convert.ToHexString(SHA256.HashData(stream));
            if (!hash.Equals("7BE5BE791159472B1FBF3C69796F7CB30DCA7AD8466C2DF70058C37116CDEE02", StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"AlexNet checkpoint SHA-256 mismatch: {hash}");
        }

        var start = new ProcessStartInfo("python")
        {
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            WorkingDirectory = repositoryRoot,
        };
        start.ArgumentList.Add(script);
        start.Environment["PYTHONUNBUFFERED"] = "1";
        start.Environment["TORCH_HOME"] = modelCache;
        string existingPythonPath = start.Environment.TryGetValue("PYTHONPATH", out string? value) ? value ?? "" : "";
        start.Environment["PYTHONPATH"] = string.IsNullOrWhiteSpace(existingPythonPath)
            ? packages : packages + Path.PathSeparator + existingPythonPath;
        _process = StartWithCleanDllDirectory(start);
        _process.ErrorDataReceived += (_, e) => { if (!string.IsNullOrWhiteSpace(e.Data)) lock (_errors) _errors.Add(e.Data); };
        _process.BeginErrorReadLine();
        _input = _process.StandardInput.BaseStream;

        string? ready = null;
        try
        {
            var deadline = Stopwatch.StartNew();
            while (deadline.Elapsed < TimeSpan.FromMinutes(2) && !_process.HasExited)
            {
                TimeSpan remaining = TimeSpan.FromMinutes(2) - deadline.Elapsed;
                string? line = ReadOutputLine(remaining);
                if (line == null) break;
                if (line.StartsWith("READY\t", StringComparison.Ordinal)) { ready = line; break; }
            }
        }
        catch { StopProcess(); throw; }
        if (ready == null)
        {
            StopProcess();
            throw new InvalidOperationException($"LPIPS evaluator did not become ready. {ErrorText()}");
        }
        Device = ready["READY\t".Length..];
    }

    public double Evaluate(Mat resultSrgb, Mat truthSrgb)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        using var result8 = NormalizeSrgbToByte(resultSrgb); using var truth8 = NormalizeSrgbToByte(truthSrgb);
        byte[] resultPng = CvInvoke.Imencode(".png", result8);
        byte[] truthPng = CvInvoke.Imencode(".png", truth8);
        Span<byte> header = stackalloc byte[8];
        BinaryPrimitives.WriteUInt32LittleEndian(header[..4], checked((uint)resultPng.Length));
        BinaryPrimitives.WriteUInt32LittleEndian(header[4..], checked((uint)truthPng.Length));
        _input.Write(header); _input.Write(resultPng); _input.Write(truthPng); _input.Flush();
        string? line;
        while ((line = ReadOutputLine(TimeSpan.FromMinutes(2))) != null)
            if (double.TryParse(line, NumberStyles.Float, CultureInfo.InvariantCulture, out double value)) return value;
        throw new InvalidOperationException($"LPIPS evaluator terminated unexpectedly. {ErrorText()}");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        try
        {
            Span<byte> stop = stackalloc byte[8]; _input.Write(stop); _input.Flush(); _input.Dispose();
            if (!_process.WaitForExit(10_000)) _process.Kill(entireProcessTree: true);
        }
        catch { if (!_process.HasExited) _process.Kill(entireProcessTree: true); }
        _process.Dispose();
    }

    private string ErrorText() { lock (_errors) return string.Join(" | ", _errors.TakeLast(8)); }

    internal static Mat NormalizeSrgbToByte(Mat value)
    {
        if (value.Depth == DepthType.Cv8U) return value.Clone();
        double scale = value.Depth switch
        {
            DepthType.Cv32F or DepthType.Cv64F => 255.0,
            DepthType.Cv16U => 1.0 / 257.0,
            _ => throw new InvalidDataException($"Unsupported LPIPS image depth: {value.Depth}"),
        };
        var result = new Mat();
        value.ConvertTo(result, DepthType.Cv8U, scale);
        return result;
    }
    private string? ReadOutputLine(TimeSpan timeout)
        => _process.StandardOutput.ReadLineAsync().WaitAsync(timeout).GetAwaiter().GetResult();

    private void StopProcess()
    {
        try { if (!_process.HasExited) _process.Kill(entireProcessTree: true); }
        catch { }
        try { _process.WaitForExit(5_000); }
        catch { }
    }

    private static Process StartWithCleanDllDirectory(ProcessStartInfo start)
    {
        var buffer = new StringBuilder(32768);
        uint length = GetDllDirectory((uint)buffer.Capacity, buffer);
        string? previous = length > 0 ? buffer.ToString() : null;
        try
        {
            SetDllDirectory(null);
            return Process.Start(start) ?? throw new InvalidOperationException("Could not start Python LPIPS evaluator");
        }
        finally { SetDllDirectory(previous); }
    }
}
