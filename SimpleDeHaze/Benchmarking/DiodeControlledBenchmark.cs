using System.Buffers.Binary;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using SimpleDeHaze.Methods;

namespace SimpleDeHaze.Benchmarking;

/// <summary>Streaming RGB-D controlled experiment described in REPRODUCIBILITY.md.</summary>
internal static class DiodeControlledBenchmark
{
    private sealed class Manifest
    {
        public string Dataset { get; set; } = "DIODE validation";
        public string SourceRoot { get; set; } = "";
        public string Seed { get; set; } = "simpledehaze-diode-v1";
        public SynthesisSpec Synthesis { get; set; } = new();
        public List<FrameSpec> Frames { get; set; } = new();
    }

    private sealed class SynthesisSpec
    {
        public List<double> TargetTransmissionAtP90Depth { get; set; } = new();
        public List<AirlightSpec> Airlights { get; set; } = new();
        public List<NoiseSpec> NoiseModels { get; set; } = new();
    }

    private sealed class FrameSpec
    {
        public string Id { get; set; } = "";
        public string Domain { get; set; } = "unknown";
        public string SceneGroup { get; set; } = "";
        public string Split { get; set; } = "";
        public string Clear { get; set; } = "";
        public string Depth { get; set; } = "";
        public string DepthMask { get; set; } = "";
    }

    private sealed class AirlightSpec
    {
        public string Id { get; set; } = "";
        public double[] RgbLinear { get; set; } = Array.Empty<double>();
    }

    private sealed class NoiseSpec
    {
        public string Id { get; set; } = "";
        public double GaussianSigma { get; set; }
        public double PoissonPeak { get; set; }
    }

    public static int Run(string manifestPath, string outputPath, string split, int limitFrames,
        int maxDimension, string? variantFilter, bool quick, bool computeLpips)
    {
        manifestPath = Path.GetFullPath(manifestPath);
        outputPath = Path.GetFullPath(outputPath);
        var manifest = JsonSerializer.Deserialize<Manifest>(File.ReadAllText(manifestPath),
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
            ?? throw new InvalidDataException($"Empty DIODE manifest: {manifestPath}");
        if (manifest.Frames.Count == 0) throw new InvalidDataException("DIODE manifest contains no frames");
        if (!Directory.Exists(manifest.SourceRoot)) throw new DirectoryNotFoundException($"DIODE sourceRoot is absent: {manifest.SourceRoot}");

        IEnumerable<FrameSpec> frameQuery = manifest.Frames;
        if (!split.Equals("all", StringComparison.OrdinalIgnoreCase))
            frameQuery = frameQuery.Where(x => x.Split.Equals(split, StringComparison.OrdinalIgnoreCase));
        FrameSpec[] frames = frameQuery.Take(Math.Max(1, quick ? Math.Min(2, limitFrames) : limitFrames)).ToArray();
        if (frames.Length == 0) throw new InvalidOperationException($"No DIODE frames for split '{split}'");

        string[] requestedVariants = string.IsNullOrWhiteSpace(variantFilter)
            ? Enumerable.Range(0, 10).Select(i => $"B{i}").ToArray()
            : variantFilter.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(x => x.ToUpperInvariant()).Distinct().ToArray();
        if (requestedVariants.Any(x => x != "G0" && (x.Length != 2 || x[0] != 'B' || x[1] < '0' || x[1] > '9')))
            throw new ArgumentException("--variants accepts a comma-separated subset of G0 and B0..B9");

        double[] targets = (quick ? manifest.Synthesis.TargetTransmissionAtP90Depth.Take(1) : manifest.Synthesis.TargetTransmissionAtP90Depth).ToArray();
        AirlightSpec[] airlights = (quick ? manifest.Synthesis.Airlights.Take(2) : manifest.Synthesis.Airlights).ToArray();
        NoiseSpec[] noises = (quick ? manifest.Synthesis.NoiseModels.Take(2) : manifest.Synthesis.NoiseModels).ToArray();
        if (targets.Length == 0 || airlights.Length == 0 || noises.Length == 0)
            throw new InvalidDataException("DIODE synthesis grid is empty");

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        using var lpips = computeLpips ? new LpipsBridge(Environment.CurrentDirectory) : null;
        using var writer = new StreamWriter(outputPath, false, new UTF8Encoding(true));
        writer.WriteLine("dataset;split;frame_id;scene_group;domain;clear_path;depth_path;depth_mask_path;hazy_artifact;transmission_artifact;recipe;random_seed;target_t_p90;airlight;airlight_r_linear;airlight_g_linear;airlight_b_linear;noise;gaussian_sigma;poisson_peak;variant;width;height;valid_depth_fraction;depth_p90;beta;t_true_mean;t_hat_rmse;airlight_l2_error;psnr;ssim;ciede2000;lpips;lpips_status;hue_error_deg;hue_chromatic_fraction;chroma_error;invalid_channel_before;invalid_channel_after;required_clip_mean;clip_pct;flat_noise_input_sigma;flat_noise_output_sigma;flat_noise_amplification;g_parallel_rmse;g_perp_rmse;mean_g_parallel;mean_g_perp;mean_alpha;projected_pixel_fraction;ms;working_set_mb;gpu_used_mb");

        int recipes = 0, rows = 0, failures = 0;
        foreach (FrameSpec frame in frames)
        {
            try
            {
                using var data = LoadFrame(manifest.SourceRoot, frame, maxDimension);
                foreach (double target in targets)
                foreach (AirlightSpec airlight in airlights)
                foreach (NoiseSpec noise in noises)
                {
                    recipes++;
                    int seed = StableSeed(manifest.Seed, frame.Id, target.ToString("R", CultureInfo.InvariantCulture), airlight.Id, noise.Id);
                    using var synthesized = Synthesize(data.ClearLinear, data.Depth, data.ValidMask, target, airlight, noise, seed);
                    using var estimated = PerturbTransmission(synthesized.Transmission, data.ValidMask, seed, out double tRmse, out double tVariance);
                    double[] trueAirBgr = ToBgr(airlight.RgbLinear);
                    double[] estimatedAir =
                    {
                        Math.Clamp(trueAirBgr[0] + 0.020, 0, 1),
                        Math.Clamp(trueAirBgr[1] - 0.015, 0, 1),
                        Math.Clamp(trueAirBgr[2] + 0.012, 0, 1)
                    };
                    using var gtSrgb = ColorSpace.ToSrgb(data.ClearLinear);
                    using var gt8 = ToByte(gtSrgb);
                    using var inputSrgb = ColorSpace.ToSrgb(synthesized.HazyLinear);
                    using var input8 = ToByte(inputSrgb);

                    string recipeId = $"{frame.Id}__t{target.ToString("0.00", CultureInfo.InvariantCulture)}__{airlight.Id}__{noise.Id}";
                    foreach (string variant in requestedVariants)
                    {
                        try
                        {
                            bool gammaDomain = variant == "G0";
                            double[] variantAir = gammaDomain ? estimatedAir.Select(LinearToSrgb).ToArray() : estimatedAir;
                            double[] variantTrueAir = gammaDomain ? trueAirBgr.Select(LinearToSrgb).ToArray() : trueAirBgr;
                            double[] variantAirVariance = Enumerable.Range(0, 3).Select(c => Square(variantAir[c] - variantTrueAir[c])).ToArray();
                            double variantAirError = Math.Sqrt(variantAirVariance.Sum());
                            var sw = Stopwatch.StartNew();
                            using var result = A2crStressBenchmark.RunVariant(gammaDomain ? "B1" : variant,
                                gammaDomain ? inputSrgb : synthesized.HazyLinear, estimated,
                                variantAir, variantAirVariance, tVariance, Math.Sqrt(synthesized.MeanNoiseVariance));
                            sw.Stop();
                            using var resultSrgb = gammaDomain ? result.RawLinear.Clone() : ColorSpace.ToSrgb(result.RawLinear);
                            using var gammaLinearResult = gammaDomain ? ColorSpace.ToLinear(resultSrgb) : null;
                            Mat resultLinear = gammaLinearResult ?? result.RawLinear;
                            var metrics = Metrics.Evaluate(resultSrgb, gt8, input8);
                            var colorError = BenchmarkColorErrors.Evaluate(resultSrgb, gtSrgb);
                            var flatNoise = FlatNoiseError(resultLinear, data.ClearLinear, synthesized.HazyLinear,
                                synthesized.NoiselessHazyLinear, data.ValidMask);
                            double lpipsValue = lpips?.Evaluate(resultSrgb, gtSrgb) ?? double.NaN;
                            double gpRmse = GainRmse(result.GainParallel, synthesized.Transmission, data.ValidMask);
                            double gqRmse = GainRmse(result.GainPerpendicular, synthesized.Transmission, data.ValidMask);
                            writer.WriteLine(string.Join(';', new[]
                            {
                                manifest.Dataset, frame.Split, frame.Id, frame.SceneGroup, frame.Domain,
                                frame.Clear, frame.Depth, frame.DepthMask, $"stream://{recipeId}/hazy", $"stream://{recipeId}/transmission",
                                recipeId, seed.ToString(CultureInfo.InvariantCulture), N(target), airlight.Id,
                                N(airlight.RgbLinear[0]), N(airlight.RgbLinear[1]), N(airlight.RgbLinear[2]),
                                noise.Id, N(noise.GaussianSigma), N(noise.PoissonPeak), variant,
                                data.ClearLinear.Cols.ToString(CultureInfo.InvariantCulture), data.ClearLinear.Rows.ToString(CultureInfo.InvariantCulture),
                                N(data.ValidFraction), N(synthesized.DepthP90), N(synthesized.Beta), N(synthesized.MeanTransmission),
                                N(tRmse), N(variantAirError), N(metrics.Psnr), N(metrics.Ssim), N(metrics.Ciede2000),
                                N(lpipsValue), computeLpips ? $"alex_v0.1_{lpips!.Device}" : "not_requested",
                                N(colorError.HueErrorDegrees), N(colorError.ChromaticPixelFraction), N(colorError.ChromaError),
                                N(result.InvalidBefore), N(result.InvalidAfter), N(result.MeanRequiredClipping),
                                N(metrics.ClipPct), N(flatNoise.InputSigma), N(flatNoise.OutputSigma), N(flatNoise.Amplification), N(gpRmse), N(gqRmse),
                                N(result.MeanGainParallel), N(result.MeanGainPerpendicular), N(result.MeanAlpha), N(result.ProjectedFraction),
                                N(sw.Elapsed.TotalMilliseconds), N(Process.GetCurrentProcess().WorkingSet64 / 1048576.0), ""
                            }));
                            rows++;
                        }
                        catch (Exception ex)
                        {
                            failures++;
                            Console.Error.WriteLine($"DIODE-VARIANT-FAIL recipe={recipeId} variant={variant}: {ex.Message}");
                        }
                    }
                    if (recipes % 10 == 0 || recipes == 1)
                        Console.WriteLine($"DIODE-CONTROLLED recipes={recipes} rows={rows} frame={frame.Id}");
                }
            }
            catch (Exception ex)
            {
                failures++;
                Console.Error.WriteLine($"DIODE-FRAME-FAIL frame={frame.Id}: {ex}");
            }
        }
        writer.Flush();
        HardwareSnapshot hardware = HardwareProbe.Capture($"Emgu.CV {typeof(CvInvoke).Assembly.GetName().Version}");
        var meta = new
        {
            generatedUtc = DateTimeOffset.UtcNow,
            manifest = manifestPath,
            output = outputPath,
            split,
            quick,
            computeLpips,
            maxDimension,
            frames = frames.Length,
            recipes,
            variants = requestedVariants,
            rows,
            failures,
            streaming = true,
            generatedImagesPersisted = false,
            colorSpace = "linear RGB (OpenCV storage order BGR)",
            transmissionEstimator = "known transmission plus deterministic biased Gaussian perturbation; controlled recovery diagnostic",
            randomSeedPolicy = "signed Int32 from the first four little-endian bytes of SHA-256(global seed, frame id, target t, airlight id, noise id)",
            shotNoise = "poisson_gaussian uses the high-count Gaussian approximation with variance signal/peak plus Gaussian variance",
            hueMetric = "circular Lab hue error weighted by reference chroma, evaluated only where reference C*_ab >= 2; coverage is reported",
            flatNoise = "high-pass luminance residual on the 20% lowest-gradient valid-depth pixels; amplification is blank for a zero-noise input",
            lpips = computeLpips
                ? $"genuine LPIPS 0.1.4, AlexNet v0.1/ImageNet trunk, streamed in-memory via {lpips!.Device}"
                : "not requested; pass --lpips to use the locally installed pretrained AlexNet evaluator",
            gpuPerRow = "not measured; CPU benchmark, gpu_used_mb intentionally blank",
            hardware
        };
        File.WriteAllText(Path.ChangeExtension(outputPath, ".meta.json"), JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"DIODE-CONTROLLED-DONE frames={frames.Length} recipes={recipes} rows={rows} failures={failures} csv={outputPath}");
        return failures == 0 ? 0 : 1;
    }

    private sealed class FrameData : IDisposable
    {
        public Mat ClearLinear { get; }
        public Mat Depth { get; }
        public Mat ValidMask { get; }
        public double ValidFraction { get; }
        public FrameData(Mat clearLinear, Mat depth, Mat validMask, double validFraction)
        { ClearLinear = clearLinear; Depth = depth; ValidMask = validMask; ValidFraction = validFraction; }
        public void Dispose() { ClearLinear.Dispose(); Depth.Dispose(); ValidMask.Dispose(); }
    }

    private sealed class Synthesized : IDisposable
    {
        public Mat HazyLinear { get; }
        public Mat NoiselessHazyLinear { get; }
        public Mat Transmission { get; }
        public double DepthP90 { get; }
        public double Beta { get; }
        public double MeanTransmission { get; }
        public double MeanNoiseVariance { get; }
        public Synthesized(Mat hazy, Mat noiselessHazy, Mat transmission, double depthP90, double beta,
            double meanTransmission, double meanNoiseVariance)
        { HazyLinear = hazy; NoiselessHazyLinear = noiselessHazy; Transmission = transmission; DepthP90 = depthP90; Beta = beta; MeanTransmission = meanTransmission; MeanNoiseVariance = meanNoiseVariance; }
        public void Dispose() { HazyLinear.Dispose(); NoiselessHazyLinear.Dispose(); Transmission.Dispose(); }
    }

    private static FrameData LoadFrame(string root, FrameSpec frame, int maxDimension)
    {
        string imagePath = ResolveBelow(root, frame.Clear);
        string depthPath = ResolveBelow(root, frame.Depth);
        string maskPath = ResolveBelow(root, frame.DepthMask);
        using var image = CvInvoke.Imread(imagePath, ImreadModes.Color);
        if (image.IsEmpty) throw new InvalidDataException($"Cannot decode DIODE image: {imagePath}");
        NpyReader.Array2D depthArray = NpyReader.ReadFloat2D(depthPath);
        NpyReader.Array2D maskArray = NpyReader.ReadFloat2D(maskPath);
        if (depthArray.Rows != image.Rows || depthArray.Columns != image.Cols || maskArray.Rows != image.Rows || maskArray.Columns != image.Cols)
            throw new InvalidDataException($"RGB/depth/mask shape mismatch for {frame.Id}: image={image.Cols}x{image.Rows}, depth={depthArray.Columns}x{depthArray.Rows}, mask={maskArray.Columns}x{maskArray.Rows}");

        using var depthFull = FloatMat(depthArray.Values, depthArray.Rows, depthArray.Columns, 1);
        using var maskFull = FloatMat(maskArray.Values, maskArray.Rows, maskArray.Columns, 1);
        Size size = Fit(image.Size, maxDimension);
        using var imageSized = Resize(image, size, Inter.Area);
        var depth = Resize(depthFull, size, Inter.Nearest);
        var mask = Resize(maskFull, size, Inter.Nearest);
        int pixels = size.Width * size.Height;
        var depths = new float[pixels]; var masks = new float[pixels]; depth.CopyTo(depths); mask.CopyTo(masks);
        int valid = 0;
        for (int i = 0; i < pixels; i++)
        {
            bool ok = masks[i] > 0.5f && float.IsFinite(depths[i]) && depths[i] > 0;
            masks[i] = ok ? 1f : 0f;
            if (ok) valid++;
        }
        if (valid < Math.Max(16, pixels / 100)) { depth.Dispose(); mask.Dispose(); throw new InvalidDataException($"Too few valid depth pixels for {frame.Id}: {valid}/{pixels}"); }
        Marshal.Copy(masks, 0, mask.DataPointer, masks.Length);
        using var srgb = new Mat(); imageSized.ConvertTo(srgb, DepthType.Cv32F, 1.0 / 255.0);
        var clear = ColorSpace.ToLinear(srgb);
        return new FrameData(clear, depth, mask, valid / (double)pixels);
    }

    private static Synthesized Synthesize(Mat clear, Mat depth, Mat validMask, double targetT,
        AirlightSpec airlight, NoiseSpec noise, int seed)
    {
        int pixels = clear.Rows * clear.Cols;
        var j = new float[pixels * 3]; var d = new float[pixels]; var mask = new float[pixels];
        clear.CopyTo(j); depth.CopyTo(d); validMask.CopyTo(mask);
        float[] validDepth = Enumerable.Range(0, pixels).Where(i => mask[i] > 0.5f).Select(i => d[i]).OrderBy(x => x).ToArray();
        double p90 = validDepth[Math.Clamp((int)Math.Ceiling(validDepth.Length * 0.90) - 1, 0, validDepth.Length - 1)];
        double beta = BetaForTarget(targetT, p90);
        double fallbackDepth = validDepth[validDepth.Length / 2];
        double[] a = ToBgr(airlight.RgbLinear);
        var transmission = new float[pixels]; var noiseless = new float[j.Length]; var observed = new float[j.Length];
        var random = new Random(seed); double sumT = 0, sumNoiseVariance = 0;
        for (int i = 0; i < pixels; i++)
        {
            double di = mask[i] > 0.5f ? d[i] : fallbackDepth;
            double ti = Math.Clamp(Math.Exp(-beta * di), 0.01, 1); transmission[i] = (float)ti; sumT += ti;
            for (int c = 0; c < 3; c++)
            {
                int k = i * 3 + c;
                double value = ti * j[k] + (1 - ti) * a[c];
                double sigma2 = PoissonGaussianVariance(value, noise.GaussianSigma, noise.PoissonPeak);
                noiseless[k] = (float)value; sumNoiseVariance += sigma2;
                observed[k] = (float)Math.Clamp(value + Math.Sqrt(sigma2) * NextGaussian(random), 0, 1);
            }
        }
        return new Synthesized(FloatMat(observed, clear.Rows, clear.Cols, 3),
            FloatMat(noiseless, clear.Rows, clear.Cols, 3), FloatMat(transmission, clear.Rows, clear.Cols, 1),
            p90, beta, sumT / pixels, sumNoiseVariance / (pixels * 3.0));
    }

    private static Mat PerturbTransmission(Mat truth, Mat validMask, int seed, out double rmse, out double variance)
    {
        int n = truth.Rows * truth.Cols; var t = new float[n]; var mask = new float[n]; truth.CopyTo(t); validMask.CopyTo(mask);
        var estimate = new float[n]; var random = new Random(seed ^ 0x51f15e); double sum2 = 0; int count = 0;
        for (int i = 0; i < n; i++)
        {
            double error = -0.015 + 0.020 * NextGaussian(random);
            estimate[i] = (float)Math.Clamp(t[i] + error, 0.01, 1);
            if (mask[i] > 0.5f) { double actualError = estimate[i] - t[i]; sum2 += actualError * actualError; count++; }
        }
        variance = sum2 / Math.Max(1, count); rmse = Math.Sqrt(variance);
        return FloatMat(estimate, truth.Rows, truth.Cols, 1);
    }

    private static double GainRmse(Mat gain, Mat transmission, Mat validMask)
    {
        int n = gain.Rows * gain.Cols; var g = new float[n]; var t = new float[n]; var mask = new float[n];
        gain.CopyTo(g); transmission.CopyTo(t); validMask.CopyTo(mask); double sum2 = 0; int count = 0;
        for (int i = 0; i < n; i++) if (mask[i] > 0.5f)
        {
            double oracle = 1.0 / Math.Max(0.08, t[i]);
            sum2 += Square(g[i] - oracle); count++;
        }
        return Math.Sqrt(sum2 / Math.Max(1, count));
    }

    private readonly record struct FlatNoiseReport(double InputSigma, double OutputSigma, double Amplification);

    private static FlatNoiseReport FlatNoiseError(Mat result, Mat clear, Mat hazy, Mat noiselessHazy, Mat validMask)
    {
        int rows = clear.Rows, cols = clear.Cols, pixels = rows * cols;
        var output = new float[pixels * 3]; var truth = new float[pixels * 3];
        var input = new float[pixels * 3]; var inputMean = new float[pixels * 3]; var mask = new float[pixels];
        result.CopyTo(output); clear.CopyTo(truth); hazy.CopyTo(input); noiselessHazy.CopyTo(inputMean); validMask.CopyTo(mask);
        var cleanY = new double[pixels]; var outputResidual = new double[pixels]; var inputResidual = new double[pixels];
        double[] weights = { 0.0722, 0.7152, 0.2126 };
        for (int i = 0; i < pixels; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                int k = i * 3 + c;
                cleanY[i] += weights[c] * truth[k];
                outputResidual[i] += weights[c] * (output[k] - truth[k]);
                inputResidual[i] += weights[c] * (input[k] - inputMean[k]);
            }
        }
        var samples = new List<(double Gradient, double Input, double Output)>();
        for (int y = 1; y + 1 < rows; y++)
        for (int x = 1; x + 1 < cols; x++)
        {
            int i = y * cols + x;
            if (mask[i] <= 0.5f) continue;
            double gradient = Math.Abs(2 * cleanY[i] - cleanY[i - 1] - cleanY[i + 1])
                + Math.Abs(2 * cleanY[i] - cleanY[i - cols] - cleanY[i + cols]);
            double inHp = inputResidual[i] - 0.25 * (inputResidual[i - 1] + inputResidual[i + 1] + inputResidual[i - cols] + inputResidual[i + cols]);
            double outHp = outputResidual[i] - 0.25 * (outputResidual[i - 1] + outputResidual[i + 1] + outputResidual[i - cols] + outputResidual[i + cols]);
            samples.Add((gradient, inHp, outHp));
        }
        if (samples.Count == 0) return new FlatNoiseReport(double.NaN, double.NaN, double.NaN);
        double[] gradients = samples.Select(x => x.Gradient).OrderBy(x => x).ToArray();
        double threshold = gradients[Math.Clamp((int)Math.Ceiling(gradients.Length * 0.20) - 1, 0, gradients.Length - 1)];
        var flat = samples.Where(x => x.Gradient <= threshold).ToArray();
        double inputSigma = StandardDeviation(flat.Select(x => x.Input));
        double outputSigma = StandardDeviation(flat.Select(x => x.Output));
        return new FlatNoiseReport(inputSigma, outputSigma, inputSigma > 1e-7 ? outputSigma / inputSigma : double.NaN);
    }

    private static double StandardDeviation(IEnumerable<double> values)
    {
        double[] data = values.ToArray(); if (data.Length == 0) return double.NaN;
        double mean = data.Average(); return Math.Sqrt(data.Average(x => Square(x - mean)));
    }

    private static string ResolveBelow(string root, string relative)
    {
        string fullRoot = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar;
        string full = Path.GetFullPath(Path.Combine(fullRoot, relative.Replace('/', Path.DirectorySeparatorChar)));
        if (!full.StartsWith(fullRoot, StringComparison.OrdinalIgnoreCase)) throw new InvalidDataException($"Manifest path escapes sourceRoot: {relative}");
        if (!File.Exists(full)) throw new FileNotFoundException("DIODE manifest file is absent", full);
        return full;
    }

    private static int StableSeed(params string[] values)
    {
        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(string.Join('\n', values)));
        return BinaryPrimitives.ReadInt32LittleEndian(hash);
    }

    private static double[] ToBgr(double[] rgb)
    {
        if (rgb.Length != 3 || rgb.Any(x => !double.IsFinite(x) || x < 0 || x > 1))
            throw new InvalidDataException("Airlight rgbLinear must contain three finite [0,1] values");
        return new[] { rgb[2], rgb[1], rgb[0] };
    }

    internal static double BetaForTarget(double targetTransmission, double depthP90)
        => -Math.Log(Math.Clamp(targetTransmission, 0.01, 0.99)) / Math.Max(1e-6, depthP90);

    internal static double PoissonGaussianVariance(double signal, double gaussianSigma, double poissonPeak)
        => Math.Max(0, gaussianSigma) * Math.Max(0, gaussianSigma)
         + (poissonPeak > 0 ? Math.Max(0, signal) / poissonPeak : 0);

    private static double LinearToSrgb(double value)
    {
        value = Math.Clamp(value, 0, 1);
        return value <= 0.0031308 ? 12.92 * value : 1.055 * Math.Pow(value, 1.0 / 2.4) - 0.055;
    }

    private static Size Fit(Size source, int maxDimension)
    {
        if (maxDimension <= 0 || Math.Max(source.Width, source.Height) <= maxDimension) return source;
        double scale = maxDimension / (double)Math.Max(source.Width, source.Height);
        return new Size(Math.Max(1, (int)Math.Round(source.Width * scale)), Math.Max(1, (int)Math.Round(source.Height * scale)));
    }

    private static Mat Resize(Mat source, Size size, Inter interpolation)
    {
        var result = new Mat();
        if (source.Size == size) source.CopyTo(result); else CvInvoke.Resize(source, result, size, 0, 0, interpolation);
        return result;
    }

    private static Mat ToByte(Mat value) { var result = new Mat(); value.ConvertTo(result, DepthType.Cv8U, 255); return result; }
    private static Mat FloatMat(float[] data, int rows, int cols, int channels)
    { var result = new Mat(rows, cols, DepthType.Cv32F, channels); Marshal.Copy(data, 0, result.DataPointer, data.Length); return result; }
    private static double NextGaussian(Random random) => Math.Sqrt(-2 * Math.Log(Math.Max(1e-12, random.NextDouble()))) * Math.Cos(2 * Math.PI * random.NextDouble());
    private static double Square(double value) => value * value;
    private static string N(double value) => double.IsFinite(value) ? value.ToString("G9", CultureInfo.InvariantCulture) : "";
}
