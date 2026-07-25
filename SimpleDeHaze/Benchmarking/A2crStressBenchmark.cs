using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

using SimpleDeHaze.Methods;

namespace SimpleDeHaze.Benchmarking;

internal static class A2crStressBenchmark
{
    internal sealed record VariantOutput(Mat RawLinear, Mat GainParallel, Mat GainPerpendicular,
        double MeanGainParallel, double MeanGainPerpendicular,
        double MeanAlpha, double ProjectedFraction, double InvalidBefore, double InvalidAfter,
        double MeanRequiredClipping) : IDisposable
    {
        public void Dispose()
        {
            RawLinear.Dispose();
            GainParallel.Dispose();
            GainPerpendicular.Dispose();
        }
    }

    public static int Run(string outputPath, bool quick)
    {
        outputPath = Path.GetFullPath(outputPath);
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        double[] transmissions = quick ? new[] { 0.08, 0.2, 0.6 } : new[] { 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
        var airlights = new Dictionary<string, double[]>
        {
            ["neutral"] = new[] { 0.82, 0.82, 0.82 },
            ["blue"] = new[] { 0.95, 0.80, 0.65 },
            ["yellow"] = new[] { 0.55, 0.85, 0.95 },
            ["gray"] = new[] { 0.62, 0.62, 0.62 },
        };
        string[] noises = quick ? new[] { "gaussian", "local-t" } : new[] { "gaussian", "poisson-gaussian", "jpeg", "color", "local-t" };

        using var writer = new StreamWriter(outputPath, false, new System.Text.UTF8Encoding(true));
        writer.WriteLine("scenario;variant;t_true;airlight;noise;psnr;ssim;ciede2000;hue_error_deg;chroma_error;invalid_channel_before;invalid_channel_after;required_clip_mean;flat_noise_x;dense_region_mse;mean_g_parallel;mean_g_perp;mean_g_difference;mean_alpha;projected_pixel_fraction;ms;working_set_mb");
        int rows = 0, failed = 0, scenario = 0;
        foreach (double tTrue in transmissions)
        foreach (var air in airlights)
        foreach (string noise in noises)
        {
            scenario++;
            try
            {
                using var clean = SyntheticClean(96, 64);
                using var hazy = Synthesize(clean, air.Value, tTrue, noise, scenario, out var trueTransmission);
                using var estimatedT = EstimatedTransmission(trueTransmission, noise, scenario, out double transmissionVariance);
                double[] aHat = { Math.Clamp(air.Value[0] + 0.040, 0, 1), Math.Clamp(air.Value[1] - 0.030, 0, 1), Math.Clamp(air.Value[2] + 0.025, 0, 1) };
                double[] aVariance = { Square(aHat[0] - air.Value[0]), Square(aHat[1] - air.Value[1]), Square(aHat[2] - air.Value[2]) };
                using var gtSrgb = ColorSpace.ToSrgb(clean); using var gt8 = ToByte(gtSrgb);
                using var inputSrgb = ColorSpace.ToSrgb(hazy); using var input8 = ToByte(inputSrgb);

                foreach (string variant in Enumerable.Range(0, 10).Select(i => $"B{i}"))
                {
                    var sw = Stopwatch.StartNew();
                    using var result = RunVariant(variant, hazy, estimatedT, aHat, aVariance, transmissionVariance);
                    sw.Stop();
                    using var srgb = ColorSpace.ToSrgb(result.RawLinear);
                    var metrics = Metrics.Evaluate(srgb, gt8, input8);
                    var colorError = BenchmarkColorErrors.Evaluate(srgb, gtSrgb);
                    double denseMse = DenseRegionMse(result.RawLinear, clean, trueTransmission, threshold: 0.2);
                    writer.WriteLine(string.Join(';', new[]
                    {
                        scenario.ToString(CultureInfo.InvariantCulture), variant, N(tTrue), air.Key, noise,
                        N(metrics.Psnr), N(metrics.Ssim), N(metrics.Ciede2000), N(colorError.HueErrorDegrees), N(colorError.ChromaError),
                        N(result.InvalidBefore), N(result.InvalidAfter), N(result.MeanRequiredClipping),
                        N(FlatNoiseRatio(result.RawLinear, clean)), N(denseMse), N(result.MeanGainParallel),
                        N(result.MeanGainPerpendicular), N(result.MeanGainParallel-result.MeanGainPerpendicular),
                        N(result.MeanAlpha), N(result.ProjectedFraction), N(sw.Elapsed.TotalMilliseconds),
                        N(Process.GetCurrentProcess().WorkingSet64 / 1048576.0)
                    }));
                    rows++;
                }
                trueTransmission.Dispose();
                Console.WriteLine($"A2CR-STRESS scenario={scenario} t={tTrue:F2} A={air.Key} noise={noise} rows={rows}");
            }
            catch (Exception ex)
            {
                failed++;
                Console.Error.WriteLine($"A2CR-STRESS-FAIL scenario={scenario}: {ex.Message}");
            }
        }
        writer.Flush();
        File.WriteAllText(Path.ChangeExtension(outputPath, ".meta.json"), System.Text.Json.JsonSerializer.Serialize(new
        {
            generated = DateTimeOffset.Now, quick, scenarios = scenario, rows, failed,
            size = new { width = 96, height = 64 },
            variants = Enumerable.Range(0, 10).Select(i => $"B{i}").ToArray(),
            note = "Controlled synthetic diagnostic; B9 uses the joint edge-aware primal-dual solver with exact per-pixel polygon feasibility."
        }, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"A2CR-STRESS-DONE scenarios={scenario} rows={rows} failed={failed} csv={outputPath}");
        return failed == 0 ? 0 : 1;
    }

    internal static VariantOutput RunVariant(string variant, Mat input, Mat tHat, double[] a, double[] aVariance,
        double tVariance, double noiseSigma = 0.012)
    {
        const double tMin = 0.08;
        if (variant is "B0" or "B1" or "B2" or "B3")
        {
            using var transmission = tHat.Clone();
            double floor = variant == "B0" ? 1e-4 : tMin;
            if (variant == "B3")
            {
                using var bound = DehazeCore.BoundaryConstraint(input, new MCvScalar(a[0], a[1], a[2]));
                CvInvoke.Max(transmission, bound, transmission);
            }
            return ScalarRecovery(input, transmission, a, floor, variant == "B2" ? 0.35 : floor);
        }

        bool noise = variant is "B5" or "B6" or "B7" or "B8" or "B9";
        bool tUncertainty = variant is "B6" or "B7" or "B8" or "B9";
        bool aUncertainty = variant is "B7" or "B8" or "B9";
        bool feasible = variant is "B8" or "B9";
        double tv = variant == "B9" ? 0.025 : 0;
        int n = tHat.Rows * tHat.Cols;
        using var varT = ConstantMap(tHat.Rows, tHat.Cols, tVariance);
        var air = new AirlightEstimate(new MCvScalar(a[0], a[1], a[2]), aVariance, Array.Empty<MCvScalar>());
        var options = new A2crRecoveryOptions(tMin, noiseSigma * noiseSigma, 1, noise, tUncertainty,
            aUncertainty, feasible, 2, 0, tv, 30, 0.08);
        using var recovered = A2crRecovery.Recover(input, tHat, varT, air, options);
        var d = recovered.Diagnostics;
        return new VariantOutput(recovered.LinearResult.Clone(), recovered.GainParallel.Clone(), recovered.GainPerpendicular.Clone(),
            d.MeanGainParallel, d.MeanGainPerpendicular,
            d.MeanAlpha, d.ProjectedPixelFraction, d.InvalidChannelFractionBefore,
            d.InvalidChannelFractionAfter, d.MeanRequiredClipping);
    }

    private static VariantOutput ScalarRecovery(Mat input, Mat transmission, double[] air, double tMin, double chromaFloor)
    {
        int pixels = input.Rows * input.Cols;
        var src = new float[pixels * 3]; var t = new float[pixels]; input.CopyTo(src); transmission.CopyTo(t);
        var output = new float[src.Length]; var gains = new float[pixels]; long invalid = 0; double clipping = 0, gain = 0;
        for (int i = 0; i < pixels; i++)
        {
            int j = i * 3; double ti = Math.Max(tMin, t[i]); gains[i] = (float)(1 / ti); gain += gains[i];
            double mean = ((src[j] - air[0]) + (src[j + 1] - air[1]) + (src[j + 2] - air[2])) / 3.0;
            for (int c = 0; c < 3; c++)
            {
                double residual = (src[j + c] - air[c]) - mean;
                double value = air[c] + mean / ti + residual / Math.Max(ti, chromaFloor);
                output[j + c] = (float)value;
                if (value < -1e-6 || value > 1 + 1e-6) invalid++;
                clipping += value < 0 ? -value : value > 1 ? value - 1 : 0;
            }
        }
        double invalidFraction = invalid / (double)(pixels * 3);
        return new VariantOutput(FloatMat(output, input.Rows, input.Cols, 3),
            FloatMat(gains, input.Rows, input.Cols, 1), FloatMat(gains, input.Rows, input.Cols, 1),
            gain / pixels, gain / pixels,
            1, 0, invalidFraction, invalidFraction, clipping / (pixels * 3));
    }

    private static Mat SyntheticClean(int width, int height)
    {
        var data = new float[width * height * 3];
        for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int i = (y * width + x) * 3; double fx = x / (double)(width - 1), fy = y / (double)(height - 1);
            bool flat = x < width / 4 && y >= height / 2;
            data[i] = (float)(flat ? 0.28 : Math.Clamp(0.08 + 0.62 * fx + 0.05 * Math.Sin(0.35 * x), 0, 1));
            data[i + 1] = (float)(flat ? 0.38 : Math.Clamp(0.10 + 0.58 * fy + 0.04 * Math.Cos(0.28 * y), 0, 1));
            data[i + 2] = (float)(flat ? 0.48 : Math.Clamp(0.12 + 0.35 * fx + 0.32 * fy + 0.04 * Math.Sin(0.2 * (x + y)), 0, 1));
        }
        return FloatMat(data, height, width, 3);
    }

    private static Mat Synthesize(Mat clean, double[] air, double tBase, string noise, int seed, out Mat trueTransmission)
    {
        int rows = clean.Rows, cols = clean.Cols, pixels = rows * cols;
        var j = new float[pixels * 3]; clean.CopyTo(j); var t = new float[pixels]; var observed = new float[j.Length];
        var random = new Random(1000 + seed); bool local = noise == "local-t";
        for (int y = 0; y < rows; y++)
        for (int x = 0; x < cols; x++)
        {
            int i = y * cols + x, k = i * 3;
            double ti = Math.Clamp(tBase + (local ? 0.12 * Math.Sin(2 * Math.PI * x / cols) * Math.Sin(Math.PI * y / rows) : 0), 0.02, 1);
            t[i] = (float)ti;
            for (int c = 0; c < 3; c++)
            {
                double value = ti * j[k + c] + (1 - ti) * air[c];
                double sigma = noise switch
                {
                    "poisson-gaussian" => Math.Sqrt(Math.Max(0, value) * 0.00012 + 0.000025),
                    "color" => c == 0 ? 0.016 : c == 1 ? 0.005 : 0.011,
                    "local-t" => 0.006,
                    "jpeg" => 0,
                    _ => 0.010,
                };
                observed[k + c] = (float)Math.Clamp(value + sigma * NextGaussian(random), 0, 1);
            }
        }
        trueTransmission = FloatMat(t, rows, cols, 1);
        var linear = FloatMat(observed, rows, cols, 3);
        if (noise != "jpeg") return linear;
        using (linear)
        using (var srgb = ColorSpace.ToSrgb(linear))
        using (var bytes = ToByte(srgb))
        {
            byte[] encoded = CvInvoke.Imencode(".jpg", bytes,
                new KeyValuePair<ImwriteFlags, int>(ImwriteFlags.JpegQuality, 35));
            using var decoded = new Mat();
            CvInvoke.Imdecode(encoded, ImreadModes.Color, decoded);
            using var decodedFloat = new Mat(); decoded.ConvertTo(decodedFloat, DepthType.Cv32F, 1.0 / 255.0);
            return ColorSpace.ToLinear(decodedFloat);
        }
    }

    private static Mat EstimatedTransmission(Mat trueTransmission, string noise, int seed, out double variance)
    {
        int n = trueTransmission.Rows * trueTransmission.Cols; var source = new float[n]; trueTransmission.CopyTo(source);
        var random = new Random(5000 + seed); var estimate = new float[n]; double sum2 = 0;
        for (int y = 0; y < trueTransmission.Rows; y++)
        for (int x = 0; x < trueTransmission.Cols; x++)
        {
            int i = y * trueTransmission.Cols + x;
            // Отрицательное смещение моделирует опасную недооценку t: именно она создаёт
            // чрезмерный gain и проверяет, что B8/B9 реально используют feasible projector.
            double error = -0.025 + 0.025 * NextGaussian(random);
            if (noise == "local-t") error -= 0.16 * Math.Exp(-Square((x - trueTransmission.Cols * 0.65) / (trueTransmission.Cols * 0.18)) - Square((y - trueTransmission.Rows * 0.45) / (trueTransmission.Rows * 0.22)));
            estimate[i] = (float)Math.Clamp(source[i] + error, 0.01, 1); sum2 += error * error;
        }
        variance = sum2 / n;
        return FloatMat(estimate, trueTransmission.Rows, trueTransmission.Cols, 1);
    }

    private static double FlatNoiseRatio(Mat result, Mat clean)
    {
        var r = new float[result.Rows * result.Cols * 3]; var c = new float[r.Length]; result.CopyTo(r); clean.CopyTo(c);
        var residual = new List<double>();
        for (int y = result.Rows / 2; y < result.Rows; y++)
        for (int x = 0; x < result.Cols / 4; x++)
        for (int channel = 0; channel < 3; channel++)
        { int i = (y * result.Cols + x) * 3 + channel; residual.Add(r[i] - c[i]); }
        double mean = residual.Average(); double std = Math.Sqrt(residual.Average(v => Square(v - mean)));
        return std / 0.01;
    }

    private static double DenseRegionMse(Mat result, Mat clean, Mat transmission, double threshold)
    {
        var r = new float[result.Rows * result.Cols * 3]; var c = new float[r.Length]; var t = new float[result.Rows * result.Cols];
        result.CopyTo(r); clean.CopyTo(c); transmission.CopyTo(t); double sum = 0; long count = 0;
        for (int i = 0; i < t.Length; i++) if (t[i] <= threshold)
            for (int channel = 0; channel < 3; channel++) { sum += Square(r[i * 3 + channel] - c[i * 3 + channel]); count++; }
        return count == 0 ? double.NaN : sum / count;
    }

    private static Mat ConstantMap(int rows, int cols, double value)
    { var result = new Mat(rows, cols, DepthType.Cv32F, 1); result.SetTo(new MCvScalar(value)); return result; }
    private static Mat ToByte(Mat float01) { var result = new Mat(); float01.ConvertTo(result, DepthType.Cv8U, 255); return result; }
    private static Mat FloatMat(float[] data, int rows, int cols, int channels)
    { var result = new Mat(rows, cols, DepthType.Cv32F, channels); Marshal.Copy(data, 0, result.DataPointer, data.Length); return result; }
    private static double NextGaussian(Random random) => Math.Sqrt(-2 * Math.Log(Math.Max(1e-12, random.NextDouble()))) * Math.Cos(2 * Math.PI * random.NextDouble());
    private static double Square(double value) => value * value;
    private static string N(double value) => double.IsFinite(value) ? value.ToString("G9", CultureInfo.InvariantCulture) : "";
}
