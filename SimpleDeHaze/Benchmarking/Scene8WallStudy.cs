using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using SimpleDeHaze.Methods;

namespace SimpleDeHaze.Benchmarking;

internal readonly record struct WallColorReport(
    int Pixels,
    double PixelFraction,
    double InputL,
    double ResultL,
    double GroundTruthL,
    double InputA,
    double ResultA,
    double GroundTruthA,
    double InputB,
    double ResultB,
    double GroundTruthB,
    double InputRedExcess,
    double ResultRedExcess,
    double GroundTruthRedExcess,
    double RedRecovery,
    double LabChromaRmse);

internal readonly record struct ScalarMapStats(
    int Count,
    double Minimum,
    double P10,
    double Median,
    double Mean,
    double P90,
    double Maximum);

/// <summary>
/// Воспроизводимый development-case для O-HAZE #08. Верхняя левая стена используется как
/// локальный failure case восстановления слабого цветового остатка в плотной дымке. Это НЕ blind
/// test: GT определяет только фиксированную оценочную маску и никогда не передаётся методам.
/// </summary>
internal static class Scene8WallStudy
{
    private const string DefaultMethodPattern =
        "канонический|A²CR|Boundary-Constrained|Color Attenuation\\+|Chromatic Airlight|Локальная дымка|HSV \\+|Transmission-aware";

    public static int Run(string inputPath, string gtPath, string outputDirectory, int maxDimension,
        string? methodPattern = null, string? roiText = null, string? parameterOverrides = null)
    {
        if (!File.Exists(inputPath)) throw new FileNotFoundException("Не найден hazy-кадр O-HAZE #08.", inputPath);
        if (!File.Exists(gtPath)) throw new FileNotFoundException("Не найден GT кадр O-HAZE #08.", gtPath);
        if (maxDimension < 64) throw new ArgumentOutOfRangeException(nameof(maxDimension));

        Directory.CreateDirectory(outputDirectory);
        using var inputFull = new Image<Bgr, byte>(inputPath);
        using var gtFull = new Image<Bgr, byte>(gtPath);
        if (inputFull.Size != gtFull.Size)
            throw new InvalidOperationException($"Hazy/GT имеют разную геометрию: {inputFull.Size} vs {gtFull.Size}.");

        double scale = Math.Min(1.0, (double)maxDimension / Math.Max(inputFull.Width, inputFull.Height));
        using var input = scale >= 1.0
            ? inputFull.Clone()
            : inputFull.Resize(Math.Max(1, (int)Math.Round(inputFull.Width * scale)),
                Math.Max(1, (int)Math.Round(inputFull.Height * scale)), Inter.Area);
        using var gt = scale >= 1.0
            ? gtFull.Clone()
            : gtFull.Resize(input.Width, input.Height, Inter.Area);

        Rectangle roi = ParseRoi(roiText, input.Width, input.Height);
        Rectangle wallArea = ParseNormalizedRoi(new[] { 0.035, 0.020, 0.220, 0.055 }, input.Width, input.Height);
        wallArea = Rectangle.Intersect(roi, wallArea);
        if (wallArea.Width < 2 || wallArea.Height < 2)
            throw new InvalidOperationException("Фиксированная область фасада не пересекается с --roi.");
        SaveInputs(input.Mat, gt.Mat, roi, wallArea, outputDirectory);

        using var gtFloat = Float01(gt.Mat);
        using var inputFloat = Float01(input.Mat);
        using var wallMask = BuildWallMask(gtFloat, roi, wallArea);
        using (var mask8 = new Mat())
        {
            wallMask.ConvertTo(mask8, DepthType.Cv8U, 255.0);
            CvInvoke.Imwrite(Path.Combine(outputDirectory, "00_wall_mask_gt_defined.png"), mask8);
        }

        string regex = string.IsNullOrWhiteSpace(methodPattern) ? DefaultMethodPattern : methodPattern;
        var methods = MethodRegistry.All.Where(m => Regex.IsMatch(m.Name, regex, RegexOptions.IgnoreCase)).ToArray();
        if (methods.Length == 0) throw new InvalidOperationException($"Ни один метод не совпал с --methods={regex}");

        string csvPath = Path.Combine(outputDirectory, "scene8-wall-study.csv");
        var summaries = new List<object>();
        var carDiagnostics = new List<object>();
        int failures = 0;
        int index = 0;
        using var writer = new StreamWriter(csvPath, false, new UTF8Encoding(true));
        writer.WriteLine(string.Join(';', Header().Select(Csv)));

        EvaluateAndWrite("Hazy input (identity)", inputFloat, null, 0.0, ref index);
        foreach (IDeHazeMethod method in methods)
        {
            var parameters = method.Parameters.ToDictionary(p => p.Key, p => p.Default, StringComparer.Ordinal);
            ApplyOverrides(parameters, parameterOverrides);
            try
            {
                var sw = Stopwatch.StartNew();
                if (method is ChromaticAnchorMethod car)
                {
                    using var execution = car.ProcessDetailed(input, parameters);
                    sw.Stop();
                    int outputIndex = index;
                    EvaluateAndWrite(method.Name, execution.Result, parameters, sw.Elapsed.TotalMilliseconds, ref index);
                    carDiagnostics.Add(SaveCarDiagnostics(execution, outputIndex, inputFloat, gtFloat,
                        wallMask, roi, outputDirectory));
                }
                else
                {
                    using var result = method.Process(input, parameters);
                    sw.Stop();
                    EvaluateAndWrite(method.Name, result, parameters, sw.Elapsed.TotalMilliseconds, ref index);
                }
            }
            catch (Exception ex)
            {
                failures++;
                writer.WriteLine(string.Join(';', FailureRow(method.Name, ex).Select(Csv)));
                Console.Error.WriteLine($"SCENE8-FAIL method='{method.Name}': {ex.GetType().Name}: {ex.Message}");
            }
        }

        var metadata = new
        {
            generated_utc = DateTimeOffset.UtcNow,
            purpose = "O-HAZE #08 upper-left reddish wall development/failure-case study",
            scientific_status = "development_case_not_blind_test",
            leakage_warning = "The scene and GT have been inspected during method development. Do not report it as an unseen test case.",
            input = Path.GetFullPath(inputPath),
            ground_truth = Path.GetFullPath(gtPath),
            input_sha256 = Hash(inputPath),
            ground_truth_sha256 = Hash(gtPath),
            source_size = new { width = inputFull.Width, height = inputFull.Height },
            processed_size = new { width = input.Width, height = input.Height },
            roi = new { roi.X, roi.Y, roi.Width, roi.Height, normalized_default = new[] { 0.02, 0.015, 0.34, 0.10 } },
            wall_area = new { wallArea.X, wallArea.Y, wallArea.Width, wallArea.Height, normalized = new[] { 0.035, 0.020, 0.220, 0.055 } },
            wall_mask = "fixed wall_area intersected with GT Lab a*>0.5 and L* in [8,75]; evaluation only",
            methods_regex = regex,
            parameter_overrides = parameterOverrides,
            methods = methods.Select(m => m.Name).ToArray(),
            failures,
            car_diagnostics = carDiagnostics,
            hardware = HardwareProbe.Capture(typeof(CvInvoke).Assembly.GetName().Version?.ToString() ?? "unknown"),
            git = GitState(),
            results = summaries,
        };
        File.WriteAllText(Path.Combine(outputDirectory, "scene8-wall-study.meta.json"),
            JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }), new UTF8Encoding(false));

        Console.WriteLine($"SCENE8-STUDY-DONE methods={methods.Length} failures={failures} size={input.Width}x{input.Height} roi={roi.X},{roi.Y},{roi.Width},{roi.Height} out={Path.GetFullPath(outputDirectory)}");
        return failures == 0 ? 0 : 1;

        void EvaluateAndWrite(string name, Mat result, IReadOnlyDictionary<string, double>? parameters,
            double milliseconds, ref int outputIndex)
        {
            using var resultOwned = result.Clone();
            using var resultRoi = new Mat(resultOwned, roi);
            using var inputRoi = new Mat(input.Mat, roi);
            using var gtRoi = new Mat(gt.Mat, roi);
            using var gtRoiFloat = Float01(gtRoi);

            Metrics.Report full = Metrics.Evaluate(resultOwned, gt.Mat, input.Mat);
            Metrics.Report local = Metrics.Evaluate(resultRoi, gtRoi, inputRoi);
            BenchmarkColorErrorReport fullColor = BenchmarkColorErrors.Evaluate(resultOwned, gtFloat);
            BenchmarkColorErrorReport roiColor = BenchmarkColorErrors.Evaluate(resultRoi, gtRoiFloat);
            WallColorReport wall = EvaluateWallColor(inputFloat, resultOwned, gtFloat, wallMask, roi);

            string parametersJson = parameters == null ? "{}" : JsonSerializer.Serialize(parameters);
            string[] row =
            {
                name, "1", "", N(milliseconds),
                N(full.Psnr), N(full.Ssim), N(full.Ciede2000), N(full.Mse), N(fullColor.HueErrorDegrees), N(fullColor.ChromaError),
                N(local.Psnr), N(local.Ssim), N(local.Ciede2000), N(local.Mse), N(roiColor.HueErrorDegrees), N(roiColor.ChromaError),
                wall.Pixels.ToString(CultureInfo.InvariantCulture), N(100.0 * wall.PixelFraction),
                N(wall.InputL), N(wall.ResultL), N(wall.GroundTruthL),
                N(wall.InputA), N(wall.ResultA), N(wall.GroundTruthA),
                N(wall.InputB), N(wall.ResultB), N(wall.GroundTruthB),
                N(wall.InputRedExcess), N(wall.ResultRedExcess), N(wall.GroundTruthRedExcess),
                N(wall.RedRecovery), N(wall.LabChromaRmse), parametersJson,
            };
            writer.WriteLine(string.Join(';', row.Select(Csv)));
            writer.Flush();

            string safe = SafeName(name);
            using var out8 = new Mat(); resultOwned.ConvertTo(out8, DepthType.Cv8U, 255.0);
            CvInvoke.Imwrite(Path.Combine(outputDirectory, $"{outputIndex:00}_{safe}_full.png"), out8);
            using (var crop = new Mat(out8, roi))
                CvInvoke.Imwrite(Path.Combine(outputDirectory, $"{outputIndex:00}_{safe}_wall.png"), crop);
            outputIndex++;

            summaries.Add(new
            {
                method = name,
                full_psnr = full.Psnr,
                full_ssim = full.Ssim,
                full_ciede2000 = full.Ciede2000,
                roi_psnr = local.Psnr,
                roi_ssim = local.Ssim,
                roi_ciede2000 = local.Ciede2000,
                wall_red_recovery = wall.RedRecovery,
                wall_lab_chroma_rmse = wall.LabChromaRmse,
            });
            Console.WriteLine($"SCENE8 method='{name}' ROI PSNR={local.Psnr:F2} SSIM={local.Ssim:F3} DE={local.Ciede2000:F2} wall a*: {wall.InputA:F2}->{wall.ResultA:F2} (GT {wall.GroundTruthA:F2}) redRecovery={wall.RedRecovery:F3}");
        }
    }

    private static object SaveCarDiagnostics(ChromaticAnchorExecution execution, int outputIndex,
        Mat inputFloat, Mat gtFloat, Mat wallMask, Rectangle roi, string outputDirectory)
    {
        const string stem = "car_diagnostics";
        SaveUnitMap(execution.Transmission, Path.Combine(outputDirectory, stem + "_transmission.png"));
        SaveUnitMap(execution.AdjustedDepth, Path.Combine(outputDirectory, stem + "_adjusted_depth_clipped.png"));
        SaveUnitMap(execution.ProjectionAlpha, Path.Combine(outputDirectory, stem + "_projection_alpha.png"));
        SaveFloatImage(execution.PhysicalSrgb, Path.Combine(outputDirectory, stem + "_physical_full.png"));
        SaveFloatImage(execution.AirlightSrgb, Path.Combine(outputDirectory, stem + "_airlight_field_full.png"));
        using (var physicalRoi = new Mat(execution.PhysicalSrgb, roi))
            SaveFloatImage(physicalRoi, Path.Combine(outputDirectory, stem + "_physical_wall.png"));
        using (var airlightRoi = new Mat(execution.AirlightSrgb, roi))
            SaveFloatImage(airlightRoi, Path.Combine(outputDirectory, stem + "_airlight_field_wall.png"));

        WallColorReport physicalWall = EvaluateWallColor(inputFloat, execution.PhysicalSrgb, gtFloat, wallMask, roi);
        ScalarMapStats transmissionFull = MapStats(execution.Transmission);
        ScalarMapStats transmissionRoi = MapStats(execution.Transmission, roi);
        ScalarMapStats transmissionWall = MapStats(execution.Transmission, roi, wallMask);
        ScalarMapStats depthWall = MapStats(execution.AdjustedDepth, roi, wallMask);
        ScalarMapStats projectionWall = MapStats(execution.ProjectionAlpha, roi, wallMask);
        MCvScalar a = execution.GlobalAirlight;
        var report = new
        {
            output_index = outputIndex,
            working_space = execution.LinearRadiance ? "linear_rgb" : "srgb_ablation",
            global_airlight_bgr_working_space = new[] { a.V0, a.V1, a.V2 },
            raw_out_of_gamut_fraction = execution.RawOutOfGamutFraction,
            gamut_projection = execution.ProjectionSummary,
            transmission_full = transmissionFull,
            transmission_roi = transmissionRoi,
            transmission_wall = transmissionWall,
            adjusted_depth_wall = depthWall,
            projection_alpha_wall = projectionWall,
            physical_inverse_wall = physicalWall,
            maps = new[]
            {
                stem + "_transmission.png",
                stem + "_adjusted_depth_clipped.png",
                stem + "_projection_alpha.png",
                stem + "_physical_full.png",
                stem + "_physical_wall.png",
                stem + "_airlight_field_full.png",
                stem + "_airlight_field_wall.png",
            },
        };
        File.WriteAllText(Path.Combine(outputDirectory, stem + ".json"),
            JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }), new UTF8Encoding(false));
        Console.WriteLine($"CAR-DIAG t_wall={transmissionWall.Mean:F4} [{transmissionWall.P10:F4},{transmissionWall.P90:F4}] " +
            $"A_bgr=({a.V0:F4},{a.V1:F4},{a.V2:F4}) outOfGamut={execution.RawOutOfGamutFraction:P1} " +
            $"projectionAlpha_wall={projectionWall.Mean:F3} physical wall a*={physicalWall.ResultA:F2}");
        return report;
    }

    private static ScalarMapStats MapStats(Mat map, Rectangle? roi = null, Mat? roiMask = null)
    {
        using var crop = roi.HasValue ? new Mat(map, roi.Value) : map.Clone();
        int count = crop.Rows * crop.Cols;
        var values = new float[count]; crop.CopyTo(values);
        float[]? mask = null;
        if (roiMask != null)
        {
            if (roiMask.Rows != crop.Rows || roiMask.Cols != crop.Cols)
                throw new ArgumentException("Размер диагностической маски не совпадает с ROI.", nameof(roiMask));
            mask = new float[count]; roiMask.CopyTo(mask);
        }
        double[] selected = Enumerable.Range(0, count)
            .Where(i => mask == null || mask[i] >= 0.5f)
            .Select(i => (double)values[i])
            .Where(double.IsFinite)
            .OrderBy(value => value)
            .ToArray();
        if (selected.Length == 0)
            return new ScalarMapStats(0, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN);
        double Percentile(double fraction)
        {
            double position = fraction * (selected.Length - 1);
            int lo = (int)Math.Floor(position), hi = (int)Math.Ceiling(position);
            double alpha = position - lo;
            return selected[lo] * (1.0 - alpha) + selected[hi] * alpha;
        }
        return new ScalarMapStats(selected.Length, selected[0], Percentile(0.10), Percentile(0.50),
            selected.Average(), Percentile(0.90), selected[^1]);
    }

    private static void SaveUnitMap(Mat map, string path)
    {
        using var clipped = map.Clone(); DehazeCore.Clamp01(clipped);
        using var bytes = new Mat(); clipped.ConvertTo(bytes, DepthType.Cv8U, 255.0);
        CvInvoke.Imwrite(path, bytes);
    }

    private static void SaveFloatImage(Mat image, string path)
    {
        using var clipped = DeHazeCPU.Clip(image.Clone());
        using var bytes = new Mat(); clipped.ConvertTo(bytes, DepthType.Cv8U, 255.0);
        CvInvoke.Imwrite(path, bytes);
    }

    private static string[] Header() => new[]
    {
        "method", "ok", "error", "ms",
        "full_psnr_raw", "full_ssim_raw", "full_ciede2000_raw", "full_mse_raw", "full_hue_error_deg", "full_chroma_error",
        "roi_psnr_raw", "roi_ssim_raw", "roi_ciede2000_raw", "roi_mse_raw", "roi_hue_error_deg", "roi_chroma_error",
        "wall_pixels", "wall_fraction_pct", "wall_input_L", "wall_result_L", "wall_gt_L",
        "wall_input_a", "wall_result_a", "wall_gt_a", "wall_input_b", "wall_result_b", "wall_gt_b",
        "wall_input_red_excess", "wall_result_red_excess", "wall_gt_red_excess", "wall_red_recovery", "wall_lab_chroma_rmse", "params_json",
    };

    private static string[] FailureRow(string method, Exception error)
    {
        var row = Enumerable.Repeat("", Header().Length).ToArray();
        row[0] = method; row[1] = "0"; row[2] = error.GetType().Name + ": " + error.Message;
        return row;
    }

    private static WallColorReport EvaluateWallColor(Mat inputFloat, Mat resultFloat, Mat gtFloat, Mat mask, Rectangle roi)
    {
        using var inputCrop = new Mat(inputFloat, roi);
        using var resultCrop = new Mat(resultFloat, roi);
        using var gtCrop = new Mat(gtFloat, roi);
        using var inputLab = new Mat(); using var resultLab = new Mat(); using var gtLab = new Mat();
        CvInvoke.CvtColor(inputCrop, inputLab, ColorConversion.Bgr2Lab);
        CvInvoke.CvtColor(resultCrop, resultLab, ColorConversion.Bgr2Lab);
        CvInvoke.CvtColor(gtCrop, gtLab, ColorConversion.Bgr2Lab);

        int pixels = roi.Width * roi.Height;
        var inputRgb = new float[pixels * 3]; var resultRgb = new float[pixels * 3]; var gtRgb = new float[pixels * 3];
        var inputValues = new float[pixels * 3]; var resultValues = new float[pixels * 3]; var gtValues = new float[pixels * 3];
        var maskValues = new float[pixels];
        inputCrop.CopyTo(inputRgb); resultCrop.CopyTo(resultRgb); gtCrop.CopyTo(gtRgb);
        inputLab.CopyTo(inputValues); resultLab.CopyTo(resultValues); gtLab.CopyTo(gtValues); mask.CopyTo(maskValues);

        int count = 0;
        double iL = 0, rL = 0, gL = 0, ia = 0, ra = 0, ga = 0, ib = 0, rb = 0, gb = 0;
        double iRed = 0, rRed = 0, gRed = 0, chromaSq = 0;
        for (int px = 0; px < pixels; px++)
        {
            if (maskValues[px] < 0.5f) continue;
            int j = px * 3;
            count++;
            iL += inputValues[j]; rL += resultValues[j]; gL += gtValues[j];
            ia += inputValues[j + 1]; ra += resultValues[j + 1]; ga += gtValues[j + 1];
            ib += inputValues[j + 2]; rb += resultValues[j + 2]; gb += gtValues[j + 2];
            iRed += inputRgb[j + 2] - 0.5 * (inputRgb[j + 1] + inputRgb[j]);
            rRed += resultRgb[j + 2] - 0.5 * (resultRgb[j + 1] + resultRgb[j]);
            gRed += gtRgb[j + 2] - 0.5 * (gtRgb[j + 1] + gtRgb[j]);
            double da = resultValues[j + 1] - gtValues[j + 1];
            double db = resultValues[j + 2] - gtValues[j + 2];
            chromaSq += da * da + db * db;
        }
        if (count == 0) return new WallColorReport(0, 0, double.NaN, double.NaN, double.NaN,
            double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN,
            double.NaN, double.NaN, double.NaN, double.NaN, double.NaN);

        iL /= count; rL /= count; gL /= count; ia /= count; ra /= count; ga /= count;
        ib /= count; rb /= count; gb /= count; iRed /= count; rRed /= count; gRed /= count;
        double denominator = gRed - iRed;
        double recovery = Math.Abs(denominator) > 1e-8 ? (rRed - iRed) / denominator : double.NaN;
        return new WallColorReport(count, count / (double)pixels, iL, rL, gL, ia, ra, ga, ib, rb, gb,
            iRed, rRed, gRed, recovery, Math.Sqrt(chromaSq / count));
    }

    private static Mat BuildWallMask(Mat gtFloat, Rectangle roi, Rectangle wallArea)
    {
        using var crop = new Mat(gtFloat, roi);
        using var lab = new Mat(); CvInvoke.CvtColor(crop, lab, ColorConversion.Bgr2Lab);
        int pixels = roi.Width * roi.Height;
        var lv = new float[pixels * 3]; var mask = new float[pixels];
        lab.CopyTo(lv);
        for (int px = 0; px < pixels; px++)
        {
            int j = px * 3;
            float l = lv[j], a = lv[j + 1];
            int x = roi.X + px % roi.Width;
            int y = roi.Y + px / roi.Width;
            mask[px] = wallArea.Contains(x, y) && a > 0.5f && l >= 8.0f && l <= 75.0f ? 1.0f : 0.0f;
        }
        return DehazeCore.MatFromFloats(mask, roi.Height, roi.Width);
    }

    private static void SaveInputs(Mat input, Mat gt, Rectangle roi, Rectangle wallArea, string outputDirectory)
    {
        CvInvoke.Imwrite(Path.Combine(outputDirectory, "00_hazy_full.png"), input);
        CvInvoke.Imwrite(Path.Combine(outputDirectory, "00_gt_full.png"), gt);
        using (var crop = new Mat(input, roi)) CvInvoke.Imwrite(Path.Combine(outputDirectory, "00_hazy_wall.png"), crop);
        using (var crop = new Mat(gt, roi)) CvInvoke.Imwrite(Path.Combine(outputDirectory, "00_gt_wall.png"), crop);
        using (var crop = new Mat(input, wallArea)) CvInvoke.Imwrite(Path.Combine(outputDirectory, "00_hazy_wall_target.png"), crop);
        using (var crop = new Mat(gt, wallArea)) CvInvoke.Imwrite(Path.Combine(outputDirectory, "00_gt_wall_target.png"), crop);
    }

    private static Mat Float01(Mat bgr8)
    {
        var value = new Mat(); bgr8.ConvertTo(value, DepthType.Cv32F, 1.0 / 255.0); return value;
    }

    private static Rectangle ParseRoi(string? text, int width, int height)
    {
        double[] values = string.IsNullOrWhiteSpace(text)
            ? new[] { 0.02, 0.015, 0.34, 0.10 }
            : text.Split(',', StringSplitOptions.TrimEntries).Select(x => double.Parse(x, CultureInfo.InvariantCulture)).ToArray();
        if (values.Length != 4) throw new ArgumentException("--roi должен содержать x,y,w,h.");
        bool fractional = values.All(v => v >= 0 && v <= 1);
        if (fractional) return ParseNormalizedRoi(values, width, height);
        int x = (int)Math.Round(values[0]);
        int y = (int)Math.Round(values[1]);
        int w = (int)Math.Round(values[2]);
        int h = (int)Math.Round(values[3]);
        x = Math.Clamp(x, 0, width - 2); y = Math.Clamp(y, 0, height - 2);
        w = Math.Clamp(w, 2, width - x); h = Math.Clamp(h, 2, height - y);
        return new Rectangle(x, y, w, h);
    }

    private static Rectangle ParseNormalizedRoi(IReadOnlyList<double> values, int width, int height)
    {
        int x = Math.Clamp((int)Math.Round(values[0] * width), 0, width - 2);
        int y = Math.Clamp((int)Math.Round(values[1] * height), 0, height - 2);
        int w = Math.Clamp((int)Math.Round(values[2] * width), 2, width - x);
        int h = Math.Clamp((int)Math.Round(values[3] * height), 2, height - y);
        return new Rectangle(x, y, w, h);
    }

    private static string SafeName(string value)
    {
        var invalid = Path.GetInvalidFileNameChars().ToHashSet();
        return new string(value.Select(ch => invalid.Contains(ch) ? '_' : ch).ToArray()).Replace(' ', '_');
    }

    private static string N(double value) => double.IsFinite(value) ? value.ToString("0.######", CultureInfo.InvariantCulture) : "";

    private static string Csv(string value)
        => value.Contains(';') || value.Contains('"') || value.Contains('\n')
            ? '"' + value.Replace("\"", "\"\"") + '"'
            : value;

    private static string Hash(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static void ApplyOverrides(Dictionary<string, double> parameters, string? text)
    {
        if (string.IsNullOrWhiteSpace(text)) return;
        foreach (string item in text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            string[] pair = item.Split('=', 2, StringSplitOptions.TrimEntries);
            if (pair.Length != 2 || !parameters.ContainsKey(pair[0])) continue;
            if (!double.TryParse(pair[1], NumberStyles.Float, CultureInfo.InvariantCulture, out double value))
                throw new ArgumentException($"Некорректное значение --params: {item}");
            parameters[pair[0]] = value;
        }
    }

    private static object GitState()
    {
        static string Run(string arguments)
        {
            try
            {
                var start = new ProcessStartInfo("git", arguments)
                {
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                using var process = Process.Start(start);
                if (process == null) return "unknown";
                string output = process.StandardOutput.ReadToEnd().Trim();
                process.WaitForExit(3000);
                return process.ExitCode == 0 ? output : "unknown";
            }
            catch { return "unknown"; }
        }
        string status = Run("status --porcelain");
        return new { commit = Run("rev-parse HEAD"), dirty = status is not "" and not "unknown" };
    }
}
