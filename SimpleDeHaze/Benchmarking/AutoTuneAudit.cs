using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using SimpleDeHaze.Methods;

namespace SimpleDeHaze.Benchmarking;

/// <summary>Воспроизводимая проверка реального покрытия и результата AutoTuner на одном кадре.</summary>
internal static class AutoTuneAudit
{
    public static int Run(string imagePath, string gtPath, string outputDirectory, string methodPattern,
        bool thorough, int maxEvaluations, int evaluationMaxDimension, int processMaxDimension,
        AutoTuneGoal goal)
    {
        if (!File.Exists(imagePath)) throw new FileNotFoundException("Не найден вход autotune audit.", imagePath);
        if (!File.Exists(gtPath)) throw new FileNotFoundException("Не найден GT autotune audit.", gtPath);
        if (maxEvaluations < 1 || evaluationMaxDimension < 64 || processMaxDimension < 64)
            throw new ArgumentOutOfRangeException(nameof(maxEvaluations));

        IDeHazeMethod[] matches = MethodRegistry.All
            .Where(method => Regex.IsMatch(method.Name, methodPattern, RegexOptions.IgnoreCase))
            .ToArray();
        if (matches.Length != 1)
            throw new InvalidOperationException($"--methods должен выбрать ровно один метод; найдено {matches.Length}.");
        IDeHazeMethod method = matches[0];

        Directory.CreateDirectory(outputDirectory);
        using var inputFull = new Image<Bgr, byte>(imagePath);
        using var gtFull = new Image<Bgr, byte>(gtPath);
        if (inputFull.Size != gtFull.Size) throw new InvalidOperationException("Hazy и GT имеют разную геометрию.");
        double scale = Math.Min(1.0, processMaxDimension / (double)Math.Max(inputFull.Width, inputFull.Height));
        using var input = scale >= 1.0 ? inputFull.Clone() : inputFull.Resize(
            Math.Max(1, (int)Math.Round(inputFull.Width * scale)),
            Math.Max(1, (int)Math.Round(inputFull.Height * scale)), Inter.Area);
        using var gt = scale >= 1.0 ? gtFull.Clone() : gtFull.Resize(input.Width, input.Height, Inter.Area);

        var start = method.Parameters.ToDictionary(parameter => parameter.Key, parameter => parameter.Default,
            StringComparer.Ordinal);
        using var initial = method.Process(input, start);
        Metrics.Report initialMetrics = Metrics.Evaluate(initial, gt.Mat, input.Mat);
        Metrics.ChromaticFidelityReport initialChroma = Metrics.ChromaticFidelity(initial, gt.Mat);
        Metrics.LocalChromaExpansionReport initialChromaExpansion = Metrics.LocalChromaExpansion(initial, input.Mat);
        double initialFullObjective = AutoTuner.Score(goal, initial, input.Mat, gt.Mat, 0.0);
        var progress = new List<object>();
        ParameterSearchResult search = AutoTuner.Audit(method, input, start, gt.Mat, goal,
            thorough, maxEvaluations, evaluationMaxDimension,
            (attempt, score) => progress.Add(new { attempt, best_score = score }));
        using var best = method.Process(input, search.Parameters);
        Metrics.Report bestMetrics = Metrics.Evaluate(best, gt.Mat, input.Mat);
        Metrics.ChromaticFidelityReport bestChroma = Metrics.ChromaticFidelity(best, gt.Mat);
        Metrics.LocalChromaExpansionReport bestChromaExpansion = Metrics.LocalChromaExpansion(best, input.Mat);
        double bestFullObjective = AutoTuner.Score(goal, best, input.Mat, gt.Mat, 0.0);

        ParamDef[] dimensions = method.Parameters.Where(parameter => thorough ? parameter.Tunable : parameter.Search).ToArray();
        var coverage = dimensions.Select(parameter =>
        {
            int expected = new[] { start[parameter.Key], parameter.Min, parameter.Max }
                .Select(parameter.Coerce).Distinct().Count();
            int observed = search.Diagnostics.DistinctValues.TryGetValue(parameter.Key, out int count) ? count : 0;
            return new { parameter = parameter.Key, expected_distinct_minimum = expected, observed_distinct = observed, ok = observed >= expected };
        }).ToArray();
        bool coverageComplete = coverage.All(item => item.ok);
        double nonDegradingTolerance = 1e-12 * Math.Max(1.0, Math.Abs(initialFullObjective));
        bool selectionNonDegrading = bestFullObjective + nonDegradingTolerance >= initialFullObjective;
        var changed = dimensions
            .Where(parameter => Math.Abs(search.Parameters[parameter.Key] - start[parameter.Key]) >
                1e-12 * Math.Max(1.0, Math.Abs(start[parameter.Key])))
            .ToDictionary(parameter => parameter.Key,
                parameter => new { from = start[parameter.Key], to = search.Parameters[parameter.Key] });

        SaveFloat(initial, Path.Combine(outputDirectory, "initial.png"));
        SaveFloat(best, Path.Combine(outputDirectory, "best.png"));
        CvInvoke.Imwrite(Path.Combine(outputDirectory, "hazy.png"), input.Mat);
        CvInvoke.Imwrite(Path.Combine(outputDirectory, "gt.png"), gt.Mat);

        var report = new
        {
            generated_utc = DateTimeOffset.UtcNow,
            scientific_status = "development_audit_not_blind_test",
            method = method.Name,
            mode = thorough ? "thorough" : "quick",
            goal = goal.ToString(),
            image = Path.GetFullPath(imagePath),
            gt = Path.GetFullPath(gtPath),
            image_sha256 = Hash(imagePath),
            gt_sha256 = Hash(gtPath),
            source_size = new { width = inputFull.Width, height = inputFull.Height },
            processed_size = new { width = input.Width, height = input.Height },
            evaluation_max_dimension = evaluationMaxDimension,
            search_diagnostics = search.Diagnostics,
            coverage_complete = coverageComplete,
            coordinate_coverage = coverage,
            changed_parameters = changed,
            start_parameters = start,
            best_parameters = search.Parameters,
            best_search_score = search.Score,
            selected_candidate = search.Diagnostics.SelectedCandidate,
            initial_full_metrics = initialMetrics,
            best_full_metrics = bestMetrics,
            initial_chromatic_fidelity = initialChroma,
            best_chromatic_fidelity = bestChroma,
            initial_local_chroma_expansion = initialChromaExpansion,
            best_local_chroma_expansion = bestChromaExpansion,
            initial_full_reference_score = AutoTuner.ReferenceScore(initialMetrics),
            best_full_reference_score = AutoTuner.ReferenceScore(bestMetrics),
            initial_full_objective = initialFullObjective,
            best_full_objective = bestFullObjective,
            selection_non_degrading = selectionNonDegrading,
            progress,
        };
        string jsonPath = Path.Combine(outputDirectory, "autotune-audit.json");
        File.WriteAllText(jsonPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }),
            new UTF8Encoding(false));
        Console.WriteLine($"AUTOTUNE-AUDIT method='{method.Name}' mode={(thorough ? "thorough" : "quick")} goal={goal} " +
            $"unique={search.Diagnostics.UniqueEvaluations} cache={search.Diagnostics.CacheHits} failed={search.Diagnostics.FailedEvaluations} " +
            $"verify={search.Diagnostics.FullResolutionVerifications}/{search.Diagnostics.FullResolutionVerificationFailures} " +
            $"rejected={search.Diagnostics.FullResolutionSafetyRejections} " +
            $"selected={search.Diagnostics.SelectedCandidate ?? "preview"} changed={changed.Count}/{dimensions.Length} " +
            $"coverage={coverageComplete} nonDegrading={selectionNonDegrading} " +
            $"fullObjective={initialFullObjective:F3}->{bestFullObjective:F3} " +
            $"fullReference={AutoTuner.ReferenceScore(initialMetrics):F3}->{AutoTuner.ReferenceScore(bestMetrics):F3} " +
            $"out={Path.GetFullPath(outputDirectory)}");
        return coverageComplete && selectionNonDegrading && double.IsFinite(search.Score) &&
            search.Diagnostics.FailedEvaluations == 0 && search.Diagnostics.FullResolutionVerificationFailures == 0 ? 0 : 1;
    }

    private static void SaveFloat(Mat value, string path)
    {
        using var clipped = DeHazeCPU.Clip(value.Clone());
        using var bytes = new Mat(); clipped.ConvertTo(bytes, DepthType.Cv8U, 255.0);
        CvInvoke.Imwrite(path, bytes);
    }

    private static string Hash(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }
}
