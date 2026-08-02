using SimpleDeHaze.Methods;

namespace SimpleDeHaze.Tests;

internal sealed class AutoTunerSearchTests
{
    public void Coverage_IndependentlyProbesEveryDimensionBeforeSoftBudget()
    {
        const int dimensions = 24; // больше прежних 16 уникальных баз Холтона
        var parameters = Enumerable.Range(0, dimensions)
            .Select(i => new ParamDef($"p{i}", $"P{i}", 0.0, 1.0, 0.5, search: true))
            .ToArray();
        var start = parameters.ToDictionary(p => p.Key, p => p.Default);
        var seen = new List<Dictionary<string, double>>();

        ParameterSearchResult result = ParameterSearch.Optimize(
            parameters,
            start,
            args =>
            {
                seen.Add(new Dictionary<string, double>(args));
                return 0.0; // плоская цель: исследование не должно зависеть от улучшений
            },
            maxEvals: 1,
            seedCount: 0);

        TestAssert.Equal(1 + 2 * dimensions, result.Diagnostics.EffectiveEvaluationLimit);
        TestAssert.Equal(1 + 2 * dimensions, result.Diagnostics.UniqueEvaluations);
        foreach (ParamDef parameter in parameters)
        {
            TestAssert.True(result.Diagnostics.DistinctValues[parameter.Key] >= 3,
                $"Параметр {parameter.Key} не был реально изменён в обе стороны.");

            bool lowAlone = seen.Any(point =>
                point[parameter.Key] == 0.0 && parameters.All(other =>
                    other.Key == parameter.Key || point[other.Key] == other.Default));
            bool highAlone = seen.Any(point =>
                point[parameter.Key] == 1.0 && parameters.All(other =>
                    other.Key == parameter.Key || point[other.Key] == other.Default));
            TestAssert.True(lowAlone && highAlone,
                $"Для {parameter.Key} нет независимой пары min/max при фиксированных остальных параметрах.");
        }
    }

    public void Search_FindsInteriorMixedScaleOptimumAndProgressNeverRegresses()
    {
        var parameters = new[]
        {
            new ParamDef("x", "continuous", 0.0, 1.0, 0.05),
            new ParamDef("odd", "odd integer", 3, 51, 3, 2, isInt: true),
            new ParamDef("log", "log", 1e-5, 1e-1, 1e-5, log: true),
            new ParamDef("step", "stepped", 0.1, 1.1, 0.1, 0.2),
        };
        var start = parameters.ToDictionary(p => p.Key, p => p.Default);
        var target = new Dictionary<string, double>
        {
            ["x"] = 0.73,
            ["odd"] = 35,
            ["log"] = 1e-3,
            ["step"] = 0.7,
        };
        var progress = new List<double>();

        double Objective(IReadOnlyDictionary<string, double> point)
        {
            double loss = 0.0;
            foreach (ParamDef p in parameters)
            {
                double delta = p.ToFraction(point[p.Key]) - p.ToFraction(target[p.Key]);
                loss += delta * delta;
            }
            return -loss;
        }

        ParameterSearchResult result = ParameterSearch.Optimize(
            parameters,
            start,
            Objective,
            maxEvals: 260,
            seedCount: 12,
            progress: (_, best) => progress.Add(best));

        TestAssert.InRange(result.Parameters["x"], 0.71, 0.75);
        TestAssert.InRange(result.Parameters["odd"], 33, 37);
        TestAssert.InRange(result.Parameters["log"] / target["log"], 0.94, 1.06);
        TestAssert.True(Math.Abs(result.Parameters["step"] - 0.7) < 1e-12,
            $"Ожидался дискретный optimum 0.7, получено {result.Parameters["step"]}.");
        TestAssert.True(result.Score > -0.002, $"Поиск остановился слишком далеко от optimum: {result.Score}.");
        for (int i = 1; i < progress.Count; i++)
            TestAssert.True(progress[i] + 1e-12 >= progress[i - 1], "Best-so-far в progress уменьшился.");
    }

    public void QuantizedCandidates_AreCachedAndDoNotConsumeBudgetAgain()
    {
        var parameter = new ParamDef("flag", "binary", 0, 1, 0, 1, isInt: true);
        var result = ParameterSearch.Optimize(
            new[] { parameter },
            new Dictionary<string, double> { [parameter.Key] = parameter.Default },
            args => args["flag"],
            maxEvals: 40,
            seedCount: 10);

        TestAssert.Equal(2, result.Diagnostics.UniqueEvaluations);
        TestAssert.True(result.Diagnostics.CacheHits > 0, "Округлённые дубликаты не попали в кэш.");
        TestAssert.Equal(1.0, result.Parameters["flag"]);
    }

    public void ParamDef_StepIsAnchoredAtMinimumAndRoundTrips()
    {
        var odd = new ParamDef("window", "odd window", 3, 51, 21, 2, isInt: true);
        TestAssert.Equal(51.0, odd.Coerce(50));
        TestAssert.Equal(21.0, odd.FromFraction(odd.ToFraction(21)));
        for (int i = 0; i <= 1000; i++)
        {
            double value = odd.FromFraction(i / 1000.0);
            TestAssert.True(value >= 3 && value <= 51 && ((int)value & 1) == 1,
                $"Получено недопустимое чётное окно {value}.");
        }
    }

    public void Halton_UsesUniquePrimeBaseBeyondSixteenDimensions()
    {
        double[] point = ParameterSearch.HaltonPoint(1, 20);
        TestAssert.True(Math.Abs(point[0] - 1.0 / 2.0) < 1e-12);
        TestAssert.True(Math.Abs(point[1] - 1.0 / 3.0) < 1e-12);
        TestAssert.True(Math.Abs(point[16] - 1.0 / 59.0) < 1e-12);
        TestAssert.True(point.Distinct().Count() == point.Length,
            "После 16-й координаты повторилась база Холтона.");
    }

    public void Registry_DefinitionsHaveUniqueKeysAndValidDefaults()
    {
        foreach (IDeHazeMethod method in MethodRegistry.All)
        {
            TestAssert.Equal(method.Parameters.Count, method.Parameters.Select(p => p.Key).Distinct(StringComparer.Ordinal).Count());
            foreach (ParamDef p in method.Parameters)
            {
                double canonical = p.Coerce(p.Default);
                TestAssert.True(Math.Abs(canonical - p.Default) <= 1e-10 * Math.Max(1.0, Math.Abs(p.Default)),
                    $"{method.Name}/{p.Key}: default {p.Default} не лежит на допустимой сетке (получено {canonical}).");
                TestAssert.True(!p.Search || p.Tunable,
                    $"{method.Name}/{p.Key}: Search=true, но Tunable=false.");
            }
        }
    }

    public void Registry_StructuralModesStayFixedDuringThoroughTune()
    {
        var structuralKeys = new HashSet<string>(StringComparer.Ordinal)
        {
            "boots", "diverse", "noiseon", "tuncon", "auncon", "feasible", "tviter",
            "fast", "scale", "quality", "linear", "strict", "csbound", "predehaze",
            "rough", "wiener", "s1", "s2", "s3", "project", "backbone",
        };

        foreach (IDeHazeMethod method in MethodRegistry.All)
            foreach (ParamDef parameter in method.Parameters.Where(p => structuralKeys.Contains(p.Key)))
                TestAssert.False(parameter.Tunable,
                    $"{method.Name}/{parameter.Key}: режимный параметр ошибочно участвует в числовом поиске.");
    }

    public void ChromaticAnchor_SearchCoordinatesAreIndependentAndActuallyProbed()
    {
        var method = new ChromaticAnchorMethod();
        var parameters = method.Parameters.Where(parameter => parameter.Search).ToArray();
        TestAssert.True(parameters.Any(parameter => parameter.Key == "denseLo"));
        TestAssert.True(parameters.Any(parameter => parameter.Key == "denseSpan"));
        TestAssert.False(method.Parameters.Any(parameter => parameter.Key == "denseHi"),
            "Зависимая координата denseHi снова появилась в пространстве поиска.");

        var start = method.Parameters.ToDictionary(parameter => parameter.Key, parameter => parameter.Default);
        ParameterSearchResult result = ParameterSearch.Optimize(parameters, start, _ => 0.0,
            maxEvals: 1, seedCount: 0);
        foreach (ParamDef parameter in parameters)
        {
            int expectedDistinct = new[] { parameter.Default, parameter.Min, parameter.Max }
                .Select(parameter.Coerce).Distinct().Count();
            TestAssert.True(result.Diagnostics.DistinctValues[parameter.Key] >= expectedDistinct,
                $"CAR/{parameter.Key}: координата не получила все уникальные default/min/max пробы.");
        }

        ParamDef lo = method.Parameters.Single(parameter => parameter.Key == "denseLo");
        ParamDef span = method.Parameters.Single(parameter => parameter.Key == "denseSpan");
        foreach (double loFraction in new[] { 0.0, 0.5, 1.0 })
        foreach (double spanFraction in new[] { 0.0, 0.5, 1.0 })
        {
            double low = lo.FromFraction(loFraction);
            double high = low + (1.0 - low) * span.FromFraction(spanFraction);
            TestAssert.True(high - low >= 0.05 - 1e-12 && high <= 1.0,
                $"Некорректная независимая пара dense gate: lo={low}, hi={high}.");
        }
    }

    public void ReferenceScore_IgnoresGtFittedDiagnosticMetrics()
    {
        Metrics.Report a = Report(psnrAligned: 5, mseAligned: 9000, ssimAligned: 0.1, deAligned: 80);
        Metrics.Report b = Report(psnrAligned: 80, mseAligned: 0, ssimAligned: 1.0, deAligned: 0);
        double scoreA = AutoTuner.ReferenceScore(a);
        double scoreB = AutoTuner.ReferenceScore(b);
        TestAssert.True(Math.Abs(scoreA - scoreB) < 1e-12,
            "Диагностическое GT-согласование всё ещё влияет на подбор параметров.");

        Metrics.Report genuinelyBetter = Report(rawPsnr: 24, rawMse: 400, rawSsim: 0.90, rawDe: 5);
        TestAssert.True(AutoTuner.ReferenceScore(genuinelyBetter) > scoreA,
            "Улучшение основных raw-метрик не повысило objective.");
    }

    public void ReferenceScore_DoesNotDoubleCountMseAndRejectsStructuralCollapse()
    {
        Metrics.Report lowMse = Report(rawMse: 100);
        Metrics.Report highMse = Report(rawMse: 6000);
        TestAssert.True(Math.Abs(AutoTuner.ReferenceScore(lowMse) - AutoTuner.ReferenceScore(highMse)) < 1e-12,
            "MSE снова учитывается отдельно от эквивалентного PSNR.");

        Metrics.Report structuredColor = Report(rawPsnr: 17.78, rawSsim: 0.637, rawDe: 12.69);
        Metrics.Report smallPsnrGainButCollapsedStructure = Report(rawPsnr: 18.21, rawSsim: 0.524, rawDe: 11.17);
        TestAssert.True(AutoTuner.ReferenceScore(structuredColor) >
            AutoTuner.ReferenceScore(smallPsnrGainButCollapsedStructure),
            "Малый выигрыш PSNR перекрыл крупную потерю структуры и цвета.");
    }

    public void NoReferenceHarshPenalty_UsesNaturalnessOnlyAsExtremeSafetyGuard()
    {
        Metrics.Report ordinary = Report(naturalness: 10);
        Metrics.Report sameDirectArtifacts = Report(naturalness: 45);
        double a = AutoTuner.NoReferenceHarshPenalty(ordinary, vivid: false);
        double b = AutoTuner.NoReferenceHarshPenalty(sameDirectArtifacts, vivid: false);
        TestAssert.True(Math.Abs(a - b) < 1e-12,
            "NaturalnessDev снова оптимизируется внутри обычного межметодного диапазона.");

        Metrics.Report implausible = Report(naturalness: 90);
        TestAssert.True(AutoTuner.NoReferenceHarshPenalty(implausible, vivid: false) > b,
            "Экстремальный NaturalnessDev больше не работает как аварийный заслон.");

        Metrics.Report actualHarshness = Report(artifact: 50, contrast: 6, edge: 20);
        TestAssert.True(AutoTuner.NoReferenceHarshPenalty(actualHarshness, vivid: false) > a,
            "Прямые признаки экстремального контраста/граней/артефактов перестали штрафоваться.");
    }

    public void FullResolutionSafety_RejectsGoodhartExtremesButAllowsBoundedImprovement()
    {
        Metrics.Report baseline = Report(naturalness: 42, artifact: 22, clip: 1.5, haze: 0.52, color: 1.76);
        Metrics.Report safe = Report(naturalness: 49, artifact: 29, clip: 2.4, haze: 0.48, color: 1.30);
        TestAssert.True(AutoTuner.PassesNoReferenceSafety(safe, baseline, AutoTuneGoal.ObjectVisibility),
            "Разумный полноразмерный кандидат ошибочно заблокирован.");

        Metrics.Report clipped = Report(naturalness: 49, artifact: 29, clip: 5.0, haze: 0.48, color: 1.30);
        Metrics.Report masked = Report(naturalness: 55, artifact: 32, clip: 2.4, haze: 0.48, color: 1.30);
        Metrics.Report rehazed = Report(naturalness: 45, artifact: 24, clip: 2.0, haze: 0.40, color: 1.30);
        TestAssert.False(AutoTuner.PassesNoReferenceSafety(clipped, baseline, AutoTuneGoal.ObjectVisibility));
        TestAssert.False(AutoTuner.PassesNoReferenceSafety(masked, baseline, AutoTuneGoal.ObjectVisibility));
        TestAssert.False(AutoTuner.PassesNoReferenceSafety(rehazed, baseline, AutoTuneGoal.ObjectVisibility));
    }

    public void Search_RejectsRunWhenEveryEvaluationFails()
    {
        bool threw = false;
        try
        {
            ParameterSearch.Optimize(
                new[] { new ParamDef("x", "x", 0, 1, 0.5) },
                new Dictionary<string, double> { ["x"] = 0.5 },
                _ => double.NaN,
                maxEvals: 3,
                seedCount: 0);
        }
        catch (InvalidOperationException) { threw = true; }
        TestAssert.True(threw, "Полностью невалидный поиск молча вернул стартовую точку.");
    }

    private static Metrics.Report Report(
        double rawPsnr = 18,
        double rawMse = 1200,
        double rawSsim = 0.72,
        double rawDe = 14,
        double psnrAligned = 18,
        double mseAligned = 1200,
        double ssimAligned = 0.72,
        double deAligned = 14,
        double naturalness = 10,
        double artifact = 10,
        double contrast = 1.5,
        double edge = 1.5,
        double clip = 0,
        double haze = 0.5,
        double color = 1)
        => new(
            HasRef: true,
            Psnr: rawPsnr,
            PsnrAligned: psnrAligned,
            Mse: rawMse,
            MseAligned: mseAligned,
            Ssim: rawSsim,
            SsimAligned: ssimAligned,
            Ciede2000: rawDe,
            Ciede2000Aligned: deAligned,
            NaturalnessDev: naturalness,
            ArtifactDev: artifact,
            FlatNoiseRatio: 1,
            HazeRemoved: haze,
            ContrastGain: contrast,
            EdgeGain: edge,
            ClipPct: clip,
            ColorRatio: color,
            Score: 50);
}
