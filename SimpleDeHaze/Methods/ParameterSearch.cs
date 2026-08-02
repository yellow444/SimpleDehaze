using System.Globalization;
using System.Text;

namespace SimpleDeHaze.Methods
{
    /// <summary>Диагностика фактически выполненного поиска, используемая регрессионными тестами.</summary>
    internal sealed record ParameterSearchDiagnostics(
        int RequestedEvaluationLimit,
        int EffectiveEvaluationLimit,
        int UniqueEvaluations,
        int CacheHits,
        int FailedEvaluations,
        IReadOnlyDictionary<string, int> DistinctValues,
        int FullResolutionVerifications = 0,
        int FullResolutionVerificationFailures = 0,
        string? SelectedCandidate = null,
        int FullResolutionSafetyRejections = 0);

    internal sealed record ParameterSearchResult(
        Dictionary<string, double> Parameters,
        double Score,
        ParameterSearchDiagnostics Diagnostics);

    /// <summary>
    /// Детерминированное безградиентное ядро автоподбора. Оно отделено от обработки изображений,
    /// чтобы свойства поиска (покрытие всех координат, дискретизация, best-so-far) можно было проверять
    /// на контролируемых функциях без OpenCV.
    /// </summary>
    internal static class ParameterSearch
    {
        private sealed record Candidate(double[] Fractions, Dictionary<string, double> Arguments, double Score);

        internal static ParameterSearchResult Optimize(
            IReadOnlyList<ParamDef> searchParams,
            IReadOnlyDictionary<string, double> startAll,
            Func<IReadOnlyDictionary<string, double>, double> objective,
            int maxEvals,
            int seedCount,
            Action<int, double>? progress = null,
            Func<bool>? cancelled = null)
        {
            ArgumentNullException.ThrowIfNull(searchParams);
            ArgumentNullException.ThrowIfNull(startAll);
            ArgumentNullException.ThrowIfNull(objective);
            if (maxEvals <= 0) throw new ArgumentOutOfRangeException(nameof(maxEvals), "Бюджет поиска должен быть положительным.");
            if (seedCount < 0) throw new ArgumentOutOfRangeException(nameof(seedCount));

            int n = searchParams.Count;
            var keys = new HashSet<string>(StringComparer.Ordinal);
            foreach (var p in searchParams)
                if (!keys.Add(p.Key)) throw new ArgumentException($"Параметр '{p.Key}' указан в поиске дважды.", nameof(searchParams));

            if (n == 0)
            {
                var emptyDiagnostics = new ParameterSearchDiagnostics(maxEvals, maxEvals, 0, 0, 0,
                    new Dictionary<string, int>(StringComparer.Ordinal));
                return new ParameterSearchResult(new Dictionary<string, double>(startAll), double.NaN, emptyDiagnostics);
            }

            // maxEvals — обычный верхний предел, но он не может отменить минимальный аудит координат:
            // текущая точка + min/max каждого параметра при фиксированных остальных координатах.
            int coverageBudget = checked(1 + 2 * n);
            int effectiveLimit = Math.Max(maxEvals, coverageBudget);
            int evaluations = 0, cacheHits = 0, failed = 0;
            bool wasCancelled = false;
            var cache = new Dictionary<string, Candidate>(StringComparer.Ordinal);
            var distinct = searchParams.ToDictionary(
                p => p.Key, _ => new HashSet<long>(), StringComparer.Ordinal);
            Candidate? bestCandidate = null;

            bool IsCancelled()
            {
                if (wasCancelled) return true;
                wasCancelled = cancelled?.Invoke() ?? false;
                return wasCancelled;
            }

            bool Stop() => evaluations >= effectiveLimit || IsCancelled();

            Candidate Canonicalize(double[] requested, double score = double.NaN)
            {
                var fractions = new double[n];
                var args = new Dictionary<string, double>(startAll, StringComparer.Ordinal);
                for (int i = 0; i < n; i++)
                {
                    double value = searchParams[i].FromFraction(requested[i]);
                    fractions[i] = searchParams[i].ToFraction(value);
                    args[searchParams[i].Key] = value;
                }
                return new Candidate(fractions, args, score);
            }

            static string CacheKey(IReadOnlyList<ParamDef> ps, IReadOnlyDictionary<string, double> args)
            {
                var sb = new StringBuilder(ps.Count * 18);
                foreach (var p in ps)
                {
                    long bits = BitConverter.DoubleToInt64Bits(args[p.Key]);
                    sb.Append(bits.ToString("X16", CultureInfo.InvariantCulture)).Append(';');
                }
                return sb.ToString();
            }

            bool TryEvaluate(double[] requested, out Candidate candidate, out bool isNew)
            {
                candidate = Canonicalize(requested);
                string key = CacheKey(searchParams, candidate.Arguments);
                if (cache.TryGetValue(key, out var cached))
                {
                    cacheHits++;
                    candidate = cached;
                    isNew = false;
                    return true;
                }
                if (Stop())
                {
                    isNew = false;
                    return false;
                }

                evaluations++;
                double score;
                try { score = objective(candidate.Arguments); }
                catch { score = double.NegativeInfinity; }
                if (!double.IsFinite(score))
                {
                    score = double.NegativeInfinity;
                    failed++;
                }

                candidate = candidate with { Score = score };
                cache.Add(key, candidate);
                for (int i = 0; i < n; i++)
                    distinct[searchParams[i].Key].Add(BitConverter.DoubleToInt64Bits(candidate.Arguments[searchParams[i].Key]));
                isNew = true;
                return true;
            }

            static bool Better(double candidate, double incumbent)
            {
                if (!double.IsFinite(candidate)) return false;
                if (!double.IsFinite(incumbent)) return true;
                double tolerance = 1e-12 * Math.Max(1.0, Math.Abs(incumbent));
                return candidate > incumbent + tolerance;
            }

            bool Consider(double[] fractions)
            {
                if (!TryEvaluate(fractions, out var candidate, out bool isNew)) return false;
                bool improved = bestCandidate == null || Better(candidate.Score, bestCandidate.Score);
                if (improved) bestCandidate = candidate;
                if (isNew) progress?.Invoke(evaluations, bestCandidate?.Score ?? double.NegativeInfinity);
                return improved;
            }

            var origin = new double[n];
            for (int i = 0; i < n; i++)
            {
                double value = startAll.TryGetValue(searchParams[i].Key, out var supplied)
                    ? supplied
                    : searchParams[i].Default;
                origin[i] = searchParams[i].ToFraction(value);
            }

            // Обязательное независимое покрытие выполняется ДО совместных seed-точек. Поэтому ни один
            // поздний параметр не останется неизменённым из-за исчерпания бюджета ранними координатами.
            Consider(origin);
            for (int i = 0; i < n && !Stop(); i++)
            {
                var low = (double[])origin.Clone();
                low[i] = 0.0;
                Consider(low);
                if (Stop()) break;
                var high = (double[])origin.Clone();
                high[i] = 1.0;
                Consider(high);
            }

            int[] bases = FirstPrimes(n);
            if (!Stop())
            {
                var defaults = new double[n];
                for (int i = 0; i < n; i++) defaults[i] = searchParams[i].ToFraction(searchParams[i].Default);
                Consider(defaults);
                if (!Stop()) Consider(Enumerable.Repeat(0.0, n).ToArray());
                if (!Stop()) Consider(Enumerable.Repeat(1.0, n).ToArray());
                for (int sample = 1; sample <= seedCount && !Stop(); sample++)
                    Consider(HaltonPoint(sample, bases));
            }

            var current = bestCandidate?.Fractions.ToArray() ?? origin;
            const double minStep = 1.0 / 1024.0;
            int haltonIndex = seedCount;
            int staleRestarts = 0;
            int sweepOffset = 0;

            while (!Stop() && staleRestarts < 3)
            {
                double h = 0.30;
                while (h > minStep && !Stop())
                {
                    double scoreBeforeSweep = bestCandidate?.Score ?? double.NegativeInfinity;
                    var beforeSweep = (double[])current.Clone();

                    // Циклический порядок не отдаёт постоянный приоритет первым параметрам, если
                    // последний локальный проход обрывается ровно на границе бюджета.
                    for (int order = 0; order < n && !Stop(); order++)
                    {
                        int i = (sweepOffset + order) % n;
                        var coordinateBase = (double[])current.Clone();

                        // Проверяем ОБА направления от одной базы. Прежний вариант останавливался на
                        // первом малом улучшении и мог не увидеть намного лучший противоположный шаг.
                        foreach (double direction in new[] { h, -h })
                        {
                            if (Stop()) break;
                            var trial = (double[])coordinateBase.Clone();
                            trial[i] = Math.Clamp(coordinateBase[i] + direction, 0.0, 1.0);
                            Consider(trial);
                        }
                        if (bestCandidate != null) current = bestCandidate.Fractions.ToArray();
                    }
                    sweepOffset = (sweepOffset + 1) % n;

                    if (bestCandidate != null && Better(bestCandidate.Score, scoreBeforeSweep))
                    {
                        // Ускоряющий паттерн-шаг по суммарному направлению улучшения за sweep.
                        var pattern = new double[n];
                        for (int i = 0; i < n; i++)
                            pattern[i] = Math.Clamp(current[i] + (current[i] - beforeSweep[i]), 0.0, 1.0);
                        if (!Stop()) Consider(pattern);
                        current = bestCandidate.Fractions.ToArray();
                    }
                    else
                    {
                        h *= 0.5;
                    }
                }

                if (Stop()) break;
                double scoreBeforeRestart = bestCandidate?.Score ?? double.NegativeInfinity;

                // Два совместных джиттера вокруг текущего best и одна новая глобальная точка.
                // Один индекс Холтона задаёт весь вектор; базы уникальны и не повторяются после 16D.
                for (int restart = 0; restart < 3 && !Stop(); restart++)
                {
                    double[] hv = HaltonPoint(++haltonIndex, bases);
                    var trial = new double[n];
                    for (int i = 0; i < n; i++)
                        trial[i] = restart < 2
                            ? Math.Clamp(current[i] + 0.35 * (hv[i] - 0.5), 0.0, 1.0)
                            : hv[i];
                    Consider(trial);
                }
                current = bestCandidate?.Fractions.ToArray() ?? current;
                staleRestarts = bestCandidate != null && Better(bestCandidate.Score, scoreBeforeRestart)
                    ? 0
                    : staleRestarts + 1;
            }

            var diagnostics = new ParameterSearchDiagnostics(
                maxEvals,
                effectiveLimit,
                evaluations,
                cacheHits,
                failed,
                distinct.ToDictionary(kv => kv.Key, kv => kv.Value.Count, StringComparer.Ordinal));

            if (bestCandidate == null)
                return new ParameterSearchResult(Canonicalize(origin).Arguments, double.NegativeInfinity, diagnostics);
            if (!double.IsFinite(bestCandidate.Score) && !wasCancelled)
                throw new InvalidOperationException("Автоподбор не получил ни одной конечной оценки: все кандидаты завершились ошибкой.");
            return new ParameterSearchResult(new Dictionary<string, double>(bestCandidate.Arguments), bestCandidate.Score, diagnostics);
        }

        internal static double[] HaltonPoint(int index, int dimensions)
            => HaltonPoint(index, FirstPrimes(dimensions));

        private static double[] HaltonPoint(int index, IReadOnlyList<int> bases)
        {
            if (index <= 0) throw new ArgumentOutOfRangeException(nameof(index));
            var point = new double[bases.Count];
            for (int i = 0; i < bases.Count; i++) point[i] = Halton(index, bases[i]);
            return point;
        }

        private static int[] FirstPrimes(int count)
        {
            if (count < 0) throw new ArgumentOutOfRangeException(nameof(count));
            var primes = new List<int>(count);
            for (int candidate = 2; primes.Count < count; candidate++)
            {
                bool prime = true;
                int limit = (int)Math.Sqrt(candidate);
                foreach (int p in primes)
                {
                    if (p > limit) break;
                    if (candidate % p == 0) { prime = false; break; }
                }
                if (prime) primes.Add(candidate);
            }
            return primes.ToArray();
        }

        private static double Halton(int index, int b)
        {
            double fraction = 1.0, result = 0.0;
            while (index > 0)
            {
                fraction /= b;
                result += fraction * (index % b);
                index /= b;
            }
            return result;
        }
    }
}
