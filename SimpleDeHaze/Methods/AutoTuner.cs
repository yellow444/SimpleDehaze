using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Авто-подбор 'оптимальных' параметров перебором по no-reference метрике качества дехейзинга.
    /// Перебор идёт на уменьшенной копии кадра (быстро даже для тяжёлых методов); найденные значения
    /// затем применяются к полному кадру.
    /// </summary>
    public static class AutoTuner
    {
        public static Dictionary<string, double> Optimize(IDeHazeMethod m, Image<Bgr, byte> input, IReadOnlyDictionary<string, double> current)
        {
            using var thumb = Thumbnail(input, 480);

            var sp = m.Parameters.Where(p => p.Search).Take(2).ToList();
            var best = new Dictionary<string, double>(current);
            if (sp.Count == 0) return best;

            var grids = sp.Select(p => Candidates(p, 4)).ToList();
            double bestScore = double.NegativeInfinity;
            foreach (var combo in Cartesian(grids))
            {
                var args = new Dictionary<string, double>(current);
                for (int i = 0; i < sp.Count; i++) args[sp[i].Key] = combo[i];
                try
                {
                    using var res = m.Process(thumb, args);
                    // без-эталонная сводная оценка: удаление дымки + детали - пересвет/перенасыщение
                    double sc = Metrics.NoRefScore(res, thumb.Mat);
                    if (sc > bestScore) { bestScore = sc; best = new Dictionary<string, double>(args); }
                }
                catch { /* пропускаем неудачные комбинации */ }
            }
            return best;
        }

        /// <summary>
        /// Тщательный (долгий) подбор ВСЕХ параметров метода по той же без-эталонной оценке.
        /// Две фазы: (1) грубый покоординатный обход - для каждого параметра прогоняем крупные шаги по
        /// всему диапазону и сдвигаемся в лучшую сторону; (2) компас-поиск с уменьшающимся шагом
        /// (крупно -> мелко) вокруг найденного места, пока есть улучшение. progress(попытки, лучший_скор).
        /// </summary>
        public static Dictionary<string, double> OptimizeThorough(
            IDeHazeMethod m, Image<Bgr, byte> input, IReadOnlyDictionary<string, double> start,
            double minColor = 0, Action<int, double>? progress = null, Func<bool>? cancelled = null, int maxEvals = 150)
        {
            using var thumb = Thumbnail(input, 340);
            var ps = m.Parameters.ToList();
            var x = new Dictionary<string, double>();
            foreach (var p in ps) x[p.Key] = start.TryGetValue(p.Key, out var v0) ? v0 : p.Default;
            if (ps.Count == 0) return x;

            int evals = 0;
            double Eval(Dictionary<string, double> args)
            {
                evals++;
                try
                {
                    using var res = m.Process(thumb, args);
                    var rep = Metrics.Evaluate(res, null, thumb.Mat);
                    double sc = rep.Score;
                    // мягкое ограничение 'не гасить цвет': если насыщенность ниже порога - большой штраф
                    if (minColor > 0 && rep.ColorRatio < minColor) sc -= 200.0 * (minColor - rep.ColorRatio);
                    return sc;
                }
                catch { return double.NegativeInfinity; }
            }
            bool Stop() => evals >= maxEvals || (cancelled?.Invoke() ?? false);

            double best = Eval(x);
            progress?.Invoke(evals, best);

            // Покоординатный спуск с БРЕКЕТИНГ-линейным поиском: по каждому параметру сначала крупные шаги
            // по всему диапазону, находим лучший узел, СУЖАЕМ диапазон до соседних узлов вокруг него и
            // бьём мельче уже внутри - и так несколько уровней. Быстрее фиксированного шага: чем ближе к
            // оптимуму, тем мельче пробы, без лишних вычислений вдали от него.
            for (int pass = 0; pass < 3 && !Stop(); pass++)
            {
                bool improved = false;
                foreach (var p in ps)
                {
                    if (Stop()) break;
                    double lo = 0, hi = 1, bestF = ValToFrac(p, x[p.Key]), bestS = best;
                    for (int level = 0; level < 3 && !Stop(); level++)   // крупно -> мельче в суженном диапазоне
                    {
                        const int n = 5;
                        for (int k = 0; k < n; k++)
                        {
                            double f = lo + (hi - lo) * k / (n - 1);
                            if (level > 0 && Math.Abs(f - bestF) < 1e-9) continue;   // центр диапазона уже посчитан
                            double s = Eval(new Dictionary<string, double>(x) { [p.Key] = FracToVal(p, f) });
                            progress?.Invoke(evals, Math.Max(best, bestS));
                            if (s > bestS) { bestS = s; bestF = f; }
                            if (Stop()) break;
                        }
                        double half = (hi - lo) / (n - 1);                            // диапазон лучших ~ соседние узлы
                        lo = Math.Max(0, bestF - half); hi = Math.Min(1, bestF + half);
                    }
                    if (bestS > best) { x[p.Key] = FracToVal(p, bestF); best = bestS; improved = true; }
                }
                if (!improved) break;   // проход без улучшений - дальше смысла нет
            }
            progress?.Invoke(evals, best);
            return x;
        }

        /// <summary>
        /// 'Авто-лучший': быстро сканирует все методы на превью (дефолтные параметры), выбирает лучший по
        /// без-эталонной оценке и тщательно настраивает его параметры. Возвращает (метод, параметры, скор).
        /// </summary>
        public static (IDeHazeMethod method, Dictionary<string, double> prms, double score) PickBest(
            IReadOnlyList<IDeHazeMethod> methods, Image<Bgr, byte> input,
            Action<string>? progress = null, Func<bool>? cancelled = null)
        {
            IDeHazeMethod best = methods[0];
            double bestScore = double.NegativeInfinity;
            using (var thumb = Thumbnail(input, 300))   // скан - мелкое превью, нужен лишь относительный рейтинг
            {
                int i = 0;
                foreach (var m in methods)
                {
                    if (cancelled?.Invoke() ?? false) break;
                    i++;
                    try
                    {
                        var def = m.Parameters.ToDictionary(p => p.Key, p => p.Default);
                        using var res = m.Process(thumb, def);
                        double s = Metrics.NoRefScore(res, thumb.Mat);
                        if (s > bestScore) { bestScore = s; best = m; }
                    }
                    catch { /* пропускаем падающие методы */ }
                    progress?.Invoke($"Авто: скан {i}/{methods.Count} - лучший '{best.Name}' ({bestScore:F0})");
                }
            }
            var start = best.Parameters.ToDictionary(p => p.Key, p => p.Default);
            var tuned = OptimizeThorough(best, input, start, 0,
                (e, sc) => progress?.Invoke($"Авто: настройка '{best.Name}'... попытка {e}, скор {sc:F0}"), cancelled);
            return (best, tuned, bestScore);
        }

        /// <summary>Положение значения в [0,1] вдоль диапазона параметра (с учётом лог-шкалы).</summary>
        private static double ValToFrac(ParamDef p, double v)
        {
            v = Math.Clamp(v, p.Min, p.Max);
            return p.Log ? Math.Log(v / p.Min) / Math.Log(p.Max / p.Min) : (v - p.Min) / (p.Max - p.Min);
        }

        /// <summary>Значение параметра по положению [0,1] (лог-шкала + округление int).</summary>
        private static double FracToVal(ParamDef p, double f)
        {
            f = Math.Clamp(f, 0, 1);
            double v = p.Log ? p.Min * Math.Pow(p.Max / p.Min, f) : p.Min + f * (p.Max - p.Min);
            if (p.IsInt) v = Math.Round(v);
            return Math.Clamp(v, p.Min, p.Max);
        }

        private static Image<Bgr, byte> Thumbnail(Image<Bgr, byte> img, int maxDim)
        {
            int w = img.Width, h = img.Height;
            double s = Math.Min(1.0, (double)maxDim / Math.Max(w, h));
            return s >= 1.0 ? img.Clone() : img.Resize((int)(w * s), (int)(h * s), Inter.Area);
        }

        private static List<double> Candidates(ParamDef p, int steps)
        {
            var list = new List<double>();
            for (int i = 0; i < steps; i++)
            {
                double frac = steps == 1 ? 0.5 : (double)i / (steps - 1);
                double v = p.Log ? p.Min * Math.Pow(p.Max / p.Min, frac) : p.Min + frac * (p.Max - p.Min);
                if (p.IsInt) v = Math.Round(v);
                list.Add(v);
            }
            return list.Distinct().ToList();
        }

        private static IEnumerable<double[]> Cartesian(List<List<double>> grids)
        {
            IEnumerable<double[]> result = new[] { Array.Empty<double>() };
            foreach (var grid in grids)
            {
                var next = new List<double[]>();
                foreach (var prefix in result)
                    foreach (var v in grid)
                    {
                        var arr = new double[prefix.Length + 1];
                        Array.Copy(prefix, arr, prefix.Length);
                        arr[prefix.Length] = v;
                        next.Add(arr);
                    }
                result = next;
            }
            return result;
        }
    }
}
