using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Авто-подбор 'оптимальных' параметров поиском по метрике качества дехейзинга (без эталона или
    /// против эталона - по выбранной цели). Перебор идёт на уменьшенной копии кадра (быстро даже для
    /// тяжёлых методов); найденные значения затем применяются к полному кадру.
    ///
    /// Ядро поиска - <see cref="ParameterSearch"/>: обязательная независимая проверка диапазона каждого
    /// параметра, детерминированный мульти-старт Холтона и паттерн-поиск Хука-Дживса. Поиск ведётся
    /// по каноническим значениям ParamDef (log/int/step), а повторные фактические точки кэшируются.
    /// </summary>
    public static class AutoTuner
    {
        /// <summary>
        /// Быстрый подбор: коротким паттерн-поиском настраивает ВСЕ параметры с флагом <c>Search</c>
        /// (остальные фиксированы на текущих значениях). Бюджет ~60 оценок на превью, затем исходная
        /// и найденная точки сравниваются на полном кадре (две проверки вне поискового бюджета).
        /// </summary>
        public static Dictionary<string, double> Optimize(
            IDeHazeMethod m, Image<Bgr, byte> input, IReadOnlyDictionary<string, double> current,
            Mat? gt = null, AutoTuneGoal goal = AutoTuneGoal.ObjectVisibility, int evalMaxDim = 480)
        {
            using var thumb = Thumbnail(input, evalMaxDim);
            using var gtThumb = ReferenceThumb(gt, thumb.Size);

            var startAll = new Dictionary<string, double>(current);
            foreach (var p in m.Parameters)
                startAll[p.Key] = p.Coerce(startAll.TryGetValue(p.Key, out var supplied) ? supplied : p.Default);

            var sp = m.Parameters.Where(p => p.Search).ToList();
            if (sp.Count == 0) return startAll;

            using var probe = NativeProbe(input, thumb);
            ParameterSearchResult search = PatternSearchResult(m, thumb, gtThumb, sp, startAll, goal, 0.0,
                maxEvals: 60, seedCount: 4, progress: null, cancelled: null, nativeProbe: probe);
            return FinalizeQuickResult(m, input, gt, goal, 0.0, startAll, search, 60, null).Parameters;
        }

        /// <summary>
        /// Тщательный (долгий) подбор ВСЕХ параметров метода. Мульти-старт (текущая точка, all-min,
        /// all-max, точки Холтона) находит хороший бассейн, затем паттерн-поиск Хука-Дживса уточняет.
        /// progress(попытки, лучший_скор). <paramref name="minColor"/> штрафует обесцвечивание.
        /// maxEvals является мягким пределом: если он меньше обязательного покрытия min/max всех
        /// координат, ядро автоматически поднимает его до 1 + 2*N уникальных попыток. После поиска
        /// исходная, грубая и уточнённая точки повторно сравниваются на полном кадре; эти 2-3
        /// верификации не входят в поисковый бюджет и не дают уточнению ухудшить целевую функцию.
        /// </summary>
        public static Dictionary<string, double> OptimizeThorough(
            IDeHazeMethod m, Image<Bgr, byte> input, IReadOnlyDictionary<string, double> start,
            double minColor = 0, Action<int, double>? progress = null, Func<bool>? cancelled = null, int maxEvals = 320,
            Mat? gt = null, AutoTuneGoal goal = AutoTuneGoal.ObjectVisibility, int evalMaxDim = 340)
        {
            using var fineThumb = Thumbnail(input, evalMaxDim);
            using var coarseThumb = Thumbnail(input, Math.Max(480, evalMaxDim));
            using var fineGt = ReferenceThumb(gt, fineThumb.Size);
            using var coarseGt = ReferenceThumb(gt, coarseThumb.Size);
            var x = new Dictionary<string, double>();
            foreach (var p in m.Parameters)
                x[p.Key] = p.Coerce(start.TryGetValue(p.Key, out var v0) ? v0 : p.Default);
            // режимные/структурные параметры (быстро/HQ, апскейл) задаёт пользователь - их не трогаем
            var ps = m.Parameters.Where(p => p.Tunable).ToList();
            if (ps.Count == 0) return x;

            using var probe = NativeProbe(input, coarseThumb);
            return ThoroughSearchResult(m, coarseThumb, coarseGt, fineThumb, fineGt, ps, x, goal, minColor,
                maxEvals, progress, cancelled, probe, input, gt).Parameters;
        }

        /// <summary>
        /// Воспроизводимый аудит фактического изображения: возвращает не только лучший набор, но и
        /// число уникальных оценок/кэш-попаданий/ошибок и количество разных значений каждой координаты.
        /// Используется исследовательским CLI и не меняет поведение GUI-подбора.
        /// </summary>
        internal static ParameterSearchResult Audit(
            IDeHazeMethod method, Image<Bgr, byte> input, IReadOnlyDictionary<string, double> start,
            Mat? gt, AutoTuneGoal goal, bool thorough, int maxEvals, int evalMaxDim,
            Action<int, double>? progress = null, Func<bool>? cancelled = null)
        {
            using var fineThumb = Thumbnail(input, evalMaxDim);
            using var coarseThumb = thorough ? Thumbnail(input, Math.Max(480, evalMaxDim)) : null;
            using var fineGt = ReferenceThumb(gt, fineThumb.Size);
            using var coarseGt = coarseThumb == null ? null : ReferenceThumb(gt, coarseThumb.Size);
            var canonical = new Dictionary<string, double>(StringComparer.Ordinal);
            foreach (ParamDef parameter in method.Parameters)
                canonical[parameter.Key] = parameter.Coerce(start.TryGetValue(parameter.Key, out double value)
                    ? value : parameter.Default);
            var dimensions = method.Parameters.Where(parameter => thorough ? parameter.Tunable : parameter.Search).ToArray();
            using var probe = NativeProbe(input, coarseThumb ?? fineThumb);
            return thorough
                ? ThoroughSearchResult(method, coarseThumb!, coarseGt, fineThumb, fineGt, dimensions,
                    canonical, goal, 0.0, maxEvals, progress, cancelled, probe, input, gt)
                : FinalizeQuickResult(method, input, gt, goal, 0.0, canonical,
                    PatternSearchResult(method, fineThumb, fineGt, dimensions, canonical, goal, 0.0,
                        maxEvals, 4, progress, cancelled, probe), maxEvals, cancelled);
        }

        private static ParameterSearchResult ThoroughSearchResult(
            IDeHazeMethod method,
            Image<Bgr, byte> coarseThumb, Mat? coarseGt,
            Image<Bgr, byte> fineThumb, Mat? fineGt,
            IReadOnlyList<ParamDef> allDimensions, IReadOnlyDictionary<string, double> startAll,
            AutoTuneGoal goal, double minColor, int maxEvals,
            Action<int, double>? progress, Func<bool>? cancelled, Image<Bgr, byte>? nativeProbe,
            Image<Bgr, byte> verificationInput, Mat? verificationGt)
        {
            ParamDef[] primary = allDimensions.Where(parameter => parameter.Search).ToArray();
            if (primary.Length == 0)
            {
                ParameterSearchResult only = PatternSearchResult(method, fineThumb, fineGt, allDimensions,
                    startAll, goal, minColor, maxEvals, Math.Clamp(maxEvals / 20, 6, 16), progress,
                    cancelled, nativeProbe);
                return FinalizeThoroughResult(method, verificationInput, verificationGt, goal, minColor,
                    startAll, only, only, maxEvals, cancelled);
            }

            int coarseBudget = Math.Min(60, Math.Max(1, maxEvals / 4));
            ParameterSearchResult coarse = PatternSearchResult(method, coarseThumb, coarseGt, primary, startAll,
                goal, minColor, coarseBudget, 4, progress, cancelled, nativeProbe);
            int offset = coarse.Diagnostics.UniqueEvaluations;
            int fineBudget = Math.Max(1, maxEvals - offset);
            Action<int, double>? fineProgress = progress == null
                ? null
                : (attempt, score) => progress(offset + attempt, Math.Max(coarse.Score, score));
            ParameterSearchResult fine = PatternSearchResult(method, fineThumb, fineGt, allDimensions,
                coarse.Parameters, goal, minColor, fineBudget, Math.Clamp(fineBudget / 20, 6, 16),
                fineProgress, cancelled, nativeProbe);

            return FinalizeThoroughResult(method, verificationInput, verificationGt, goal, minColor,
                startAll, coarse, fine, maxEvals, cancelled);
        }

        private static ParameterSearchResult FinalizeQuickResult(
            IDeHazeMethod method, Image<Bgr, byte> verificationInput, Mat? verificationGt,
            AutoTuneGoal goal, double minColor, IReadOnlyDictionary<string, double> startAll,
            ParameterSearchResult search, int maxEvals, Func<bool>? cancelled)
        {
            ParameterSearchResult result = FinalizeThoroughResult(method, verificationInput, verificationGt,
                goal, minColor, startAll, search, search, maxEvals, cancelled);
            string selected = result.Diagnostics.SelectedCandidate switch
            {
                "coarse" or "fine" => "quick",
                string value => value,
                _ => "cancelled",
            };
            return result with { Diagnostics = result.Diagnostics with { SelectedCandidate = selected } };
        }

        private static ParameterSearchResult FinalizeThoroughResult(
            IDeHazeMethod method, Image<Bgr, byte> verificationInput, Mat? verificationGt,
            AutoTuneGoal goal, double minColor, IReadOnlyDictionary<string, double> startAll,
            ParameterSearchResult coarse, ParameterSearchResult fine, int maxEvals, Func<bool>? cancelled)
        {
            Dictionary<string, double> selected = fine.Parameters;
            double selectedScore = fine.Score;
            string selectedCandidate = "fine";
            int verificationEvaluations = 0;
            int verificationFailures = 0;
            int safetyRejections = 0;

            // Отмена должна быть быстрой: полноразмерные контрольные прогоны намеренно пропускаются.
            if (!(cancelled?.Invoke() ?? false))
            {
                using var gtFull = ReferenceThumb(verificationGt, verificationInput.Size);
                var candidates = new (string Name, IReadOnlyDictionary<string, double> Parameters)[]
                {
                    ("start", startAll),
                    ("coarse", coarse.Parameters),
                    ("fine", fine.Parameters),
                };
                var seen = new HashSet<string>(StringComparer.Ordinal);
                Dictionary<string, double>? verifiedBest = null;
                double verifiedBestScore = double.NegativeInfinity;
                string? verifiedBestName = null;
                Metrics.Report? safetyBaseline = null;

                foreach ((string name, IReadOnlyDictionary<string, double> parameters) in candidates)
                {
                    string key = string.Join(";", method.Parameters.Select(parameter =>
                        BitConverter.DoubleToInt64Bits(parameters[parameter.Key]).ToString("X16")));
                    if (!seen.Add(key)) continue;

                    verificationEvaluations++;
                    try
                    {
                        using Mat result = method.Process(verificationInput, parameters);
                        if (goal != AutoTuneGoal.Reference)
                        {
                            Metrics.Report candidateMetrics = Metrics.Evaluate(result, null, verificationInput.Mat);
                            if (name == "start") safetyBaseline = candidateMetrics;
                            else if (safetyBaseline.HasValue &&
                                !PassesNoReferenceSafety(candidateMetrics, safetyBaseline.Value, goal))
                            {
                                safetyRejections++;
                                continue;
                            }
                        }
                        double score = Score(goal, result, verificationInput.Mat, gtFull, minColor);
                        if (!double.IsFinite(score))
                        {
                            verificationFailures++;
                            continue;
                        }

                        double tolerance = 1e-12 * Math.Max(1.0, Math.Abs(verifiedBestScore));
                        if (verifiedBest == null || score > verifiedBestScore + tolerance)
                        {
                            verifiedBest = new Dictionary<string, double>(parameters, StringComparer.Ordinal);
                            verifiedBestScore = score;
                            verifiedBestName = name;
                        }
                    }
                    catch
                    {
                        verificationFailures++;
                    }
                }

                if (verifiedBest == null)
                    throw new InvalidOperationException("Полноразмерная верификация не получила ни одного корректного кандидата.");
                selected = verifiedBest;
                selectedScore = verifiedBestScore;
                selectedCandidate = verifiedBestName!;
            }

            bool oneSearchStage = ReferenceEquals(coarse, fine);
            var diagnostics = new ParameterSearchDiagnostics(
                maxEvals,
                coarse.Diagnostics.EffectiveEvaluationLimit + (oneSearchStage ? 0 : fine.Diagnostics.EffectiveEvaluationLimit),
                coarse.Diagnostics.UniqueEvaluations + (oneSearchStage ? 0 : fine.Diagnostics.UniqueEvaluations),
                coarse.Diagnostics.CacheHits + (oneSearchStage ? 0 : fine.Diagnostics.CacheHits),
                coarse.Diagnostics.FailedEvaluations + (oneSearchStage ? 0 : fine.Diagnostics.FailedEvaluations),
                fine.Diagnostics.DistinctValues,
                verificationEvaluations,
                verificationFailures,
                selectedCandidate,
                safetyRejections);
            return new ParameterSearchResult(selected, selectedScore, diagnostics);
        }

        internal static bool PassesNoReferenceSafety(Metrics.Report candidate, Metrics.Report baseline,
            AutoTuneGoal goal)
        {
            bool vivid = goal == AutoTuneGoal.Vivid;
            double clipLimit = Math.Max(vivid ? 5.0 : 3.0, baseline.ClipPct + (vivid ? 2.0 : 1.0));
            double naturalnessLimit = Math.Max(vivid ? 60.0 : 50.0,
                baseline.NaturalnessDev + (vivid ? 12.0 : 8.0));
            double artifactLimit = Math.Max(vivid ? 35.0 : 30.0,
                baseline.ArtifactDev + (vivid ? 10.0 : 8.0));
            double minimumHazeRemoval = vivid ? 0.0 : Math.Max(0.0, baseline.HazeRemoved - 0.05);
            double minimumColor = vivid ? 0.75 : Math.Min(0.80, 0.65 * baseline.ColorRatio);
            double maximumColor = Math.Max(vivid ? 2.25 : 2.0, 1.35 * baseline.ColorRatio);
            return candidate.ClipPct <= clipLimit &&
                candidate.NaturalnessDev <= naturalnessLimit &&
                candidate.ArtifactDev <= artifactLimit &&
                candidate.HazeRemoved >= minimumHazeRemoval &&
                candidate.ColorRatio >= minimumColor && candidate.ColorRatio <= maximumColor;
        }

        /// <summary>
        /// Проба на НАТИВНОМ разрешении: маленький кроп оригинала в 100% масштабе из самой «плоской»
        /// (задымлённой) зоны. Нужна, потому что параметры-радиусы (патч, окно A, sigma фильтров),
        /// подобранные на превью, ведут себя иначе на полном кадре: даунскейл превью прячет пиксельные
        /// артефакты («кракле», зерно), и подбор выбирает то, что на полном размере выглядит ужасно.
        /// Каждый кандидат дополнительно прогоняется на этой пробе, и шум в плоских зонах штрафуется.
        /// null - если превью и так почти полноразмерное (артефакты видны в основной оценке).
        /// </summary>
        private static Image<Bgr, byte>? NativeProbe(Image<Bgr, byte> input, Image<Bgr, byte> thumb)
        {
            if (thumb.Width >= (int)(input.Width * 0.9)) return null;
            int cs = Math.Min(384, Math.Min(input.Width, input.Height));
            if (cs < 96) return null;

            // ищем самый гладкий блок (там кракле заметнее всего): карта локального σ на даунскейле
            using var gray = new Mat();
            CvInvoke.CvtColor(input.Mat, gray, ColorConversion.Bgr2Gray);
            double s = Math.Min(1.0, 400.0 / Math.Max(input.Width, input.Height));
            using var small = new Mat();
            CvInvoke.Resize(gray, small, new Size(Math.Max(8, (int)(input.Width * s)), Math.Max(8, (int)(input.Height * s))), 0, 0, Inter.Area);
            using var g32 = new Mat();
            small.ConvertTo(g32, DepthType.Cv32F, 1.0 / 255.0);
            var ks = new Size(21, 21); var anc = new Point(-1, -1);
            using var mean = new Mat(); CvInvoke.Blur(g32, mean, ks, anc);
            using var mean2 = new Mat();
            using (var sq = new Mat()) { CvInvoke.Multiply(g32, g32, sq); CvInvoke.Blur(sq, mean2, ks, anc); }
            using var var0 = new Mat();
            using (var m2 = new Mat()) { CvInvoke.Multiply(mean, mean, m2); CvInvoke.Subtract(mean2, m2, var0); }

            // усредняем σ² по будущему окну кропа и берём минимум (центр самой гладкой зоны)
            int win = Math.Max(5, (int)(cs * s) | 1);
            using var zone = new Mat();
            CvInvoke.Blur(var0, zone, new Size(win, win), anc);
            double mn = 0, mx = 0; Point pMin = default, pMax = default;
            CvInvoke.MinMaxLoc(zone, ref mn, ref mx, ref pMin, ref pMax);

            int cx = (int)(pMin.X / s), cy = (int)(pMin.Y / s);
            int x0 = Math.Clamp(cx - cs / 2, 0, input.Width - cs);
            int y0 = Math.Clamp(cy - cs / 2, 0, input.Height - cs);
            using var crop = new Mat(input.Mat, new Rectangle(x0, y0, cs, cs));
            return crop.ToImage<Bgr, byte>();
        }

        /// <summary>
        /// Изображенческий адаптер над тестируемым отдельно безградиентным ядром ParameterSearch.
        /// Возвращает полный набор аргументов с лучшей фактически измеренной оценкой.
        /// </summary>
        private static ParameterSearchResult PatternSearchResult(
            IDeHazeMethod m, Image<Bgr, byte> thumb, Mat? gtThumb,
            IReadOnlyList<ParamDef> searchParams, IReadOnlyDictionary<string, double> startAll,
            AutoTuneGoal goal, double minColor, int maxEvals, int seedCount,
            Action<int, double>? progress, Func<bool>? cancelled, Image<Bgr, byte>? nativeProbe = null)
        {
            var result = ParameterSearch.Optimize(
                searchParams,
                startAll,
                args =>
                {
                    using var res = m.Process(thumb, args);
                    double score = Score(goal, res, thumb.Mat, gtThumb, minColor);
                    if (nativeProbe != null && double.IsFinite(score))
                    {
                        // Ошибка нативной пробы больше не даёт кандидату несправедливое преимущество:
                        // исключение пометит всю точку как неуспешную в ParameterSearch.
                        using var probeResult = m.Process(nativeProbe, args);
                        score -= 0.7 * Math.Max(0.0, FlatZoneNoise(probeResult, nativeProbe.Mat) - 45.0);
                    }
                    return score;
                },
                maxEvals,
                seedCount,
                progress,
                cancelled);
            return result;
        }

        /// <summary>
        /// 'Авто-лучший': быстро сканирует все методы на превью (дефолтные параметры), выбирает лучший по
        /// без-эталонной оценке и тщательно настраивает его параметры. Возвращает (метод, параметры, скор).
        /// </summary>
        public static (IDeHazeMethod method, Dictionary<string, double> prms, double score) PickBest(
            IReadOnlyList<IDeHazeMethod> methods, Image<Bgr, byte> input,
            Action<string>? progress = null, Func<bool>? cancelled = null,
            Mat? gt = null, AutoTuneGoal goal = AutoTuneGoal.ObjectVisibility)
        {
            ArgumentNullException.ThrowIfNull(methods);
            if (methods.Count == 0) throw new ArgumentException("Список методов для автоподбора пуст.", nameof(methods));

            IDeHazeMethod? best = null;
            double bestScore = double.NegativeInfinity;
            using (var thumb = Thumbnail(input, 300))   // скан - мелкое превью, нужен лишь относительный рейтинг
            using (var gtThumb = ReferenceThumb(gt, thumb.Size))
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
                        double s = Score(goal, res, thumb.Mat, gtThumb, 0.0);
                        if (double.IsFinite(s) && s > bestScore) { bestScore = s; best = m; }
                    }
                    catch { /* пропускаем падающие методы */ }
                    string bestName = best?.Name ?? "пока нет успешных";
                    progress?.Invoke($"Авто: скан {i}/{methods.Count} - лучший '{bestName}' ({bestScore:F0})");
                }
            }
            if (best == null)
                throw new InvalidOperationException("Ни один метод не удалось корректно оценить на превью.");

            var start = best.Parameters.ToDictionary(p => p.Key, p => p.Default);
            var tuned = OptimizeThorough(best, input, start, 0,
                (e, sc) => progress?.Invoke($"Авто: настройка '{best.Name}'... попытка {e}, скор {sc:F0}"), cancelled,
                gt: gt, goal: goal);

            // Возвращаем оценку настроенной точки, а не устаревший score дефолта из первого скана.
            using var tunedThumb = Thumbnail(input, 300);
            using var tunedGt = ReferenceThumb(gt, tunedThumb.Size);
            using var tunedResult = best.Process(tunedThumb, tuned);
            double tunedScore = Score(goal, tunedResult, tunedThumb.Mat, tunedGt, 0.0);
            return (best, tuned, tunedScore);
        }

        internal static double Score(AutoTuneGoal goal, Mat resultFloat01, Mat input8, Mat? gt8, double minColor)
        {
            Metrics.Report rep;
            double sc;
            if (goal == AutoTuneGoal.Reference && gt8 != null)
            {
                rep = Metrics.Evaluate(resultFloat01, gt8, input8);
                sc = ReferenceScore(rep);
                Metrics.ChromaticFidelityReport chroma = Metrics.ChromaticFidelity(resultFloat01, gt8);
                if (chroma.ChromaticPixels >= 32 && chroma.ChromaticPixelFraction >= 0.0005)
                    sc -= 0.15 * (100.0 - chroma.Score);
                // При наличии GT не используем NaturalnessDev: эта безэталонная эвристика может
                // награждать обесцвечивание, даже когда SSIM и цвет относительно GT стали хуже.
                // Оставляем только прямые заслоны от высокочастотного шума и явных артефактов.
                sc -= 0.45 * Math.Max(0.0, FlatZoneNoise(resultFloat01, input8) - 45.0)
                    + 0.4 * Math.Max(0.0, rep.ArtifactDev - 32.0);
            }
            else
            {
                rep = Metrics.Evaluate(resultFloat01, null, input8);
                sc = goal switch
                {
                    AutoTuneGoal.Vivid => VividScore(resultFloat01, input8, rep),
                    AutoTuneGoal.ObjectVisibility => ObjectVisibilityScore(resultFloat01, input8, rep),
                    _ => ObjectVisibilityScore(resultFloat01, input8, rep),
                };
            }

            if (minColor > 0 && rep.ColorRatio < minColor)
                sc -= 200.0 * (minColor - rep.ColorRatio);
            return sc;
        }

        internal static double ReferenceScore(Metrics.Report r)
        {
            // Только RAW-метрики являются честной целевой функцией: aligned-варианты используют GT
            // для аффинной правки самого результата и потому остаются исключительно диагностикой.
            double psnr = double.IsPositiveInfinity(r.Psnr) ? 100.0 : (double.IsFinite(r.Psnr) ? r.Psnr : 0.0);
            double ssim = double.IsFinite(r.Ssim) ? r.Ssim : 0.0;
            double de = double.IsFinite(r.Ciede2000) ? r.Ciede2000 : 35.0;

            double psnrTerm = 100.0 * Math.Clamp((psnr - 8.0) / 18.0, 0.0, 1.0);
            double ssimTerm = 100.0 * Math.Clamp(ssim, 0.0, 1.0);
            double colorTerm = 100.0 * Math.Clamp(1.0 - de / 35.0, 0.0, 1.0);
            double artifactPenalty = 1.6 * r.ClipPct + 18.0 * Math.Clamp(r.ColorRatio - 1.45, 0.0, 2.0);
            // MSE не добавляем отдельным слагаемым: при фиксированном диапазоне это та же ошибка,
            // что PSNR, только в монотонно преобразованной шкале. Двойной учёт раньше давал 62%
            // веса пиксельной ошибке и позволял серому/размытому экстремуму победить структуру и цвет.
            return 0.35 * psnrTerm + 0.40 * ssimTerm + 0.25 * colorTerm - artifactPenalty;
        }

        private static double ObjectVisibilityScore(Mat resultFloat01, Mat input8, Metrics.Report r)
        {
            using var r8full = new Mat();
            resultFloat01.ConvertTo(r8full, DepthType.Cv8U, 255.0);
            using var r8 = Down(r8full, 560);
            using var inp = ResizeTo(input8, r8.Size);
            using var grR = Gray(r8);
            using var grI = Gray(inp);

            var local = new List<double>();
            const int gridX = 5, gridY = 4;
            for (int gy = 0; gy < gridY; gy++)
            {
                int y0 = gy * r8.Rows / gridY;
                int y1 = (gy + 1) * r8.Rows / gridY;
                for (int gx = 0; gx < gridX; gx++)
                {
                    int x0 = gx * r8.Cols / gridX;
                    int x1 = (gx + 1) * r8.Cols / gridX;
                    var roi = new Rectangle(x0, y0, Math.Max(1, x1 - x0), Math.Max(1, y1 - y0));
                    using var ri = new Mat(grI, roi);
                    using var rr = new Mat(grR, roi);
                    double cGain = Std(rr) / (Std(ri) + 1e-6);
                    double eGain = MeanGrad(rr) / (MeanGrad(ri) + 1e-6);
                    double lGain = MeanAbsLap(rr) / (MeanAbsLap(ri) + 1e-6);
                    double tile =
                        38.0 * Math.Clamp((cGain - 0.92) / 1.65, 0.0, 1.0) +
                        42.0 * Math.Clamp((eGain - 0.95) / 2.80, 0.0, 1.0) +
                        20.0 * Math.Clamp((lGain - 0.95) / 3.20, 0.0, 1.0);
                    local.Add(tile);
                }
            }

            local.Sort();
            int weakN = Math.Max(1, local.Count / 3);
            double weak = local.Take(weakN).Average();
            double mean = local.Average();
            double haze = 100.0 * Math.Sqrt(Math.Clamp(r.HazeRemoved, 0.0, 1.0));
            double global = 0.48 * weak + 0.28 * mean + 0.24 * haze;
            double clipPenalty = 1.9 * Math.Clamp(r.ClipPct - 3.0, 0.0, 30.0);
            double colorPenalty = 16.0 * Math.Clamp(r.ColorRatio - 1.75, 0.0, 2.0);
            // штраф за усиленный шум: высокочастотная энергия результата там, где ВХОД локально плоский
            // (густая дымка) - это не объекты, а раздутый JPEG/сенсорный шум. Без него подбор крутит
            // omega/t_min/refine в крайности и награждает 'грани' от шума (Грани x20+).
            double noisePenalty = 0.55 * Math.Max(0.0, FlatZoneNoise(resultFloat01, input8) - 45.0);
            // Анти-Goodhart заслон по непосредственно наблюдаемым экстремумам. NaturalnessDev имеет
            // разный baseline у разных методов, поэтому внутри обычного диапазона не оптимизируется;
            // в helper она используется лишь как аварийный ограничитель явного экстремума > 50.
            double harshPenalty = NoReferenceHarshPenalty(r, vivid: false);
            double localColorPenalty = LocalChromaExpansionPenalty(
                Metrics.LocalChromaExpansion(resultFloat01, input8), vivid: false);
            return global - clipPenalty - colorPenalty - noisePenalty - harshPenalty - localColorPenalty;
        }

        private static double VividScore(Mat resultFloat01, Mat input8, Metrics.Report r)
        {
            double haze = 100.0 * Math.Sqrt(Math.Clamp(r.HazeRemoved, 0.0, 1.0));
            double contrast = 100.0 * Bell(r.ContrastGain, 1.75, 1.05);
            double edge = 100.0 * Bell(r.EdgeGain, 2.25, 1.45);
            double color = 100.0 * Bell(r.ColorRatio, 1.55, 0.75);
            double clipPenalty = 2.4 * Math.Clamp(r.ClipPct - 2.5, 0.0, 30.0);
            double grayPenalty = 30.0 * Math.Clamp(1.05 - r.ColorRatio, 0.0, 1.0);
            double noisePenalty = 0.40 * Math.Max(0.0, FlatZoneNoise(resultFloat01, input8) - 50.0);
            // Сочность != выжженный контраст; пороги чуть свободнее, чем у цели «объекты».
            double harshPenalty = NoReferenceHarshPenalty(r, vivid: true);
            double localColorPenalty = LocalChromaExpansionPenalty(
                Metrics.LocalChromaExpansion(resultFloat01, input8), vivid: true);
            return 0.24 * haze + 0.27 * contrast + 0.24 * edge + 0.25 * color - clipPenalty - grayPenalty - noisePenalty - harshPenalty - localColorPenalty;
        }

        internal static double LocalChromaExpansionPenalty(Metrics.LocalChromaExpansionReport expansion, bool vivid)
        {
            double scale = vivid ? 0.55 : 1.0;
            return scale * (0.40 * expansion.P95Excess +
                2.0 * expansion.MeanExcess +
                20.0 * Math.Max(0.0, expansion.ExplodedPixelFraction - 0.02));
        }

        internal static double NoReferenceHarshPenalty(Metrics.Report r, bool vivid)
            => vivid
                ? 10.0 * Math.Max(0.0, r.ContrastGain - 4.5) +
                  1.0 * Math.Max(0.0, r.EdgeGain - 14.0) +
                  0.4 * Math.Max(0.0, r.ArtifactDev - 32.0) +
                  1.5 * Math.Max(0.0, r.NaturalnessDev - 55.0)
                : 12.0 * Math.Max(0.0, r.ContrastGain - 4.0) +
                  1.2 * Math.Max(0.0, r.EdgeGain - 12.0) +
                  0.5 * Math.Max(0.0, r.ArtifactDev - 30.0) +
                  2.0 * Math.Max(0.0, r.NaturalnessDev - 50.0);

        /// <summary>
        /// Прокси шума: средняя |Laplacian| результата, взвешенная 'плоскостью входа' (локальный σ входа мал
        /// = густая дымка, восстановимой высокой частоты нет). Высокое значение = усиленный шум, а не детали.
        /// Считается на 560px (как и плиточный анализ). Чем чище результат, тем ниже (GT ~70, шумный >150).
        /// </summary>
        private static double FlatZoneNoise(Mat resultFloat01, Mat input8)
        {
            using var r8full = new Mat();
            resultFloat01.ConvertTo(r8full, DepthType.Cv8U, 255.0);
            using var r8 = Down(r8full, 560);
            using var inp = ResizeTo(input8, r8.Size);
            using var grR = Gray(r8);
            using var grI = Gray(inp);

            using var gi = new Mat(); grI.ConvertTo(gi, DepthType.Cv32F);
            var ks = new Size(11, 11); var anc = new Point(-1, -1);
            using var mean = new Mat(); CvInvoke.Blur(gi, mean, ks, anc);
            using var mean2 = new Mat();
            using (var sq = new Mat()) { CvInvoke.Multiply(gi, gi, sq); CvInvoke.Blur(sq, mean2, ks, anc); }
            using var var0 = new Mat();
            using (var m2 = new Mat()) { CvInvoke.Multiply(mean, mean, m2); CvInvoke.Subtract(mean2, m2, var0); }
            using (var z = new Mat(var0.Size, DepthType.Cv32F, 1)) { z.SetTo(new MCvScalar(0)); CvInvoke.Max(var0, z, var0); }
            using var istd = new Mat(); CvInvoke.Sqrt(var0, istd);
            using var flat = new Mat();
            istd.ConvertTo(flat, DepthType.Cv32F, -1.0 / 8.0, 1.0);   // (8 - σ_local)/8, σ в 8-битных единицах
            DehazeCore.Clamp01(flat);

            using var gr = new Mat(); grR.ConvertTo(gr, DepthType.Cv32F);
            using var lap = new Mat(); CvInvoke.Laplacian(gr, lap, DepthType.Cv32F, 3);
            using var alap = new Mat();
            using (var zz = new Mat(lap.Size, DepthType.Cv32F, 1)) { zz.SetTo(new MCvScalar(0)); CvInvoke.AbsDiff(lap, zz, alap); }
            using var num = new Mat(); CvInvoke.Multiply(alap, flat, num);
            double s = CvInvoke.Sum(num).V0;
            double w = CvInvoke.Sum(flat).V0;
            return s / (w + 1e-6);
        }

        private static double Bell(double x, double center, double width)
        {
            double d = Math.Abs(x - center) / Math.Max(1e-6, width);
            return Math.Clamp(1.0 - d * d, 0.0, 1.0);
        }

        private static Image<Bgr, byte> Thumbnail(Image<Bgr, byte> img, int maxDim)
        {
            int w = img.Width, h = img.Height;
            double s = Math.Min(1.0, (double)maxDim / Math.Max(w, h));
            return s >= 1.0 ? img.Clone() : img.Resize((int)(w * s), (int)(h * s), Inter.Area);
        }

        private static Mat? ReferenceThumb(Mat? gt, Size size)
        {
            if (gt == null) return null;
            var o = new Mat();
            if (gt.Size.Equals(size)) gt.CopyTo(o);
            else CvInvoke.Resize(gt, o, size, 0, 0, Inter.Area);
            return o;
        }

        private static Mat Down(Mat m, int maxDim)
        {
            int w = m.Cols, h = m.Rows;
            double s = Math.Min(1.0, (double)maxDim / Math.Max(w, h));
            var o = new Mat();
            if (s >= 1.0) m.CopyTo(o);
            else CvInvoke.Resize(m, o, new Size(Math.Max(1, (int)(w * s)), Math.Max(1, (int)(h * s))), 0, 0, Inter.Area);
            return o;
        }

        private static Mat ResizeTo(Mat m, Size sz)
        {
            var o = new Mat();
            if (m.Size.Equals(sz)) m.CopyTo(o);
            else CvInvoke.Resize(m, o, sz, 0, 0, Inter.Area);
            return o;
        }

        private static Mat Gray(Mat bgr8)
        {
            var g = new Mat();
            CvInvoke.CvtColor(bgr8, g, ColorConversion.Bgr2Gray);
            return g;
        }

        private static double Std(Mat gray8)
        {
            MCvScalar mean = default, std = default;
            CvInvoke.MeanStdDev(gray8, ref mean, ref std);
            return std.V0;
        }

        private static double MeanGrad(Mat gray8)
        {
            using var gx = new Mat();
            using var gy = new Mat();
            CvInvoke.Sobel(gray8, gx, DepthType.Cv32F, 1, 0, 3);
            CvInvoke.Sobel(gray8, gy, DepthType.Cv32F, 0, 1, 3);
            using var mag = new Mat();
            using (var gx2 = new Mat())
            using (var gy2 = new Mat())
            {
                CvInvoke.Multiply(gx, gx, gx2);
                CvInvoke.Multiply(gy, gy, gy2);
                CvInvoke.Add(gx2, gy2, mag);
            }
            CvInvoke.Sqrt(mag, mag);
            return CvInvoke.Mean(mag).V0;
        }

        private static double MeanAbsLap(Mat gray8)
        {
            using var lap = new Mat();
            CvInvoke.Laplacian(gray8, lap, DepthType.Cv32F, 3);
            using var zero = new Mat(lap.Size, DepthType.Cv32F, 1);
            zero.SetTo(new MCvScalar(0));
            using var abs = new Mat();
            CvInvoke.AbsDiff(lap, zero, abs);
            return CvInvoke.Mean(abs).V0;
        }

    }
}
