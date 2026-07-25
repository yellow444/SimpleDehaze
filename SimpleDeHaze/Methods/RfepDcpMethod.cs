using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Boundary-Constrained Prior Fusion (в прежних документах - RFEP-DCP).
    ///
    /// ВАЖНО о новизне: сама нижняя граница на пропускание НЕ является вкладом этого проекта.
    /// Она известна как boundary constraint из работы Meng et al., "Efficient Image Dehazing with
    /// Boundary Constraint and Contextual Regularization", ICCV 2013. Здесь она используется как
    /// готовый результат. Отличия метода - в композиции: смешивание DCP и robust-HSV приоров по
    /// карте доверия, проекция до и после edge-aware уточнения, и (это уже собственный вывод)
    /// граница, согласованная с ФАКТИЧЕСКОЙ chroma-safe формулой восстановления -
    /// см. <see cref="DehazeCore.ChromaSafeLowerBound"/>.
    /// </summary>
    public sealed class RfepDcpMethod : IDeHazeMethod
    {
        public string Name => "Boundary-Constrained Prior Fusion (быв. RFEP)";

        public string Description =>
            "Смешивание двух приоров + проекция на допустимую область пропускания.\n\n" +
            "Граница НЕ новая: это boundary constraint из Meng et al., ICCV 2013 -\n" +
            "    t >= max_c max( (I_c-A_c)/(1-A_c), (A_c-I_c)/A_c ),\n" +
            "она следует из требования 0 <= J_c <= 1 для СТАНДАРТНОЙ инверсии.\n\n" +
            "Что здесь своё:\n" +
            "1. карта t собирается из DCP и robust-HSV приора по карте доверия (sky/bright aware);\n" +
            "2. проекция применяется до и после fast guided filter, чтобы сглаживание не вернуло\n" +
            "   недопустимые значения;\n" +
            "3. режим «граница» = chroma-safe: выведена отдельная кусочная область допустимости именно\n" +
            "   для формулы J_c = A_c + d̄/max(t,t_min) + δ_c/max(t,chromaFloor), которую и использует\n" +
            "   восстановление. Классический t_box для неё избыточно консервативен.\n\n" +
            "Честно о гарантиях: строгая допустимость выполняется ТОЛЬКО при strict=1\n" +
            "(жёсткая проекция t=max(t,bound), без ослабления bscale/bmax). При strict=0 метод\n" +
            "намеренно ослабляет границу ради более сильного дехейзинга - гарантии тогда нет,\n" +
            "а доля нарушений измеряется командой --mathtest.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",  "omega - доля удаляемой дымки",       0.3,  0.98, 0.95, search: true),
            new ParamDef("patch",  "Патч тёмного канала",                1,    15,   5,    1, isInt: true),
            new ParamDef("alpha",  "alpha - robust HSV prior",           0.2,  2.0,  1.0,  search: true),
            new ParamDef("tauV",   "tau_v - bright threshold",           0.45, 0.85, 0.62),
            new ParamDef("tauS",   "tau_s - saturation threshold",       0.08, 0.40, 0.22),
            new ParamDef("rho",    "rho - сила проекции",                0.0,  1.0,  0.80, search: true),
            new ParamDef("bscale", "Ослабление границы (1 = строгая)",   0.4,  1.2,  0.85),
            new ParamDef("bmax",   "Потолок границы",                    0.60, 1.0,  0.92),
            new ParamDef("strict", "Строгая допустимость (0/1)",         0,    1,    0,    1, isInt: true, tunable: false),
            new ParamDef("csbound","Граница под chroma-safe (0/1)",      0,    1,    1,    1, isInt: true, tunable: false),
            new ParamDef("linear", "Линейный радианс (0/1)",             0,    1,    0,    1, isInt: true, tunable: false),
            new ParamDef("tsky",   "t_sky - мягкость неба/белого",       0.45, 0.9,  0.68),
            new ParamDef("min",    "t_min - нижний порог t",             0.01, 0.5,  0.08),
            new ParamDef("chroma", "chromaFloor - защита цвета",         0.08, 0.7,  0.35),
            new ParamDef("refine", "Радиус fast Guided Filter",          5,    120,  48,   1, isInt: true),
            new ParamDef("eps",    "eps - регуляризация GF",             1e-5, 1e-2, 1e-3, log: true),
            new ParamDef("fast",   "Ускорение GF (1 = full)",            1,    8,    4,    1, isInt: true, tunable: false),
            new ParamDef("tone",   "Восстановление тона",                0.0,  1.0,  0.15),
            new ParamDef("color",  "Потолок усиления цветности",          1.0,  1.6,  1.25),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"], alpha = p["alpha"], tauV = p["tauV"], tauS = p["tauS"];
            double rho = p["rho"], bscale = p["bscale"], bmax = p["bmax"], tSky = p["tsky"];
            double tmin = p["min"], chromaFloor = p["chroma"], eps = p["eps"], tone = p["tone"], color = p["color"];
            int patch = (int)p["patch"], refine = (int)p["refine"], fast = (int)p["fast"];
            bool strict = p.TryGetValue("strict", out var st) && st >= 0.5;
            bool csBound = !p.TryGetValue("csbound", out var cb) || cb >= 0.5;
            bool linear = p.TryGetValue("linear", out var ln) && ln >= 0.5;

            // строгий режим: граница берётся как есть и проекция жёсткая - только так утверждение
            // «результат не выходит за RGB-куб до клиппинга» действительно выполняется
            if (strict) { bscale = 1.0; bmax = 1.0; rho = 1.0; }

            using var I = ColorSpace.Normalize(input, linear);
            using var darkA = DehazeCore.DarkChannel(I, patch);
            var A = DehazeCore.Atmospheric(I, darkA, 0.001);

            using var norm = DehazeCore.NormByA(I, A);
            using var darkNorm = DehazeCore.DarkChannel(norm, patch);
            using var tD = new Mat();
            darkNorm.ConvertTo(tD, DepthType.Cv32F, -omega, 1.0);
            DehazeCore.Clamp01(tD);

            using var sky = DehazeCore.SkyMask(I);
            using var tH = RobustHsvTransmission(I, sky, alpha, patch, tSky);
            using var wD = DcpConfidence(I, sky, alpha, tauV, tauS);
            using var tMix = Fuse(tD, tH, wD);
            DehazeCore.RaiseInSky(tMix, sky, tSky);
            DehazeCore.Clamp01(tMix);

            // граница: либо классический boundary constraint (для стандартной инверсии),
            // либо согласованная с фактическим chroma-safe восстановлением
            using var tBox = LowerBound(I, A, csBound, tmin, chromaFloor, bscale, bmax);
            using var tProj = ProjectToEnvelope(tMix, tBox, rho);
            using var tRef = Refiners.FastGuided(I, tProj, refine, eps, fast);
            DehazeCore.Clamp01(tRef);
            using var tFinal = ProjectToEnvelope(tRef, tBox, strict ? 1.0 : Math.Min(1.0, rho + 0.20));

            using var recovered = DehazeCore.Recover(I, tFinal, A, tmin, chromaFloor);
            using var srgb = ColorSpace.Encode(recovered, linear);
            using var toned = DehazeCore.RestoreTone(srgb, tone);
            return DehazeCore.LimitColorfulness(toned, input.Mat, color);
        }

        /// <summary>
        /// Диагностика для --mathtest: доля пикселей, где итоговая карта t лежит ниже допустимой
        /// границы (то есть восстановление гарантированно упирается в клиппинг).
        /// Возвращает (violation до проекции, violation после проекции).
        /// </summary>
        public static (double before, double after) MeasureViolation(Image<Bgr, byte> input,
            IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"], alpha = p["alpha"], tauV = p["tauV"], tauS = p["tauS"];
            double rho = p["rho"], bscale = p["bscale"], bmax = p["bmax"], tSky = p["tsky"];
            double tmin = p["min"], chromaFloor = p["chroma"], eps = p["eps"];
            int patch = (int)p["patch"], refine = (int)p["refine"], fast = (int)p["fast"];
            bool strict = p.TryGetValue("strict", out var st) && st >= 0.5;
            bool csBound = !p.TryGetValue("csbound", out var cb) || cb >= 0.5;
            bool linear = p.TryGetValue("linear", out var ln) && ln >= 0.5;
            if (strict) { bscale = 1.0; bmax = 1.0; rho = 1.0; }

            using var I = ColorSpace.Normalize(input, linear);
            using var darkA = DehazeCore.DarkChannel(I, patch);
            var A = DehazeCore.Atmospheric(I, darkA, 0.001);
            using var norm = DehazeCore.NormByA(I, A);
            using var darkNorm = DehazeCore.DarkChannel(norm, patch);
            using var tD = new Mat();
            darkNorm.ConvertTo(tD, DepthType.Cv32F, -omega, 1.0);
            DehazeCore.Clamp01(tD);

            using var sky = DehazeCore.SkyMask(I);
            using var tH = RobustHsvTransmission(I, sky, alpha, patch, tSky);
            using var wD = DcpConfidence(I, sky, alpha, tauV, tauS);
            using var tMix = Fuse(tD, tH, wD);
            DehazeCore.RaiseInSky(tMix, sky, tSky);
            DehazeCore.Clamp01(tMix);

            // нарушения считаем относительно СТРОГОЙ границы, без ослабления bscale/bmax
            using var strictBound = csBound
                ? DehazeCore.ChromaSafeLowerBound(I, A, tmin, chromaFloor)
                : DehazeCore.BoundaryConstraint(I, A);

            using var tBox = LowerBound(I, A, csBound, tmin, chromaFloor, bscale, bmax);
            using var tProj = ProjectToEnvelope(tMix, tBox, rho);
            using var tRefined = Refiners.FastGuided(I, tProj, refine, eps, fast);
            DehazeCore.Clamp01(tRefined);
            using var tFinal = ProjectToEnvelope(tRefined, tBox, strict ? 1.0 : Math.Min(1.0, rho + 0.20));

            return (DehazeCore.ViolationRate(tMix, strictBound), DehazeCore.ViolationRate(tFinal, strictBound));
        }

        private static Mat LowerBound(Mat i01, MCvScalar a, bool chromaSafe, double tmin, double chromaFloor,
            double scale, double maxBound)
        {
            var bound = chromaSafe
                ? DehazeCore.ChromaSafeLowerBound(i01, a, tmin, chromaFloor)
                : DehazeCore.BoundaryConstraint(i01, a);

            if (Math.Abs(scale - 1.0) > 1e-6)
                CvInvoke.Multiply(bound, new ScalarArray(scale), bound);      // намеренное ослабление
            if (maxBound < 1.0)
                using (var cap = new Mat(bound.Size, DepthType.Cv32F, 1))
                {
                    cap.SetTo(new MCvScalar(maxBound));
                    CvInvoke.Min(bound, cap, bound);
                }
            DehazeCore.Clamp01(bound);
            return bound;
        }

        private static Mat Fuse(Mat tD, Mat tH, Mat wD)
        {
            var t = new Mat();
            using var diff = new Mat();
            CvInvoke.Subtract(tD, tH, diff);
            CvInvoke.Multiply(wD, diff, diff);
            CvInvoke.Add(tH, diff, t);
            return t;
        }

        private static Mat ProjectToEnvelope(Mat t, Mat lower, double rho)
        {
            var projected = t.Clone();
            using var deficit = new Mat();
            CvInvoke.Subtract(lower, t, deficit);
            using var zero = new Mat(deficit.Size, DepthType.Cv32F, 1);
            zero.SetTo(new MCvScalar(0));
            CvInvoke.Max(deficit, zero, deficit);
            CvInvoke.AddWeighted(projected, 1.0, deficit, Math.Clamp(rho, 0.0, 1.0), 0.0, projected);
            DehazeCore.Clamp01(projected);
            return projected;
        }

        private static Mat RobustHsvTransmission(Mat i01, Mat sky, double alpha, int patch, double tSky)
        {
            using var hsv = new Mat();
            CvInvoke.CvtColor(i01, hsv, ColorConversion.Bgr2Hsv);
            var hc = hsv.Split();
            using var S = hc[1];
            using var V = hc[2];
            hc[0].Dispose();

            using var q = new Mat();
            CvInvoke.Subtract(V, S, q);
            var (mu, sigma) = RobustStats(q);

            using var z = new Mat();
            q.ConvertTo(z, DepthType.Cv32F, 1.0 / (sigma + 1e-6), -mu / (sigma + 1e-6));
            using (var zero = new Mat(z.Size, DepthType.Cv32F, 1))
            {
                zero.SetTo(new MCvScalar(0));
                CvInvoke.Max(z, zero, z);
            }
            using (var zmax = new Mat(z.Size, DepthType.Cv32F, 1))
            {
                zmax.SetTo(new MCvScalar(3.5));
                CvInvoke.Min(z, zmax, z);
            }

            using var elem = CvInvoke.GetStructuringElement(
                ElementShape.Rectangle,
                new System.Drawing.Size(2 * Math.Max(1, patch) + 1, 2 * Math.Max(1, patch) + 1),
                new System.Drawing.Point(-1, -1));
            CvInvoke.Erode(z, z, elem, new System.Drawing.Point(-1, -1), 1, BorderType.Reflect101, default);

            using var tH = new Mat();
            z.ConvertTo(tH, DepthType.Cv32F, -alpha);
            CvInvoke.Exp(tH, tH);
            DehazeCore.Clamp01(tH);

            using var flat = DehazeCore.Flatness(i01, 8.0);
            using var edge = new Mat();
            flat.ConvertTo(edge, DepthType.Cv32F, -1.0, 1.0);
            using var safe = new Mat();
            S.ConvertTo(safe, DepthType.Cv32F, 0.20, tSky);
            CvInvoke.AddWeighted(safe, 1.0, edge, 0.10, 0.0, safe);
            DehazeCore.Clamp01(safe);

            using var safeH = new Mat();
            CvInvoke.Max(tH, safe, safeH);
            var t = new Mat();
            using var diff = new Mat();
            CvInvoke.Subtract(safeH, tH, diff);
            CvInvoke.Multiply(sky, diff, diff);
            CvInvoke.Add(tH, diff, t);
            DehazeCore.Clamp01(t);
            return t;
        }

        private static Mat DcpConfidence(Mat i01, Mat sky, double alpha, double tauV, double tauS)
        {
            using var hsv = new Mat();
            CvInvoke.CvtColor(i01, hsv, ColorConversion.Bgr2Hsv);
            var hc = hsv.Split();
            hc[0].Dispose();
            using var S = hc[1];
            using var V = hc[2];

            using var vExcess = PositivePart(V, tauV);
            using var sDeficit = new Mat();
            S.ConvertTo(sDeficit, DepthType.Cv32F, -1.0, tauS);
            DehazeCore.Clamp01(sDeficit);

            double lambda = 3.0 * alpha;
            using var vPenalty = new Mat();
            vExcess.ConvertTo(vPenalty, DepthType.Cv32F, -lambda);
            CvInvoke.Exp(vPenalty, vPenalty);
            using var sPenalty = new Mat();
            sDeficit.ConvertTo(sPenalty, DepthType.Cv32F, -lambda);
            CvInvoke.Exp(sPenalty, sPenalty);
            using var notSky = new Mat();
            sky.ConvertTo(notSky, DepthType.Cv32F, -1.0, 1.0);

            var wD = new Mat();
            CvInvoke.Multiply(notSky, vPenalty, wD);
            CvInvoke.Multiply(wD, sPenalty, wD);
            DehazeCore.Clamp01(wD);
            return wD;
        }

        private static Mat PositivePart(Mat m, double threshold)
        {
            var res = new Mat();
            m.ConvertTo(res, DepthType.Cv32F, 1.0, -threshold);
            using var zero = new Mat(res.Size, DepthType.Cv32F, 1);
            zero.SetTo(new MCvScalar(0));
            CvInvoke.Max(res, zero, res);
            return res;
        }

        private static (double mu, double sigma) RobustStats(Mat q)
        {
            int n = q.Rows * q.Cols;
            var data = new float[n];
            q.CopyTo(data);
            Array.Sort(data);
            double q25 = Percentile(data, 0.25);
            double q50 = Percentile(data, 0.50);
            double q75 = Percentile(data, 0.75);
            double sigma = Math.Max(0.03, (q75 - q25) / 1.349);
            return (q50, sigma);
        }

        private static double Percentile(float[] sorted, double p)
        {
            if (sorted.Length == 0) return 0;
            double pos = Math.Clamp(p, 0, 1) * (sorted.Length - 1);
            int lo = (int)Math.Floor(pos);
            int hi = Math.Min(sorted.Length - 1, lo + 1);
            double t = pos - lo;
            return sorted[lo] * (1.0 - t) + sorted[hi] * t;
        }
    }
}
