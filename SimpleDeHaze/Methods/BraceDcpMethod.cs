using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// BRACE-DCP: confidence fusion of DCP and robust HSV bright-region prior.
    /// </summary>
    public sealed class BraceDcpMethod : IDeHazeMethod
    {
        public string Name => "BRACE-DCP (bright-region aware)";

        public string Description =>
            "BRACE-DCP: Bright-Region Adaptive Confidence Enhanced DCP.\n\n" +
            "Идея: DCP силён на цветных/текстурных областях, но часто ошибается на небе, снегу,\n" +
            "белых стенах и пересветах. Поэтому считаем две карты t: классическую t_D от dark channel\n" +
            "и robust HSV-карту t_H по квантилизованному (V-S), затем смешиваем их по доверию к DCP.\n\n" +
            "w_D высоко на обычных объектах и низко в ярких малонасыщенных гладких зонах.\n" +
            "В bright/sky-зонах HSV-оценка получает мягкий floor t_sky, чтобы не выжигать цвет и небо.\n" +
            "После смешивания карта t уточняется fast guided filter, а восстановление использует\n" +
            "chroma-safe Recover: яркость очищается сильнее, хрома усиливается мягче.\n\n" +
            "Формула: t_D = 1 − ω·min_c min_Ω(I_c/A_c); t_H = clamp(α·q(V−S)) с floor t_sky в ярких зонах;\n" +
            "t = w_D·t_D + (1−w_D)·t_H (w_D — доверие к DCP, низкое на небе/пересвете), уточн. guided filter;\n" +
            "J = (I − A)/max(t, t_min) + A (chroma-safe).\n\n" +
            "Параметры: omega/patch - DCP; alpha - сила HSV-prior; tauV/tauS - bright/low-sat thresholds; fast - ускорение GF.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",  "omega - доля удаляемой дымки",       0.3,  0.98, 0.95, search: true),
            new ParamDef("patch",  "Патч тёмного канала",                1,    15,   5,    1, isInt: true),
            new ParamDef("alpha",  "alpha - сила robust HSV prior",      0.2,  2.0,  1.1,  search: true),
            new ParamDef("tauV",   "tau_v - порог яркости",              0.45, 0.85, 0.62),
            new ParamDef("tauS",   "tau_s - порог насыщенности",         0.08, 0.40, 0.22),
            new ParamDef("tsky",   "t_sky - мягкость неба/белого",       0.45, 0.9,  0.68, search: true),
            new ParamDef("min",    "t_min - нижний порог t",             0.01, 0.5,  0.08),
            new ParamDef("chroma", "chromaFloor - защита цвета",         0.08, 0.7,  0.35),
            new ParamDef("refine", "Радиус fast Guided Filter",          5,    120,  48,   1, isInt: true),
            new ParamDef("eps",    "eps - регуляризация GF",             1e-5, 1e-2, 1e-3, log: true),
            new ParamDef("fast",   "Ускорение GF (1 = full)",            1,    8,    4,    1, isInt: true, tunable: false),
            new ParamDef("tone",   "Восстановление тона",                0.0,  1.0,  0.2),
            new ParamDef("color",  "Потолок усиления цветности",          1.0,  1.6,  1.25),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"], alpha = p["alpha"], tauV = p["tauV"], tauS = p["tauS"], tSky = p["tsky"];
            double tmin = p["min"], chromaFloor = p["chroma"], eps = p["eps"], tone = p["tone"];
            int patch = (int)p["patch"], refine = (int)p["refine"], fast = (int)p["fast"];

            using var I = DehazeCore.Normalize(input);
            using var darkA = DehazeCore.DarkChannel(I, patch);
            var A = DehazeCore.Atmospheric(I, darkA, 0.001);

            using var norm = DehazeCore.NormByA(I, A);
            using var darkNorm = DehazeCore.DarkChannel(norm, patch);
            using var tD = new Mat();
            darkNorm.ConvertTo(tD, DepthType.Cv32F, -omega, 1.0);
            DehazeCore.Clamp01(tD);

            using var sky = DehazeCore.SkyMask(I);
            using var tH = BuildHsvBrightTransmission(I, sky, alpha, patch, tSky);
            using var wD = DcpConfidence(I, sky, alpha, tauV, tauS);

            using var tRaw = new Mat();
            using (var diff = new Mat())
            {
                CvInvoke.Subtract(tD, tH, diff);
                CvInvoke.Multiply(wD, diff, diff);
                CvInvoke.Add(tH, diff, tRaw);
            }

            DehazeCore.RaiseInSky(tRaw, sky, tSky);
            DehazeCore.Clamp01(tRaw);

            using var tRef = Refiners.FastGuided(I, tRaw, refine, eps, fast);
            DehazeCore.Clamp01(tRef);

            using var recovered = DehazeCore.Recover(I, tRef, A, tmin, chromaFloor);
            using var toned = DehazeCore.RestoreTone(recovered, tone);
            return DehazeCore.LimitColorfulness(toned, input.Mat, p["color"]);
        }

        private static Mat BuildHsvBrightTransmission(Mat i01, Mat sky, double alpha, int patch, double tSky)
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

            int r = Math.Max(1, patch);
            using var elem = CvInvoke.GetStructuringElement(
                ElementShape.Rectangle,
                new System.Drawing.Size(2 * r + 1, 2 * r + 1),
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
            using (var diff = new Mat())
            {
                CvInvoke.Subtract(safeH, tH, diff);
                CvInvoke.Multiply(sky, diff, diff);
                CvInvoke.Add(tH, diff, t);
            }

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
