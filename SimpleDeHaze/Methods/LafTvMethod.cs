using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// LAF-TV/WLS: low-resolution local airlight field with edge-aware regularization.
    /// </summary>
    public sealed class LafTvMethod : IDeHazeMethod
    {
        public string Name => "LAF-TV/WLS (low-res airlight)";

        public string Description =>
            "Local Airlight Field with low-resolution TV/WLS-style regularization.\n\n" +
            "Глобальный A заменяется гладким полем A(x), полезным для неравномерного неба,\n" +
            "боковой засветки и локального glare. Поле A строится на уменьшенной копии кадра:\n" +
            "яркие, малонасыщенные и гладкие пиксели получают большой airlight-confidence q,\n" +
            "затем A_g = blur(q*I_g)/blur(q) и дополнительно сглаживается WLS на low-res сетке.\n\n" +
            "После апсемплинга A(x) считаем DCP уже по I/A(x), уточняем t fast guided filter и\n" +
            "восстанавливаем J с chroma-safe локальной моделью. Это практичный low-res вариант LAF-TV\n" +
            "из TEMP.md: вместо тяжёлого TV/PCG используется быстрый matrix-free WLS.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",   "omega - доля удаляемой дымки", 0.3,  0.98, 0.95, search: true),
            new ParamDef("patch",   "Патч тёмного канала",          1,    15,   5,    1, isInt: true),
            new ParamDef("grid",    "Downsample-фактор A(x)",       4,    24,   12,   1, isInt: true),
            new ParamDef("aRadius", "Окно low-res A(x)",            1,    24,   6,    1, isInt: true),
            new ParamDef("lambda",  "lambda - WLS гладкость A(x)",  0.2,  80,   8,    log: true, search: true),
            new ParamDef("iters",   "Итераций WLS для A(x)",        2,    40,   12,   1, isInt: true),
            new ParamDef("min",     "t_min - нижний порог t",       0.01, 0.5,  0.1),
            new ParamDef("chroma",  "chromaFloor - защита цвета",   0.08, 0.7,  0.35),
            new ParamDef("refine",  "Радиус fast Guided Filter",    5,    120,  48,   1, isInt: true),
            new ParamDef("eps",     "eps - регуляризация GF",       1e-5, 1e-2, 1e-3, log: true),
            new ParamDef("fast",    "Ускорение GF (1 = full)",      1,    8,    4,    1, isInt: true, tunable: false),
            new ParamDef("tone",    "Восстановление тона",          0.0,  1.0,  0.1),
            new ParamDef("color",   "Потолок усиления цветности",    1.0,  1.6,  1.25),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"], lambda = p["lambda"], tmin = p["min"], chroma = p["chroma"];
            double eps = p["eps"], tone = p["tone"], color = p["color"];
            int patch = (int)p["patch"], grid = (int)p["grid"], aRadius = (int)p["aRadius"];
            int iters = (int)p["iters"], refine = (int)p["refine"], fast = (int)p["fast"];

            using var I = DehazeCore.Normalize(input);
            using var small = Downsample(I, grid);
            var aSmall = EstimateLowResAirlight(small, aRadius, lambda, iters);
            var aField = UpsampleAirlight(I, aSmall, refine, eps, fast);
            foreach (var a in aSmall) a.Dispose();

            using var norm = DivideByLocalAirlight(I, aField);
            using var dark = DehazeCore.DarkChannel(norm, patch);
            using var tRaw = new Mat();
            dark.ConvertTo(tRaw, DepthType.Cv32F, -omega, 1.0);
            DehazeCore.Clamp01(tRaw);

            using var t = Refiners.FastGuided(I, tRaw, refine, eps, fast);
            DehazeCore.Clamp01(t);

            using var recovered = RecoverLocal(I, t, aField, tmin, chroma);
            foreach (var a in aField) a.Dispose();

            using var toned = DehazeCore.RestoreTone(recovered, tone);
            return DehazeCore.LimitColorfulness(toned, input.Mat, color);
        }

        private static Mat Downsample(Mat src, int factor)
        {
            int f = Math.Max(1, factor);
            var small = new Mat();
            CvInvoke.Resize(
                src,
                small,
                new System.Drawing.Size(Math.Max(1, src.Cols / f), Math.Max(1, src.Rows / f)),
                0,
                0,
                Inter.Area);
            return small;
        }

        private static Mat[] EstimateLowResAirlight(Mat small, int radius, double lambda, int iters)
        {
            int r = Math.Max(1, radius);
            var anc = new System.Drawing.Point(-1, -1);
            var ks = new System.Drawing.Size(2 * r + 1, 2 * r + 1);

            using var D = DehazeCore.DarkChannel(small, 1);
            using var S = DehazeCore.Saturation(small);
            using var flat = DehazeCore.Flatness(small, 2.0);

            using var q = new Mat();
            CvInvoke.Multiply(D, D, q);
            using (var oneMinusS = new Mat())
            using (var satWeight = new Mat())
            {
                S.ConvertTo(oneMinusS, DepthType.Cv32F, -1.0, 1.0);
                CvInvoke.Pow(oneMinusS, 1.5, satWeight);
                CvInvoke.Multiply(q, satWeight, q);
            }
            CvInvoke.Multiply(q, flat, q);
            CvInvoke.Add(q, new ScalarArray(1e-4), q);

            using var den = new Mat();
            CvInvoke.Blur(q, den, ks, anc);
            CvInvoke.Add(den, new ScalarArray(1e-6), den);

            var ch = small.Split();
            var result = new Mat[3];
            for (int c = 0; c < 3; c++)
            {
                using var weighted = new Mat();
                CvInvoke.Multiply(q, ch[c], weighted);
                using var blurred = new Mat();
                CvInvoke.Blur(weighted, blurred, ks, anc);
                using var a0 = new Mat();
                CvInvoke.Divide(blurred, den, a0);
                result[c] = Refiners.Wls(small, a0, lambda, iters);
                ClampAirlight(result[c]);
                ch[c].Dispose();
            }

            return result;
        }

        private static Mat[] UpsampleAirlight(Mat guide, Mat[] lowResA, int refine, double eps, int fast)
        {
            var result = new Mat[3];
            for (int c = 0; c < 3; c++)
            {
                using var up = new Mat();
                CvInvoke.Resize(lowResA[c], up, guide.Size, 0, 0, Inter.Linear);
                result[c] = Refiners.FastGuided(guide, up, refine, eps, fast);
                ClampAirlight(result[c]);
            }
            return result;
        }

        private static Mat DivideByLocalAirlight(Mat i01, Mat[] aField)
        {
            var ch = i01.Split();
            using var vec = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                var nc = new Mat();
                CvInvoke.Divide(ch[c], aField[c], nc);
                vec.Push(nc);
                nc.Dispose();
                ch[c].Dispose();
            }

            var norm = new Mat();
            CvInvoke.Merge(vec, norm);
            return norm;
        }

        private static Mat RecoverLocal(Mat i01, Mat tSingle, Mat[] aField, double tmin, double chromaFloor)
        {
            using var tLum = new Mat();
            using (var tm = new Mat(tSingle.Size, DepthType.Cv32F, 1))
            {
                tm.SetTo(new MCvScalar(tmin));
                CvInvoke.Max(tSingle, tm, tLum);
            }

            double cf = Math.Max(tmin, chromaFloor);
            using var tChroma = new Mat();
            using (var tc = new Mat(tSingle.Size, DepthType.Cv32F, 1))
            {
                tc.SetTo(new MCvScalar(cf));
                CvInvoke.Max(tSingle, tc, tChroma);
            }

            var ch = i01.Split();
            var d = new Mat[3];
            for (int c = 0; c < 3; c++)
            {
                d[c] = new Mat();
                CvInvoke.Subtract(ch[c], aField[c], d[c]);
                ch[c].Dispose();
            }

            using var dbar = new Mat();
            CvInvoke.Add(d[0], d[1], dbar);
            CvInvoke.Add(dbar, d[2], dbar);
            dbar.ConvertTo(dbar, DepthType.Cv32F, 1.0 / 3.0);

            using var lumPart = new Mat();
            CvInvoke.Divide(dbar, tLum, lumPart);

            using var outv = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                using var delta = new Mat();
                CvInvoke.Subtract(d[c], dbar, delta);
                var jc = new Mat();
                CvInvoke.Divide(delta, tChroma, jc);
                CvInvoke.Add(jc, lumPart, jc);
                CvInvoke.Add(jc, aField[c], jc);
                outv.Push(jc);
                jc.Dispose();
                d[c].Dispose();
            }

            using var J = new Mat();
            CvInvoke.Merge(outv, J);
            return DeHazeCPU.Clip(J.Clone());
        }

        private static void ClampAirlight(Mat a)
        {
            using (var lo = new Mat(a.Size, DepthType.Cv32F, 1))
            {
                lo.SetTo(new MCvScalar(0.05));
                CvInvoke.Max(a, lo, a);
            }
            using (var hi = new Mat(a.Size, DepthType.Cv32F, 1))
            {
                hi.SetTo(new MCvScalar(1.0));
                CvInvoke.Min(a, hi, a);
            }
        }
    }
}
