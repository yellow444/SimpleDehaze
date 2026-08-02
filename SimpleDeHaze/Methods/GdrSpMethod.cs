using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// GDR-SP: Gradient-Domain Recovery with Screened Poisson.
    /// </summary>
    public sealed class GdrSpMethod : IDeHazeMethod
    {
        public string Name => "GDR-SP (screened Poisson recovery)";

        public string Description =>
            "Gradient-Domain Recovery with Screened Poisson.\n\n" +
            "DCP оценивает A и t, обычный Recover даёт J0. Затем ищем J, близкое к J0,\n" +
            "но с градиентами, усиленными согласно толщине дымки:\n" +
            "    min_J mu(x)||J-J0||^2 + lambda||grad J - s(x) grad I||^2,\n" +
            "    s(x)=min(1/(t+eps), smax),  mu(x)=mu0+mu1*t(x).\n\n" +
            "Получается screened-Poisson система. Решаем её по яркости, затем возвращаем цветность из J0:\n" +
            "так меньше дрейф цвета и в 3 раза меньше дорогих итераций, чем при поканальном Poisson.\n" +
            "Это recovery-stage против halo/ringing от жёсткой попиксельной инверсии.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",  "omega - доля удаляемой дымки", 0.3,  0.98, 0.95, search: true),
            new ParamDef("patch",  "Патч тёмного канала",          1,    15,   5,    1, isInt: true),
            new ParamDef("min",    "t_min - нижний порог t",       0.01, 0.5,  0.08),
            new ParamDef("chroma", "chromaFloor - защита цвета",   0.08, 0.7,  0.35),
            new ParamDef("lambda", "lambda - вес градиентов",      0.02, 0.5,  0.10, log: true, search: true),
            new ParamDef("smax",   "s_max - потолок усиления grad", 2.0,  8.0,  5.0),
            new ParamDef("mu0",    "mu0 - базовое доверие J0",      0.05, 2.0,  0.40),
            new ParamDef("mu1",    "mu1 - доверие J0 по t",         0.05, 2.0,  0.60),
            new ParamDef("iters",  "Итераций Якоби",               5,    120,  35,   1, isInt: true),
            new ParamDef("refine", "Радиус fast Guided Filter",    5,    120,  48,   1, isInt: true),
            new ParamDef("eps",    "eps - регуляризация GF",       1e-5, 1e-2, 1e-3, log: true),
            new ParamDef("fast",   "Ускорение GF (1 = full)",      1,    8,    4,    1, isInt: true, tunable: false),
            new ParamDef("tone",   "Быстрый тон/контраст",          0.0,  0.6,  0.12),
            new ParamDef("color",  "Потолок усиления цветности",    1.0,  1.6,  1.18),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"], tmin = p["min"], chroma = p["chroma"], lambda = p["lambda"];
            double smax = p["smax"], mu0 = p["mu0"], mu1 = p["mu1"], eps = p["eps"], tone = p["tone"], color = p["color"];
            int patch = (int)p["patch"], iters = (int)p["iters"], refine = (int)p["refine"], fast = (int)p["fast"];

            using var I = DehazeCore.Normalize(input);
            using var darkA = DehazeCore.DarkChannel(I, patch);
            var A = DehazeCore.Atmospheric(I, darkA, 0.001);
            using var norm = DehazeCore.NormByA(I, A);
            using var dark = DehazeCore.DarkChannel(norm, patch);
            using var tRaw = new Mat();
            dark.ConvertTo(tRaw, DepthType.Cv32F, -omega, 1.0);
            DehazeCore.Clamp01(tRaw);

            using var t = Refiners.FastGuided(I, tRaw, refine, eps, fast);
            DehazeCore.Clamp01(t);

            using var j0 = DehazeCore.Recover(I, t, A, tmin, chroma);
            using var solvedY = ScreenedPoissonLuminance(I, j0, t, lambda, smax, mu0, mu1, tmin, iters);
            using var solved = ApplyLuminance(j0, solvedY);
            using var toned = DehazeCore.RestoreToneFast(solved, tone);
            return DehazeCore.LimitColorfulness(toned, input.Mat, color);
        }

        private static Mat ScreenedPoissonLuminance(Mat i01, Mat j0, Mat t, double lambda, double smax, double mu0, double mu1, double tmin, int iters)
        {
            using var tc = new Mat();
            using (var tm = new Mat(t.Size, DepthType.Cv32F, 1))
            {
                tm.SetTo(new MCvScalar(tmin));
                CvInvoke.Max(t, tm, tc);
            }

            using var scale = new Mat();
            using (var te = new Mat())
            using (var ones = new Mat(t.Size, DepthType.Cv32F, 1))
            using (var sm = new Mat(t.Size, DepthType.Cv32F, 1))
            {
                CvInvoke.Add(tc, new ScalarArray(1e-4), te);
                ones.SetTo(new MCvScalar(1.0));
                CvInvoke.Divide(ones, te, scale);
                sm.SetTo(new MCvScalar(smax));
                CvInvoke.Min(scale, sm, scale);
            }

            using var mu = new Mat();
            t.ConvertTo(mu, DepthType.Cv32F, mu1, mu0);
            using var den = new Mat();
            CvInvoke.Add(mu, new ScalarArray(4.0 * lambda), den);

            using var iGray = new Mat();
            using var jGray = new Mat();
            CvInvoke.CvtColor(i01, iGray, ColorConversion.Bgr2Gray);
            CvInvoke.CvtColor(j0, jGray, ColorConversion.Bgr2Gray);

            using var gxI = ForwardDiffX(iGray);
            using var gyI = ForwardDiffY(iGray);
            CvInvoke.Multiply(gxI, scale, gxI);
            CvInvoke.Multiply(gyI, scale, gyI);

            using var divg = Divergence(gxI, gyI);
            using var muJ0 = new Mat();
            CvInvoke.Multiply(mu, jGray, muJ0);

            var cur = jGray.Clone();
            for (int it = 0; it < iters; it++)
            {
                using var sumN = NeighborSum(cur);
                using var num = new Mat();
                CvInvoke.AddWeighted(muJ0, 1.0, sumN, lambda, 0.0, num);
                CvInvoke.AddWeighted(num, 1.0, divg, -lambda, 0.0, num);
                CvInvoke.Divide(num, den, cur);
            }

            DehazeCore.Clamp01(cur);
            return cur;
        }

        private static Mat ApplyLuminance(Mat bgr01, Mat targetY)
        {
            using var currentY = new Mat();
            CvInvoke.CvtColor(bgr01, currentY, ColorConversion.Bgr2Gray);
            using var denom = new Mat();
            CvInvoke.Add(currentY, new ScalarArray(1e-4), denom);
            using var gain = new Mat();
            CvInvoke.Divide(targetY, denom, gain);
            DehazeCore.Clamp(gain, 0.55, 1.75);

            var ch = bgr01.Split();
            using var outv = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                CvInvoke.Multiply(ch[c], gain, ch[c]);
                outv.Push(ch[c]);
            }

            using var result = new Mat();
            CvInvoke.Merge(outv, result);
            foreach (var c in ch) c.Dispose();
            return DeHazeCPU.Clip(result.Clone());
        }

        private static Mat ForwardDiffX(Mat m)
        {
            using var e = DehazeCore.Shift(m, -1, 0);
            var d = new Mat();
            CvInvoke.Subtract(e, m, d);
            return d;
        }

        private static Mat ForwardDiffY(Mat m)
        {
            using var s = DehazeCore.Shift(m, 0, -1);
            var d = new Mat();
            CvInvoke.Subtract(s, m, d);
            return d;
        }

        private static Mat Divergence(Mat gx, Mat gy)
        {
            using var gxW = DehazeCore.Shift(gx, 1, 0);
            using var gyN = DehazeCore.Shift(gy, 0, 1);
            using var dx = new Mat();
            using var dy = new Mat();
            CvInvoke.Subtract(gx, gxW, dx);
            CvInvoke.Subtract(gy, gyN, dy);
            var div = new Mat();
            CvInvoke.Add(dx, dy, div);
            return div;
        }

        private static Mat NeighborSum(Mat m)
        {
            using var lap = new Mat();
            CvInvoke.Laplacian(m, lap, DepthType.Cv32F, 1, 1.0, 0.0, BorderType.Replicate);
            var sum = new Mat();
            CvInvoke.AddWeighted(lap, 1.0, m, 4.0, 0.0, sum);
            return sum;
        }
    }
}
