using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// GPU-версии итеративных уточнителей карты t (вход/выход - GpuMat, без загрузки/выгрузки внутри).
    /// Используются из GpuCore, который держит весь конвейер на видеокарте.
    /// </summary>
    internal static class GpuRefiners
    {
        /// <summary>Сдвиг GpuMat на (dx,dy) с реплицированной границей.</summary>
        private static GpuMat Shift(GpuMat m, int dx, int dy)
        {
            using var b = new GpuMat();
            CudaInvoke.CopyMakeBorder(m, b, 1, 1, 1, 1, BorderType.Replicate, new MCvScalar());
            using var roi = b.ColRange(1 - dx, 1 - dx + m.Size.Width).RowRange(1 - dy, 1 - dy + m.Size.Height);
            var outp = new GpuMat();
            roi.CopyTo(outp);
            return outp;
        }

        private static GpuMat EdgeWeight(GpuMat gray, int dx, int dy, double sigma)
        {
            using var sh = Shift(gray, dx, dy);
            var w = new GpuMat();
            using (var d1 = new GpuMat()) using (var d2 = new GpuMat())
            {
                CudaInvoke.Subtract(gray, sh, d1);
                CudaInvoke.Subtract(sh, gray, d2);
                CudaInvoke.Max(d1, d2, w);                       // |gray - sh|
            }
            CudaInvoke.Multiply(w, new ScalarArray(-1.0 / sigma), w);
            CudaInvoke.Exp(w, w);                                // exp(-|*|/σ)
            return w;
        }

        private static void AccumulateWeighted(GpuMat num, GpuMat w, GpuMat tShift, double lambda)
        {
            using var tmp = new GpuMat();
            CudaInvoke.Multiply(w, tShift, tmp);
            CudaInvoke.Multiply(tmp, new ScalarArray(lambda), tmp);
            CudaInvoke.Add(num, tmp, num);
        }

        /// <summary>WLS (matting-Laplacian-style) уточнение t на GPU. Вход/выход - GpuMat.</summary>
        public static GpuMat WlsCore(GpuMat gGuide, GpuMat gT, double lambda, int iters)
        {
            const double sigmaC = 0.1;
            using var gray = new GpuMat();
            CudaInvoke.CvtColor(gGuide, gray, ColorConversion.Bgr2Gray);

            using var wE = EdgeWeight(gray, -1, 0, sigmaC);
            using var wS = EdgeWeight(gray, 0, -1, sigmaC);
            using var wW = Shift(wE, 1, 0);
            using var wN = Shift(wS, 0, 1);

            using var den = new GpuMat();
            CudaInvoke.Add(wE, wW, den);
            CudaInvoke.Add(den, wS, den);
            CudaInvoke.Add(den, wN, den);
            CudaInvoke.Multiply(den, new ScalarArray(lambda), den);
            CudaInvoke.Add(den, new ScalarArray(1.0), den);

            using var tTilde = new GpuMat(); gT.CopyTo(tTilde);
            var cur = new GpuMat(); gT.CopyTo(cur);
            for (int it = 0; it < iters; it++)
            {
                using var tE = Shift(cur, -1, 0); using var tW = Shift(cur, 1, 0);
                using var tS = Shift(cur, 0, -1); using var tN = Shift(cur, 0, 1);
                using var num = new GpuMat(); tTilde.CopyTo(num);
                AccumulateWeighted(num, wE, tE, lambda);
                AccumulateWeighted(num, wW, tW, lambda);
                AccumulateWeighted(num, wS, tS, lambda);
                AccumulateWeighted(num, wN, tN, lambda);
                CudaInvoke.Divide(num, den, cur);
            }
            return cur;
        }

        /// <summary>Beltrami / анизотропная диффузия t на GPU. Вход/выход - GpuMat.</summary>
        public static GpuMat BeltramiCore(GpuMat gGuide, GpuMat gT, int iters, double kappa)
        {
            using var gray = new GpuMat();
            CudaInvoke.CvtColor(gGuide, gray, ColorConversion.Bgr2Gray);

            using var c = new GpuMat();
            using (var gE = Shift(gray, -1, 0)) using (var gW = Shift(gray, 1, 0))
            using (var gS = Shift(gray, 0, -1)) using (var gN = Shift(gray, 0, 1))
            using (var gx = new GpuMat()) using (var gy = new GpuMat())
            using (var gx2 = new GpuMat()) using (var gy2 = new GpuMat())
            {
                CudaInvoke.Subtract(gE, gW, gx); CudaInvoke.Multiply(gx, new ScalarArray(0.5), gx);
                CudaInvoke.Subtract(gS, gN, gy); CudaInvoke.Multiply(gy, new ScalarArray(0.5), gy);
                CudaInvoke.Multiply(gx, gx, gx2); CudaInvoke.Multiply(gy, gy, gy2);
                CudaInvoke.Add(gx2, gy2, c);
            }
            CudaInvoke.Multiply(c, new ScalarArray(-1.0 / (kappa * kappa)), c);
            CudaInvoke.Exp(c, c);

            var cur = new GpuMat(); gT.CopyTo(cur);
            const double dt = 0.2;
            for (int it = 0; it < iters; it++)
            {
                using var lap = new GpuMat();
                using (var n = Shift(cur, 0, 1)) using (var s = Shift(cur, 0, -1))
                using (var e = Shift(cur, -1, 0)) using (var w = Shift(cur, 1, 0))
                {
                    CudaInvoke.Add(n, s, lap); CudaInvoke.Add(lap, e, lap); CudaInvoke.Add(lap, w, lap);
                    using var c4 = new GpuMat();
                    CudaInvoke.Multiply(cur, new ScalarArray(4.0), c4);
                    CudaInvoke.Subtract(lap, c4, lap);
                }
                CudaInvoke.Multiply(lap, c, lap);
                using var step = new GpuMat();
                CudaInvoke.Multiply(lap, new ScalarArray(dt), step);
                CudaInvoke.Add(cur, step, cur);
            }
            return cur;
        }
    }
}
