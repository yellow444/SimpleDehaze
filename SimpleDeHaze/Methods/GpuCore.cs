using System.Drawing;

using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Полный конвейер классического DCP на GPU (GpuMat): нормализация, тёмный канал, трансмиссия,
    /// восстановление - всё на CUDA; уточнитель t тоже на GPU. На CPU остаётся только дешёвая
    /// оценка атмосферного света (по выгруженному тёмному каналу).
    /// </summary>
    internal static class GpuCore
    {
        public static Mat Run(Image<Bgr, byte> img, double omega, int patch, double tmin, Func<GpuMat, GpuMat, GpuMat> refineGpu, double chromaFloor = 0.4)
        {
            using var I = new Mat();
            img.Mat.ConvertTo(I, DepthType.Cv32F, 1.0 / 255.0);
            using var gI = new GpuMat(I);

            using var dark = DarkChannel(gI, patch);
            var a = Atmospheric(dark, img, 0.001);
            using var tRaw = RawTransmission(gI, a, omega, patch);
            using var tRef = refineGpu(gI, tRaw);
            using var J = Recover(gI, tRef, a, tmin, chromaFloor);
            return J.ToMat();
        }

        private static GpuMat DarkChannel(GpuMat gI, int patch)
        {
            var ch = gI.Split();
            var dc = new GpuMat();
            CudaInvoke.Min(ch[0], ch[1], dc);
            CudaInvoke.Min(dc, ch[2], dc);
            foreach (var c in ch) c.Dispose();
            int k = Math.Max(1, patch);
            var elem = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(2 * k + 1, 2 * k + 1), new Point(-1, -1));
            using var erode = new CudaMorphologyFilter(MorphOp.Erode, dc.Depth, 1, elem, new Point(-1, -1), 1);
            erode.Apply(dc, dc);
            return dc;
        }

        private static MCvScalar Atmospheric(GpuMat dark, Image<Bgr, byte> img, double topPercent)
        {
            using var darkCpu = dark.ToMat();
            int W = darkCpu.Cols, H = darkCpu.Rows, n = W * H;
            var d = new float[n]; darkCpu.CopyTo(d);
            var data = img.Data;   // byte[H,W,3], BGR

            const int B = 256;
            var hist = new int[B + 1];
            for (int i = 0; i < n; i++) { int bin = (int)(d[i] * B); hist[bin < 0 ? 0 : (bin > B ? B : bin)]++; }
            int k = Math.Max(1, (int)(n * topPercent));
            int need = k, thrBin = 0;
            for (int bin = B; bin >= 0; bin--) { need -= hist[bin]; if (need <= 0) { thrBin = bin; break; } }
            float thr = thrBin / (float)B;

            double sb = 0, sg = 0, sr = 0; int cnt = 0;
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    if (d[y * W + x] >= thr) { sb += data[y, x, 0]; sg += data[y, x, 1]; sr += data[y, x, 2]; cnt++; }
                }
            if (cnt == 0) cnt = 1;
            return new MCvScalar(sb / cnt / 255.0, sg / cnt / 255.0, sr / cnt / 255.0);
        }

        private static GpuMat RawTransmission(GpuMat gI, MCvScalar a, double omega, int patch)
        {
            var ch = gI.Split();
            double[] av = { a.V0, a.V1, a.V2 };
            var dch = new GpuMat[3];
            for (int c = 0; c < 3; c++)
            {
                dch[c] = new GpuMat();
                CudaInvoke.Divide(ch[c], new ScalarArray(av[c]), dch[c]);
                ch[c].Dispose();
            }
            using var norm = new GpuMat();
            using (var v = new VectorOfGpuMat(dch)) CudaInvoke.Merge(v, norm);
            foreach (var x in dch) x.Dispose();

            using var dc = DarkChannel(norm, patch);
            var t = new GpuMat();
            CudaInvoke.Multiply(dc, new ScalarArray(-omega), t);   // t = 1 - ω*dc
            CudaInvoke.Add(t, new ScalarArray(1.0), t);
            return t;
        }

        /// <summary>Восстановление с защитой от перенасыщения (см. <see cref="DehazeCore.Recover"/>): яркость / t_lum, хрома / t_chroma.</summary>
        private static GpuMat Recover(GpuMat gI, GpuMat t, MCvScalar a, double tmin, double chromaFloor)
        {
            using var tLum = new GpuMat();
            using (var tm = new GpuMat()) { t.CopyTo(tm); tm.SetTo(new MCvScalar(tmin)); CudaInvoke.Max(t, tm, tLum); }

            double cf = Math.Max(tmin, chromaFloor);
            using var tChroma = new GpuMat();
            if (cf > tmin) { using var tcm = new GpuMat(); t.CopyTo(tcm); tcm.SetTo(new MCvScalar(cf)); CudaInvoke.Max(t, tcm, tChroma); }
            else tLum.CopyTo(tChroma);

            var ch = gI.Split();
            double[] av = { a.V0, a.V1, a.V2 };
            var d = new GpuMat[3];
            for (int c = 0; c < 3; c++) { d[c] = new GpuMat(); CudaInvoke.Subtract(ch[c], new ScalarArray(av[c]), d[c]); ch[c].Dispose(); }

            using var dbar = new GpuMat();
            CudaInvoke.Add(d[0], d[1], dbar); CudaInvoke.Add(dbar, d[2], dbar);
            CudaInvoke.Multiply(dbar, new ScalarArray(1.0 / 3.0), dbar);
            using var lumPart = new GpuMat(); CudaInvoke.Divide(dbar, tLum, lumPart);   // d / t_lum

            var outc = new GpuMat[3];
            for (int c = 0; c < 3; c++)
            {
                outc[c] = new GpuMat();
                CudaInvoke.Subtract(d[c], dbar, outc[c]);          // хрома δ_c
                CudaInvoke.Divide(outc[c], tChroma, outc[c]);      // / t_chroma (слабее)
                CudaInvoke.Add(outc[c], lumPart, outc[c]);
                CudaInvoke.Add(outc[c], new ScalarArray(av[c]), outc[c]);
                d[c].Dispose();
            }
            var J = new GpuMat();
            using (var v = new VectorOfGpuMat(outc)) CudaInvoke.Merge(v, J);
            foreach (var x in outc) x.Dispose();
            return DeHazeGPU.Clip(J);   // клип [0,1] на месте
        }
    }
}
