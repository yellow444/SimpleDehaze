using System.Drawing;
using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Различные способы уточнения карты пропускания t (вход: гайд I - BGR float, и грубая t - 1 канал float).
    /// Все возвращают новую уточнённую t (1 канал float).
    /// </summary>
    internal static class Refiners
    {
        private static Mat MatFromFloats(float[] data, int rows, int cols)
        {
            var m = new Mat(rows, cols, DepthType.Cv32F, 1);
            Marshal.Copy(data, 0, m.DataPointer, data.Length);
            return m;
        }

        // ---------- Fractional Laplacian (изотропное частотное сглаживание через ДПФ) ----------
        public static Mat Fractional(Mat _guide, Mat t, double alpha, double lambda)
        {
            int rows = t.Rows, cols = t.Cols;
            int oh = CvInvoke.GetOptimalDFTSize(rows), ow = CvInvoke.GetOptimalDFTSize(cols);
            using var padded = new Mat();
            CvInvoke.CopyMakeBorder(t, padded, 0, oh - rows, 0, ow - cols, BorderType.Reflect101);

            using var zeros = new Mat(padded.Size, DepthType.Cv32F, 1); zeros.SetTo(new MCvScalar(0));
            using var complex = new Mat();
            using (var planes = new VectorOfMat()) { planes.Push(padded); planes.Push(zeros); CvInvoke.Merge(planes, complex); }
            CvInvoke.Dft(complex, complex, DxtType.Forward, 0);

            var mask = new float[oh * ow];
            System.Threading.Tasks.Parallel.For(0, oh, v =>
            {
                double fv = Math.Min(v, oh - v) / (double)oh;
                int row = v * ow;
                for (int u = 0; u < ow; u++)
                {
                    double fu = Math.Min(u, ow - u) / (double)ow;
                    double xi2 = fu * fu + fv * fv;
                    mask[row + u] = (float)(1.0 / (1.0 + lambda * Math.Pow(xi2, alpha)));
                }
            });
            using var maskMat = MatFromFloats(mask, oh, ow);

            var cch = complex.Split();
            CvInvoke.Multiply(cch[0], maskMat, cch[0]);
            CvInvoke.Multiply(cch[1], maskMat, cch[1]);
            using (var merged = new Mat())
            {
                using (var v2 = new VectorOfMat()) { v2.Push(cch[0]); v2.Push(cch[1]); CvInvoke.Merge(v2, merged); }
                CvInvoke.Dft(merged, merged, DxtType.Inverse | DxtType.Scale, 0);
                var rch = merged.Split();
                var res = new Mat(rch[0], new Rectangle(0, 0, cols, rows)).Clone();
                foreach (var c in rch) c.Dispose();
                foreach (var c in cch) c.Dispose();
                return res;
            }
        }

        // ---------- Beltrami / анизотропная диффузия с краевым торможением по гайду ----------
        public static Mat Beltrami(Mat guide, Mat t, int iters, double kappa)
        {
            using var gray = new Mat();
            CvInvoke.CvtColor(guide, gray, ColorConversion.Bgr2Gray);
            using var gx = new Mat(); using var gy = new Mat();
            CvInvoke.Sobel(gray, gx, DepthType.Cv32F, 1, 0, 3);
            CvInvoke.Sobel(gray, gy, DepthType.Cv32F, 0, 1, 3);
            using var g2 = new Mat();
            using (var gx2 = new Mat()) { CvInvoke.Multiply(gx, gx, gx2); using var gy2 = new Mat(); CvInvoke.Multiply(gy, gy, gy2); CvInvoke.Add(gx2, gy2, g2); }
            using var c = new Mat();
            g2.ConvertTo(c, DepthType.Cv32F, -1.0 / (kappa * kappa));
            CvInvoke.Exp(c, c);                                  // c = exp(-|∇I|^2/κ^2) ∈ (0,1]

            var cur = t.Clone();
            const double dt = 0.2;
            for (int it = 0; it < iters; it++)
            {
                using var lap = new Mat();
                CvInvoke.Laplacian(cur, lap, DepthType.Cv32F, 1, 1.0, 0.0, BorderType.Reflect101);  // Δt одной операцией
                CvInvoke.Multiply(lap, c, lap);                         // c*Δt
                using var upd = new Mat();
                CvInvoke.AddWeighted(cur, 1.0, lap, dt, 0.0, upd);       // cur + dt*c*Δt
                upd.CopyTo(cur);
            }
            return cur;
        }

        // ---------- WLS (полноразмерный matting-Laplacian-style global solve, взвешенный Якоби) ----------
        public static Mat Wls(Mat guide, Mat t, double lambda, int iters)
        {
            using var gray = new Mat();
            CvInvoke.CvtColor(guide, gray, ColorConversion.Bgr2Gray);
            const double sigmaC = 0.1;
            using var wE = new Mat();
            using (var e = DehazeCore.Shift(gray, -1, 0)) using (var de = new Mat()) { CvInvoke.AbsDiff(gray, e, de); de.ConvertTo(wE, DepthType.Cv32F, -1.0 / sigmaC); }
            CvInvoke.Exp(wE, wE);
            using var wS = new Mat();
            using (var s = DehazeCore.Shift(gray, 0, -1)) using (var ds = new Mat()) { CvInvoke.AbsDiff(gray, s, ds); ds.ConvertTo(wS, DepthType.Cv32F, -1.0 / sigmaC); }
            CvInvoke.Exp(wS, wS);

            // инварианты цикла (веса и знаменатель не зависят от t) - считаем один раз
            using var wW = DehazeCore.Shift(wE, 1, 0);
            using var wN = DehazeCore.Shift(wS, 0, 1);
            using var den = new Mat(wE.Size, DepthType.Cv32F, 1); den.SetTo(new MCvScalar(1.0));
            using (var ws = new Mat())
            {
                CvInvoke.Add(wE, wW, ws); CvInvoke.Add(ws, wS, ws); CvInvoke.Add(ws, wN, ws);
                CvInvoke.AddWeighted(den, 1.0, ws, lambda, 0, den);   // den = 1 + λ*Σw
            }

            using var tTilde = t.Clone();
            var cur = t.Clone();
            for (int it = 0; it < iters; it++)
            {
                using var tE = DehazeCore.Shift(cur, -1, 0); using var tW = DehazeCore.Shift(cur, 1, 0);
                using var tS = DehazeCore.Shift(cur, 0, -1); using var tN = DehazeCore.Shift(cur, 0, 1);

                using var num = tTilde.Clone();
                using (var tmp = new Mat())
                {
                    CvInvoke.Multiply(wE, tE, tmp); CvInvoke.AddWeighted(num, 1.0, tmp, lambda, 0, num);
                    CvInvoke.Multiply(wW, tW, tmp); CvInvoke.AddWeighted(num, 1.0, tmp, lambda, 0, num);
                    CvInvoke.Multiply(wS, tS, tmp); CvInvoke.AddWeighted(num, 1.0, tmp, lambda, 0, num);
                    CvInvoke.Multiply(wN, tN, tmp); CvInvoke.AddWeighted(num, 1.0, tmp, lambda, 0, num);
                }
                CvInvoke.Divide(num, den, cur);
            }
            return cur;
        }

        // ---------- MST Tree Filter (минимальное остовное дерево + двухпроходная агрегация) ----------
        public static Mat Mst(Mat guide, Mat t, double sigma)
        {
            int W = guide.Cols, H = guide.Rows, n = W * H;
            var ch = guide.Split();
            var b = new float[n]; var g = new float[n]; var r = new float[n];
            ch[0].CopyTo(b); ch[1].CopyTo(g); ch[2].CopyTo(r);
            foreach (var c in ch) c.Dispose();
            var tv = new float[n]; t.CopyTo(tv);
            if (n <= 1) return t.Clone();

            int m = (W - 1) * H + W * (H - 1);
            var ep = new int[m]; var eq = new int[m]; var ew = new float[m];
            int idx = 0;
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                {
                    int p = y * W + x;
                    if (x < W - 1) { int q = p + 1; ep[idx] = p; eq[idx] = q; ew[idx] = ColorDist(b, g, r, p, q); idx++; }
                    if (y < H - 1) { int q = p + W; ep[idx] = p; eq[idx] = q; ew[idx] = ColorDist(b, g, r, p, q); idx++; }
                }

            var order = Enumerable.Range(0, m).ToArray();
            var ewKey = (float[])ew.Clone();
            Array.Sort(ewKey, order);   // сортировка индексов по весам примитивным сравнением (быстрее делегата)

            var parent = new int[n]; for (int i = 0; i < n; i++) parent[i] = i;
            var rank = new byte[n];
            int Find(int a) { while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; }

            var teP = new int[n - 1]; var teQ = new int[n - 1]; var teW = new float[n - 1]; int tc = 0;
            foreach (var e in order)
            {
                if (tc == n - 1) break;
                int a = Find(ep[e]), cc = Find(eq[e]);
                if (a != cc)
                {
                    if (rank[a] < rank[cc]) { (a, cc) = (cc, a); }
                    parent[cc] = a; if (rank[a] == rank[cc]) rank[a]++;
                    teP[tc] = ep[e]; teQ[tc] = eq[e]; teW[tc] = ew[e]; tc++;
                }
            }

            var deg = new int[n];
            for (int i = 0; i < tc; i++) { deg[teP[i]]++; deg[teQ[i]]++; }
            var head = new int[n + 1]; for (int i = 0; i < n; i++) head[i + 1] = head[i] + deg[i];
            var adjTo = new int[2 * tc]; var adjW = new float[2 * tc]; var cur = new int[n]; Array.Copy(head, cur, n);
            for (int i = 0; i < tc; i++)
            {
                int p = teP[i], q = teQ[i]; float w = teW[i];
                adjTo[cur[p]] = q; adjW[cur[p]] = w; cur[p]++;
                adjTo[cur[q]] = p; adjW[cur[q]] = w; cur[q]++;
            }

            var par = new int[n]; var sPar = new float[n]; var ordr = new int[n]; var vis = new bool[n];
            int qt = 0; ordr[qt++] = 0; vis[0] = true; par[0] = -1; sPar[0] = 0f;
            for (int qh = 0; qh < qt; qh++)
            {
                int p = ordr[qh];
                for (int k = head[p]; k < head[p + 1]; k++)
                {
                    int q = adjTo[k];
                    if (!vis[q]) { vis[q] = true; par[q] = p; sPar[q] = (float)Math.Exp(-adjW[k] / sigma); ordr[qt++] = q; }
                }
            }

            var aggV = new float[n]; var aggW = new float[n];
            for (int i = 0; i < n; i++) { aggV[i] = tv[i]; aggW[i] = 1f; }
            for (int i = n - 1; i >= 0; i--) { int p = ordr[i]; int pa = par[p]; if (pa >= 0) { float s = sPar[p]; aggV[pa] += s * aggV[p]; aggW[pa] += s * aggW[p]; } }

            var outV = new float[n]; var outW = new float[n];
            for (int i = 0; i < n; i++)
            {
                int p = ordr[i]; int pa = par[p];
                if (pa < 0) { outV[p] = aggV[p]; outW[p] = aggW[p]; }
                else { float s = sPar[p]; outV[p] = aggV[p] + s * (outV[pa] - s * aggV[p]); outW[p] = aggW[p] + s * (outW[pa] - s * aggW[p]); }
            }

            var res = new float[n];
            for (int i = 0; i < n; i++) res[i] = outW[i] > 1e-12f ? outV[i] / outW[i] : tv[i];
            return MatFromFloats(res, H, W);
        }

        // ---------- Total Variation (ROF), primal-dual Chambolle-Pock (изотропно, без гайда) ----------
        public static Mat Tv(Mat _guide, Mat tTilde, double lambda, int iters)
        {
            int W = tTilde.Cols, H = tTilde.Rows;
            using var t = new Mat(); tTilde.CopyTo(t);
            using var tbar = new Mat(); tTilde.CopyTo(tbar);
            using var px = new Mat(t.Size, DepthType.Cv32F, 1); px.SetTo(new MCvScalar(0));
            using var py = new Mat(t.Size, DepthType.Cv32F, 1); py.SetTo(new MCvScalar(0));
            double tau = 0.25, sigma = 0.25, ratio = tau / Math.Max(1e-6, lambda);
            for (int it = 0; it < iters; it++)
            {
                // градиент: прямые разности по ROI (без Shift-аллокаций), граница Replicate => край = 0
                using (var gx = DiffX(tbar, W, H, true)) using (var gy = DiffY(tbar, W, H, true))
                {
                    CvInvoke.AddWeighted(px, 1.0, gx, sigma, 0, px);
                    CvInvoke.AddWeighted(py, 1.0, gy, sigma, 0, py);
                }
                using (var nrm = new Mat())
                {
                    using (var px2 = new Mat()) using (var py2 = new Mat()) { CvInvoke.Multiply(px, px, px2); CvInvoke.Multiply(py, py, py2); CvInvoke.Add(px2, py2, nrm); }
                    CvInvoke.Sqrt(nrm, nrm);
                    using (var one = new Mat(nrm.Size, DepthType.Cv32F, 1)) { one.SetTo(new MCvScalar(1.0)); CvInvoke.Max(nrm, one, nrm); }
                    CvInvoke.Divide(px, nrm, px); CvInvoke.Divide(py, nrm, py);   // проекция ||p||<=1
                }
                using var tNew = new Mat();
                // дивергенция: обратные разности (адъюнкт градиента)
                using (var dx = DiffX(px, W, H, false)) using (var dy = DiffY(py, W, H, false)) using (var div = new Mat())
                {
                    CvInvoke.Add(dx, dy, div);
                    CvInvoke.AddWeighted(t, 1.0, div, tau, 0, tNew);
                    CvInvoke.AddWeighted(tNew, 1.0, tTilde, ratio, 0, tNew);
                    CvInvoke.Multiply(tNew, new ScalarArray(1.0 / (1.0 + ratio)), tNew);   // прокс data-term
                }
                CvInvoke.AddWeighted(tNew, 2.0, t, -1.0, 0, tbar);   // экстраполяция
                tNew.CopyTo(t);
            }
            return t.Clone();
        }

        /// <summary>Конечная разность по X через ROI: forward => in(x+1)-in(x) (край-справа=0); backward => in(x)-in(x-1) (край-слева=0).</summary>
        private static Mat DiffX(Mat m, int W, int H, bool forward)
        {
            var o = new Mat(m.Size, DepthType.Cv32F, 1); o.SetTo(new MCvScalar(0));
            if (W > 1)
            {
                int dstX = forward ? 0 : 1;
                using var dst = new Mat(o, new Rectangle(dstX, 0, W - 1, H));
                using var right = new Mat(m, new Rectangle(1, 0, W - 1, H));
                using var left = new Mat(m, new Rectangle(0, 0, W - 1, H));
                CvInvoke.Subtract(right, left, dst);   // (in[1:] - in[:-1]) кладём в forward->[:-1] / backward->[1:]
            }
            return o;
        }

        /// <summary>Конечная разность по Y через ROI (см. <see cref="DiffX"/>).</summary>
        private static Mat DiffY(Mat m, int W, int H, bool forward)
        {
            var o = new Mat(m.Size, DepthType.Cv32F, 1); o.SetTo(new MCvScalar(0));
            if (H > 1)
            {
                int dstY = forward ? 0 : 1;
                using var dst = new Mat(o, new Rectangle(0, dstY, W, H - 1));
                using var down = new Mat(m, new Rectangle(0, 1, W, H - 1));
                using var up = new Mat(m, new Rectangle(0, 0, W, H - 1));
                CvInvoke.Subtract(down, up, dst);
            }
            return o;
        }

        // ---------- Weighted Guided Filter (одноканальный гайд по яркости, адаптивная ε) ----------
        public static Mat Wgif(Mat guideColor, Mat t, int r, double eps)
        {
            using var I = new Mat();
            CvInvoke.CvtColor(guideColor, I, ColorConversion.Bgr2Gray);
            var ks = new Size(2 * r + 1, 2 * r + 1);
            var anc = new Point(-1, -1);

            using var II = new Mat(); CvInvoke.Multiply(I, I, II);
            using var Ip = new Mat(); CvInvoke.Multiply(I, t, Ip);
            using var meanI = new Mat(); CvInvoke.Blur(I, meanI, ks, anc);
            using var meanP = new Mat(); CvInvoke.Blur(t, meanP, ks, anc);
            using var corrI = new Mat(); CvInvoke.Blur(II, corrI, ks, anc);
            using var corrIp = new Mat(); CvInvoke.Blur(Ip, corrIp, ks, anc);
            using var varI = new Mat(); using (var m2 = new Mat()) { CvInvoke.Multiply(meanI, meanI, m2); CvInvoke.Subtract(corrI, m2, varI); }
            using var covIp = new Mat(); using (var mm = new Mat()) { CvInvoke.Multiply(meanI, meanP, mm); CvInvoke.Subtract(corrIp, mm, covIp); }

            // edge-aware вес Γ: локальная дисперсия 3x3, нормированная средним
            using var var3 = new Mat();
            using (var m3 = new Mat()) using (var c3 = new Mat()) using (var sq = new Mat())
            {
                CvInvoke.Blur(I, m3, new Size(3, 3), anc);
                CvInvoke.Blur(II, c3, new Size(3, 3), anc);
                CvInvoke.Multiply(m3, m3, sq);
                CvInvoke.Subtract(c3, sq, var3);
            }
            CvInvoke.Add(var3, new ScalarArray(1e-6), var3);
            double meanVar = Math.Max(1e-9, CvInvoke.Mean(var3).V0);
            using var Gamma = new Mat(); var3.ConvertTo(Gamma, DepthType.Cv32F, 1.0 / meanVar);   // (σ^2+η)/mean

            using var epsG = new Mat();
            using (var num = new Mat(Gamma.Size, DepthType.Cv32F, 1)) { num.SetTo(new MCvScalar(eps)); CvInvoke.Divide(num, Gamma, epsG); }  // ε/Γ
            using var denom = new Mat(); CvInvoke.Add(varI, epsG, denom);
            using var a = new Mat(); CvInvoke.Divide(covIp, denom, a);
            using var b = new Mat(); using (var aMi = new Mat()) { CvInvoke.Multiply(a, meanI, aMi); CvInvoke.Subtract(meanP, aMi, b); }
            using var meanA = new Mat(); CvInvoke.Blur(a, meanA, ks, anc);
            using var meanB = new Mat(); CvInvoke.Blur(b, meanB, ks, anc);

            var q = new Mat(); CvInvoke.Multiply(meanA, I, q); CvInvoke.Add(q, meanB, q);
            return q;
        }

        private static float ColorDist(float[] b, float[] g, float[] r, int p, int q)
        {
            float db = b[p] - b[q], dg = g[p] - g[q], dr = r[p] - r[q];
            return MathF.Sqrt(db * db + dg * dg + dr * dr);
        }
    }
}
