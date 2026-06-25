using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>Gradient-Domain DCP: восстановление J через screened-Poisson (меньше halo). См. docs/methods/gradient-domain-dcp.md.</summary>
    public sealed class GradientDomainMethod : IDeHazeMethod
    {
        public string Name => "DCP - Gradient Domain";

        public string Description =>
            "DCP оценивает A и t как обычно, но финальное J восстанавливается не попиксельно, а через\n" +
            "согласованные градиенты (screened-Poisson) - меньше halo/перешарпа у границ глубины.\n\n" +
            "Целевой градиент: ∇J ~ (∇I - (J0-A)*∇t)/max(t,t_min); решаем (μ-λΔ)J = μJ0 - λ*div(g)\n" +
            "итерациями Якоби, где μ мало там, где t резко меняется (доверяем градиентам).\n\n" +
            "Параметры: λ - вес градиентов; iters; ω, patch - DCP.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",  "ω - доля удаляемой дымки", 0.3, 0.95, 0.5, search: true),
            new ParamDef("patch",  "Патч тёмного канала",      1, 15, 5, 1, isInt: true),
            new ParamDef("min",    "t_min - нижний порог t",   0.01, 0.5, 0.1),
            new ParamDef("lambda", "λ - вес градиентов",       0.05, 1.0, 0.25, log: true, search: true),
            new ParamDef("iters",  "Итераций Якоби",           10, 80, 30, 1, isInt: true),
            new ParamDef("refine", "Радиус Guided Filter",     5, 120, 50, 1, isInt: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"], tmin = p["min"], lam = p["lambda"];
            int patch = (int)p["patch"], iters = (int)p["iters"], refine = (int)p["refine"];

            using var I = DehazeCore.Normalize(input);
            using var darkA = DehazeCore.DarkChannel(I, patch);
            var A = DehazeCore.Atmospheric(I, darkA, 0.001);
            using var norm = DehazeCore.NormByA(I, A);
            using var D = DehazeCore.DarkChannel(norm, patch);
            using var tRaw = new Mat(); D.ConvertTo(tRaw, DepthType.Cv32F, -omega, 1.0);
            using var t = new Mat(); XImgprocInvoke.GuidedFilter(I, tRaw, t, refine, 1e-3);
            using var tc = new Mat();
            using (var tm = new Mat(t.Size, DepthType.Cv32F, 1)) { tm.SetTo(new MCvScalar(tmin)); CvInvoke.Max(t, tm, tc); }

            // dxt, dyt и μ - общие для всех каналов
            using var dxt = new Mat(); using (var s = DehazeCore.Shift(t, -1, 0)) CvInvoke.Subtract(s, t, dxt);
            using var dyt = new Mat(); using (var s = DehazeCore.Shift(t, 0, -1)) CvInvoke.Subtract(s, t, dyt);
            using var mu = new Mat();
            using (var d2 = new Mat()) { CvInvoke.Multiply(dxt, dxt, mu); CvInvoke.Multiply(dyt, dyt, d2); CvInvoke.Add(mu, d2, mu); }
            CvInvoke.Multiply(mu, new ScalarArray(20.0), mu);
            CvInvoke.Add(mu, new ScalarArray(1.0), mu);
            using (var ones = new Mat(mu.Size, DepthType.Cv32F, 1)) { ones.SetTo(new MCvScalar(1.0)); CvInvoke.Divide(ones, mu, mu); }   // μ = 1/(1+20|∇t|^2)
            using var den = new Mat(); CvInvoke.Add(mu, new ScalarArray(4.0 * lam), den);   // μ + 4λ

            using var J0 = DehazeCore.Recover(I, t, A, tmin);
            var ich = I.Split(); var j0 = J0.Split();
            double[] av = { A.V0, A.V1, A.V2 };
            using var outv = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                using var dxI = new Mat(); using (var s = DehazeCore.Shift(ich[c], -1, 0)) CvInvoke.Subtract(s, ich[c], dxI);
                using var dyI = new Mat(); using (var s = DehazeCore.Shift(ich[c], 0, -1)) CvInvoke.Subtract(s, ich[c], dyI);
                using var JmA = new Mat(); CvInvoke.Subtract(j0[c], new ScalarArray(av[c]), JmA);
                using var gx = new Mat(); using (var tmp = new Mat()) { CvInvoke.Multiply(JmA, dxt, tmp); CvInvoke.Subtract(dxI, tmp, gx); CvInvoke.Divide(gx, tc, gx); }
                using var gy = new Mat(); using (var tmp = new Mat()) { CvInvoke.Multiply(JmA, dyt, tmp); CvInvoke.Subtract(dyI, tmp, gy); CvInvoke.Divide(gy, tc, gy); }
                using var divg = new Mat();
                using (var gxw = DehazeCore.Shift(gx, 1, 0)) using (var gyn = DehazeCore.Shift(gy, 0, 1)) using (var dx = new Mat()) using (var dy = new Mat())
                { CvInvoke.Subtract(gx, gxw, dx); CvInvoke.Subtract(gy, gyn, dy); CvInvoke.Add(dx, dy, divg); }
                using var muJ0 = new Mat(); CvInvoke.Multiply(mu, j0[c], muJ0);

                var Jc = j0[c].Clone();
                for (int it = 0; it < iters; it++)
                {
                    using var sumN = new Mat();
                    // Σ 4-соседей = Δ + 4*Jc (Laplacian работает там, где Filter2D даёт size-assert)
                    using (var lap = new Mat()) { CvInvoke.Laplacian(Jc, lap, DepthType.Cv32F, 1, 1.0, 0.0, BorderType.Replicate); CvInvoke.AddWeighted(lap, 1.0, Jc, 4.0, 0.0, sumN); }
                    using var num = new Mat();
                    CvInvoke.AddWeighted(muJ0, 1.0, sumN, lam, 0, num);     // μJ0 + λ*Σneighbors
                    CvInvoke.AddWeighted(num, 1.0, divg, -lam, 0, num);     // - λ*div(g)
                    CvInvoke.Divide(num, den, Jc);                          // /(μ+4λ)
                }
                outv.Push(Jc); Jc.Dispose();
            }
            using var J = new Mat(); CvInvoke.Merge(outv, J);
            foreach (var c in ich) c.Dispose();
            foreach (var c in j0) c.Dispose();
            return DeHazeCPU.Clip(J.Clone());
        }
    }
}
