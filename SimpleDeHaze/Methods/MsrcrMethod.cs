using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace SimpleDeHaze.Methods
{
    /// <summary>MSRCR - Multi-Scale Retinex с цветовой реставрацией (per-channel + насыщенность). См. docs/methods/other-methods.md.</summary>
    public sealed class MsrcrMethod : IDeHazeMethod
    {
        public string Name => "MSRCR (Retinex + цвет)";

        public string Description =>
            "Multi-Scale Retinex with Color Restoration (Jobson, 1997). В отличие от MSR по яркости,\n" +
            "Retinex считается по КАЖДОМУ каналу (с сохранением средней яркости канала), затем\n" +
            "добавляется восстановление насыщенности (color restoration), которого 'голому' MSR не хватает.\n\n" +
            "Шаги: per-channel MSR R_c = Σ_s(log I_c - log(G_s*I_c)) -> нормировка по среднему канала ->\n" +
            "усиление насыщенности (restore).\n\n" +
            "Параметры: gain - контраст; restore - реставрация цвета; масштабы σ.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("gain",    "Контраст (нормировка)",  0.5, 3.0, 1.3, search: true),
            new ParamDef("restore", "Реставрация цвета",      0.0, 1.5, 0.4, search: true),
            new ParamDef("small",   "Малый масштаб σ",        5,  40,  12, 1, isInt: true),
            new ParamDef("mid",     "Средний масштаб σ",      20, 120, 50, 1, isInt: true),
            new ParamDef("large",   "Большой масштаб σ",      60, 300, 130, 1, isInt: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double gain = p["gain"], restore = p["restore"];
            double[] sigmas = { (int)p["small"], (int)p["mid"], (int)p["large"] };
            const double eps = 1e-2;

            using var I = new Mat();
            input.Mat.ConvertTo(I, DepthType.Cv32F, 1.0 / 255.0);
            var ch = I.Split();

            using var outv = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                using var logI = new Mat();
                using (var ie = new Mat()) { CvInvoke.Add(ch[c], new ScalarArray(eps), ie); CvInvoke.Log(ie, logI); }
                using var msr = new Mat(ch[c].Size, DepthType.Cv32F, 1); msr.SetTo(new MCvScalar(0));
                foreach (var s in sigmas)
                {
                    using var blur = DehazeCore.FastGaussian(ch[c], s);   // большой σ - через прореживание (быстро)
                    using var lb = new Mat(); using (var be = new Mat()) { CvInvoke.Add(blur, new ScalarArray(eps), be); CvInvoke.Log(be, lb); }
                    using var d = new Mat(); CvInvoke.Subtract(logI, lb, d); CvInvoke.Add(msr, d, msr);
                }
                CvInvoke.Multiply(msr, new ScalarArray(1.0 / sigmas.Length), msr);

                MCvScalar m = default, sd = default; CvInvoke.MeanStdDev(msr, ref m, ref sd);
                double s2 = sd.V0 < 1e-6 ? 1.0 : sd.V0;
                double meanCh = CvInvoke.Mean(ch[c]).V0;
                var o = new Mat();
                CvInvoke.Subtract(msr, new ScalarArray(m.V0), o);
                CvInvoke.Multiply(o, new ScalarArray(gain * 0.15 / s2), o);
                CvInvoke.Add(o, new ScalarArray(meanCh), o);
                using (var z = new Mat(o.Size, DepthType.Cv32F, 1)) { z.SetTo(new MCvScalar(0)); CvInvoke.Max(o, z, o); }
                DehazeCore.SoftHighlight(o, 0.85);   // мягкий roll-off светов вместо жёсткого клипа
                outv.Push(o); o.Dispose();
                ch[c].Dispose();
            }
            using var outBgr = new Mat(); CvInvoke.Merge(outv, outBgr);
            using var clipped = DeHazeCPU.Clip(outBgr.Clone());

            // color restoration: усиление насыщенности
            if (restore > 1e-3)
            {
                using var hsv = new Mat(); CvInvoke.CvtColor(clipped, hsv, ColorConversion.Bgr2Hsv);
                var hc = hsv.Split();
                CvInvoke.Multiply(hc[1], new ScalarArray(1.0 + restore), hc[1]);
                using (var hi = new Mat(hc[1].Size, DepthType.Cv32F, 1)) { hi.SetTo(new MCvScalar(1.0)); CvInvoke.Min(hc[1], hi, hc[1]); }
                using (var v = new VectorOfMat(hc)) CvInvoke.Merge(v, hsv);
                foreach (var x in hc) x.Dispose();
                var res = new Mat(); CvInvoke.CvtColor(hsv, res, ColorConversion.Hsv2Bgr);
                return res;
            }
            return clipped.Clone();
        }
    }
}
