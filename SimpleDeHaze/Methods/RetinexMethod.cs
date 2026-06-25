using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace SimpleDeHaze.Methods
{
    /// <summary>Multi-Scale Retinex по яркости - классическое усиление видимости в дымке (цвет сохраняется).</summary>
    public sealed class RetinexMethod : IDeHazeMethod
    {
        public string Name => "Multi-Scale Retinex";

        public string Description =>
            "Multi-Scale Retinex (Jobson, 1997). Модель: наблюдение = освещённость x отражение;\n" +
            "Retinex оценивает отражение, убирая плавную освещённость (~ дымку/перепад света).\n\n" +
            "R(x) = Σ_s [ log I(x) - log( G_s * I )(x) ]  по нескольким масштабам s.\n\n" +
            "Здесь применяется к каналу яркости (Y), цвет (Cr,Cb) сохраняется - без 'серого' эффекта.\n" +
            "Шаги: BGR->YCrCb -> MSR по Y на масштабах {малый,средний,большой} -> нормировка (gain) -> обратно.\n\n" +
            "Параметры: gain - контраст результата; mid/large - масштабы размытия.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("gain",  "Контраст (нормировка)", 0.5, 3.0, 1.5, search: true),
            new ParamDef("small", "Малый масштаб σ",        5,  40,  12, 1, isInt: true),
            new ParamDef("mid",   "Средний масштаб σ",      20, 120, 50, 1, isInt: true),
            new ParamDef("large", "Большой масштаб σ",      60, 300, 130, 1, isInt: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double gain = p["gain"];
            double[] sigmas = { (int)p["small"], (int)p["mid"], (int)p["large"] };

            using var I = new Mat();
            input.Mat.ConvertTo(I, DepthType.Cv32F, 1.0 / 255.0);
            using var ycc = new Mat();
            CvInvoke.CvtColor(I, ycc, ColorConversion.Bgr2YCrCb);
            var ch = ycc.Split();                  // Y, Cr, Cb (float)
            using var Y = ch[0];

            using var logY = new Mat();
            using (var ye = new Mat()) { CvInvoke.Add(Y, new ScalarArray(1e-3), ye); CvInvoke.Log(ye, logY); }

            using var msr = new Mat(Y.Size, DepthType.Cv32F, 1); msr.SetTo(new MCvScalar(0));
            foreach (var s in sigmas)
            {
                using var blur = DehazeCore.FastGaussian(Y, s);   // большой σ - через прореживание (быстро)
                using var logB = new Mat();
                using (var be = new Mat()) { CvInvoke.Add(blur, new ScalarArray(1e-3), be); CvInvoke.Log(be, logB); }
                using var diff = new Mat();
                CvInvoke.Subtract(logY, logB, diff);
                CvInvoke.Add(msr, diff, msr);
            }
            CvInvoke.Multiply(msr, new ScalarArray(1.0 / sigmas.Length), msr);

            // нормировка по среднему/СКО (gain управляет растяжением) -> [0,1]
            MCvScalar mean = default, std = default;
            CvInvoke.MeanStdDev(msr, ref mean, ref std);
            double sd = std.V0 < 1e-6 ? 1.0 : std.V0;
            double yMean = CvInvoke.Mean(Y).V0;            // сохранить исходную яркость сцены
            using var newY = new Mat();
            CvInvoke.Subtract(msr, new ScalarArray(mean.V0), newY);
            CvInvoke.Multiply(newY, new ScalarArray(gain * 0.15 / sd), newY);
            CvInvoke.Add(newY, new ScalarArray(yMean), newY);
            using (var lo = new Mat(newY.Size, DepthType.Cv32F, 1)) { lo.SetTo(new MCvScalar(0)); CvInvoke.Max(newY, lo, newY); }
            DehazeCore.SoftHighlight(newY, 0.85);   // мягкий roll-off вместо жёсткого клипа -> не 'выжигаем' света

            newY.CopyTo(ch[0]);
            using var outYcc = new Mat();
            using (var v = new VectorOfMat(ch)) CvInvoke.Merge(v, outYcc);
            ch[1].Dispose(); ch[2].Dispose();
            var res = new Mat();
            CvInvoke.CvtColor(outYcc, res, ColorConversion.YCrCb2Bgr);
            return DeHazeCPU.Clip(res);
        }
    }
}
