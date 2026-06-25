using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace SimpleDeHaze.Methods
{
    /// <summary>Tarel & Hautière - оценка атмосферной вуали через медианную фильтрацию. См. docs/methods/other-methods.md.</summary>
    public sealed class TarelMethod : IDeHazeMethod
    {
        public string Name => "Tarel (atmospheric veil)";

        public string Description =>
            "Tarel & Hautière (2009): оценивает 'атмосферную вуаль' V напрямую через медианы,\n" +
            "без тёмного канала и без решения систем.\n\n" +
            "I_min = min_c I_c;  W = median(I_min) - median(|I_min - median(I_min)|);\n" +
            "V = max(min(p*W, I_min), 0);  J_c = (I_c - V)/(1 - V).\n\n" +
            "Быстро и интерпретируемо; p - 'процент' убираемой дымки, sv - окно медианы.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("p",   "p - доля удаляемой вуали", 0.1, 1.0, 0.35, search: true),
            new ParamDef("sv",  "Окно медианы (нечёт.)",    3, 51, 21, 2, isInt: true),
            new ParamDef("min", "Нижний порог (1-V)",       0.02, 0.5, 0.1),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double pf = p["p"]; double dmin = p["min"];
            int sv = (int)p["sv"]; if (sv % 2 == 0) sv++;

            // I_min и медианы - в 8U (medianBlur с большим окном работает на 8U)
            using var Imin8 = DehazeCore.MinChannel(input.Mat);
            using var med = new Mat(); CvInvoke.MedianBlur(Imin8, med, sv);
            using var absdev = new Mat(); CvInvoke.AbsDiff(Imin8, med, absdev);
            using var medAbs = new Mat(); CvInvoke.MedianBlur(absdev, medAbs, sv);
            using var W = new Mat(); CvInvoke.Subtract(med, medAbs, W);     // 8U: max(med-medAbs,0)
            using var pW = new Mat(); W.ConvertTo(pW, DepthType.Cv8U, pf);  // p*W
            using var V = new Mat(); CvInvoke.Min(pW, Imin8, V);            // V = min(p*W, I_min)

            // восстановление в float: J_c = (I_c - V)/max(1 - V, min)
            using var If = new Mat(); input.Mat.ConvertTo(If, DepthType.Cv32F, 1.0 / 255.0);
            using var Vf = new Mat(); V.ConvertTo(Vf, DepthType.Cv32F, 1.0 / 255.0);
            using var denom = new Mat(); Vf.ConvertTo(denom, DepthType.Cv32F, -1.0, 1.0);   // 1 - V
            using (var dm = new Mat(denom.Size, DepthType.Cv32F, 1)) { dm.SetTo(new MCvScalar(dmin)); CvInvoke.Max(denom, dm, denom); }

            var ch = If.Split();
            using var outv = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                var jc = new Mat();
                CvInvoke.Subtract(ch[c], Vf, jc);
                CvInvoke.Divide(jc, denom, jc);
                outv.Push(jc); jc.Dispose(); ch[c].Dispose();
            }
            using var J = new Mat();
            CvInvoke.Merge(outv, J);
            return DeHazeCPU.Clip(J.Clone());
        }
    }
}
