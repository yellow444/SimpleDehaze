using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>Energy-Based DCP (упрощённо): доверительная смесь t_DCP и t_CAP + WLS. См. docs/methods/energy-based-dcp.md.</summary>
    public sealed class EnergyBasedDcpMethod : IDeHazeMethod
    {
        public string Name => "DCP - Energy-Based (DCP+CAP)";

        public string Description =>
            "Объединяет два приора в одну (упрощённую) энергию: оценку DCP (t_D) и Color Attenuation\n" +
            "Prior по HSV (t_S), смешанные по доверию к DCP, плюс edge-aware гладкость (WLS).\n\n" +
            "Доверие к DCP: λ_D = σ(3*S + 4*tex - 2*B) (мала на небе/гладком/белом).\n" +
            "t = t_S + λ_D*(t_D - t_S), затем WLS-уточнение. Минимальный прототип без clipping-терма.\n\n" +
            "Параметры: ω - сила DCP; β - сила CAP; λ - гладкость WLS; iters.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",  "ω - доля удаляемой дымки", 0.3, 0.95, 0.5, search: true),
            new ParamDef("beta",   "β - сила CAP",             0.5, 2.0, 1.0),
            new ParamDef("patch",  "Патч тёмного канала",      1, 15, 5, 1, isInt: true),
            new ParamDef("min",    "t_min - нижний порог t",   0.01, 0.5, 0.1),
            new ParamDef("lambda", "λ - гладкость WLS",        1, 300, 40, log: true, search: true),
            new ParamDef("iters",  "Итераций WLS",             5, 60, 25, 1, isInt: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"], beta = p["beta"], tmin = p["min"], lambda = p["lambda"];
            int patch = (int)p["patch"], iters = (int)p["iters"];
            var anc = new Point(-1, -1);

            using var I = DehazeCore.Normalize(input);
            using var darkA = DehazeCore.DarkChannel(I, patch);
            var A = DehazeCore.Atmospheric(I, darkA, 0.001);
            using var norm = DehazeCore.NormByA(I, A);
            using var DA = DehazeCore.DarkChannel(norm, patch);
            using var tD = new Mat(); DA.ConvertTo(tD, DepthType.Cv32F, -omega, 1.0);

            // t_S из CAP (HSV): d = 0.1218 + 0.9597*V - 0.7802*S ; t_S = exp(-β*d)
            using var hsv = new Mat(); CvInvoke.CvtColor(I, hsv, ColorConversion.Bgr2Hsv);
            var hc = hsv.Split();
            using var S = hc[1]; using var V = hc[2]; hc[0].Dispose();
            using var d = new Mat(); CvInvoke.AddWeighted(V, 0.9597, S, -0.7802, 0.1218, d, DepthType.Cv32F);
            using var tS = new Mat(); d.ConvertTo(tS, DepthType.Cv32F, -beta); CvInvoke.Exp(tS, tS);

            // доверие λ_D = σ(3*S + 4*tex - 2*B - 0.5)
            using var B = DehazeCore.BrightChannel(I, patch * 2);
            using var gray = new Mat(); CvInvoke.CvtColor(I, gray, ColorConversion.Bgr2Gray);
            using var tex = new Mat();
            using (var m = new Mat()) using (var g2 = new Mat()) using (var mg2 = new Mat()) using (var m2 = new Mat())
            {
                CvInvoke.Blur(gray, m, new Size(5, 5), anc);
                CvInvoke.Multiply(gray, gray, g2);
                CvInvoke.Blur(g2, mg2, new Size(5, 5), anc);
                CvInvoke.Multiply(m, m, m2);
                CvInvoke.Subtract(mg2, m2, tex);                  // локальная дисперсия
            }
            using var score = new Mat(); S.ConvertTo(score, DepthType.Cv32F, 3.0, 0.6);   // смещение к DCP (он сильнее на этом датасете)
            CvInvoke.AddWeighted(score, 1.0, tex, 4.0, 0, score);
            CvInvoke.AddWeighted(score, 1.0, B, -2.0, 0, score);
            using var lamD = new Mat();
            using (var neg = new Mat()) { score.ConvertTo(neg, DepthType.Cv32F, -1.0); CvInvoke.Exp(neg, neg); CvInvoke.Add(neg, new ScalarArray(1.0), neg); using var ones = new Mat(neg.Size, DepthType.Cv32F, 1); ones.SetTo(new MCvScalar(1.0)); CvInvoke.Divide(ones, neg, lamD); }

            // t = t_S + λ_D*(t_D - t_S)
            using var tRaw = new Mat();
            using (var diff = new Mat()) { CvInvoke.Subtract(tD, tS, diff); CvInvoke.Multiply(lamD, diff, diff); CvInvoke.Add(tS, diff, tRaw); }

            using var t = Refiners.Wls(I, tRaw, lambda, iters);
            return DehazeCore.Recover(I, t, A, tmin);
        }
    }
}
