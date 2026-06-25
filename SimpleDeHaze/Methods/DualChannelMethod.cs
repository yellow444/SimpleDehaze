using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>Dual-Channel Confidence: DCP + bright/saturation priors по довериям (лучше небо/белое). См. docs/methods/dual-channel-confidence-prior.md.</summary>
    public sealed class DualChannelMethod : IDeHazeMethod
    {
        public string Name => "DCP - Dual-Channel (sky-aware)";

        public string Description =>
            "DCP плюс 'небесный' приор: на небе/снегу/белых объектах тёмный канал врёт, поэтому\n" +
            "там доверие к DCP снижается и используется более мягкая оценка t.\n\n" +
            "Доверие к DCP: w_D = σ(3*S + 2*G - 2.5*B - 0.3), где S - насыщенность, G - края,\n" +
            "B - светлый канал (индикатор неба). Смешиваем:  t = t_B + w_D*(t_DCP - t_B).\n\n" +
            "Параметры: ω - сила DCP; t_sky - базовая трансмиссия неба; patch, refine/ε.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega", "ω - доля удаляемой дымки", 0.3, 0.95, 0.5, search: true),
            new ParamDef("patch", "Патч тёмного канала",      1, 15, 5, 1, isInt: true),
            new ParamDef("tsky",  "t_sky (мягкость неба)",    0.4, 0.95, 0.7, search: true),
            new ParamDef("min",   "t_min - нижний порог t",   0.01, 0.5, 0.08),
            new ParamDef("refine","Радиус Guided Filter",     5, 120, 50, 1, isInt: true),
            new ParamDef("eps",   "ε - регуляризация GF",     1e-5, 1e-2, 1e-3, log: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"]; int patch = (int)p["patch"];
            double tsky = p["tsky"], tmin = p["min"]; int refine = (int)p["refine"]; double eps = p["eps"];

            using var I = DehazeCore.Normalize(input);
            using var darkA = DehazeCore.DarkChannel(I, patch);
            var A = DehazeCore.Atmospheric(I, darkA, 0.001);
            using var norm = DehazeCore.NormByA(I, A);
            using var DA = DehazeCore.DarkChannel(norm, patch);
            using var tD = new Mat(); DA.ConvertTo(tD, DepthType.Cv32F, -omega, 1.0);

            using var B = DehazeCore.BrightChannel(I, patch * 2);
            using var S = DehazeCore.Saturation(I);
            using var flat = DehazeCore.Flatness(I, 8.0);
            using var G = new Mat(); flat.ConvertTo(G, DepthType.Cv32F, -1.0, 1.0);   // G = 1 - flat ∈[0,1], высоко на краях

            // score = 3*S + 2*G - 2.5*B - 0.3 ; w_D = sigmoid(score)
            using var score = new Mat(); S.ConvertTo(score, DepthType.Cv32F, 3.0, -0.3);
            CvInvoke.AddWeighted(score, 1.0, G, 2.0, 0, score);
            CvInvoke.AddWeighted(score, 1.0, B, -2.5, 0, score);
            using var wD = new Mat();
            using (var neg = new Mat())
            {
                score.ConvertTo(neg, DepthType.Cv32F, -1.0);
                CvInvoke.Exp(neg, neg);                                // exp(-score)
                CvInvoke.Add(neg, new ScalarArray(1.0), neg);          // 1 + exp(-score)
                using var ones = new Mat(neg.Size, DepthType.Cv32F, 1); ones.SetTo(new MCvScalar(1.0));
                CvInvoke.Divide(ones, neg, wD);                        // σ(score)
            }

            // t_B = clip( t_sky + 0.25*S + 0.15*G )
            using var tB = new Mat(); S.ConvertTo(tB, DepthType.Cv32F, 0.25, tsky);
            CvInvoke.AddWeighted(tB, 1.0, G, 0.15, 0, tB);

            // t = t_B + w_D*(t_D - t_B)
            using var tRaw = new Mat();
            using (var diff = new Mat()) { CvInvoke.Subtract(tD, tB, diff); CvInvoke.Multiply(wD, diff, diff); CvInvoke.Add(tB, diff, tRaw); }

            using var t = new Mat();
            XImgprocInvoke.GuidedFilter(I, tRaw, t, refine, eps);
            return DehazeCore.Recover(I, t, A, tmin);
        }
    }
}
