using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// PF-SFGF: Pyramid-Fused DCP with Spectral Gain and Fast Guided Filter.
    /// </summary>
    public sealed class PfSfgfMethod : IDeHazeMethod
    {
        public string Name => "PF-SFGF (pyramid + fast GF)";

        public string Description =>
            "Pyramid-Fused DCP with Spectral Gain and Fast Guided Filter.\n\n" +
            "Считает три DCP-карты t на разных радиусах: малый радиус держит границы и тонкие детали,\n" +
            "средний даёт стабильную основную оценку, большой сглаживает плоские дальние области.\n" +
            "Карты смешиваются по градиентному доверию: у границ больше вес малого окна, на гладком - большого.\n\n" +
            "После fusion карта t уточняется fast guided filter с downsample, затем изображение\n" +
            "восстанавливается chroma-safe формулой. Финальный spectral gain - ограниченный band-pass\n" +
            "boost средних частот, зависящий от средней плотности дымки; он не трогает DC и почти не\n" +
            "разгоняет шумовые высокие частоты.\n\n" +
            "Формула: t_i = 1 − ω_i·min_c min_Ω_{r_i}(I_c/A_c), i ∈ {r1,r2,r3}; t = Σ w_i·t_i (w_i — градиентное\n" +
            "доверие: у границ выше вес малого окна), уточн. fast guided filter; J = (I − A)/max(t, t_min) + A;\n" +
            "финал — band-pass gain средних частот (не трогает DC и высокие).\n\n" +
            "Параметры: omega/r1/r2/r3 - DCP-пирамида; fast/refine - fast GF; gain - mid-frequency boost.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",  "omega - базовая сила DCP",        0.3,  0.98, 0.95, search: true),
            new ParamDef("r1",     "Малый радиус",                    1,    8,    3,    1, isInt: true),
            new ParamDef("r2",     "Средний радиус",                  3,    16,   7,    1, isInt: true),
            new ParamDef("r3",     "Большой радиус",                  8,    32,   15,   1, isInt: true),
            new ParamDef("tsky",   "t_sky - мягкость неба/белого",    0.45, 0.9,  0.66),
            new ParamDef("min",    "t_min - нижний порог t",          0.01, 0.5,  0.08),
            new ParamDef("chroma", "chromaFloor - защита цвета",      0.08, 0.7,  0.35),
            new ParamDef("refine", "Радиус fast Guided Filter",       5,    120,  48,   1, isInt: true),
            new ParamDef("eps",    "eps - регуляризация GF",          1e-5, 1e-2, 1e-3, log: true),
            new ParamDef("fast",   "Ускорение GF (1 = full)",         1,    8,    4,    1, isInt: true, tunable: false),
            new ParamDef("gain",   "Mid-frequency spectral gain",     0.0,  0.22, 0.12, search: true),
            new ParamDef("tone",   "Восстановление тона",             0.0,  1.0,  0.15),
            new ParamDef("color",  "Потолок усиления цветности",       1.0,  1.6,  1.25),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"], tSky = p["tsky"], tmin = p["min"], chromaFloor = p["chroma"];
            double eps = p["eps"], gain = p["gain"], tone = p["tone"], color = p["color"];
            int r1 = (int)p["r1"], r2 = (int)p["r2"], r3 = (int)p["r3"];
            int refine = (int)p["refine"], fast = (int)p["fast"];

            using var I = DehazeCore.Normalize(input);
            using var darkA = DehazeCore.DarkChannel(I, Math.Max(3, r2));
            var A = DehazeCore.Atmospheric(I, darkA, 0.001);
            using var norm = DehazeCore.NormByA(I, A);

            using var tSmall = Transmission(norm, r1, Math.Clamp(omega - 0.02, 0.05, 0.99));
            using var tMid = Transmission(norm, r2, omega);
            using var tLarge = Transmission(norm, r3, Math.Clamp(omega + 0.02, 0.05, 0.99));

            using var flat = DehazeCore.Flatness(I, 8.0);
            using var edge = new Mat();
            flat.ConvertTo(edge, DepthType.Cv32F, -1.0, 1.0);

            using var wSmall = new Mat();
            edge.ConvertTo(wSmall, DepthType.Cv32F, 0.85, 0.15);
            using var wLarge = new Mat();
            flat.ConvertTo(wLarge, DepthType.Cv32F, 0.85, 0.15);
            using var wMid = new Mat();
            CvInvoke.Multiply(edge, flat, wMid);
            wMid.ConvertTo(wMid, DepthType.Cv32F, 2.0, 0.20);

            using var tRaw = WeightedAverage(tSmall, tMid, tLarge, wSmall, wMid, wLarge);
            using (var sky = DehazeCore.SkyMask(I))
                DehazeCore.RaiseInSky(tRaw, sky, tSky);
            DehazeCore.Clamp01(tRaw);

            using var tRef = Refiners.FastGuided(I, tRaw, refine, eps, fast);
            DehazeCore.Clamp01(tRef);

            using var recovered = DehazeCore.Recover(I, tRef, A, tmin, chromaFloor);
            using var boosted = ApplySpectralGain(recovered, tRef, gain);
            using var toned = DehazeCore.RestoreTone(boosted, tone);
            return DehazeCore.LimitColorfulness(toned, input.Mat, color);
        }

        private static Mat Transmission(Mat normByA, int radius, double omega)
        {
            using var dark = DehazeCore.DarkChannel(normByA, Math.Max(1, radius));
            var t = new Mat();
            dark.ConvertTo(t, DepthType.Cv32F, -omega, 1.0);
            DehazeCore.Clamp01(t);
            return t;
        }

        private static Mat WeightedAverage(Mat t1, Mat t2, Mat t3, Mat w1, Mat w2, Mat w3)
        {
            using var sumW = new Mat();
            CvInvoke.Add(w1, w2, sumW);
            CvInvoke.Add(sumW, w3, sumW);
            CvInvoke.Add(sumW, new ScalarArray(1e-6), sumW);

            using var num = new Mat();
            using (var tmp = new Mat())
            {
                CvInvoke.Multiply(w1, t1, num);
                CvInvoke.Multiply(w2, t2, tmp);
                CvInvoke.Add(num, tmp, num);
                CvInvoke.Multiply(w3, t3, tmp);
                CvInvoke.Add(num, tmp, num);
            }

            var fused = new Mat();
            CvInvoke.Divide(num, sumW, fused);
            return fused;
        }

        private static Mat ApplySpectralGain(Mat bgr01, Mat t, double gain)
        {
            double haze = Math.Clamp(1.0 - CvInvoke.Mean(t).V0, 0.0, 1.0);
            double amount = Math.Clamp(gain * haze, 0.0, 0.18);
            if (amount <= 1e-4)
                return bgr01.Clone();

            using var fine = DehazeCore.FastGaussian(bgr01, 1.2);
            using var coarse = DehazeCore.FastGaussian(bgr01, 12.0);
            using var band = new Mat();
            CvInvoke.Subtract(fine, coarse, band);

            var boosted = new Mat();
            CvInvoke.AddWeighted(bgr01, 1.0, band, amount, 0.0, boosted);
            return DeHazeCPU.Clip(boosted);
        }
    }
}
