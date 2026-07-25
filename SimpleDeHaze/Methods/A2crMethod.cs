using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Airlight-Aligned, Uncertainty-Aware Convex Recovery. The novelty status is a research
    /// hypothesis: this implementation establishes numerical contracts, not a priority claim.
    /// </summary>
    public sealed class A2crMethod : IDeHazeMethod
    {
        public string Name => "A²CR-Dehaze (dual-gain recovery, эксперимент)";

        public string Description =>
            "Экспериментальный training-free recovery operator в линейном RGB.\n\n" +
            "d=I-A раскладывается относительно направления u=A/||A||: p=uuᵀd, q=(I-uuᵀ)d.\n" +
            "Вместо одного 1/t используются два аналитических gain:\n" +
            "J=A+g∥p+g⊥q,  g*=(S·t+U)/(S(t²+σt²)+N+U).\n\n" +
            "A оценивается bootstrap'ом; DCP/CAP/haze-line карты объединяются weighted median в\n" +
            "optical-depth D=-ln(t), а weighted MAD становится σt². Последний шаг движется от\n" +
            "безопасной точки (g∥,g⊥)=(1,1) к предложенным gains лишь до границы RGB-куба.\n" +
            "При tv>0 совместный primal-dual solver оптимизирует оба gain-поля внутри точного\n" +
            "per-pixel RGB-feasible многоугольника, с edge-aware TV и coupling.\n\n" +
            "Это кандидат на исследовательский метод, не доказанный claim мировой новизны.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",   "ω DCP",                              0.3, 0.99, 0.95, search: true),
            new ParamDef("patch",   "Радиус DCP",                         1,   15,   7, 1, isInt: true),
            new ParamDef("top",     "Top для A, %",                       0.01, 2.0, 0.1),
            new ParamDef("min",     "t_min / максимум gain",              0.02, 0.5, 0.08),
            new ParamDef("boots",   "Bootstrap samples (1/3/5)",          1,    5,   3, 2, isInt: true, tunable: false),
            new ParamDef("diverse", "Добавить CAP + haze-lines (0/1)",     0,    1,   1, 1, isInt: true, tunable: false),
            new ParamDef("tunc",    "Множитель uncertainty(t)",           0,    3,   1),
            new ParamDef("tufloor", "Минимальная σ_t",                    0, 0.10, 0.01),
            new ParamDef("noise",   "σ шума linear RGB",                  0, 0.05, 0.006),
            new ParamDef("airunc",  "Множитель uncertainty(A)",           0,   10,   1),
            new ParamDef("radius",  "Радиус локальной энергии S",         0,   20,   4, 1, isInt: true),
            new ParamDef("couple",  "Связь μ между dual gains",           0,  0.1,   0),
            new ParamDef("noiseon", "Учитывать шум (0/1)",                0,    1,   1, 1, isInt: true, tunable: false),
            new ParamDef("tuncon",  "Учитывать uncertainty(t) (0/1)",     0,    1,   1, 1, isInt: true, tunable: false),
            new ParamDef("auncon",  "Учитывать uncertainty(A) (0/1)",     0,    1,   1, 1, isInt: true, tunable: false),
            new ParamDef("feasible","RGB-feasible projection (0/1)",      0,    1,   1, 1, isInt: true, tunable: false),
            new ParamDef("tv",      "Вес TV по gain-картам",              0, 0.10,   0),
            new ParamDef("tviter",  "Итерации joint primal-dual TV",       1,   80,  30, 1, isInt: true, tunable: false),
            new ParamDef("tvedge",  "Масштаб edge-aware TV",            0.01, 0.5, 0.08, log: true),
            new ParamDef("refine",  "Радиус guided refinement t",         3,  120,  40, 1, isInt: true),
            new ParamDef("eps",     "ε guided refinement",             1e-5, 1e-2, 1e-3, log: true),
            new ParamDef("fast",    "Ускорение guided filter",            1,    8,   4, 1, isInt: true, tunable: false),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            using var detailed = Execute(input, p);
            return detailed.SrgbResult.Clone();
        }

        internal static A2crExecution Execute(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            int patch = Math.Max(1, (int)p["patch"]), bootCount = NormalizeBootstrapCount((int)p["boots"]);
            double omega = p["omega"], tMin = p["min"], top = p["top"] / 100.0;
            using var linear = ColorSpace.NormalizeLinear(input);
            var airlight = AirlightBootstrap.Estimate(linear, patch, top, bootCount);

            var maps = new List<(Mat Transmission, double Weight)>();
            try
            {
                int[] offsets = bootCount == 1 ? new[] { 0 } : bootCount == 3 ? new[] { -1, 0, 1 } : new[] { -2, -1, 0, 1, 2 };
                foreach (int offset in offsets)
                {
                    int radius = Math.Max(1, patch + offset * Math.Max(1, patch / 3));
                    double localOmega = Math.Clamp(omega + 0.015 * offset, 0.1, 0.999);
                    maps.Add((DehazeCore.RawTransmission(linear, airlight.Value, localOmega, radius), offset == 0 ? 1.0 : 0.75));
                }
                if (p["diverse"] >= 0.5)
                {
                    maps.Add((CapTransmission(linear, tMin, patch), 1.0));
                    maps.Add((HazeLineTransmission(linear, airlight.Value, omega: 0.65, binsPerAxis: 18, tMin), 1.0));
                }

                using var fused = OpticalDepthFusion.Fuse(maps, tMin, p["tunc"], p["tufloor"]);
                using var tRefined = Refiners.FastGuided(linear, fused.Transmission, (int)p["refine"], p["eps"], (int)p["fast"]);
                DehazeCore.Clamp01(tRefined);
                using (var floor = new Mat(tRefined.Size, DepthType.Cv32F, 1))
                {
                    floor.SetTo(new MCvScalar(tMin));
                    CvInvoke.Max(tRefined, floor, tRefined);
                }

                var options = new A2crRecoveryOptions(tMin, p["noise"] * p["noise"], p["airunc"],
                    p["noiseon"] >= 0.5, p["tuncon"] >= 0.5, p["auncon"] >= 0.5,
                    p["feasible"] >= 0.5, (int)p["radius"], p["couple"], p["tv"],
                    (int)p["tviter"], p["tvedge"]);
                var recovery = A2crRecovery.Recover(linear, tRefined, fused.TransmissionVariance, airlight, options);
                var srgb = ColorSpace.ToSrgb(recovery.LinearResult);
                return new A2crExecution(srgb, recovery, tRefined.Clone(), fused.TransmissionVariance.Clone(),
                    fused.SigmaDepth.Clone(), airlight);
            }
            finally
            {
                foreach (var item in maps) item.Transmission.Dispose();
            }
        }

        private static int NormalizeBootstrapCount(int count) => count <= 1 ? 1 : count <= 3 ? 3 : 5;

        private static Mat CapTransmission(Mat inputLinear, double tMin, int radius)
        {
            using var srgb = ColorSpace.ToSrgb(inputLinear);
            using var hsv = new Mat();
            CvInvoke.CvtColor(srgb, hsv, ColorConversion.Bgr2Hsv);
            var channels = hsv.Split();
            using var saturation = channels[1]; using var value = channels[2]; channels[0].Dispose();
            using var depth = new Mat();
            CvInvoke.AddWeighted(value, 0.959710, saturation, -0.780245, 0.121779, depth, DepthType.Cv32F);
            using var element = CvInvoke.GetStructuringElement(ElementShape.Rectangle,
                new System.Drawing.Size(2 * radius + 1, 2 * radius + 1), new System.Drawing.Point(-1, -1));
            CvInvoke.Erode(depth, depth, element, new System.Drawing.Point(-1, -1), 1, BorderType.Reflect101, default);
            var transmission = new Mat();
            depth.ConvertTo(transmission, DepthType.Cv32F, -1.0);
            CvInvoke.Exp(transmission, transmission);
            using var floor = new Mat(transmission.Size, DepthType.Cv32F, 1); floor.SetTo(new MCvScalar(tMin));
            CvInvoke.Max(transmission, floor, transmission); DehazeCore.Clamp01(transmission);
            return transmission;
        }

        private static Mat HazeLineTransmission(Mat inputLinear, MCvScalar airlight, double omega, int binsPerAxis, double tMin)
        {
            int pixels = inputLinear.Rows * inputLinear.Cols;
            var input = new float[pixels * 3]; inputLinear.CopyTo(input);
            int bins = binsPerAxis * binsPerAxis * binsPerAxis;
            var index = new int[pixels]; var radius = new float[pixels]; var maximum = new float[bins];
            double[] a = { airlight.V0, airlight.V1, airlight.V2 };
            for (int i = 0; i < pixels; i++)
            {
                int j = i * 3;
                float db = input[j] - (float)a[0], dg = input[j + 1] - (float)a[1], dr = input[j + 2] - (float)a[2];
                float length = MathF.Sqrt(db * db + dg * dg + dr * dr); radius[i] = length;
                if (length < 1e-7f) { index[i] = -1; continue; }
                int qb = Math.Clamp((int)((db / length + 1) * 0.5 * binsPerAxis), 0, binsPerAxis - 1);
                int qg = Math.Clamp((int)((dg / length + 1) * 0.5 * binsPerAxis), 0, binsPerAxis - 1);
                int qr = Math.Clamp((int)((dr / length + 1) * 0.5 * binsPerAxis), 0, binsPerAxis - 1);
                int bin = (qb * binsPerAxis + qg) * binsPerAxis + qr; index[i] = bin;
                maximum[bin] = Math.Max(maximum[bin], length);
            }
            var output = new float[pixels];
            for (int i = 0; i < pixels; i++)
            {
                float raw = index[i] >= 0 && maximum[index[i]] > 1e-7 ? radius[i] / maximum[index[i]] : 1;
                output[i] = Math.Clamp(1f - (float)omega * (1f - raw), (float)tMin, 1f);
            }
            return DehazeCore.MatFromFloats(output, inputLinear.Rows, inputLinear.Cols);
        }
    }

    internal sealed class A2crExecution : IDisposable
    {
        public Mat SrgbResult { get; }
        public A2crRecoveryResult Recovery { get; }
        public Mat Transmission { get; }
        public Mat TransmissionVariance { get; }
        public Mat SigmaDepth { get; }
        public AirlightEstimate Airlight { get; }

        public A2crExecution(Mat srgbResult, A2crRecoveryResult recovery, Mat transmission,
            Mat transmissionVariance, Mat sigmaDepth, AirlightEstimate airlight)
        {
            SrgbResult = srgbResult; Recovery = recovery; Transmission = transmission;
            TransmissionVariance = transmissionVariance; SigmaDepth = sigmaDepth; Airlight = airlight;
        }

        public void Dispose()
        {
            SrgbResult.Dispose(); Recovery.Dispose(); Transmission.Dispose(); TransmissionVariance.Dispose(); SigmaDepth.Dispose();
        }
    }
}
