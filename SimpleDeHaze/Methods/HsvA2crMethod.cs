using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// HSV companion to A²CR. The atmospheric inverse is still solved in linear RGB; this
    /// stage only performs a bounded, uncertainty-aware recovery in cylindrical HSV space.
    /// Hue is represented by a 2-D unit vector, so the 0/360 degree seam is never averaged
    /// as an ordinary scalar.
    /// </summary>
    public sealed class HsvA2crMethod : IDeHazeMethod
    {
        private static readonly A2crMethod BaseMethod = new();

        public string Name => "HSV²CR (круговая цветность + A²CR, эксперимент)";

        public string Description =>
            "Экспериментальное HSV-дополнение к A²CR. Физическая модель дымки и оценка t,A " +
            "остаются в линейном RGB. Полученный A²CR-результат служит предложением, а HSV²CR " +
            "отдельно принимает или подавляет изменения яркости V и круговой цветности " +
            "c=(S cos H,S sin H) по локальной неопределённости transmission.\n\n" +
            "w=SΔ/(SΔ+λU+ε), x=x_input+w(x_A²CR-x_input). Hue не усредняется как число: " +
            "359° и 1° смешиваются через 0°. Выпуклая интерполяция V и диска цветности " +
            "конструктивно сохраняет 0≤V,S≤1. Это исследовательский оператор, а не заявление " +
            "о физической линейности HSV или доказанной мировой новизне.";

        public IReadOnlyList<ParamDef> Parameters { get; } = BaseMethod.Parameters.Concat(new[]
        {
            new ParamDef("hsvv",     "HSV²CR: сила восстановления V",       0, 1, 1, search: true),
            new ParamDef("hsvc",     "HSV²CR: сила цветности",             0, 1, 0.9, search: true),
            new ParamDef("hsvuv",    "HSV²CR: штраф uncertainty для V",    0, 10, 1),
            new ParamDef("hsvuc",    "HSV²CR: штраф uncertainty цвета",    0, 10, 2),
            new ParamDef("hsvsat",   "HSV²CR: порог ненадёжного Hue",   0.005, 0.3, 0.05, log: true),
            new ParamDef("hsvfloor", "HSV²CR: стабилизатор риска",       1e-8, 1e-2, 1e-6, log: true),
        }).ToArray();

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            using var execution = A2crMethod.Execute(input, p);
            using var inputSrgb = DehazeCore.Normalize(input);
            var options = new HsvA2crOptions(
                p["hsvv"], p["hsvc"], p["hsvuv"], p["hsvuc"], p["hsvsat"], p["hsvfloor"]);
            return HsvA2crRecovery.Recover(inputSrgb, execution.SrgbResult,
                execution.Transmission, execution.TransmissionVariance, options);
        }
    }

    internal sealed record HsvA2crOptions(
        double ValueStrength,
        double ChromaStrength,
        double ValueUncertaintyPenalty,
        double ChromaUncertaintyPenalty,
        double HueSaturationFloor,
        double RiskFloor);

    internal readonly record struct HsvA2crPixel(
        double HueDegrees,
        double Saturation,
        double Value,
        double ValueWeight,
        double ChromaWeight);

    internal static class HsvA2crRecovery
    {
        public static Mat Recover(Mat inputSrgb, Mat proposalSrgb, Mat transmission,
            Mat transmissionVariance, HsvA2crOptions options)
        {
            Validate(inputSrgb, proposalSrgb, transmission, transmissionVariance);
            int rows = inputSrgb.Rows, cols = inputSrgb.Cols, pixels = rows * cols;

            using var inputHsv = new Mat();
            using var proposalHsv = new Mat();
            CvInvoke.CvtColor(inputSrgb, inputHsv, ColorConversion.Bgr2Hsv);
            CvInvoke.CvtColor(proposalSrgb, proposalHsv, ColorConversion.Bgr2Hsv);

            var source = new float[pixels * 3];
            var proposal = new float[pixels * 3];
            var t = new float[pixels];
            var variance = new float[pixels];
            inputHsv.CopyTo(source); proposalHsv.CopyTo(proposal);
            transmission.CopyTo(t); transmissionVariance.CopyTo(variance);

            var output = new float[pixels * 3];
            for (int i = 0; i < pixels; i++)
            {
                int j = i * 3;
                double ti = Math.Clamp(t[i], 1e-4f, 1f);
                double relativeUncertainty = Math.Max(0, variance[i]) /
                    (ti * ti + Math.Max(0, variance[i]) + 1e-12);
                var recovered = BlendPixel(
                    source[j], source[j + 1], source[j + 2],
                    proposal[j], proposal[j + 1], proposal[j + 2],
                    relativeUncertainty, options);
                output[j] = (float)recovered.HueDegrees;
                output[j + 1] = (float)recovered.Saturation;
                output[j + 2] = (float)recovered.Value;
            }

            using var outputHsv = new Mat(rows, cols, DepthType.Cv32F, 3);
            Marshal.Copy(output, 0, outputHsv.DataPointer, output.Length);
            var result = new Mat();
            CvInvoke.CvtColor(outputHsv, result, ColorConversion.Hsv2Bgr);
            DehazeCore.Clamp01(result);
            return result;
        }

        internal static HsvA2crPixel BlendPixel(double inputHue, double inputSaturation, double inputValue,
            double proposalHue, double proposalSaturation, double proposalValue,
            double relativeUncertainty, HsvA2crOptions options)
        {
            double hi = DegreesToRadians(WrapHue(inputHue));
            double hp = DegreesToRadians(WrapHue(proposalHue));
            double si = Math.Clamp(inputSaturation, 0, 1);
            double sp = Math.Clamp(proposalSaturation, 0, 1);
            double vi = Math.Clamp(inputValue, 0, 1);
            double vp = Math.Clamp(proposalValue, 0, 1);

            double cix = si * Math.Cos(hi), ciy = si * Math.Sin(hi);
            double cpx = sp * Math.Cos(hp), cpy = sp * Math.Sin(hp);
            double dx = cpx - cix, dy = cpy - ciy, dv = vp - vi;
            double signalV = dv * dv;
            double signalC = dx * dx + dy * dy;

            double uncertainty = Math.Clamp(double.IsFinite(relativeUncertainty) ? relativeUncertainty : 1, 0, 1);
            double satFloor = Math.Clamp(options.HueSaturationFloor, 1e-6, 1);
            double hueInstability = Math.Clamp((satFloor - Math.Min(si, sp)) / satFloor, 0, 1);
            double floor = Math.Max(1e-12, options.RiskFloor);
            double valueWeight = Math.Clamp(options.ValueStrength, 0, 1) * signalV /
                (signalV + Math.Max(0, options.ValueUncertaintyPenalty) * uncertainty + floor);
            double chromaWeight = Math.Clamp(options.ChromaStrength, 0, 1) * signalC /
                (signalC + Math.Max(0, options.ChromaUncertaintyPenalty) *
                    (uncertainty + hueInstability * hueInstability) + floor);

            double value = Math.Clamp(vi + valueWeight * dv, 0, 1);
            double cx = cix + chromaWeight * dx, cy = ciy + chromaWeight * dy;
            double saturation = Math.Clamp(Math.Sqrt(cx * cx + cy * cy), 0, 1);
            double hue = saturation <= 1e-12
                ? WrapHue(inputHue)
                : WrapHue(Math.Atan2(cy, cx) * 180.0 / Math.PI);
            return new HsvA2crPixel(hue, saturation, value, valueWeight, chromaWeight);
        }

        internal static double CircularDistanceDegrees(double a, double b)
        {
            double d = Math.Abs(WrapHue(a) - WrapHue(b));
            return Math.Min(d, 360 - d);
        }

        private static double WrapHue(double hue)
        {
            if (!double.IsFinite(hue)) return 0;
            hue %= 360;
            return hue < 0 ? hue + 360 : hue;
        }

        private static double DegreesToRadians(double degrees) => degrees * Math.PI / 180.0;

        private static void Validate(Mat input, Mat proposal, Mat transmission, Mat variance)
        {
            if (input.Depth != DepthType.Cv32F || input.NumberOfChannels != 3 ||
                proposal.Depth != DepthType.Cv32F || proposal.NumberOfChannels != 3)
                throw new ArgumentException("HSV²CR expects BGR float input and proposal");
            if (proposal.Size != input.Size || transmission.Size != input.Size || variance.Size != input.Size ||
                transmission.Depth != DepthType.Cv32F || transmission.NumberOfChannels != 1 ||
                variance.Depth != DepthType.Cv32F || variance.NumberOfChannels != 1)
                throw new ArgumentException("HSV²CR maps and proposal must match the input");
        }
    }
}
