using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.CvEnum;

namespace SimpleDeHaze.Methods
{
    internal readonly record struct GamutProjectionSummary(
        double ProjectedPixelFraction,
        double MeanAlpha,
        double InvalidChannelFractionBefore,
        double InvalidChannelFractionAfter);

    internal sealed class GamutProjectionResult : IDisposable
    {
        public Mat Result { get; }
        public Mat Alpha { get; }
        public GamutProjectionSummary Summary { get; }

        public GamutProjectionResult(Mat result, Mat alpha, GamutProjectionSummary summary)
        {
            Result = result;
            Alpha = alpha;
            Summary = summary;
        }

        public void Dispose()
        {
            Result.Dispose();
            Alpha.Dispose();
        }
    }

    /// <summary>
    /// Проекция кандидата восстановления на RGB-куб вдоль луча от наблюдаемого пикселя:
    /// P(α)=I+α(J_raw-I), α∈[0,1]. Поскольку I уже лежит в [0,1]³, допустимое множество α —
    /// непустой отрезок. Берётся его максимальная верхняя граница, поэтому изменение ослабляется
    /// только настолько, насколько требует gamut, а соотношение поканальных приращений сохраняется.
    /// </summary>
    internal static class GamutProjector
    {
        public static GamutProjectionResult ProjectFromInput(Mat input01, Mat candidate, double boundaryMargin = 0.995)
        {
            if (input01.Depth != DepthType.Cv32F || candidate.Depth != DepthType.Cv32F ||
                input01.NumberOfChannels != 3 || candidate.NumberOfChannels != 3 || input01.Size != candidate.Size)
                throw new ArgumentException("Gamut projection expects equally-sized BGR float Mats.");

            boundaryMargin = Math.Clamp(boundaryMargin, 0.0, 1.0);
            int pixels = input01.Rows * input01.Cols;
            var input = new float[pixels * 3];
            var raw = new float[pixels * 3];
            var output = new float[pixels * 3];
            var alphaMap = new float[pixels];
            input01.CopyTo(input);
            candidate.CopyTo(raw);

            long projected = 0, invalidBefore = 0, invalidAfter = 0;
            double alphaSum = 0;
            for (int pixel = 0; pixel < pixels; pixel++)
            {
                int offset = pixel * 3;
                double alpha = 1.0;
                bool finite = true;
                for (int channel = 0; channel < 3; channel++)
                {
                    double source = input[offset + channel];
                    double target = raw[offset + channel];
                    if (!double.IsFinite(source) || !double.IsFinite(target))
                    {
                        finite = false;
                        invalidBefore++;
                        continue;
                    }
                    if (target < 0.0 || target > 1.0) invalidBefore++;
                    alpha = Math.Min(alpha, MaxStep(source, target - source));
                }
                if (!finite) alpha = 0.0;
                alpha = Math.Clamp(alpha, 0.0, 1.0);
                if (alpha < 1.0 - 1e-7)
                {
                    projected++;
                    alpha *= boundaryMargin;
                }
                alphaMap[pixel] = (float)alpha;
                alphaSum += alpha;

                for (int channel = 0; channel < 3; channel++)
                {
                    double source = input[offset + channel];
                    double target = raw[offset + channel];
                    double value = finite ? source + alpha * (target - source) : source;
                    value = Math.Clamp(value, 0.0, 1.0);
                    if (!double.IsFinite(value) || value < -1e-7 || value > 1.0 + 1e-7) invalidAfter++;
                    output[offset + channel] = (float)value;
                }
            }

            var summary = new GamutProjectionSummary(
                pixels == 0 ? 0 : projected / (double)pixels,
                pixels == 0 ? 1 : alphaSum / pixels,
                pixels == 0 ? 0 : invalidBefore / (double)(pixels * 3),
                pixels == 0 ? 0 : invalidAfter / (double)(pixels * 3));
            return new GamutProjectionResult(ToMat(output, input01.Rows, input01.Cols, 3),
                ToMat(alphaMap, input01.Rows, input01.Cols, 1), summary);
        }

        internal static double MaxStep(double source, double delta)
        {
            if (delta > 1e-15) return Math.Max(0.0, (1.0 - source) / delta);
            if (delta < -1e-15) return Math.Max(0.0, source / -delta);
            return 1.0;
        }

        private static Mat ToMat(float[] data, int rows, int cols, int channels)
        {
            var result = new Mat(rows, cols, DepthType.Cv32F, channels);
            Marshal.Copy(data, 0, result.DataPointer, data.Length);
            return result;
        }
    }
}
