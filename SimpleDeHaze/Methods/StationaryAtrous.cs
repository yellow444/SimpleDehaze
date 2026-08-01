using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.CvEnum;

namespace SimpleDeHaze.Methods
{
    internal readonly record struct UtawDiagnostics(
        double MeanReliability, double MeanAbsoluteValueDelta, double GamutLimitedPixelFraction);

    internal sealed class UtawResult : IDisposable
    {
        public Mat LinearResult { get; }
        public UtawDiagnostics Diagnostics { get; }

        public UtawResult(Mat linearResult, UtawDiagnostics diagnostics)
        {
            LinearResult = linearResult;
            Diagnostics = diagnostics;
        }

        public void Dispose() => LinearResult.Dispose();
    }

    /// <summary>
    /// Undecimated B3-spline à trous decomposition. The transform itself is established prior art;
    /// the experimental part is the coefficient reliability derived jointly from recovered signal
    /// power, recovery gains and optical-depth disagreement.
    /// </summary>
    internal static class StationaryAtrous
    {
        private static readonly float[] B3 = { 1f / 16, 4f / 16, 6f / 16, 4f / 16, 1f / 16 };

        public static UtawResult EnhanceHcvValue(Mat recoveredLinear, Mat gainValue, Mat gainChroma,
            Mat sigmaDepth, AirlightEstimate airlight, int levels, int energyRadius,
            double detailBoost, double uncertaintyWeight, double noiseVariance, double bandLimit)
        {
            if (recoveredLinear.Depth != DepthType.Cv32F || recoveredLinear.NumberOfChannels != 3)
                throw new ArgumentException("UTAW input must be linear BGR float", nameof(recoveredLinear));
            levels = Math.Clamp(levels, 1, 5);
            energyRadius = Math.Clamp(energyRadius, 0, 20);
            detailBoost = Math.Clamp(detailBoost, 0, 3);
            uncertaintyWeight = Math.Max(0, uncertaintyWeight);
            bandLimit = Math.Clamp(bandLimit, 0, 0.25);

            double[] a =
            {
                Math.Max(0.02, airlight.Value.V0),
                Math.Max(0.02, airlight.Value.V1),
                Math.Max(0.02, airlight.Value.V2),
            };
            using var value = NormalizedValue(recoveredLinear, a);
            var bands = new List<Mat>(levels);
            var current = value.Clone();
            try
            {
                for (int level = 0; level < levels; level++)
                {
                    var smooth = Smooth(current, 1 << level);
                    var band = new Mat();
                    CvInvoke.Subtract(current, smooth, band);
                    bands.Add(band);
                    current.Dispose();
                    current = smooth;
                }

                using var gainSquared = new Mat();
                CvInvoke.Multiply(gainValue, gainValue, gainSquared);
                using var sigmaSquared = new Mat();
                CvInvoke.Multiply(sigmaDepth, sigmaDepth, sigmaSquared);
                double normalizedNoise = Math.Max(0, noiseVariance) *
                    a.Select(channel => 1.0 / (channel * channel)).Average();

                using var enhancedValue = current.Clone();
                double reliabilitySum = 0;
                long reliabilityCount = (long)value.Rows * value.Cols * levels;
                for (int level = 0; level < levels; level++)
                {
                    using var squared = new Mat();
                    CvInvoke.Multiply(bands[level], bands[level], squared);
                    using var localPower = new Mat();
                    if (energyRadius > 0)
                    {
                        int size = 2 * energyRadius + 1;
                        CvInvoke.Blur(squared, localPower, new System.Drawing.Size(size, size),
                            new System.Drawing.Point(-1, -1));
                    }
                    else squared.CopyTo(localPower);

                    using var uncertainty = new Mat();
                    gainSquared.ConvertTo(uncertainty, DepthType.Cv32F, normalizedNoise);
                    using var depthTerm = new Mat();
                    sigmaSquared.ConvertTo(depthTerm, DepthType.Cv32F,
                        uncertaintyWeight * (1.0 + 0.35 * level));
                    CvInvoke.Add(uncertainty, depthTerm, uncertainty);

                    using var signalPower = new Mat();
                    CvInvoke.Subtract(localPower, uncertainty, signalPower);
                    using (var zero = new Mat(signalPower.Size, DepthType.Cv32F, 1))
                    {
                        zero.SetTo(new Emgu.CV.Structure.MCvScalar(0));
                        CvInvoke.Max(signalPower, zero, signalPower);
                    }
                    using var denominator = new Mat();
                    CvInvoke.Add(signalPower, uncertainty, denominator);
                    denominator.ConvertTo(denominator, DepthType.Cv32F, 1, 1e-8);
                    using var reliability = new Mat();
                    CvInvoke.Divide(signalPower, denominator, reliability);
                    reliabilitySum += CvInvoke.Sum(reliability).V0;

                    double scaleFraction = levels == 1 ? 1 : (double)level / (levels - 1);
                    double cap = detailBoost * (0.65 + 0.35 * scaleFraction);
                    using var delta = new Mat();
                    CvInvoke.Multiply(bands[level], reliability, delta, cap);
                    if (bandLimit > 0) DehazeCore.Clamp(delta, -bandLimit, bandLimit);
                    using var contribution = new Mat();
                    CvInvoke.Add(bands[level], delta, contribution);
                    CvInvoke.Add(enhancedValue, contribution, enhancedValue);
                }

                using var valueDelta = new Mat();
                CvInvoke.Subtract(enhancedValue, value, valueDelta);
                var output = ApplyFeasibleValueDelta(recoveredLinear, valueDelta, a,
                    out double meanAbsoluteDelta, out double limitedFraction);
                return new UtawResult(output, new UtawDiagnostics(
                    reliabilityCount > 0 ? reliabilitySum / reliabilityCount : 0,
                    meanAbsoluteDelta, limitedFraction));
            }
            finally
            {
                current.Dispose();
                foreach (var band in bands) band.Dispose();
            }
        }

        internal static Mat Smooth(Mat source, int spacing)
        {
            spacing = Math.Max(1, spacing);
            int length = 1 + 4 * spacing;
            var coefficients = new float[length];
            for (int i = 0; i < B3.Length; i++) coefficients[i * spacing] = B3[i];
            using var kernel = DehazeCore.MatFromFloats(coefficients, 1, length);
            var result = new Mat();
            CvInvoke.SepFilter2D(source, result, DepthType.Cv32F, kernel, kernel,
                new System.Drawing.Point(-1, -1), 0, BorderType.Reflect101);
            return result;
        }

        private static Mat NormalizedValue(Mat linear, IReadOnlyList<double> airlight)
        {
            var channels = linear.Split();
            try
            {
                using var b = new Mat(); using var g = new Mat(); using var r = new Mat();
                channels[0].ConvertTo(b, DepthType.Cv32F, 1.0 / airlight[0]);
                channels[1].ConvertTo(g, DepthType.Cv32F, 1.0 / airlight[1]);
                channels[2].ConvertTo(r, DepthType.Cv32F, 1.0 / airlight[2]);
                var value = new Mat();
                CvInvoke.Max(b, g, value); CvInvoke.Max(value, r, value);
                return value;
            }
            finally
            {
                foreach (var channel in channels) channel.Dispose();
            }
        }

        private static Mat ApplyFeasibleValueDelta(Mat input, Mat valueDelta, IReadOnlyList<double> airlight,
            out double meanAbsoluteDelta, out double limitedFraction)
        {
            int pixels = input.Rows * input.Cols;
            var source = new float[pixels * 3]; var delta = new float[pixels];
            input.CopyTo(source); valueDelta.CopyTo(delta);
            var output = new float[source.Length];
            double sum = 0; long limited = 0;
            for (int i = 0; i < pixels; i++)
            {
                int j = i * 3;
                double requested = delta[i], applied = requested;
                if (requested > 0)
                {
                    for (int channel = 0; channel < 3; channel++)
                        applied = Math.Min(applied, (1.0 - source[j + channel]) / airlight[channel]);
                }
                else if (requested < 0)
                {
                    for (int channel = 0; channel < 3; channel++)
                        applied = Math.Max(applied, -source[j + channel] / airlight[channel]);
                }
                if (Math.Abs(applied - requested) > 1e-8) limited++;
                sum += Math.Abs(applied);
                for (int channel = 0; channel < 3; channel++)
                    output[j + channel] = (float)Math.Clamp(
                        source[j + channel] + airlight[channel] * applied, 0, 1);
            }
            meanAbsoluteDelta = pixels > 0 ? sum / pixels : 0;
            limitedFraction = pixels > 0 ? limited / (double)pixels : 0;
            var result = new Mat(input.Rows, input.Cols, DepthType.Cv32F, 3);
            Marshal.Copy(output, 0, result.DataPointer, output.Length);
            return result;
        }
    }
}
