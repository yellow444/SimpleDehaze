using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using SimpleDeHaze.Methods;

namespace SimpleDeHaze.Tests;

public sealed class PhysicalCoreTests
{
    public void SrgbCurve_RoundTripsAllByteValues()
    {
        using var src = new Mat(1, 256, DepthType.Cv32F, 1);
        var values = Enumerable.Range(0, 256).Select(x => x / 255f).ToArray();
        Marshal.Copy(values, 0, src.DataPointer, values.Length);
        using var linear = ColorSpace.ToLinear(src);
        using var roundTrip = ColorSpace.ToSrgb(linear);
        var actual = new float[values.Length];
        roundTrip.CopyTo(actual);
        TestAssert.InRange(actual.Zip(values).Max(x => Math.Abs(x.First - x.Second)), 0, 1e-5);
    }

    public void BoundaryConstraint_KeepsStandardRecoveryInsideRgbCube()
    {
        var random = new Random(20260728);
        for (int i = 0; i < 10_000; i++)
        {
            double[] a = Enumerable.Range(0, 3).Select(_ => 0.2 + 0.75 * random.NextDouble()).ToArray();
            double[] input = Enumerable.Range(0, 3).Select(_ => random.NextDouble()).ToArray();
            double t = Math.Max(1e-6, BoxBound(input, a));
            for (int c = 0; c < 3; c++)
                TestAssert.InRange((input[c] - a[c]) / t + a[c], -1e-9, 1 + 1e-9);
        }
    }

    public void ChromaSafeBound_KeepsActualRecoveryInsideRgbCube()
    {
        var random = new Random(20260728);
        for (int i = 0; i < 10_000; i++)
        {
            double[] a = Enumerable.Range(0, 3).Select(_ => 0.2 + 0.75 * random.NextDouble()).ToArray();
            double[] input = Enumerable.Range(0, 3).Select(_ => random.NextDouble()).ToArray();
            double tMin = 0.02 + 0.1 * random.NextDouble();
            double chroma = tMin + 0.5 * random.NextDouble();
            double t = DehazeCore.ChromaSafeAt(input[0], input[1], input[2], a, tMin, chroma);
            double mean = input.Zip(a).Average(x => x.First - x.Second);
            for (int c = 0; c < 3; c++)
            {
                double recovered = a[c] + mean / Math.Max(t, tMin) + ((input[c] - a[c]) - mean) / Math.Max(t, chroma);
                TestAssert.InRange(recovered, -2e-6, 1 + 2e-6);
            }
        }
    }

    public void IdentityMetrics_AreExactWithinNumericalTolerance()
    {
        using var bytes = new Mat(16, 16, DepthType.Cv8U, 3);
        bytes.SetTo(new MCvScalar(40, 100, 180));
        using var f = new Mat();
        bytes.ConvertTo(f, DepthType.Cv32F, 1.0 / 255.0);
        var report = Metrics.Evaluate(f, bytes, bytes);
        TestAssert.True(report.Psnr >= 90, $"PSNR={report.Psnr}");
        TestAssert.InRange(report.Ssim, 0.999, 1.001);
        TestAssert.InRange(report.Ciede2000, 0, 1e-9);
        // Для идеально плоского кадра обе оценки шума равны нулю; отношение определено как 0.
        TestAssert.InRange(report.FlatNoiseRatio, 0, 1e-9);
    }

    public void GamutProjection_PreservesRayAndKeepsEveryChannelInsideCube()
    {
        const int pixels = 512;
        var random = new Random(20260730);
        var inputValues = new float[pixels * 3];
        var candidateValues = new float[pixels * 3];
        for (int i = 0; i < inputValues.Length; i++)
        {
            inputValues[i] = (float)random.NextDouble();
            candidateValues[i] = (float)(-2.0 + 5.0 * random.NextDouble());
        }
        using var input = FloatMat(inputValues, 16, 32, 3);
        using var candidate = FloatMat(candidateValues, 16, 32, 3);
        using var projection = GamutProjector.ProjectFromInput(input, candidate, 1.0);
        var output = new float[inputValues.Length]; projection.Result.CopyTo(output);
        var alpha = new float[pixels]; projection.Alpha.CopyTo(alpha);

        for (int pixel = 0; pixel < pixels; pixel++)
        {
            TestAssert.InRange(alpha[pixel], 0, 1);
            int offset = pixel * 3;
            for (int channel = 0; channel < 3; channel++)
            {
                int i = offset + channel;
                TestAssert.InRange(output[i], -1e-6, 1 + 1e-6);
                double expected = inputValues[i] + alpha[pixel] * (candidateValues[i] - inputValues[i]);
                TestAssert.InRange(Math.Abs(output[i] - expected), 0, 2e-6);
            }
        }
        TestAssert.InRange(projection.Summary.InvalidChannelFractionAfter, 0, 1e-12);
        TestAssert.True(projection.Summary.ProjectedPixelFraction > 0.5);
    }

    public void ChromaticAnchor_ExactlyInvertsConstantHazeModelWithoutRegularization()
    {
        const int rows = 4, cols = 5;
        double[] clean = { 0.20, 0.32, 0.51 };
        double[] air = { 0.82, 0.75, 0.68 };
        const double t = 0.40;
        var observed = new MCvScalar(
            t * clean[0] + (1 - t) * air[0],
            t * clean[1] + (1 - t) * air[1],
            t * clean[2] + (1 - t) * air[2]);
        using var input = new Mat(rows, cols, DepthType.Cv32F, 3); input.SetTo(observed);
        using var transmission = new Mat(rows, cols, DepthType.Cv32F, 1); transmission.SetTo(new MCvScalar(t));
        var airField = air.Select(value =>
        {
            var channel = new Mat(rows, cols, DepthType.Cv32F, 1);
            channel.SetTo(new MCvScalar(value));
            return channel;
        }).ToArray();
        try
        {
            using var recovered = LocalHazeCore.RecoverChromaticAnchor(input, transmission, airField,
                new MCvScalar(air[0], air[1], air[2]), 0.01, 0.01, 1.0, 0.0, 0.0);
            var values = new float[rows * cols * 3]; recovered.CopyTo(values);
            for (int pixel = 0; pixel < rows * cols; pixel++)
                for (int channel = 0; channel < 3; channel++)
                    TestAssert.InRange(Math.Abs(values[pixel * 3 + channel] - clean[channel]), 0, 2e-5);
        }
        finally
        {
            foreach (var channel in airField) channel.Dispose();
        }
    }

    public void DenseHazeGate_IsSmoothMonotoneAndHasDeclaredEndpoints()
    {
        // density = 1-t: [0, .60, .70, .80, 1]. Для lo=.60/hi=.80 smoothstep даёт [0,0,.5,1,1].
        float[] transmissionValues = { 1.0f, 0.40f, 0.30f, 0.20f, 0.0f };
        using var transmission = FloatMat(transmissionValues, 1, transmissionValues.Length, 1);
        using var gate = ChromaticAnchorMethod.DenseHazeGate(transmission, 0.60, 0.80);
        var actual = new float[transmissionValues.Length]; gate.CopyTo(actual);
        double[] expected = { 0.0, 0.0, 0.5, 1.0, 1.0 };
        for (int i = 0; i < actual.Length; i++)
        {
            TestAssert.InRange(Math.Abs(actual[i] - expected[i]), 0, 2e-6);
            if (i > 0) TestAssert.True(actual[i] + 1e-7 >= actual[i - 1], "Dense-haze gate must be monotone in density.");
        }
    }

    private static double BoxBound(double[] input, double[] a)
    {
        double bound = 0;
        for (int c = 0; c < 3; c++)
        {
            bound = Math.Max(bound, (input[c] - a[c]) / Math.Max(1e-4, 1 - a[c]));
            bound = Math.Max(bound, (a[c] - input[c]) / Math.Max(1e-4, a[c]));
        }
        return Math.Clamp(bound, 0, 1);
    }

    private static Mat FloatMat(float[] values, int rows, int cols, int channels)
    {
        var result = new Mat(rows, cols, DepthType.Cv32F, channels);
        Marshal.Copy(values, 0, result.DataPointer, values.Length);
        return result;
    }
}
