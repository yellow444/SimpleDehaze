using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using SimpleDeHaze.Methods;

namespace SimpleDeHaze.Tests;

public sealed class HcvA2crTests
{
    public void ExactDualGainInverse_RecoversAtmosphericGroundTruth()
    {
        const int rows = 10, cols = 100, pixels = rows * cols;
        double[] air = { 0.76, 0.84, 0.93 };
        var random = new Random(8101);
        var clean = new float[pixels * 3]; var hazy = new float[pixels * 3]; var transmission = new float[pixels];
        for (int i = 0; i < pixels; i++)
        {
            double t = 0.1 + 0.85 * random.NextDouble(); transmission[i] = (float)t;
            for (int channel = 0; channel < 3; channel++)
            {
                int index = i * 3 + channel;
                clean[index] = (float)(0.02 + 0.96 * random.NextDouble());
                hazy[index] = (float)(t * clean[index] + (1 - t) * air[channel]);
            }
        }

        using var input = ToMat(hazy, rows, cols, 3);
        using var tMap = ToMat(transmission, rows, cols, 1);
        using var variance = new Mat(rows, cols, DepthType.Cv32F, 1); variance.SetTo(new MCvScalar(0));
        var estimate = new AirlightEstimate(new MCvScalar(air[0], air[1], air[2]),
            new[] { 0.0, 0.0, 0.0 }, new[] { new MCvScalar(air[0], air[1], air[2]) });
        var options = new HcvA2crRecoveryOptions(0.02, 0, 0,
            false, false, false, true, 0, 0, 0);
        using var result = HcvA2crRecovery.Recover(input, tMap, variance, estimate, options);
        var actual = new float[clean.Length]; result.LinearResult.CopyTo(actual);
        double maximum = actual.Zip(clean, (x, y) => Math.Abs(x - y)).Max();
        TestAssert.InRange(maximum, 0, 3e-5);
        TestAssert.InRange(result.Diagnostics.InvalidChannelFractionAfter, 0, 1e-12);
    }

    public void FeasibleProjector_KeepsRandomHcvProposalsInsideRgbCube()
    {
        var random = new Random(8102);
        for (int sample = 0; sample < 20_000; sample++)
        {
            double[] air = Enumerable.Range(0, 3).Select(_ => 0.2 + 0.79 * random.NextDouble()).ToArray();
            float[] input = Enumerable.Range(0, 3).Select(_ => (float)random.NextDouble()).ToArray();
            double xb = input[0] / air[0], xg = input[1] / air[1], xr = input[2] / air[2];
            double value = Math.Max(xb, Math.Max(xg, xr));
            var p = new float[3]; var q = new float[3];
            for (int channel = 0; channel < 3; channel++)
            {
                p[channel] = (float)(air[channel] * (value - 1));
                q[channel] = (float)(input[channel] - air[channel] - p[channel]);
            }
            double proposedV = 1 + 11 * random.NextDouble();
            double proposedC = 1 + 11 * random.NextDouble();
            var projected = HcvA2crFeasibleProjector.Project(air, p, q, proposedV, proposedC, 12);
            TestAssert.InRange(projected.B, -3e-7, 1 + 3e-7);
            TestAssert.InRange(projected.G, -3e-7, 1 + 3e-7);
            TestAssert.InRange(projected.R, -3e-7, 1 + 3e-7);
            double identityDistance = Math.Sqrt((proposedV - 1) * (proposedV - 1) +
                                                (proposedC - 1) * (proposedC - 1));
            TestAssert.True(projected.Distance <= identityDistance + 1e-7);
        }
    }

    public void AtrousZeroBoost_IsIdentityAndConstantSafe()
    {
        const int rows = 17, cols = 19, pixels = rows * cols;
        var random = new Random(8103);
        var source = Enumerable.Range(0, pixels * 3)
            .Select(_ => (float)(0.08 + 0.72 * random.NextDouble())).ToArray();
        using var input = ToMat(source, rows, cols, 3);
        using var gains = new Mat(rows, cols, DepthType.Cv32F, 1); gains.SetTo(new MCvScalar(1));
        using var sigma = new Mat(rows, cols, DepthType.Cv32F, 1); sigma.SetTo(new MCvScalar(0));
        var airlight = new AirlightEstimate(new MCvScalar(0.86, 0.89, 0.92),
            new[] { 0.0, 0.0, 0.0 }, new[] { new MCvScalar(0.86, 0.89, 0.92) });
        using var result = StationaryAtrous.EnhanceHcvValue(input, gains, gains, sigma,
            airlight, 4, 3, 0, 2, 0, 0.04);
        var actual = new float[source.Length]; result.LinearResult.CopyTo(actual);
        TestAssert.InRange(actual.Zip(source, (x, y) => Math.Abs(x - y)).Max(), 0, 3e-6);

        var constant = Enumerable.Repeat(0.4f, pixels * 3).ToArray();
        using var constantInput = ToMat(constant, rows, cols, 3);
        using var constantResult = StationaryAtrous.EnhanceHcvValue(constantInput, gains, gains, sigma,
            airlight, 4, 3, 1.5, 2, 0, 0.04);
        var constantActual = new float[constant.Length]; constantResult.LinearResult.CopyTo(constantActual);
        TestAssert.InRange(constantActual.Zip(constant, (x, y) => Math.Abs(x - y)).Max(), 0, 3e-6);
    }

    public void FamilyBalancedFusion_PreventsCorrelatedDcpMajority()
    {
        var owned = new List<Mat>();
        try
        {
            Mat Constant(float value)
            {
                var map = new Mat(3, 4, DepthType.Cv32F, 1); map.SetTo(new MCvScalar(value));
                owned.Add(map); return map;
            }
            var dcp = Enumerable.Range(0, 5)
                .Select(_ => (Transmission: Constant(0.2f), Weight: 1.0)).ToArray();
            var cap = new[] { (Transmission: Constant(0.7f), Weight: 1.0) };
            var haze = new[] { (Transmission: Constant(0.8f), Weight: 1.0) };
            using var balanced = OpticalDepthFusion.FuseFamilies(
                new IReadOnlyList<(Mat Transmission, double Weight)>[] { dcp, cap, haze }, 0.05, 1, 0);
            var values = new float[12]; balanced.Transmission.CopyTo(values);
            TestAssert.InRange(values.Min(), 0.69999, 0.70001);
            TestAssert.InRange(values.Max(), 0.69999, 0.70001);
            var sigma = new float[12]; balanced.SigmaDepth.CopyTo(sigma);
            TestAssert.True(sigma.All(value => value > 0 && float.IsFinite(value)));
        }
        finally
        {
            foreach (var map in owned) map.Dispose();
        }
    }

    public void RefinedTransmissionVariance_UsesTheRefinedMeanAndDepthSigma()
    {
        float[] refinedValues = { 0.2f, 0.4f, 0.8f };
        float[] sigmaValues = { 0.5f, 0.25f, 0.125f };
        using var refined = ToMat(refinedValues, 1, 3, 1);
        using var sigmaDepth = ToMat(sigmaValues, 1, 3, 1);
        using var variance = OpticalDepthFusion.PropagateVariance(refined, sigmaDepth, 0.01);
        var actual = new float[3]; variance.CopyTo(actual);
        for (int i = 0; i < actual.Length; i++)
        {
            double expected = refinedValues[i] * refinedValues[i] * sigmaValues[i] * sigmaValues[i] + 0.0001;
            TestAssert.InRange(Math.Abs(actual[i] - expected), 0, 2e-8);
        }
    }

    public void RegisteredPipelines_ProduceFiniteFeasibleOutput()
    {
        using var input = RandomImage(36, 28, 8104);
        foreach (IDeHazeMethod method in new IDeHazeMethod[]
                 { new HcvA2crMethod(), new HcvRgbA2crFusionMethod(), new HcvA2crUtawMethod() })
        {
            var parameters = method.Parameters.ToDictionary(definition => definition.Key, definition => definition.Default);
            parameters["patch"] = 3; parameters["refine"] = 5; parameters["radius"] = 2;
            using var output = method.Process(input, parameters);
            var values = new float[output.Rows * output.Cols * output.NumberOfChannels]; output.CopyTo(values);
            TestAssert.True(values.All(float.IsFinite));
            TestAssert.InRange(values.Min(), 0, 1); TestAssert.InRange(values.Max(), 0, 1);
        }
    }

    private static Image<Bgr, byte> RandomImage(int width, int height, int seed)
    {
        var input = new Image<Bgr, byte>(width, height);
        var bytes = new byte[width * height * 3]; new Random(seed).NextBytes(bytes);
        Marshal.Copy(bytes, 0, input.Mat.DataPointer, bytes.Length);
        return input;
    }

    private static Mat ToMat(float[] data, int rows, int cols, int channels)
    {
        var mat = new Mat(rows, cols, DepthType.Cv32F, channels);
        Marshal.Copy(data, 0, mat.DataPointer, data.Length);
        return mat;
    }
}
