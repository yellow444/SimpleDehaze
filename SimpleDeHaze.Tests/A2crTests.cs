using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using SimpleDeHaze.Methods;

namespace SimpleDeHaze.Tests;

public sealed class A2crTests
{
    public void RiskGain_RecoversClassicalInverseAndMinimizesQuadratic()
    {
        foreach (double t in new[] { 0.08, 0.2, 0.55, 1.0 })
        {
            double classical = A2crRisk.OptimalGain(0.4, t, 0, 0, 0, 0.05);
            TestAssert.InRange(Math.Abs(classical - 1.0 / t), 0, 1e-10);
        }

        var random = new Random(20260728);
        for (int i = 0; i < 5_000; i++)
        {
            double s = random.NextDouble(), t = 0.05 + 0.95 * random.NextDouble();
            double vt = 0.05 * random.NextDouble(), n = 0.05 * random.NextDouble(), u = 0.05 * random.NextDouble();
            double g = A2crRisk.OptimalGain(s, t, vt, n, u, 0.05);
            double risk = A2crRisk.Risk(g, s, t, vt, n, u);
            double left = A2crRisk.Risk(Math.Max(1, g - 1e-4), s, t, vt, n, u);
            double right = A2crRisk.Risk(Math.Min(20, g + 1e-4), s, t, vt, n, u);
            TestAssert.True(risk <= left + 1e-10 && risk <= right + 1e-10, $"g={g}, R={risk}/{left}/{right}");
        }
        double safe = A2crRisk.OptimalGain(0.1, 0.1, 0, 0, 1e12, 0.05);
        TestAssert.InRange(Math.Abs(safe - 1), 0, 1e-10);
    }

    public void FeasibleProjector_KeepsEveryRandomPixelInsideCube()
    {
        var random = new Random(20260728);
        for (int i = 0; i < 50_000; i++)
        {
            double b = random.NextDouble(), g = random.NextDouble(), r = random.NextDouble();
            double ab = random.NextDouble(), ag = random.NextDouble(), ar = random.NextDouble();
            double gp = 1 + 19 * random.NextDouble(), gq = 1 + 19 * random.NextDouble();
            var projected = A2crFeasibleProjector.Project(b, g, r, ab, ag, ar, gp, gq);
            TestAssert.InRange(projected.B, -2e-12, 1 + 2e-12);
            TestAssert.InRange(projected.G, -2e-12, 1 + 2e-12);
            TestAssert.InRange(projected.R, -2e-12, 1 + 2e-12);
            TestAssert.InRange(projected.Alpha, 0, 1);
            TestAssert.InRange(projected.GainParallel, 1, gp + 1e-12);
            TestAssert.InRange(projected.GainPerpendicular, 1, gq + 1e-12);
        }
    }

    public void EuclideanProjector_IsFeasibleAndNeverFartherThanRayProjection()
    {
        var random = new Random(20260730);
        for (int i = 0; i < 20_000; i++)
        {
            double b = random.NextDouble(), g = random.NextDouble(), r = random.NextDouble();
            double ab = random.NextDouble(), ag = random.NextDouble(), ar = random.NextDouble();
            double gp = 1 + 19 * random.NextDouble(), gq = 1 + 19 * random.NextDouble();
            var euclidean = A2crFeasibleProjector.ProjectEuclidean(b, g, r, ab, ag, ar, gp, gq, 20);
            var ray = A2crFeasibleProjector.Project(b, g, r, ab, ag, ar, gp, gq);
            TestAssert.InRange(euclidean.B, -3e-10, 1 + 3e-10);
            TestAssert.InRange(euclidean.G, -3e-10, 1 + 3e-10);
            TestAssert.InRange(euclidean.R, -3e-10, 1 + 3e-10);
            TestAssert.InRange(euclidean.GainParallel, 1 - 3e-10, 20 + 3e-10);
            TestAssert.InRange(euclidean.GainPerpendicular, 1 - 3e-10, 20 + 3e-10);
            double rayDistance = Math.Sqrt((ray.GainParallel - gp) * (ray.GainParallel - gp) +
                                           (ray.GainPerpendicular - gq) * (ray.GainPerpendicular - gq));
            TestAssert.True(euclidean.Distance <= rayDistance + 2e-9,
                $"Euclidean distance {euclidean.Distance} exceeds feasible ray {rayDistance}");
        }
    }

    public void CachedFeasiblePolygon_MatchesExactProjection()
    {
        var random = new Random(20260731);
        var polygonP = new float[A2crFeasibleProjector.MaximumPolygonVertices];
        var polygonQ = new float[A2crFeasibleProjector.MaximumPolygonVertices];
        for (int i = 0; i < 5_000; i++)
        {
            double b = random.NextDouble(), g = random.NextDouble(), r = random.NextDouble();
            double ab = random.NextDouble(), ag = random.NextDouble(), ar = random.NextDouble();
            double gp = 0.5 + 22 * random.NextDouble(), gq = 0.5 + 22 * random.NextDouble();
            int count = A2crFeasibleProjector.BuildPolygon(b, g, r, ab, ag, ar, 20, polygonP, polygonQ);
            var cached = A2crFeasibleProjector.ProjectToPolygon(gp, gq,
                polygonP.AsSpan(0, count), polygonQ.AsSpan(0, count));
            var exact = A2crFeasibleProjector.ProjectEuclidean(b, g, r, ab, ag, ar, gp, gq, 20);
            TestAssert.InRange(Math.Abs(cached.Distance - exact.Distance), 0, 2e-5);
            TestAssert.InRange(Math.Abs(cached.GainParallel - exact.GainParallel), 0, 3e-5);
            TestAssert.InRange(Math.Abs(cached.GainPerpendicular - exact.GainPerpendicular), 0, 3e-5);
        }
    }

    public void Recovery_NoUncertainty_EqualsScalarAtmosphericInverse()
    {
        using var input = FloatMat(new[] { 0.50f, 0.55f, 0.60f }, 1, 1, 3);
        using var t = FloatMat(new[] { 0.5f }, 1, 1, 1);
        using var variance = FloatMat(new[] { 0f }, 1, 1, 1);
        var air = new AirlightEstimate(new MCvScalar(0.8, 0.85, 0.9), new[] { 0d, 0d, 0d }, Array.Empty<MCvScalar>());
        var options = new A2crRecoveryOptions(0.05, 0, 0, false, false, false, false, 0, 0, 0, 0, 0.08);
        using var recovered = A2crRecovery.Recover(input, t, variance, air, options);
        var actual = new float[3]; recovered.LinearResult.CopyTo(actual);
        double[] expected = { 0.8 + (0.5 - 0.8) / 0.5, 0.85 + (0.55 - 0.85) / 0.5, 0.9 + (0.6 - 0.9) / 0.5 };
        for (int c = 0; c < 3; c++) TestAssert.InRange(Math.Abs(actual[c] - expected[c]), 0, 2e-6);
        TestAssert.InRange(Math.Abs(recovered.Diagnostics.MeanGainParallel - 2), 0, 2e-6);
        TestAssert.InRange(Math.Abs(recovered.Diagnostics.MeanGainPerpendicular - 2), 0, 2e-6);
    }

    public void OpticalDepthFusion_UsesMedianAndReportsDisagreement()
    {
        using var a = FloatMat(new[] { 0.8f }, 1, 1, 1);
        using var b = FloatMat(new[] { 0.5f }, 1, 1, 1);
        using var c = FloatMat(new[] { 0.2f }, 1, 1, 1);
        using var fused = OpticalDepthFusion.Fuse(new[] { (a, 1d), (b, 1d), (c, 1d) }, 0.01, 1, 0);
        var transmission = new float[1]; var variance = new float[1];
        fused.Transmission.CopyTo(transmission); fused.TransmissionVariance.CopyTo(variance);
        TestAssert.InRange(Math.Abs(transmission[0] - 0.5), 0, 1e-6);
        TestAssert.True(variance[0] > 0);
    }

    public void TvRefiner_PreservesConstantGain()
    {
        var source = Enumerable.Repeat(3f, 64).ToArray();
        var result = A2crTvRefiner.Denoise(source, 8, 8, 0.05, 20, 1, 20);
        TestAssert.InRange(result.Max(x => Math.Abs(x - 3f)), 0, 1e-6);
    }

    public void TvRefiner_ReducesTotalVariation()
    {
        var source = new float[16 * 16];
        for (int y = 0; y < 16; y++)
        for (int x = 0; x < 16; x++) source[y * 16 + x] = (x + y) % 2 == 0 ? 2 : 8;
        var result = A2crTvRefiner.Denoise(source, 16, 16, 0.2, 30, 1, 20);
        TestAssert.True(TotalVariation(result, 16, 16) < TotalVariation(source, 16, 16));
    }

    public void JointTvSolver_DecreasesObjectiveAndKeepsEveryPixelFeasible()
    {
        const int rows = 10, cols = 12, pixels = rows * cols;
        var random = new Random(20260801);
        var input = new float[pixels * 3];
        var initialP = new float[pixels]; var initialQ = new float[pixels];
        var ap = new float[pixels]; var bp = new float[pixels];
        var aq = new float[pixels]; var bq = new float[pixels];
        double[] air = { 0.82, 0.86, 0.90 };
        for (int i = 0; i < pixels; i++)
        {
            int j = i * 3;
            input[j] = (float)random.NextDouble(); input[j + 1] = (float)random.NextDouble(); input[j + 2] = (float)random.NextDouble();
            initialP[i] = (float)(1 + 8 * random.NextDouble());
            initialQ[i] = (float)(1 + 8 * random.NextDouble());
            ap[i] = (float)(0.02 + 0.2 * random.NextDouble());
            aq[i] = (float)(0.02 + 0.2 * random.NextDouble());
            bp[i] = ap[i] * (float)(1 + 3 * random.NextDouble());
            bq[i] = aq[i] * (float)(1 + 3 * random.NextDouble());
        }

        var result = A2crTvRefiner.SolveJoint(initialP, initialQ, ap, bp, aq, bq,
            input, air, rows, cols, 0.04, 0.08, 0.01, 60, 20, true);
        TestAssert.True(double.IsFinite(result.ObjectiveBefore) && double.IsFinite(result.ObjectiveAfter));
        TestAssert.True(result.ObjectiveAfter <= result.ObjectiveBefore + 1e-6,
            $"Joint objective increased: {result.ObjectiveBefore} -> {result.ObjectiveAfter}");
        TestAssert.True(result.Iterations > 0);
        for (int i = 0; i < pixels; i++)
        {
            int j = i * 3;
            var safe = A2crFeasibleProjector.Project(input[j], input[j + 1], input[j + 2],
                air[0], air[1], air[2], result.GainParallel[i], result.GainPerpendicular[i]);
            TestAssert.InRange(Math.Abs(safe.Alpha - 1), 0, 3e-5);
        }
    }

    public void Method_DefaultPipeline_ProducesFiniteFeasibleOutput()
    {
        using var input = new Image<Bgr, byte>(40, 30);
        var bytes = new byte[40 * 30 * 3]; new Random(77).NextBytes(bytes);
        Marshal.Copy(bytes, 0, input.Mat.DataPointer, bytes.Length);
        var method = new A2crMethod();
        var parameters = method.Parameters.ToDictionary(x => x.Key, x => x.Default);
        parameters["patch"] = 3; parameters["refine"] = 5; parameters["fast"] = 1;
        using var execution = A2crMethod.Execute(input, parameters);
        var values = new float[40 * 30 * 3]; execution.SrgbResult.CopyTo(values);
        TestAssert.True(values.All(float.IsFinite));
        TestAssert.InRange(values.Min(), 0, 1);
        TestAssert.InRange(values.Max(), 0, 1);
        TestAssert.InRange(execution.Recovery.Diagnostics.InvalidChannelFractionAfter, 0, 0);

        if (CudaBackend.IsAvailable)
        {
            parameters[CudaBackend.ParameterKey] = 1;
            using var gpuExecution = A2crMethod.Execute(input, parameters);
            using var difference = new Mat();
            CvInvoke.AbsDiff(execution.SrgbResult, gpuExecution.SrgbResult, difference);
            double maximum = 0;
            foreach (var channel in difference.Split())
            {
                double minimum = 0, channelMaximum = 0;
                var minPoint = new System.Drawing.Point(); var maxPoint = new System.Drawing.Point();
                CvInvoke.MinMaxLoc(channel, ref minimum, ref channelMaximum, ref minPoint, ref maxPoint);
                maximum = Math.Max(maximum, channelMaximum); channel.Dispose();
            }
            TestAssert.InRange(maximum, 0, 2e-4);
            TestAssert.InRange(gpuExecution.Recovery.Diagnostics.InvalidChannelFractionAfter, 0, 0);
        }
    }

    public void HsvRecovery_InterpolatesHueAcrossCircularSeam()
    {
        var options = new HsvA2crOptions(1, 0.5, 1, 1, 0.05, 1e-12);
        var recovered = HsvA2crRecovery.BlendPixel(359, 1, 0.5, 1, 1, 0.5, 0, options);
        TestAssert.InRange(HsvA2crRecovery.CircularDistanceDegrees(recovered.HueDegrees, 0), 0, 1e-4);
        TestAssert.True(HsvA2crRecovery.CircularDistanceDegrees(recovered.HueDegrees, 180) > 179);
        TestAssert.InRange(recovered.Saturation, 0, 1);
    }

    public void HsvRecovery_UncertaintyShrinksBothIndependentUpdates()
    {
        var options = new HsvA2crOptions(1, 1, 2, 2, 0.05, 1e-9);
        var certain = HsvA2crRecovery.BlendPixel(30, 0.7, 0.3, 80, 0.9, 0.8, 0, options);
        var uncertain = HsvA2crRecovery.BlendPixel(30, 0.7, 0.3, 80, 0.9, 0.8, 1, options);
        TestAssert.True(certain.ValueWeight > uncertain.ValueWeight);
        TestAssert.True(certain.ChromaWeight > uncertain.ChromaWeight);
        TestAssert.InRange(uncertain.Value, 0, 1);
        TestAssert.InRange(uncertain.Saturation, 0, 1);
    }

    public void HsvMethod_DefaultPipeline_ProducesFiniteFeasibleOutput()
    {
        using var input = new Image<Bgr, byte>(32, 24);
        var bytes = new byte[32 * 24 * 3]; new Random(117).NextBytes(bytes);
        Marshal.Copy(bytes, 0, input.Mat.DataPointer, bytes.Length);
        var method = new HsvA2crMethod();
        var parameters = method.Parameters.ToDictionary(x => x.Key, x => x.Default);
        parameters["patch"] = 3; parameters["refine"] = 5; parameters["fast"] = 1;
        using var output = method.Process(input, parameters);
        var values = new float[32 * 24 * 3]; output.CopyTo(values);
        TestAssert.True(values.All(float.IsFinite));
        TestAssert.InRange(values.Min(), 0, 1);
        TestAssert.InRange(values.Max(), 0, 1);
    }

    private static double TotalVariation(float[] values, int rows, int cols)
    {
        double total = 0;
        for (int y = 0; y < rows; y++)
        for (int x = 0; x < cols; x++)
        {
            int i = y * cols + x;
            if (x + 1 < cols) total += Math.Abs(values[i + 1] - values[i]);
            if (y + 1 < rows) total += Math.Abs(values[i + cols] - values[i]);
        }
        return total;
    }

    private static Mat FloatMat(float[] values, int rows, int cols, int channels)
    {
        var mat = new Mat(rows, cols, DepthType.Cv32F, channels);
        Marshal.Copy(values, 0, mat.DataPointer, values.Length);
        return mat;
    }
}
