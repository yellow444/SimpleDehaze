using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using SimpleDeHaze.Methods;

namespace SimpleDeHaze.Tests;

public sealed class C3rTests
{
    public void OpticalFusion_EndpointWeightCannotErasePriorDisagreement()
    {
        using var cap = FloatMat(new[] { 0.2f });
        using var dcp = FloatMat(new[] { 0.8f });
        using var meanCap = new Mat(); using var varCap = new Mat();
        using var meanDcp = new Mat(); using var varDcp = new Mat();
        HsvC3rMethod.FuseTransmissionInOpticalDepth(cap, dcp, 1, 0.05, meanCap, varCap);
        HsvC3rMethod.FuseTransmissionInOpticalDepth(cap, dcp, 0, 0.05, meanDcp, varDcp);

        var mc = new float[1]; var md = new float[1];
        var vc = new float[1]; var vd = new float[1];
        meanCap.CopyTo(mc); meanDcp.CopyTo(md); varCap.CopyTo(vc); varDcp.CopyTo(vd);
        TestAssert.InRange(Math.Abs(mc[0] - 0.2), 0, 1e-6);
        TestAssert.InRange(Math.Abs(md[0] - 0.8), 0, 1e-6);
        TestAssert.True(vc[0] > 0 && vd[0] > 0);
        TestAssert.InRange(Math.Abs(vc[0] / (mc[0] * mc[0]) - vd[0] / (md[0] * md[0])), 0, 1e-5);
    }

    public void RiskGain_RecoversInverseAndFallsBackUnderUncertainty()
    {
        using var signal = FloatMat(new[] { 0.4f });
        using var noise = FloatMat(new[] { 0f });
        using var zero = FloatMat(new[] { 0f });
        using var t = FloatMat(new[] { 0.5f });
        using var certain = HsvC3rMethod.ComputeRiskGain(signal, noise, zero, t, zero, 10);
        var certainValue = new float[1]; certain.CopyTo(certainValue);
        TestAssert.InRange(Math.Abs(certainValue[0] - 2), 0, 2e-6);

        using var variance = FloatMat(new[] { 0.2f });
        using var identityPenalty = FloatMat(new[] { 100f });
        using var guarded = HsvC3rMethod.ComputeRiskGain(signal, noise, identityPenalty, t, variance, 10);
        var guardedValue = new float[1]; guarded.CopyTo(guardedValue);
        TestAssert.True(guardedValue[0] >= 1 && guardedValue[0] < certainValue[0]);
        TestAssert.InRange(Math.Abs(guardedValue[0] - 1), 0, 0.01);
    }

    public void ConeProjection_IsFeasibleAndRemainsOnInputCandidateSegment()
    {
        var random = new Random(20260801);
        for (int iteration = 0; iteration < 20_000; iteration++)
        {
            double inputValue = random.NextDouble();
            double inputSaturation = random.NextDouble();
            double angle = 2 * Math.PI * random.NextDouble();
            double inputCx = inputValue * inputSaturation * Math.Cos(angle);
            double inputCy = inputValue * inputSaturation * Math.Sin(angle);
            double candidateValue = -2 + 4 * random.NextDouble();
            double candidateCx = -2 + 4 * random.NextDouble();
            double candidateCy = -2 + 4 * random.NextDouble();
            double cap = inputSaturation + (1 - inputSaturation) * random.NextDouble();

            HsvC3rMethod.ProjectAlongInputRay(inputValue, inputCx, inputCy,
                candidateValue, candidateCx, candidateCy, cap,
                out double value, out double cx, out double cy);

            TestAssert.InRange(value, -2e-9, 1 + 2e-9);
            TestAssert.True(cx * cx + cy * cy <= cap * cap * value * value + 2e-8);
            double[] delta = { candidateValue - inputValue, candidateCx - inputCx, candidateCy - inputCy };
            double[] moved = { value - inputValue, cx - inputCx, cy - inputCy };
            int pivot = Array.IndexOf(delta.Select(Math.Abs).ToArray(), delta.Select(Math.Abs).Max());
            double alpha = Math.Abs(delta[pivot]) > 1e-12 ? moved[pivot] / delta[pivot] : 0;
            TestAssert.InRange(alpha, -2e-7, 1 + 2e-7);
            for (int k = 0; k < 3; k++)
                TestAssert.InRange(Math.Abs(moved[k] - alpha * delta[k]), 0, 3e-6);
        }
    }

    public void DefaultPipeline_ProducesFiniteFeasibleOutput()
    {
        using var input = new Image<Bgr, byte>(32, 24);
        var bytes = new byte[32 * 24 * 3]; new Random(301).NextBytes(bytes);
        Marshal.Copy(bytes, 0, input.Mat.DataPointer, bytes.Length);
        var method = new HsvC3rMethod();
        var parameters = method.Parameters.ToDictionary(x => x.Key, x => x.Default);
        parameters["patch"] = 3; parameters["rmin"] = 3;
        parameters["rguide"] = 5; parameters["energy"] = 3;
        using var output = method.Process(input, parameters);
        var values = new float[32 * 24 * 3]; output.CopyTo(values);
        TestAssert.True(values.All(float.IsFinite));
        TestAssert.InRange(values.Min(), 0, 1);
        TestAssert.InRange(values.Max(), 0, 1);
    }

    private static Mat FloatMat(float[] values)
    {
        var mat = new Mat(1, values.Length, DepthType.Cv32F, 1);
        Marshal.Copy(values, 0, mat.DataPointer, values.Length);
        return mat;
    }
}
