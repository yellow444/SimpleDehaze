using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using SimpleDeHaze.Methods;

namespace SimpleDeHaze.Tests;

public sealed class TransmissionAwareTests
{
    public void BandGate_IsBoundedMonotoneAndCoarseSafe()
    {
        double previous = -1;
        for (int i = 0; i <= 1000; i++)
        {
            double t = i / 1000.0;
            double gate = ContourOps.TransmissionBandGate(t, 0.12, 0.45, 0.25, 0.0);
            TestAssert.InRange(gate, 0.25, 1.0);
            TestAssert.True(gate + 1e-12 >= previous, "Transmission gate must be monotone in t");
            previous = gate;
        }

        TestAssert.InRange(Math.Abs(ContourOps.TransmissionBandGate(0, 0.12, 0.45, 1.0) - 1), 0, 1e-12);
        TestAssert.InRange(Math.Abs(ContourOps.TransmissionBandGate(1, 0.12, 0.45, 0.0) - 1), 0, 1e-12);
        TestAssert.True(ContourOps.TransmissionBandGate(0, 0.12, 0.45, 0.5, 1.0) >= 0.4);
    }

    public void EdgeAwareBands_UnitGainReconstructsHsvInput()
    {
        using var input = RandomImage(32, 24, 701);
        using var inputFloat = DehazeCore.Normalize(input);
        using var transmission = new Mat(input.Height, input.Width, DepthType.Cv32F, 1);
        transmission.SetTo(new MCvScalar(1));
        using var output = ContourOps.TransmissionEdgeAwareBands(inputFloat, transmission, 3,
            1, 1, 1, 0.12, 0.45, 8, 0.18, true, null, 0);
        using var difference = new Mat(); CvInvoke.AbsDiff(inputFloat, output, difference);
        double maxError = 0;
        foreach (var channel in difference.Split())
        {
            double channelMin = 0, channelMax = 0;
            var minPoint = new System.Drawing.Point(); var maxPoint = new System.Drawing.Point();
            CvInvoke.MinMaxLoc(channel, ref channelMin, ref channelMax, ref minPoint, ref maxPoint);
            maxError = Math.Max(maxError, channelMax);
            channel.Dispose();
        }
        TestAssert.InRange(maxError, 0, 2e-4);
    }

    public void HsvAndEdgeVariants_ProduceFiniteFeasibleOutput()
    {
        using var input = RandomImage(36, 28, 702);
        var method = new TransScaleLaplacianMethod();
        var parameters = method.Parameters.ToDictionary(x => x.Key, x => x.Default);
        parameters["patch"] = 3; parameters["aRadius"] = 20;
        parameters["rguide"] = 5; parameters["levels"] = 4; parameters["smooth"] = 0;

        foreach (var variant in new[] { (Space: 1.0, Basis: 0.0), (Space: 0.0, Basis: 1.0), (Space: 1.0, Basis: 1.0) })
        {
            parameters["space"] = variant.Space;
            parameters["basis"] = variant.Basis;
            using var output = method.Process(input, parameters);
            var values = new float[output.Rows * output.Cols * output.NumberOfChannels];
            output.CopyTo(values);
            TestAssert.True(values.All(float.IsFinite));
            TestAssert.InRange(values.Min(), 0, 1);
            TestAssert.InRange(values.Max(), 0, 1);
        }
    }

    public void RegisteredHsvEdgeVariant_FreezesModesAndUsesValidatedDefaults()
    {
        using var input = RandomImage(36, 28, 703);
        var method = new TransmissionAwareHsvEdgeMethod();
        TestAssert.False(method.Parameters.Any(parameter => parameter.Key is "space" or "basis"));
        var parameters = method.Parameters.ToDictionary(x => x.Key, x => x.Default);
        TestAssert.InRange(Math.Abs(parameters["gFine"] - 0.5), 0, 1e-12);
        TestAssert.InRange(Math.Abs(parameters["gMid"] - 1.6), 0, 1e-12);
        TestAssert.InRange(Math.Abs(parameters["gCoarse"] - 1.2), 0, 1e-12);
        parameters["patch"] = 3; parameters["aRadius"] = 20;
        parameters["rguide"] = 5; parameters["levels"] = 4; parameters["smooth"] = 0;
        using var output = method.Process(input, parameters);
        var values = new float[output.Rows * output.Cols * output.NumberOfChannels];
        output.CopyTo(values);
        TestAssert.True(values.All(float.IsFinite));
        TestAssert.InRange(values.Min(), 0, 1);
        TestAssert.InRange(values.Max(), 0, 1);
    }

    private static Image<Bgr, byte> RandomImage(int width, int height, int seed)
    {
        var input = new Image<Bgr, byte>(width, height);
        var bytes = new byte[width * height * 3]; new Random(seed).NextBytes(bytes);
        Marshal.Copy(bytes, 0, input.Mat.DataPointer, bytes.Length);
        return input;
    }
}
