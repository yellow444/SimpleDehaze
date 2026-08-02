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

        foreach (var variant in new[]
                 {
                     (Space: 1.0, Basis: 0.0), (Space: 0.0, Basis: 1.0),
                     (Space: 1.0, Basis: 1.0), (Space: 0.0, Basis: 2.0),
                     (Space: 1.0, Basis: 2.0),
                 })
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
        TestAssert.False(method.Parameters.Any(parameter => parameter.Key is
            "wiener" or "uNoise" or "uUnc" or "uRadius" or "uLimit"));
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

    public void RegisteredHsvUtawVariant_ExposesOnlyEffectiveSearchCoordinates()
    {
        var generic = new TransScaleLaplacianMethod();
        foreach (string key in new[] { "edgeS", "edgeR", "uNoise", "uUnc", "uRadius", "uLimit" })
            TestAssert.False(generic.Parameters.Single(parameter => parameter.Key == key).Tunable,
                $"generic basis-specific coordinate {key} must stay fixed during auto-tuning");

        var edge = new TransmissionAwareHsvEdgeMethod();
        foreach (string key in new[] { "edgeS", "edgeR" })
            TestAssert.True(edge.Parameters.Single(parameter => parameter.Key == key).Tunable);

        var method = new TransmissionAwareHsvUtawMethod();
        TestAssert.False(method.Parameters.Any(parameter => parameter.Key is
            "space" or "basis" or "wiener" or "edgeS" or "edgeR"));
        foreach (string key in new[] { "uNoise", "uUnc", "uLimit" })
        {
            TestAssert.True(method.Parameters.Single(parameter => parameter.Key == key).Search,
                $"{key} must participate in quick AutoTuner search");
            TestAssert.True(method.Parameters.Single(parameter => parameter.Key == key).Tunable);
        }
        TestAssert.False(method.Parameters.Single(parameter => parameter.Key == "uRadius").Search);
        TestAssert.True(method.Parameters.Single(parameter => parameter.Key == "uRadius").Tunable);
        var defaults = method.Parameters.ToDictionary(parameter => parameter.Key, parameter => parameter.Default);
        TestAssert.InRange(Math.Abs(defaults["gFine"] - 0.0), 0, 1e-12);
        TestAssert.InRange(Math.Abs(defaults["gMid"] - 1.3), 0, 1e-12);
        TestAssert.InRange(Math.Abs(defaults["gCoarse"] - 1.1), 0, 1e-12);
        TestAssert.InRange(Math.Abs(defaults["uNoise"] - 0.006), 0, 1e-12);
        TestAssert.InRange(Math.Abs(defaults["uUnc"] - 2.0), 0, 1e-12);
        TestAssert.InRange(Math.Abs(defaults["uRadius"] - 3.0), 0, 1e-12);
        TestAssert.InRange(Math.Abs(defaults["uLimit"] - 0.025), 0, 1e-12);
    }

    public void AtrousUnitGain_ReconstructsHsvAndFloatLabInputs()
    {
        using var input = RandomImage(31, 25, 704);
        using var inputFloat = DehazeCore.Normalize(input);
        using var transmission = new Mat(input.Height, input.Width, DepthType.Cv32F, 1);
        using var sigma = new Mat(input.Height, input.Width, DepthType.Cv32F, 1);
        transmission.SetTo(new MCvScalar(1)); sigma.SetTo(new MCvScalar(0));
        foreach (bool hsv in new[] { false, true })
        {
            using var output = ContourOps.TransmissionAtrousBands(inputFloat, transmission, sigma,
                4, 1, 1, 1, 0.12, 0.45, hsv, 0, 1, 3, 0.04);
            using var difference = new Mat(); CvInvoke.AbsDiff(inputFloat, output, difference);
            double maximum = 0;
            foreach (var channel in difference.Split())
            {
                double minimum = 0, channelMaximum = 0;
                var minPoint = new System.Drawing.Point(); var maxPoint = new System.Drawing.Point();
                CvInvoke.MinMaxLoc(channel, ref minimum, ref channelMaximum, ref minPoint, ref maxPoint);
                maximum = Math.Max(maximum, channelMaximum); channel.Dispose();
            }
            // OpenCV's float Lab round-trip is approximate near the sRGB gamut boundary.
            TestAssert.InRange(maximum, 0, hsv ? 2e-4 : 5e-3);
        }
    }

    public void AtrousCpuGpu_StayNumericallyEquivalentWhenCudaExists()
    {
        if (!GpuStationaryAtrous.IsAvailable) return;
        using var input = RandomImage(64, 48, 705);
        using var inputFloat = DehazeCore.Normalize(input);
        using var transmission = new Mat(input.Height, input.Width, DepthType.Cv32F, 1);
        using var sigma = new Mat(input.Height, input.Width, DepthType.Cv32F, 1);
        transmission.SetTo(new MCvScalar(0.63)); sigma.SetTo(new MCvScalar(0.11));
        using var cpu = ContourOps.TransmissionAtrousBands(inputFloat, transmission, sigma,
            4, 0.5, 1.6, 1.2, 0.12, 0.45, true, 0.004, 1, 3, 0.04);
        using var gpu = GpuStationaryAtrous.TransmissionAtrousHsv(inputFloat, transmission, sigma,
            4, 0.5, 1.6, 1.2, 0.12, 0.45, 0.004, 1, 3, 0.04);
        using var difference = new Mat(); CvInvoke.AbsDiff(cpu, gpu, difference);
        double maximum = 0;
        foreach (var channel in difference.Split())
        {
            double minimum = 0, channelMaximum = 0;
            var minPoint = new System.Drawing.Point(); var maxPoint = new System.Drawing.Point();
            CvInvoke.MinMaxLoc(channel, ref minimum, ref channelMaximum, ref minPoint, ref maxPoint);
            maximum = Math.Max(maximum, channelMaximum); channel.Dispose();
        }
        TestAssert.InRange(maximum, 0, 7e-4);

        using var laplacianCpu = ContourOps.TransmissionScaleLaplacian(inputFloat, transmission,
            4, 0.5, 1.6, 1.2, 0.12, 0.45, null, 0.08);
        using var laplacianGpu = GpuTransmissionLaplacian.Run(inputFloat, transmission,
            4, 0.5, 1.6, 1.2, 0.12, 0.45, null, 0.08);
        using var laplacianDifference = new Mat();
        CvInvoke.AbsDiff(laplacianCpu, laplacianGpu, laplacianDifference);
        double laplacianMaximum = 0;
        foreach (var channel in laplacianDifference.Split())
        {
            double minimum = 0, channelMaximum = 0;
            var minPoint = new System.Drawing.Point(); var maxPoint = new System.Drawing.Point();
            CvInvoke.MinMaxLoc(channel, ref minimum, ref channelMaximum, ref minPoint, ref maxPoint);
            laplacianMaximum = Math.Max(laplacianMaximum, channelMaximum); channel.Dispose();
        }
        TestAssert.InRange(laplacianMaximum, 0, 7e-4);
    }

    public void Registry_RecommendsValidatedTransmissionVariantsAndKeepsAblationsOut()
    {
        TestAssert.True(MethodRegistry.Recommended.Contains("Transmission-aware Laplacian (эксперимент)"));
        TestAssert.True(MethodRegistry.Recommended.Contains("Transmission-aware HSV UTAW (эксперимент)"));
        TestAssert.False(MethodRegistry.Recommended.Contains("Transmission-aware HSV Edge Bands (эксперимент)"));
        TestAssert.False(MethodRegistry.All.Any(method => method.Name.Contains("GPU", StringComparison.OrdinalIgnoreCase)));

        var cudaMethods = MethodRegistry.All
            .Where(method => method.Parameters.Any(parameter => parameter.Key == CudaBackend.ParameterKey))
            .ToArray();
        TestAssert.Equal(7, cudaMethods.Length);
        TestAssert.True(cudaMethods.Any(method => method is A2crMethod));
        TestAssert.True(cudaMethods.Any(method => method is DcpCpuMethod));
        TestAssert.True(cudaMethods.Any(method => method is BeltramiMethod));
        TestAssert.True(cudaMethods.Any(method => method is MattingMethod));
        TestAssert.True(cudaMethods.Any(method => method is TransScaleLaplacianMethod));
        TestAssert.True(cudaMethods.Any(method => method is TransmissionAwareHsvUtawMethod));
        TestAssert.True(cudaMethods.Any(method => method is HsvA2crMethod));
        foreach (var method in cudaMethods)
        {
            var backend = method.Parameters.Single(parameter => parameter.Key == CudaBackend.ParameterKey);
            TestAssert.Equal(0.0, backend.Default);
            TestAssert.True(backend.IsInt);
            TestAssert.False(backend.Search);
            TestAssert.False(backend.Tunable);
            TestAssert.Equal(CudaBackend.IsAvailable, backend.IsEnabled);
        }
        string c3rName = MethodRegistry.All.Single(method => method is HsvC3rMethod).Name;
        TestAssert.False(MethodRegistry.Recommended.Contains(c3rName));
        string hsvA2crName = MethodRegistry.All.Single(method => method is HsvA2crMethod).Name;
        TestAssert.False(MethodRegistry.Recommended.Contains(hsvA2crName));
        TestAssert.False(MethodRegistry.Recommended.Contains(MethodRegistry.All.Single(method => method is HcvA2crMethod).Name));
        TestAssert.False(MethodRegistry.Recommended.Contains(MethodRegistry.All.Single(method => method is HcvRgbA2crFusionMethod).Name));
        TestAssert.False(MethodRegistry.Recommended.Contains(MethodRegistry.All.Single(method => method is HcvA2crUtawMethod).Name));
    }

    private static Image<Bgr, byte> RandomImage(int width, int height, int seed)
    {
        var input = new Image<Bgr, byte>(width, height);
        var bytes = new byte[width * height * 3]; new Random(seed).NextBytes(bytes);
        Marshal.Copy(bytes, 0, input.Mat.DataPointer, bytes.Length);
        return input;
    }
}
