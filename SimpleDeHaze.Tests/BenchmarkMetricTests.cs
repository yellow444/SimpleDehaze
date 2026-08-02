using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.CvEnum;

using SimpleDeHaze.Benchmarking;
using SimpleDeHaze.Methods;

namespace SimpleDeHaze.Tests;

public sealed class BenchmarkMetricTests
{
    public void HueError_IgnoresAchromaticPixelsAndIsZeroForIdentity()
    {
        using var image = new Mat(1, 2, DepthType.Cv32F, 3);
        Marshal.Copy(new[] { 0f, 0f, 1f, 0.5f, 0.5f, 0.5f }, 0, image.DataPointer, 6);
        var report = BenchmarkColorErrors.Evaluate(image, image);
        TestAssert.InRange(report.HueErrorDegrees, 0, 1e-9);
        TestAssert.InRange(report.ChromaError, 0, 1e-9);
        TestAssert.InRange(report.ChromaticPixelFraction, 0.49, 0.51);
    }

    public void LpipsByteConversion_PreservesBytesAndScalesUnitFloats()
    {
        byte[] bytes = { 0, 1, 127, 255 };
        using var byteInput = new Mat(1, bytes.Length, DepthType.Cv8U, 1);
        Marshal.Copy(bytes, 0, byteInput.DataPointer, bytes.Length);
        using var byteOutput = LpipsBridge.NormalizeSrgbToByte(byteInput);
        var byteActual = new byte[bytes.Length];
        Marshal.Copy(byteOutput.DataPointer, byteActual, 0, byteActual.Length);
        TestAssert.True(byteActual.SequenceEqual(bytes),
            $"LPIPS byte input was rescaled: {string.Join(',', byteActual)}");

        float[] unit = { 0f, 1f / 255f, 0.5f, 1f };
        using var floatInput = new Mat(1, unit.Length, DepthType.Cv32F, 1);
        Marshal.Copy(unit, 0, floatInput.DataPointer, unit.Length);
        using var floatOutput = LpipsBridge.NormalizeSrgbToByte(floatInput);
        var floatActual = new byte[unit.Length];
        Marshal.Copy(floatOutput.DataPointer, floatActual, 0, floatActual.Length);
        TestAssert.Equal((byte)0, floatActual[0]);
        TestAssert.Equal((byte)1, floatActual[1]);
        TestAssert.InRange(floatActual[2], 127, 128);
        TestAssert.Equal((byte)255, floatActual[3]);
    }

    public void Ciede2000_MatchesSharmaReferencePairs()
    {
        var samples = new[]
        {
            (L1: 50.0, A1: 2.6772, B1: -79.7751, L2: 50.0, A2: 0.0, B2: -82.7485, Expected: 2.0425),
            (L1: 50.0, A1: 3.1571, B1: -77.2803, L2: 50.0, A2: 0.0, B2: -82.7485, Expected: 2.8615),
            (L1: 50.0, A1: 2.8361, B1: -74.0200, L2: 50.0, A2: 0.0, B2: -82.7485, Expected: 3.4412),
            (L1: 50.0, A1: -1.3802, B1: -84.2814, L2: 50.0, A2: 0.0, B2: -82.7485, Expected: 1.0000),
        };
        foreach (var sample in samples)
        {
            double forward = Metrics.DeltaE2000(sample.L1, sample.A1, sample.B1,
                sample.L2, sample.A2, sample.B2);
            double reverse = Metrics.DeltaE2000(sample.L2, sample.A2, sample.B2,
                sample.L1, sample.A1, sample.B1);
            TestAssert.InRange(Math.Abs(forward - sample.Expected), 0, 5e-5);
            TestAssert.InRange(Math.Abs(reverse - sample.Expected), 0, 5e-5);
        }
    }

    public void ChromaticFidelity_DetectsColorCollapseAndIsNeutralForGrayGt()
    {
        byte[] redBytes = Enumerable.Repeat(new byte[] { 0, 0, 255 }, 16).SelectMany(x => x).ToArray();
        float[] redFloats = Enumerable.Repeat(new float[] { 0, 0, 1 }, 16).SelectMany(x => x).ToArray();
        float[] grayFloats = Enumerable.Repeat(new float[] { 0.5f, 0.5f, 0.5f }, 16).SelectMany(x => x).ToArray();
        byte[] grayBytes = Enumerable.Repeat(new byte[] { 128, 128, 128 }, 16).SelectMany(x => x).ToArray();
        using var redGt = new Mat(4, 4, DepthType.Cv8U, 3);
        using var exactRed = new Mat(4, 4, DepthType.Cv32F, 3);
        using var collapsed = new Mat(4, 4, DepthType.Cv32F, 3);
        using var grayGt = new Mat(4, 4, DepthType.Cv8U, 3);
        Marshal.Copy(redBytes, 0, redGt.DataPointer, redBytes.Length);
        Marshal.Copy(redFloats, 0, exactRed.DataPointer, redFloats.Length);
        Marshal.Copy(grayFloats, 0, collapsed.DataPointer, grayFloats.Length);
        Marshal.Copy(grayBytes, 0, grayGt.DataPointer, grayBytes.Length);

        Metrics.ChromaticFidelityReport exact = Metrics.ChromaticFidelity(exactRed, redGt);
        Metrics.ChromaticFidelityReport lost = Metrics.ChromaticFidelity(collapsed, redGt);
        Metrics.ChromaticFidelityReport noSignal = Metrics.ChromaticFidelity(collapsed, grayGt);
        TestAssert.InRange(exact.Score, 99.99, 100.0);
        TestAssert.InRange(lost.Score, 0.0, 10.0);
        TestAssert.Equal(16, lost.ChromaticPixels);
        TestAssert.InRange(noSignal.Score, 99.99, 100.0);
        TestAssert.Equal(0, noSignal.ChromaticPixels);
    }

    public void LocalChromaExpansion_DetectsLocalizedColorExplosion()
    {
        const int pixels = 100;
        byte[] grayBytes = Enumerable.Repeat(new byte[] { 128, 128, 128 }, pixels).SelectMany(x => x).ToArray();
        float[] grayFloats = Enumerable.Repeat(new float[] { 0.5f, 0.5f, 0.5f }, pixels).SelectMany(x => x).ToArray();
        float[] partlyRed = (float[])grayFloats.Clone();
        for (int pixel = 0; pixel < 10; pixel++)
        {
            int offset = pixel * 3;
            partlyRed[offset] = 0;
            partlyRed[offset + 1] = 0;
            partlyRed[offset + 2] = 1;
        }
        using var input = new Mat(10, 10, DepthType.Cv8U, 3);
        using var unchanged = new Mat(10, 10, DepthType.Cv32F, 3);
        using var exploded = new Mat(10, 10, DepthType.Cv32F, 3);
        Marshal.Copy(grayBytes, 0, input.DataPointer, grayBytes.Length);
        Marshal.Copy(grayFloats, 0, unchanged.DataPointer, grayFloats.Length);
        Marshal.Copy(partlyRed, 0, exploded.DataPointer, partlyRed.Length);

        Metrics.LocalChromaExpansionReport stable = Metrics.LocalChromaExpansion(unchanged, input);
        Metrics.LocalChromaExpansionReport bad = Metrics.LocalChromaExpansion(exploded, input);
        TestAssert.InRange(stable.MeanExcess, 0, 1e-12);
        TestAssert.InRange(stable.P95Excess, 0, 1e-12);
        TestAssert.InRange(bad.ExplodedPixelFraction, 0.09, 0.11);
        TestAssert.True(bad.P95Excess > 20 && bad.MeanExcess > 5,
            "Локальное цветовое пятно растворилось в средней метрике.");
        TestAssert.True(AutoTuner.LocalChromaExpansionPenalty(bad, vivid: false) >
            AutoTuner.LocalChromaExpansionPenalty(stable, vivid: false));
    }
}
