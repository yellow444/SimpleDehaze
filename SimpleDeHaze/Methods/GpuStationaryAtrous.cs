using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace SimpleDeHaze.Methods
{
    /// <summary>CUDA counterpart of HSV-V <see cref="ContourOps.TransmissionAtrousBands"/>.</summary>
    internal static class GpuStationaryAtrous
    {
        private static readonly float[] B3 = { 1f / 16, 4f / 16, 6f / 16, 4f / 16, 1f / 16 };

        public static bool IsAvailable => CudaBackend.IsAvailable;

        public static Mat TransmissionAtrousHsv(Mat bgr01, Mat tMap, Mat sigmaDepth, int levels,
            double gFine, double gMid, double gCoarse, double tLo, double tHi,
            double noiseSigma, double uncertaintyWeight, int energyRadius, double deltaLimit)
        {
            if (!IsAvailable) throw new InvalidOperationException("CUDA device is not available");
            levels = Math.Clamp(levels, 2, 5);
            energyRadius = Math.Clamp(energyRadius, 0, 20);
            double span = Math.Max(1e-3, tHi - tLo);

            using var gBgr = new GpuMat(bgr01);
            using var gT = new GpuMat(tMap);
            using var gSigma = new GpuMat(sigmaDepth);
            using var gHsv = new GpuMat(); CudaInvoke.CvtColor(gBgr, gHsv, ColorConversion.Bgr2Hsv);
            var channels = gHsv.Split();
            var bands = new List<GpuMat>(levels);
            var current = new GpuMat(); channels[2].CopyTo(current);
            try
            {
                for (int level = 0; level < levels; level++)
                {
                    var smooth = Smooth(current, 1 << level);
                    var band = new GpuMat(); CudaInvoke.Subtract(current, smooth, band); bands.Add(band);
                    current.Dispose(); current = smooth;
                }

                using var tSquared = new GpuMat(); CudaInvoke.Multiply(gT, gT, tSquared);
                CudaInvoke.Add(tSquared, new ScalarArray(1e-6), tSquared);
                using var sigmaSquared = new GpuMat(); CudaInvoke.Multiply(gSigma, gSigma, sigmaSquared);
                using var reconstructed = new GpuMat(); current.CopyTo(reconstructed);
                for (int level = 0; level < levels; level++)
                {
                    double sf = (double)level / Math.Max(1, levels - 1);
                    double baseGain = level == 0 ? gFine :
                        gMid + (double)(level - 1) / Math.Max(1, levels - 2) * (gCoarse - gMid);
                    using var normalizedT = new GpuMat();
                    CudaInvoke.Multiply(gT, new ScalarArray(1.0 / span), normalizedT);
                    CudaInvoke.Add(normalizedT, new ScalarArray(-tLo / span), normalizedT);
                    Clamp(normalizedT, 0, 1);
                    using (var square = new GpuMat())
                    using (var cube = new GpuMat())
                    {
                        CudaInvoke.Multiply(normalizedT, normalizedT, square);
                        CudaInvoke.Multiply(square, normalizedT, cube);
                        CudaInvoke.Multiply(square, new ScalarArray(3), normalizedT);
                        CudaInvoke.Multiply(cube, new ScalarArray(2), cube);
                        CudaInvoke.Subtract(normalizedT, cube, normalizedT);
                    }
                    using var transmissionGate = new GpuMat();
                    CudaInvoke.Multiply(normalizedT, new ScalarArray(1 - sf), transmissionGate);
                    CudaInvoke.Add(transmissionGate, new ScalarArray(sf), transmissionGate);

                    using var bandSquared = new GpuMat(); CudaInvoke.Multiply(bands[level], bands[level], bandSquared);
                    using var power = new GpuMat();
                    if (energyRadius > 0)
                    {
                        int size = 2 * energyRadius + 1;
                        using var box = new CudaBoxFilter(DepthType.Cv32F, 1, DepthType.Cv32F, 1,
                            new System.Drawing.Size(size, size), new System.Drawing.Point(-1, -1),
                            BorderType.Reflect101, new MCvScalar());
                        box.Apply(bandSquared, power);
                    }
                    else bandSquared.CopyTo(power);

                    using var noiseBudget = new GpuMat();
                    using (var numerator = new GpuMat(tMap.Rows, tMap.Cols, DepthType.Cv32F, 1))
                    {
                        numerator.SetTo(new MCvScalar(Math.Max(0, noiseSigma * noiseSigma)));
                        CudaInvoke.Divide(numerator, tSquared, noiseBudget);
                    }
                    using var disagreement = new GpuMat();
                    CudaInvoke.Multiply(sigmaSquared,
                        new ScalarArray(Math.Max(0, uncertaintyWeight) * (1 + 0.35 * level)), disagreement);
                    CudaInvoke.Add(noiseBudget, disagreement, noiseBudget);
                    using var signalPower = new GpuMat(); CudaInvoke.Subtract(power, noiseBudget, signalPower);
                    CudaInvoke.Max(signalPower, new ScalarArray(0), signalPower);
                    using var denominator = new GpuMat(); CudaInvoke.Add(signalPower, noiseBudget, denominator);
                    CudaInvoke.Add(denominator, new ScalarArray(1e-8), denominator);
                    using var reliability = new GpuMat(); CudaInvoke.Divide(signalPower, denominator, reliability);
                    CudaInvoke.Multiply(reliability, transmissionGate, reliability);
                    using var delta = new GpuMat(); CudaInvoke.Multiply(bands[level], reliability, delta);
                    CudaInvoke.Multiply(delta, new ScalarArray(baseGain - 1), delta);
                    if (deltaLimit > 0) Clamp(delta, -deltaLimit, deltaLimit);
                    using var contribution = new GpuMat(); CudaInvoke.Add(bands[level], delta, contribution);
                    CudaInvoke.Add(reconstructed, contribution, reconstructed);
                }
                Clamp(reconstructed, 0, 1);
                reconstructed.CopyTo(channels[2]);
                using (var vector = new VectorOfGpuMat(channels)) CudaInvoke.Merge(vector, gHsv);
                using var gOutput = new GpuMat(); CudaInvoke.CvtColor(gHsv, gOutput, ColorConversion.Hsv2Bgr);
                var output = gOutput.ToMat(); DehazeCore.Clamp01(output); return output;
            }
            finally
            {
                current.Dispose();
                foreach (var band in bands) band.Dispose();
                foreach (var channel in channels) channel.Dispose();
            }
        }

        private static GpuMat Smooth(GpuMat source, int spacing)
        {
            spacing = Math.Max(1, spacing);
            int length = 1 + 4 * spacing;
            var coefficients = new float[length];
            for (int i = 0; i < B3.Length; i++) coefficients[i * spacing] = B3[i];
            using var horizontal = DehazeCore.MatFromFloats(coefficients, 1, length);
            using var vertical = DehazeCore.MatFromFloats(coefficients, length, 1);
            using var filterX = new CudaLinearFilter(DepthType.Cv32F, 1, DepthType.Cv32F, 1,
                horizontal, new System.Drawing.Point(-1, -1), BorderType.Reflect101, new MCvScalar());
            using var filterY = new CudaLinearFilter(DepthType.Cv32F, 1, DepthType.Cv32F, 1,
                vertical, new System.Drawing.Point(-1, -1), BorderType.Reflect101, new MCvScalar());
            using var temporary = new GpuMat(); filterX.Apply(source, temporary);
            var result = new GpuMat(); filterY.Apply(temporary, result); return result;
        }

        private static void Clamp(GpuMat value, double minimum, double maximum)
        {
            CudaInvoke.Max(value, new ScalarArray(minimum), value);
            CudaInvoke.Min(value, new ScalarArray(maximum), value);
        }
    }
}
