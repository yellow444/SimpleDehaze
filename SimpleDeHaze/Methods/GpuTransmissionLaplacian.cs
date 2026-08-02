using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;

namespace SimpleDeHaze.Methods
{
    /// <summary>CUDA counterpart of the Lab-L transmission-aware Laplacian pyramid stage.</summary>
    internal static class GpuTransmissionLaplacian
    {
        public static Mat Run(Mat bgr01, Mat tMap, int levels,
            double gFine, double gMid, double gCoarse, double tLo, double tHi,
            Mat? richMap = null, double bandLimit = 0)
        {
            CudaBackend.RequireAvailable();
            levels = Math.Clamp(levels, 2, 7);
            double span = Math.Max(1e-3, tHi - tLo);
            const double coarseWeight = 0.8;

            using var bgr8 = new Mat(); bgr01.ConvertTo(bgr8, DepthType.Cv8U, 255.0);
            using var lab = new Mat(); CvInvoke.CvtColor(bgr8, lab, ColorConversion.Bgr2Lab);
            var channels = lab.Split();
            using var luminanceCpu = new Mat();
            channels[0].ConvertTo(luminanceCpu, DepthType.Cv32F, 1.0 / 255.0);
            using var luminance = new GpuMat(luminanceCpu);
            using var transmission = new GpuMat(tMap);
            using var richness = richMap == null ? null : new GpuMat(richMap);

            var gauss = new List<GpuMat> { Clone(luminance) };
            var transmissions = new List<GpuMat> { Clone(transmission) };
            var bands = new List<GpuMat>();
            try
            {
                for (int level = 1; level < levels; level++)
                {
                    var down = new GpuMat(); CudaInvoke.PyrDown(gauss[level - 1], down); gauss.Add(down);
                    var downT = new GpuMat(); CudaInvoke.PyrDown(transmissions[level - 1], downT); transmissions.Add(downT);
                }
                for (int level = 0; level < levels - 1; level++)
                {
                    using var up = Up(gauss[level + 1], gauss[level].Size);
                    var band = new GpuMat(); CudaInvoke.Subtract(gauss[level], up, band); bands.Add(band);
                }

                var current = Clone(gauss[levels - 1]);
                try
                {
                    for (int level = levels - 2; level >= 0; level--)
                    {
                        using var up = Up(current, gauss[level].Size);
                        double scaleFraction = (double)level / Math.Max(1, levels - 2);
                        double baseGain = level == 0 ? gFine
                            : gMid + (double)(level - 1) / Math.Max(1, levels - 2) * (gCoarse - gMid);

                        using var normalized = new GpuMat();
                        transmissions[level].ConvertTo(normalized, DepthType.Cv32F, 1.0 / span, -tLo / span);
                        Clamp(normalized, 0, 1);
                        using (var square = new GpuMat())
                        using (var cube = new GpuMat())
                        {
                            CudaInvoke.Multiply(normalized, normalized, square);
                            CudaInvoke.Multiply(square, normalized, cube);
                            CudaInvoke.Multiply(square, new ScalarArray(3), normalized);
                            CudaInvoke.Multiply(cube, new ScalarArray(2), cube);
                            CudaInvoke.Subtract(normalized, cube, normalized);
                        }
                        using var gate = new GpuMat();
                        normalized.ConvertTo(gate, DepthType.Cv32F, 1.0 - scaleFraction, scaleFraction);

                        if (richness != null)
                        {
                            using var resizedRichness = Resize(richness, bands[level].Size);
                            using var weightedRichness = new GpuMat();
                            resizedRichness.ConvertTo(weightedRichness, DepthType.Cv32F,
                                coarseWeight * scaleFraction);
                            CudaInvoke.Max(gate, weightedRichness, gate);
                        }

                        using var contribution = new GpuMat();
                        CudaInvoke.Multiply(bands[level], gate, contribution);
                        if (bandLimit > 1e-6) Clamp(contribution, -bandLimit, bandLimit);
                        using var scaled = new GpuMat();
                        contribution.ConvertTo(scaled, DepthType.Cv32F, baseGain);
                        var next = new GpuMat(); CudaInvoke.Add(up, scaled, next);
                        current.Dispose(); current = next;
                    }

                    Clamp(current, 0, 1);
                    using var currentCpu = current.ToMat();
                    currentCpu.ConvertTo(channels[0], DepthType.Cv8U, 255.0);
                    using (var vector = new VectorOfMat(channels)) CvInvoke.Merge(vector, lab);
                    using var output8 = new Mat(); CvInvoke.CvtColor(lab, output8, ColorConversion.Lab2Bgr);
                    var result = new Mat(); output8.ConvertTo(result, DepthType.Cv32F, 1.0 / 255.0);
                    return result;
                }
                finally { current.Dispose(); }
            }
            finally
            {
                foreach (var item in gauss) item.Dispose();
                foreach (var item in transmissions) item.Dispose();
                foreach (var item in bands) item.Dispose();
                foreach (var channel in channels) channel.Dispose();
            }
        }

        private static GpuMat Clone(GpuMat source)
        {
            var result = new GpuMat(); source.CopyTo(result); return result;
        }

        private static GpuMat Up(GpuMat source, System.Drawing.Size target)
        {
            using var doubled = new GpuMat(); CudaInvoke.PyrUp(source, doubled);
            return Resize(doubled, target);
        }

        private static GpuMat Resize(GpuMat source, System.Drawing.Size target)
        {
            var result = new GpuMat();
            if (source.Size.Equals(target)) source.CopyTo(result);
            else CudaInvoke.Resize(source, result, target, interpolation: Inter.Linear);
            return result;
        }

        private static void Clamp(GpuMat value, double minimum, double maximum)
        {
            CudaInvoke.Max(value, new ScalarArray(minimum), value);
            CudaInvoke.Min(value, new ScalarArray(maximum), value);
        }
    }
}
