using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.CvEnum;

namespace SimpleDeHaze.Methods
{
    internal sealed class OpticalDepthEstimate : IDisposable
    {
        public Mat Transmission { get; }
        public Mat TransmissionVariance { get; }
        public Mat SigmaDepth { get; }

        public OpticalDepthEstimate(Mat transmission, Mat transmissionVariance, Mat sigmaDepth)
        {
            Transmission = transmission;
            TransmissionVariance = transmissionVariance;
            SigmaDepth = sigmaDepth;
        }

        public void Dispose()
        {
            Transmission.Dispose();
            TransmissionVariance.Dispose();
            SigmaDepth.Dispose();
        }
    }

    internal static class OpticalDepthFusion
    {
        /// <summary>Weighted median/MAD fusion in D=-log(t), followed by first-order variance propagation.</summary>
        public static OpticalDepthEstimate Fuse(IReadOnlyList<(Mat Transmission, double Weight)> ensemble,
            double minimumTransmission, double uncertaintyScale, double uncertaintyFloor)
        {
            if (ensemble.Count == 0) throw new ArgumentException("Transmission ensemble is empty", nameof(ensemble));
            int rows = ensemble[0].Transmission.Rows, cols = ensemble[0].Transmission.Cols, n = rows * cols;
            var maps = new float[ensemble.Count][];
            for (int k = 0; k < ensemble.Count; k++)
            {
                if (ensemble[k].Transmission.Rows != rows || ensemble[k].Transmission.Cols != cols || ensemble[k].Transmission.NumberOfChannels != 1)
                    throw new ArgumentException("Transmission maps must have identical single-channel shape", nameof(ensemble));
                maps[k] = new float[n];
                ensemble[k].Transmission.CopyTo(maps[k]);
            }

            float tMin = (float)Math.Clamp(minimumTransmission, 1e-4, 1);
            var tOut = new float[n]; var varOut = new float[n]; var sigmaOut = new float[n];
            Span<double> depths = stackalloc double[ensemble.Count];
            Span<double> deviations = stackalloc double[ensemble.Count];
            Span<double> weights = stackalloc double[ensemble.Count];
            for (int i = 0; i < n; i++)
            {
                for (int k = 0; k < ensemble.Count; k++)
                {
                    depths[k] = -Math.Log(Math.Clamp(maps[k][i], tMin, 1f));
                    weights[k] = Math.Max(0, ensemble[k].Weight);
                }
                double median = WeightedMedian(depths, weights);
                for (int k = 0; k < ensemble.Count; k++) deviations[k] = Math.Abs(depths[k] - median);
                double sigmaD = 1.4826 * WeightedMedian(deviations, weights) * Math.Max(0, uncertaintyScale);
                double t = Math.Exp(-median);
                double sigmaT2 = t * t * sigmaD * sigmaD + Math.Max(0, uncertaintyFloor) * Math.Max(0, uncertaintyFloor);
                tOut[i] = (float)t;
                varOut[i] = (float)sigmaT2;
                sigmaOut[i] = (float)sigmaD;
            }

            return new OpticalDepthEstimate(ToMat(tOut, rows, cols), ToMat(varOut, rows, cols), ToMat(sigmaOut, rows, cols));
        }

        internal static double WeightedMedian(ReadOnlySpan<double> values, ReadOnlySpan<double> weights)
        {
            if (values.Length != weights.Length || values.Length == 0) throw new ArgumentException("Invalid weighted sample");
            Span<int> order = stackalloc int[values.Length];
            for (int i = 0; i < order.Length; i++) order[i] = i;
            for (int i = 1; i < order.Length; i++)
            {
                int key = order[i], j = i - 1;
                while (j >= 0 && values[order[j]] > values[key]) { order[j + 1] = order[j]; j--; }
                order[j + 1] = key;
            }
            double total = 0;
            for (int i = 0; i < weights.Length; i++) total += Math.Max(0, weights[i]);
            if (total <= 0) return values[order[order.Length / 2]];
            double cumulative = 0;
            foreach (int index in order)
            {
                cumulative += Math.Max(0, weights[index]);
                if (cumulative >= total * 0.5) return values[index];
            }
            return values[order[^1]];
        }

        private static Mat ToMat(float[] data, int rows, int cols)
        {
            var mat = new Mat(rows, cols, DepthType.Cv32F, 1);
            Marshal.Copy(data, 0, mat.DataPointer, data.Length);
            return mat;
        }
    }
}
