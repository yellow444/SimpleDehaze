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

        /// <summary>
        /// Two-stage fusion that gives each prior family one vote regardless of the number of
        /// correlated perturbations inside it. Total uncertainty combines between-family MAD and
        /// mean within-family robust variance in optical-depth space.
        /// </summary>
        public static OpticalDepthEstimate FuseFamilies(
            IReadOnlyList<IReadOnlyList<(Mat Transmission, double Weight)>> families,
            double minimumTransmission, double uncertaintyScale, double uncertaintyFloor)
        {
            if (families.Count == 0 || families.Any(family => family.Count == 0))
                throw new ArgumentException("Transmission families must be non-empty", nameof(families));
            var familyEstimates = new List<OpticalDepthEstimate>(families.Count);
            try
            {
                foreach (var family in families)
                    familyEstimates.Add(Fuse(family, minimumTransmission, 1.0, 0.0));
                using var between = Fuse(familyEstimates
                    .Select(estimate => (estimate.Transmission, 1.0)).ToArray(),
                    minimumTransmission, 1.0, 0.0);

                int rows = between.Transmission.Rows, cols = between.Transmission.Cols, pixels = rows * cols;
                var transmission = new float[pixels]; var betweenSigma = new float[pixels];
                between.Transmission.CopyTo(transmission); between.SigmaDepth.CopyTo(betweenSigma);
                var within = familyEstimates.Select(estimate =>
                {
                    var data = new float[pixels]; estimate.SigmaDepth.CopyTo(data); return data;
                }).ToArray();
                var sigma = new float[pixels]; var variance = new float[pixels];
                double scale = Math.Max(0, uncertaintyScale);
                double floor2 = Math.Max(0, uncertaintyFloor) * Math.Max(0, uncertaintyFloor);
                for (int i = 0; i < pixels; i++)
                {
                    double withinVariance = within.Average(map => map[i] * map[i]);
                    double sigmaDepth = scale * Math.Sqrt(betweenSigma[i] * betweenSigma[i] + withinVariance);
                    sigma[i] = (float)sigmaDepth;
                    variance[i] = (float)(transmission[i] * transmission[i] * sigmaDepth * sigmaDepth + floor2);
                }
                return new OpticalDepthEstimate(between.Transmission.Clone(),
                    ToMat(variance, rows, cols), ToMat(sigma, rows, cols));
            }
            finally
            {
                foreach (var estimate in familyEstimates) estimate.Dispose();
            }
        }

        /// <summary>
        /// Re-expresses optical-depth uncertainty around a refined transmission mean:
        /// Var(t) ~= t_refined^2 Var(D) + sigma_floor^2. The refinement itself is deterministic;
        /// this first-order propagation only fixes the otherwise inconsistent mean/variance pair.
        /// </summary>
        public static Mat PropagateVariance(Mat refinedTransmission, Mat sigmaDepth,
            double transmissionSigmaFloor)
        {
            if (refinedTransmission.NumberOfChannels != 1 || sigmaDepth.NumberOfChannels != 1 ||
                refinedTransmission.Rows != sigmaDepth.Rows || refinedTransmission.Cols != sigmaDepth.Cols)
                throw new ArgumentException("Transmission and optical-depth sigma must be matching scalar maps");
            using var tSquared = new Mat();
            using var sigmaSquared = new Mat();
            CvInvoke.Multiply(refinedTransmission, refinedTransmission, tSquared);
            CvInvoke.Multiply(sigmaDepth, sigmaDepth, sigmaSquared);
            var variance = new Mat();
            CvInvoke.Multiply(tSquared, sigmaSquared, variance);
            double floor = Math.Max(0, transmissionSigmaFloor);
            if (floor > 0) CvInvoke.Add(variance, new ScalarArray(floor * floor), variance);
            return variance;
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
