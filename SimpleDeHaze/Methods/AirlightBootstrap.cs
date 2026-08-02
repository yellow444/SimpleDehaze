using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    internal sealed record AirlightEstimate(MCvScalar Value, double[] ChannelVariance, MCvScalar[] Samples)
    {
        public (double Parallel, double Perpendicular) AxisVariances()
        {
            double[] a = { Value.V0, Value.V1, Value.V2 };
            double norm2 = a.Sum(x => x * x);
            if (norm2 < 1e-20)
            {
                double iso = ChannelVariance.Average();
                return (iso, iso);
            }
            double parallel = Enumerable.Range(0, 3).Sum(c => a[c] * a[c] / norm2 * ChannelVariance[c]);
            double perpendicular = Math.Max(0, (ChannelVariance.Sum() - parallel) / 2.0);
            return (parallel, perpendicular);
        }
    }

    internal static class AirlightBootstrap
    {
        public static AirlightEstimate Estimate(Mat inputLinear, int patch, double topFraction, int samples)
        {
            samples = samples <= 1 ? 1 : samples <= 3 ? 3 : 5;
            int[] offsets = samples == 1 ? new[] { 0 } : samples == 3 ? new[] { -1, 0, 1 } : new[] { -2, -1, 0, 1, 2 };
            var estimates = new List<MCvScalar>(offsets.Length);
            foreach (int offset in offsets)
            {
                int radius = Math.Max(1, patch + offset * Math.Max(1, patch / 3));
                double top = Math.Clamp(topFraction * Math.Pow(1.5, offset), 1e-5, 0.05);
                using var dark = DehazeCore.DarkChannel(inputLinear, radius);
                estimates.Add(DehazeCore.Atmospheric(inputLinear, dark, top));
            }

            double[] median = new double[3];
            double[] variance = new double[3];
            for (int c = 0; c < 3; c++)
            {
                double[] values = estimates.Select(x => Channel(x, c)).OrderBy(x => x).ToArray();
                median[c] = values[values.Length / 2];
                variance[c] = values.Select(x => (x - median[c]) * (x - median[c])).Average();
            }
            return new AirlightEstimate(new MCvScalar(median[0], median[1], median[2]), variance, estimates.ToArray());
        }

        private static double Channel(MCvScalar value, int channel) => channel switch
        {
            0 => value.V0,
            1 => value.V1,
            _ => value.V2,
        };
    }
}
