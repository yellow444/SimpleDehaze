using SimpleDeHaze.Benchmarking;

namespace SimpleDeHaze.Tests;

public sealed class DiodeFormulaTests
{
    public void Beta_ReachesTargetTransmissionAtP90Depth()
    {
        foreach (double depthP90 in new[] { 0.5, 3.0, 25.0 })
        foreach (double target in new[] { 0.8, 0.6, 0.4, 0.2, 0.1 })
        {
            double beta = DiodeControlledBenchmark.BetaForTarget(target, depthP90);
            double actual = Math.Exp(-beta * depthP90);
            TestAssert.InRange(Math.Abs(actual - target), 0, 1e-12);
        }
    }

    public void PoissonGaussianVariance_MatchesDeclaredHighCountApproximation()
    {
        double clean = DiodeControlledBenchmark.PoissonGaussianVariance(0.5, 0, 0);
        TestAssert.InRange(clean, 0, 0);

        double gaussian = DiodeControlledBenchmark.PoissonGaussianVariance(0.5, 0.01, 0);
        TestAssert.InRange(Math.Abs(gaussian - 0.0001), 0, 1e-15);

        double mixed = DiodeControlledBenchmark.PoissonGaussianVariance(0.5, 0.01, 100);
        TestAssert.InRange(Math.Abs(mixed - 0.0051), 0, 1e-15);
    }
}
