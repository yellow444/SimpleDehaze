namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Closed-form minimum of the A²CR per-component quadratic risk:
    /// R(g)=S[(g*t-1)^2+g^2*var(t)]+N*g^2+U*(1-g)^2.
    /// S, N and U are energies/variances and therefore must be non-negative.
    /// </summary>
    internal static class A2crRisk
    {
        public static double OptimalGain(double signalEnergy, double transmission,
            double transmissionVariance, double noiseVariance, double airlightVariance,
            double minimumTransmission)
        {
            double s = Math.Max(0, signalEnergy);
            double t = Math.Clamp(transmission, 0, 1);
            double vt = Math.Max(0, transmissionVariance);
            double n = Math.Max(0, noiseVariance);
            double u = Math.Max(0, airlightVariance);
            double tMin = Math.Clamp(minimumTransmission, 1e-4, 1);

            double denominator = s * (t * t + vt) + n + u;
            double gain = denominator > 1e-20 ? (s * t + u) / denominator : 1.0;
            if (!double.IsFinite(gain)) gain = 1.0;
            return Math.Clamp(gain, 1.0, 1.0 / tMin);
        }

        public static double Risk(double gain, double signalEnergy, double transmission,
            double transmissionVariance, double noiseVariance, double airlightVariance)
        {
            double bias = gain * transmission - 1.0;
            return Math.Max(0, signalEnergy) * (bias * bias + gain * gain * Math.Max(0, transmissionVariance))
                 + Math.Max(0, noiseVariance) * gain * gain
                 + Math.Max(0, airlightVariance) * (1.0 - gain) * (1.0 - gain);
        }
    }
}
