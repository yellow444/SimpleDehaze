using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.CvEnum;

namespace SimpleDeHaze.Methods
{
    internal sealed record HcvA2crRecoveryOptions(
        double MinimumTransmission,
        double NoiseVariance,
        double AirlightUncertaintyScale,
        bool UseNoise,
        bool UseTransmissionUncertainty,
        bool UseAirlightUncertainty,
        bool UseFeasibility,
        int SignalRadius,
        double GainCoupling,
        double HueUncertaintyScale);

    internal readonly record struct HcvA2crDiagnosticSummary(
        double MeanGainValue,
        double MeanGainChroma,
        double MeanHueConfidence,
        double ProjectedPixelFraction,
        double InvalidChannelFractionBefore,
        double InvalidChannelFractionAfter,
        double MeanRequiredClipping,
        double MeanProjectionDistance);

    internal sealed class HcvA2crRecoveryResult : IDisposable
    {
        public Mat LinearResult { get; }
        public Mat GainValue { get; }
        public Mat GainChroma { get; }
        public Mat HueConfidence { get; }
        public Mat ProjectionDistance { get; }
        public HcvA2crDiagnosticSummary Diagnostics { get; }

        public HcvA2crRecoveryResult(Mat linearResult, Mat gainValue, Mat gainChroma,
            Mat hueConfidence, Mat projectionDistance, HcvA2crDiagnosticSummary diagnostics)
        {
            LinearResult = linearResult;
            GainValue = gainValue;
            GainChroma = gainChroma;
            HueConfidence = hueConfidence;
            ProjectionDistance = projectionDistance;
            Diagnostics = diagnostics;
        }

        public void Dispose()
        {
            LinearResult.Dispose();
            GainValue.Dispose();
            GainChroma.Dispose();
            HueConfidence.Dispose();
            ProjectionDistance.Dispose();
        }
    }

    /// <summary>
    /// Exact airlight-normalized HCV coordinate recovery. For X=I/A the atmospheric model is
    /// X=1+t(Y-1). Consequently Hue is invariant, C(X)=t*C(Y), and V(X)-1=t*(V(Y)-1).
    /// The implementation avoids an explicit Hue angle: q_c=A_c*(X_c-V_X) stores its piecewise
    /// linear hue/chroma direction exactly, while p_c=A_c*(V_X-1) stores the value-offset axis.
    /// </summary>
    internal static class HcvA2crRecovery
    {
        public static HcvA2crRecoveryResult Recover(Mat inputLinear, Mat transmission,
            Mat transmissionVariance, AirlightEstimate airlight, HcvA2crRecoveryOptions options)
        {
            if (inputLinear.Depth != DepthType.Cv32F || inputLinear.NumberOfChannels != 3)
                throw new ArgumentException("HCV-A²CR input must be linear BGR float", nameof(inputLinear));
            int rows = inputLinear.Rows, cols = inputLinear.Cols, pixels = rows * cols;
            if (transmission.Rows != rows || transmission.Cols != cols ||
                transmissionVariance.Rows != rows || transmissionVariance.Cols != cols)
                throw new ArgumentException("HCV-A²CR maps must match the input size");

            var input = new float[pixels * 3]; var t = new float[pixels]; var varT = new float[pixels];
            inputLinear.CopyTo(input); transmission.CopyTo(t); transmissionVariance.CopyTo(varT);
            double[] a =
            {
                Math.Max(0.02, airlight.Value.V0),
                Math.Max(0.02, airlight.Value.V1),
                Math.Max(0.02, airlight.Value.V2),
            };

            var pComponents = new float[pixels * 3];
            var qComponents = new float[pixels * 3];
            var valueEnergy = new float[pixels];
            var chromaEnergy = new float[pixels];
            var chroma = new float[pixels];
            for (int i = 0; i < pixels; i++)
            {
                int j = i * 3;
                double xb = input[j] / a[0], xg = input[j + 1] / a[1], xr = input[j + 2] / a[2];
                double value = Math.Max(xb, Math.Max(xg, xr));
                double minimum = Math.Min(xb, Math.Min(xg, xr));
                double c = Math.Max(0, value - minimum);
                double valueOffset = value - 1.0;
                chroma[i] = (float)c;
                valueEnergy[i] = (float)(valueOffset * valueOffset);
                chromaEnergy[i] = (float)(c * c);
                for (int channel = 0; channel < 3; channel++)
                {
                    int k = j + channel;
                    double pc = a[channel] * valueOffset;
                    pComponents[k] = (float)pc;
                    qComponents[k] = (float)(input[k] - a[channel] - pc);
                }
            }

            valueEnergy = LocalMean(valueEnergy, rows, cols, options.SignalRadius);
            chromaEnergy = LocalMean(chromaEnergy, rows, cols, options.SignalRadius);
            double normalizedNoise = options.UseNoise
                ? Math.Max(0, options.NoiseVariance) * a.Select(x => 1.0 / (x * x)).Average()
                : 0;
            double valueNoise = normalizedNoise;
            double chromaNoise = 2.0 * normalizedNoise;
            var valueAirlightUncertainty = new float[pixels];
            var chromaAirlightUncertainty = new float[pixels];
            if (options.UseAirlightUncertainty && airlight.Samples.Length > 1)
                EstimateCoordinateAirlightUncertainty(input, pixels, airlight.Samples,
                    options.AirlightUncertaintyScale, valueAirlightUncertainty, chromaAirlightUncertainty);
            float gainMax = (float)(1.0 / Math.Clamp(options.MinimumTransmission, 1e-4, 1));

            var gainV = new float[pixels]; var gainC = new float[pixels]; var hueConfidence = new float[pixels];
            for (int i = 0; i < pixels; i++)
            {
                double ti = Math.Clamp(t[i], options.MinimumTransmission, 1);
                double vti = options.UseTransmissionUncertainty ? Math.Max(0, varT[i]) : 0;
                double inverseAttenuation2 = Math.Max(1e-10, ti * ti + vti);
                double sv = Math.Max(0, valueEnergy[i] - valueNoise) / inverseAttenuation2;
                double sc = Math.Max(0, chromaEnergy[i] - chromaNoise) / inverseAttenuation2;
                double gv = A2crRisk.OptimalGain(sv, ti, vti, valueNoise,
                    valueAirlightUncertainty[i], options.MinimumTransmission);
                double gc = A2crRisk.OptimalGain(sc, ti, vti, chromaNoise,
                    chromaAirlightUncertainty[i], options.MinimumTransmission);

                double coupling = Math.Max(0, options.GainCoupling);
                if (coupling > 0)
                {
                    double av = sv * (ti * ti + vti) + valueNoise + valueAirlightUncertainty[i];
                    double ac = sc * (ti * ti + vti) + chromaNoise + chromaAirlightUncertainty[i];
                    double bv = sv * ti + valueAirlightUncertainty[i];
                    double bc = sc * ti + chromaAirlightUncertainty[i];
                    double determinant = (av + coupling) * (ac + coupling) - coupling * coupling;
                    if (determinant > 1e-20)
                    {
                        gv = Math.Clamp((bv * (ac + coupling) + coupling * bc) / determinant, 1, gainMax);
                        gc = Math.Clamp((bc * (av + coupling) + coupling * bv) / determinant, 1, gainMax);
                    }
                }

                double c2 = chroma[i] * chroma[i];
                double qh = c2 / (c2 + chromaNoise +
                    chromaAirlightUncertainty[i] +
                    Math.Max(0, options.HueUncertaintyScale) * vti + 1e-12);
                gc = 1.0 + qh * (gc - 1.0);
                gainV[i] = (float)gv;
                gainC[i] = (float)gc;
                hueConfidence[i] = (float)Math.Clamp(qh, 0, 1);
            }

            var output = new float[pixels * 3]; var projectionDistance = new float[pixels];
            double sumV = 0, sumC = 0, sumHue = 0, sumDistance = 0, clipAmount = 0;
            long invalidBefore = 0, invalidAfter = 0, projected = 0;
            for (int i = 0; i < pixels; i++)
            {
                int j = i * 3;
                double rawB = a[0] + gainV[i] * pComponents[j] + gainC[i] * qComponents[j];
                double rawG = a[1] + gainV[i] * pComponents[j + 1] + gainC[i] * qComponents[j + 1];
                double rawR = a[2] + gainV[i] * pComponents[j + 2] + gainC[i] * qComponents[j + 2];
                invalidBefore += Invalid(rawB) + Invalid(rawG) + Invalid(rawR);
                clipAmount += ClipDistance(rawB) + ClipDistance(rawG) + ClipDistance(rawR);

                double outB = rawB, outG = rawG, outR = rawR, distance = 0;
                if (options.UseFeasibility)
                {
                    var safe = HcvA2crFeasibleProjector.Project(a, pComponents.AsSpan(j, 3),
                        qComponents.AsSpan(j, 3), gainV[i], gainC[i], gainMax);
                    gainV[i] = (float)safe.GainValue;
                    gainC[i] = (float)safe.GainChroma;
                    outB = safe.B; outG = safe.G; outR = safe.R; distance = safe.Distance;
                    if (safe.WasProjected) projected++;
                }
                output[j] = (float)Math.Clamp(outB, 0, 1);
                output[j + 1] = (float)Math.Clamp(outG, 0, 1);
                output[j + 2] = (float)Math.Clamp(outR, 0, 1);
                invalidAfter += Invalid(output[j]) + Invalid(output[j + 1]) + Invalid(output[j + 2]);
                projectionDistance[i] = (float)distance;
                sumV += gainV[i]; sumC += gainC[i]; sumHue += hueConfidence[i]; sumDistance += distance;
            }

            var diagnostics = new HcvA2crDiagnosticSummary(
                sumV / pixels, sumC / pixels, sumHue / pixels,
                projected / (double)pixels,
                invalidBefore / (double)(pixels * 3),
                invalidAfter / (double)(pixels * 3),
                clipAmount / (pixels * 3), sumDistance / pixels);
            return new HcvA2crRecoveryResult(ToMat(output, rows, cols, 3),
                ToMat(gainV, rows, cols, 1), ToMat(gainC, rows, cols, 1),
                ToMat(hueConfidence, rows, cols, 1), ToMat(projectionDistance, rows, cols, 1), diagnostics);
        }

        private static int Invalid(double value) => value < -1e-6 || value > 1 + 1e-6 ? 1 : 0;
        private static double ClipDistance(double value) => value < 0 ? -value : value > 1 ? value - 1 : 0;

        private static float[] LocalMean(float[] source, int rows, int cols, int radius)
        {
            if (radius <= 0) return source;
            using var src = ToMat(source, rows, cols, 1);
            using var dst = new Mat();
            int size = 2 * radius + 1;
            CvInvoke.Blur(src, dst, new System.Drawing.Size(size, size), new System.Drawing.Point(-1, -1));
            var result = new float[source.Length]; dst.CopyTo(result); return result;
        }

        private static void EstimateCoordinateAirlightUncertainty(float[] input, int pixels,
            IReadOnlyList<Emgu.CV.Structure.MCvScalar> samples, double scale,
            float[] valueVariance, float[] chromaVariance)
        {
            scale = Math.Max(0, scale);
            Span<double> values = stackalloc double[samples.Count];
            Span<double> chromas = stackalloc double[samples.Count];
            for (int i = 0; i < pixels; i++)
            {
                int j = i * 3;
                for (int sample = 0; sample < samples.Count; sample++)
                {
                    double ab = Math.Max(0.02, samples[sample].V0);
                    double ag = Math.Max(0.02, samples[sample].V1);
                    double ar = Math.Max(0.02, samples[sample].V2);
                    double xb = input[j] / ab, xg = input[j + 1] / ag, xr = input[j + 2] / ar;
                    double maximum = Math.Max(xb, Math.Max(xg, xr));
                    double minimum = Math.Min(xb, Math.Min(xg, xr));
                    values[sample] = maximum - 1.0;
                    chromas[sample] = maximum - minimum;
                }
                double centerV = Median(values), centerC = Median(chromas);
                double varianceV = 0, varianceC = 0;
                for (int sample = 0; sample < samples.Count; sample++)
                {
                    varianceV += (values[sample] - centerV) * (values[sample] - centerV);
                    varianceC += (chromas[sample] - centerC) * (chromas[sample] - centerC);
                }
                valueVariance[i] = (float)(scale * varianceV / samples.Count);
                chromaVariance[i] = (float)(scale * varianceC / samples.Count);
            }
        }

        private static double Median(ReadOnlySpan<double> values)
        {
            Span<double> sorted = stackalloc double[values.Length];
            values.CopyTo(sorted);
            sorted.Sort();
            return sorted[sorted.Length / 2];
        }

        private static Mat ToMat(float[] data, int rows, int cols, int channels)
        {
            var mat = new Mat(rows, cols, DepthType.Cv32F, channels);
            Marshal.Copy(data, 0, mat.DataPointer, data.Length);
            return mat;
        }
    }

    internal readonly record struct HcvA2crProjection(
        double GainValue, double GainChroma, double B, double G, double R,
        double Distance, bool WasProjected);

    internal static class HcvA2crFeasibleProjector
    {
        public static HcvA2crProjection Project(ReadOnlySpan<double> airlight,
            ReadOnlySpan<float> valueComponents, ReadOnlySpan<float> chromaComponents,
            double gainValue, double gainChroma, double gainMax)
        {
            if (airlight.Length != 3 || valueComponents.Length != 3 || chromaComponents.Length != 3)
                throw new ArgumentException("HCV projector requires three channels");
            gainMax = Math.Max(1, gainMax);
            Span<double> ca = stackalloc double[10];
            Span<double> cb = stackalloc double[10];
            Span<double> cc = stackalloc double[10];
            ca[0] = -1; ca[1] = 1; cb[0] = cb[1] = 0; cc[0] = -1; cc[1] = gainMax;
            ca[2] = ca[3] = 0; cb[2] = -1; cb[3] = 1; cc[2] = -1; cc[3] = gainMax;
            for (int channel = 0; channel < 3; channel++)
            {
                int lower = 4 + channel * 2, upper = lower + 1;
                double p = valueComponents[channel], q = chromaComponents[channel];
                ca[lower] = -p; cb[lower] = -q; cc[lower] = airlight[channel];
                ca[upper] = p; cb[upper] = q; cc[upper] = 1.0 - airlight[channel];
            }

            double bestV = 1, bestC = 1;
            double bestDistance2 = SquaredDistance(gainValue, gainChroma, bestV, bestC);
            if (Feasible(gainValue, gainChroma, ca, cb, cc))
            {
                bestV = gainValue; bestC = gainChroma; bestDistance2 = 0;
            }
            else
            {
                for (int i = 0; i < ca.Length; i++)
                {
                    double norm2 = ca[i] * ca[i] + cb[i] * cb[i];
                    if (norm2 <= 1e-24) continue;
                    double excess = ca[i] * gainValue + cb[i] * gainChroma - cc[i];
                    ConsiderCandidate(gainValue - excess * ca[i] / norm2,
                        gainChroma - excess * cb[i] / norm2, gainValue, gainChroma,
                        ca, cb, cc, ref bestV, ref bestC, ref bestDistance2);
                }
                for (int i = 0; i < ca.Length; i++)
                for (int j = i + 1; j < ca.Length; j++)
                {
                    double determinant = ca[i] * cb[j] - ca[j] * cb[i];
                    if (Math.Abs(determinant) <= 1e-18) continue;
                    ConsiderCandidate((cc[i] * cb[j] - cc[j] * cb[i]) / determinant,
                        (ca[i] * cc[j] - ca[j] * cc[i]) / determinant,
                        gainValue, gainChroma, ca, cb, cc,
                        ref bestV, ref bestC, ref bestDistance2);
                }
            }

            double b = airlight[0] + bestV * valueComponents[0] + bestC * chromaComponents[0];
            double g = airlight[1] + bestV * valueComponents[1] + bestC * chromaComponents[1];
            double r = airlight[2] + bestV * valueComponents[2] + bestC * chromaComponents[2];
            double distance = Math.Sqrt(Math.Max(0, bestDistance2));
            return new HcvA2crProjection(bestV, bestC, b, g, r, distance, distance > 1e-10);

        }

        private static void ConsiderCandidate(double candidateV, double candidateC,
            double proposedV, double proposedC, ReadOnlySpan<double> a,
            ReadOnlySpan<double> b, ReadOnlySpan<double> c,
            ref double bestV, ref double bestC, ref double bestDistance2)
        {
            if (!double.IsFinite(candidateV) || !double.IsFinite(candidateC) ||
                !Feasible(candidateV, candidateC, a, b, c)) return;
            double distance2 = SquaredDistance(proposedV, proposedC, candidateV, candidateC);
            if (distance2 < bestDistance2)
            {
                bestDistance2 = distance2; bestV = candidateV; bestC = candidateC;
            }
        }

        private static bool Feasible(double x, double y, ReadOnlySpan<double> a,
            ReadOnlySpan<double> b, ReadOnlySpan<double> c)
        {
            for (int i = 0; i < a.Length; i++)
                if (a[i] * x + b[i] * y > c[i] + 2e-10) return false;
            return true;
        }

        private static double SquaredDistance(double x, double y, double px, double py)
        {
            double dx = x - px, dy = y - py;
            return dx * dx + dy * dy;
        }
    }
}
