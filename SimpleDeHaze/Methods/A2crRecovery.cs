using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    internal sealed record A2crRecoveryOptions(
        double MinimumTransmission,
        double NoiseVariance,
        double AirlightUncertaintyScale,
        bool UseNoise,
        bool UseTransmissionUncertainty,
        bool UseAirlightUncertainty,
        bool UseFeasibility,
        int SignalRadius,
        double GainCoupling,
        double TvWeight,
        int TvIterations,
        double TvEdgeScale);

    internal readonly record struct A2crDiagnosticSummary(
        double MeanGainParallel, double MeanGainPerpendicular, double MeanAlpha,
        double ProjectedPixelFraction, double InvalidChannelFractionBefore,
        double InvalidChannelFractionAfter, double MeanRequiredClipping,
        bool JointTvUsed, int JointTvIterations, double JointTvRelativeChange,
        double JointObjectiveBefore, double JointObjectiveAfter,
        double JointConstraintProjectionFraction);

    internal sealed class A2crRecoveryResult : IDisposable
    {
        public Mat LinearResult { get; }
        public Mat GainParallel { get; }
        public Mat GainPerpendicular { get; }
        public Mat ProjectionAlpha { get; }
        public A2crDiagnosticSummary Diagnostics { get; }

        public A2crRecoveryResult(Mat linearResult, Mat gainParallel, Mat gainPerpendicular,
            Mat projectionAlpha, A2crDiagnosticSummary diagnostics)
        {
            LinearResult = linearResult;
            GainParallel = gainParallel;
            GainPerpendicular = gainPerpendicular;
            ProjectionAlpha = projectionAlpha;
            Diagnostics = diagnostics;
        }

        public void Dispose()
        {
            LinearResult.Dispose();
            GainParallel.Dispose();
            GainPerpendicular.Dispose();
            ProjectionAlpha.Dispose();
        }
    }

    internal static class A2crRecovery
    {
        public static A2crRecoveryResult Recover(Mat inputLinear, Mat transmission, Mat transmissionVariance,
            AirlightEstimate airlight, A2crRecoveryOptions options, bool useCudaLocalEnergy = false)
        {
            if (inputLinear.Depth != DepthType.Cv32F || inputLinear.NumberOfChannels != 3)
                throw new ArgumentException("A²CR input must be BGR float", nameof(inputLinear));
            int rows = inputLinear.Rows, cols = inputLinear.Cols, pixels = rows * cols;
            if (transmission.Rows != rows || transmission.Cols != cols || transmissionVariance.Rows != rows || transmissionVariance.Cols != cols)
                throw new ArgumentException("A²CR maps must match the input size");

            var input = new float[pixels * 3]; var t = new float[pixels]; var varT = new float[pixels];
            inputLinear.CopyTo(input); transmission.CopyTo(t); transmissionVariance.CopyTo(varT);
            double[] a = { airlight.Value.V0, airlight.Value.V1, airlight.Value.V2 };
            double norm = Math.Sqrt(a.Sum(x => x * x));
            double[] u = norm > 1e-12 ? a.Select(x => x / norm).ToArray() : new[] { 1 / Math.Sqrt(3.0), 1 / Math.Sqrt(3.0), 1 / Math.Sqrt(3.0) };
            var axisVariance = airlight.AxisVariances();

            var parallelEnergy = new float[pixels]; var perpendicularEnergy = new float[pixels];
            var pComponents = new float[pixels * 3]; var qComponents = new float[pixels * 3];
            for (int i = 0; i < pixels; i++)
            {
                int j = i * 3;
                double db = input[j] - a[0], dg = input[j + 1] - a[1], dr = input[j + 2] - a[2];
                double dot = u[0] * db + u[1] * dg + u[2] * dr;
                double q2 = 0;
                for (int c = 0; c < 3; c++)
                {
                    double d = input[j + c] - a[c];
                    double pc = u[c] * dot;
                    double qc = d - pc;
                    pComponents[j + c] = (float)pc;
                    qComponents[j + c] = (float)qc;
                    q2 += qc * qc;
                }
                parallelEnergy[i] = (float)(dot * dot);
                perpendicularEnergy[i] = (float)(q2 / 2.0);
            }

            parallelEnergy = LocalMean(parallelEnergy, rows, cols, options.SignalRadius, useCudaLocalEnergy);
            perpendicularEnergy = LocalMean(perpendicularEnergy, rows, cols, options.SignalRadius, useCudaLocalEnergy);
            double noise = options.UseNoise ? Math.Max(0, options.NoiseVariance) : 0;
            double up = options.UseAirlightUncertainty ? Math.Max(0, axisVariance.Parallel * options.AirlightUncertaintyScale) : 0;
            double uq = options.UseAirlightUncertainty ? Math.Max(0, axisVariance.Perpendicular * options.AirlightUncertaintyScale) : 0;
            float gainMax = (float)(1.0 / Math.Clamp(options.MinimumTransmission, 1e-4, 1));
            var gainP = new float[pixels]; var gainQ = new float[pixels];
            var riskAP = new float[pixels]; var riskBP = new float[pixels];
            var riskAQ = new float[pixels]; var riskBQ = new float[pixels];
            for (int i = 0; i < pixels; i++)
            {
                double ti = Math.Clamp(t[i], options.MinimumTransmission, 1);
                double vti = options.UseTransmissionUncertainty ? Math.Max(0, varT[i]) : 0;
                double denominator = Math.Max(1e-10, ti * ti + vti);
                double sp = Math.Max(0, parallelEnergy[i] - noise) / denominator;
                double sq = Math.Max(0, perpendicularEnergy[i] - noise) / denominator;

                double ap = sp * (ti * ti + vti) + noise + up;
                double aq = sq * (ti * ti + vti) + noise + uq;
                double bp = sp * ti + up;
                double bq = sq * ti + uq;
                riskAP[i] = (float)ap; riskAQ[i] = (float)aq;
                riskBP[i] = (float)bp; riskBQ[i] = (float)bq;
                double mu = Math.Max(0, options.GainCoupling);
                double determinant = (ap + mu) * (aq + mu) - mu * mu;
                double gp, gq;
                if (mu > 0 && determinant > 1e-20)
                {
                    gp = (bp * (aq + mu) + mu * bq) / determinant;
                    gq = (bq * (ap + mu) + mu * bp) / determinant;
                    gp = Math.Clamp(gp, 1, gainMax);
                    gq = Math.Clamp(gq, 1, gainMax);
                }
                else
                {
                    gp = A2crRisk.OptimalGain(sp, ti, vti, noise, up, options.MinimumTransmission);
                    gq = A2crRisk.OptimalGain(sq, ti, vti, noise, uq, options.MinimumTransmission);
                }
                gainP[i] = (float)gp; gainQ[i] = (float)gq;
            }

            A2crJointTvResult? jointTv = null;
            if (options.TvWeight > 0 && options.TvIterations > 0)
            {
                jointTv = A2crTvRefiner.SolveJoint(gainP, gainQ,
                    riskAP, riskBP, riskAQ, riskBQ, input, a, rows, cols,
                    options.TvWeight, options.TvEdgeScale, options.GainCoupling,
                    options.TvIterations, gainMax, options.UseFeasibility);
                gainP = jointTv.Value.GainParallel;
                gainQ = jointTv.Value.GainPerpendicular;
            }

            var output = new float[pixels * 3]; var alpha = new float[pixels];
            double sumP = 0, sumQ = 0, sumAlpha = 0, clipAmount = 0;
            long invalidBefore = 0, invalidAfter = 0, projected = 0;
            for (int i = 0; i < pixels; i++)
            {
                int j = i * 3;
                double rawB = a[0] + gainP[i] * pComponents[j] + gainQ[i] * qComponents[j];
                double rawG = a[1] + gainP[i] * pComponents[j + 1] + gainQ[i] * qComponents[j + 1];
                double rawR = a[2] + gainP[i] * pComponents[j + 2] + gainQ[i] * qComponents[j + 2];
                invalidBefore += Invalid(rawB) + Invalid(rawG) + Invalid(rawR);
                clipAmount += ClipDistance(rawB) + ClipDistance(rawG) + ClipDistance(rawR);

                double outB = rawB, outG = rawG, outR = rawR, ai = 1;
                if (options.UseFeasibility)
                {
                    var safe = A2crFeasibleProjector.Project(input[j], input[j + 1], input[j + 2],
                        a[0], a[1], a[2], gainP[i], gainQ[i]);
                    gainP[i] = (float)safe.GainParallel; gainQ[i] = (float)safe.GainPerpendicular;
                    ai = safe.Alpha; outB = safe.B; outG = safe.G; outR = safe.R;
                    if (ai < 1 - 1e-7) projected++;
                }
                alpha[i] = (float)ai;
                output[j] = (float)outB; output[j + 1] = (float)outG; output[j + 2] = (float)outR;
                invalidAfter += Invalid(outB) + Invalid(outG) + Invalid(outR);
                sumP += gainP[i]; sumQ += gainQ[i]; sumAlpha += ai;
            }

            var diagnostics = new A2crDiagnosticSummary(sumP / pixels, sumQ / pixels, sumAlpha / pixels,
                projected / (double)pixels, invalidBefore / (double)(pixels * 3), invalidAfter / (double)(pixels * 3),
                clipAmount / (pixels * 3), jointTv.HasValue,
                jointTv?.Iterations ?? 0, jointTv?.RelativeChange ?? double.NaN,
                jointTv?.ObjectiveBefore ?? double.NaN, jointTv?.ObjectiveAfter ?? double.NaN,
                jointTv?.ConstraintProjectionFraction ?? 0);
            return new A2crRecoveryResult(ToMat(output, rows, cols, 3), ToMat(gainP, rows, cols, 1),
                ToMat(gainQ, rows, cols, 1), ToMat(alpha, rows, cols, 1), diagnostics);
        }

        private static int Invalid(double value) => value < -1e-6 || value > 1 + 1e-6 ? 1 : 0;
        private static double ClipDistance(double value) => value < 0 ? -value : value > 1 ? value - 1 : 0;

        private static float[] LocalMean(float[] source, int rows, int cols, int radius, bool cuda)
        {
            if (radius <= 0) return source;
            using var src = ToMat(source, rows, cols, 1);
            int size = 2 * radius + 1;
            using var dst = new Mat();
            if (cuda)
            {
                CudaBackend.RequireAvailable();
                using var gpuSource = new GpuMat(src);
                using var gpuResult = new GpuMat();
                using var filter = new CudaBoxFilter(DepthType.Cv32F, 1, DepthType.Cv32F, 1,
                    new System.Drawing.Size(size, size), new System.Drawing.Point(-1, -1),
                    BorderType.Reflect101, new MCvScalar());
                filter.Apply(gpuSource, gpuResult);
                gpuResult.Download(dst);
            }
            else CvInvoke.Blur(src, dst, new System.Drawing.Size(size, size), new System.Drawing.Point(-1, -1));
            var result = new float[source.Length]; dst.CopyTo(result); return result;
        }

        private static Mat ToMat(float[] data, int rows, int cols, int channels)
        {
            var mat = new Mat(rows, cols, DepthType.Cv32F, channels);
            Marshal.Copy(data, 0, mat.DataPointer, data.Length);
            return mat;
        }
    }
}
