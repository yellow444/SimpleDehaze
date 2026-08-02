using Emgu.CV;
using Emgu.CV.CvEnum;

namespace SimpleDeHaze.Benchmarking;

internal readonly record struct BenchmarkColorErrorReport(
    double HueErrorDegrees, double ChromaError, double ChromaticPixelFraction);

internal static class BenchmarkColorErrors
{
    /// <summary>
    /// Lab hue is undefined near the neutral axis. Report a reference-chroma-weighted circular
    /// error only where C*_ab(gt) >= 2, plus the evaluated pixel fraction. Chroma error uses all pixels.
    /// </summary>
    public static BenchmarkColorErrorReport Evaluate(Mat resultSrgb, Mat groundTruthSrgb)
    {
        using var resultLab = new Mat(); using var truthLab = new Mat();
        CvInvoke.CvtColor(resultSrgb, resultLab, ColorConversion.Bgr2Lab);
        CvInvoke.CvtColor(groundTruthSrgb, truthLab, ColorConversion.Bgr2Lab);
        var result = new float[resultLab.Rows * resultLab.Cols * 3];
        var truth = new float[result.Length]; resultLab.CopyTo(result); truthLab.CopyTo(truth);
        double hueWeightedSum = 0, hueWeight = 0, chromaSum = 0; int chromatic = 0;
        for (int i = 0; i < result.Length; i += 3)
        {
            double resultChroma = Math.Sqrt(result[i + 1] * result[i + 1] + result[i + 2] * result[i + 2]);
            double truthChroma = Math.Sqrt(truth[i + 1] * truth[i + 1] + truth[i + 2] * truth[i + 2]);
            chromaSum += Math.Abs(resultChroma - truthChroma);
            if (truthChroma < 2.0) continue;
            double resultHue = Math.Atan2(result[i + 2], result[i + 1]);
            double truthHue = Math.Atan2(truth[i + 2], truth[i + 1]);
            double delta = Math.Abs(resultHue - truthHue); if (delta > Math.PI) delta = 2 * Math.PI - delta;
            double weight = Math.Min(50.0, truthChroma);
            hueWeightedSum += delta * 180.0 / Math.PI * weight;
            hueWeight += weight; chromatic++;
        }
        int pixels = resultLab.Rows * resultLab.Cols;
        return new BenchmarkColorErrorReport(hueWeight > 0 ? hueWeightedSum / hueWeight : 0,
            chromaSum / Math.Max(1, pixels), chromatic / (double)Math.Max(1, pixels));
    }
}
