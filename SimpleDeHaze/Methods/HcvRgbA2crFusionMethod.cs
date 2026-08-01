using System.Runtime.InteropServices;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Validity-gated convex fusion of two recovery coordinate systems sharing one physical scene
    /// estimate. The gate uses HCV hue confidence, gamut projection distance, RGB projection
    /// headroom and forward re-hazing residual rather than a fixed HSV/RGB mixing coefficient.
    /// </summary>
    public sealed class HcvRgbA2crFusionMethod : IDeHazeMethod
    {
        public string Name => "HCV↔RGB A²CR validity fusion (эксперимент)";

        public string Description =>
            "RGB A²CR и airlight-normalized HCV-A²CR получают один и тот же A, t и uncertainty. " +
            "Их допустимые linear-RGB результаты смешиваются выпукло по аналитической validity: " +
            "HCV hue-confidence, расстоянию gamut-проекции и ошибке прямого re-hazing; RGB-ветвь " +
            "штрафуется за сильную feasible-проекцию.\n\n" +
            "HSV recovery, hue preservation и airlight-centered geometry известны. Проверяемая " +
            "гипотеза здесь — validity gate между двумя uncertainty-aware recovery systems, а не " +
            "сам факт использования двух цветовых пространств.";

        public IReadOnlyList<ParamDef> Parameters { get; } = HcvA2crMethod.PhysicalParameters
            .Concat(new[]
            {
                new ParamDef("hcvPrior", "Fusion: prior веса HCV", 0.05, 0.95, 0.5),
                new ParamDef("fwdTau", "Fusion: scale forward residual", 1e-5, 0.05, 0.002, log: true),
                new ParamDef("projTau", "Fusion: scale HCV projection", 0.01, 3, 0.5, log: true),
            })
            .ToArray();

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            using var scene = A2crMethod.EstimateScene(input, p);
            var rgbOptions = new A2crRecoveryOptions(p["min"], p["noise"] * p["noise"], p["airunc"],
                p["noiseon"] >= 0.5, p["tuncon"] >= 0.5, p["auncon"] >= 0.5,
                p["feasible"] >= 0.5, (int)p["radius"], p["couple"], 0, 1, 0.08);
            using var rgb = A2crRecovery.Recover(scene.LinearInput, scene.Transmission,
                scene.TransmissionVariance, scene.Airlight, rgbOptions);
            using var hcv = HcvA2crRecovery.Recover(scene.LinearInput, scene.Transmission,
                scene.TransmissionVariance, scene.Airlight, HcvA2crMethod.CreateRecoveryOptions(p));
            using var fused = Blend(scene, rgb, hcv, p["hcvPrior"], p["fwdTau"], p["projTau"]);
            return ColorSpace.ToSrgb(fused);
        }

        internal static Mat Blend(A2crSceneEstimate scene, A2crRecoveryResult rgb,
            HcvA2crRecoveryResult hcv, double hcvPrior, double forwardTau, double projectionTau)
        {
            int pixels = scene.LinearInput.Rows * scene.LinearInput.Cols;
            var input = new float[pixels * 3]; var rgbData = new float[pixels * 3]; var hcvData = new float[pixels * 3];
            var t = new float[pixels]; var hue = new float[pixels]; var hcvDistance = new float[pixels];
            var rgbAlpha = new float[pixels];
            scene.LinearInput.CopyTo(input); rgb.LinearResult.CopyTo(rgbData); hcv.LinearResult.CopyTo(hcvData);
            scene.Transmission.CopyTo(t); hcv.HueConfidence.CopyTo(hue);
            hcv.ProjectionDistance.CopyTo(hcvDistance); rgb.ProjectionAlpha.CopyTo(rgbAlpha);
            double[] air = { scene.Airlight.Value.V0, scene.Airlight.Value.V1, scene.Airlight.Value.V2 };
            hcvPrior = Math.Clamp(hcvPrior, 0.001, 0.999);
            forwardTau = Math.Max(1e-10, forwardTau);
            projectionTau = Math.Max(1e-10, projectionTau);
            var output = new float[input.Length];
            for (int i = 0; i < pixels; i++)
            {
                int j = i * 3;
                double errorRgb = 0, errorHcv = 0;
                for (int channel = 0; channel < 3; channel++)
                {
                    double predictedRgb = t[i] * rgbData[j + channel] + (1 - t[i]) * air[channel];
                    double predictedHcv = t[i] * hcvData[j + channel] + (1 - t[i]) * air[channel];
                    errorRgb += (predictedRgb - input[j + channel]) * (predictedRgb - input[j + channel]);
                    errorHcv += (predictedHcv - input[j + channel]) * (predictedHcv - input[j + channel]);
                }
                double scoreHcv = Math.Clamp(hue[i], 0, 1) *
                    Math.Exp(-errorHcv / forwardTau) * Math.Exp(-hcvDistance[i] / projectionTau);
                double scoreRgb = (0.25 + 0.75 * Math.Clamp(rgbAlpha[i], 0, 1)) *
                    Math.Exp(-errorRgb / forwardTau);
                double weightedHcv = hcvPrior * scoreHcv;
                double weightedRgb = (1 - hcvPrior) * scoreRgb;
                double weight = weightedHcv + weightedRgb > 1e-20
                    ? weightedHcv / (weightedHcv + weightedRgb) : 0;
                for (int channel = 0; channel < 3; channel++)
                    output[j + channel] = (float)Math.Clamp(
                        weight * hcvData[j + channel] + (1 - weight) * rgbData[j + channel], 0, 1);
            }
            var result = new Mat(scene.LinearInput.Rows, scene.LinearInput.Cols, DepthType.Cv32F, 3);
            Marshal.Copy(output, 0, result.DataPointer, output.Length);
            return result;
        }
    }
}
