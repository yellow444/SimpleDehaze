using System.Text.Json;

using Emgu.CV;
using Emgu.CV.CvEnum;

namespace SimpleDeHaze.Methods
{
    internal static class A2crDiagnostics
    {
        public static void Save(A2crExecution execution, string directory,
            IReadOnlyDictionary<string, double>? parameters = null)
        {
            directory = Path.GetFullPath(directory);
            Directory.CreateDirectory(directory);
            using (var result8 = new Mat())
            {
                execution.SrgbResult.ConvertTo(result8, DepthType.Cv8U, 255);
                CvInvoke.Imwrite(Path.Combine(directory, "result.png"), result8);
            }
            SaveHeatmap(execution.Transmission, Path.Combine(directory, "transmission.png"));
            SaveHeatmap(execution.TransmissionVariance, Path.Combine(directory, "transmission-variance.png"));
            SaveHeatmap(execution.SigmaDepth, Path.Combine(directory, "sigma-depth.png"));
            SaveHeatmap(execution.Recovery.GainParallel, Path.Combine(directory, "gain-parallel.png"));
            SaveHeatmap(execution.Recovery.GainPerpendicular, Path.Combine(directory, "gain-perpendicular.png"));
            SaveHeatmap(execution.Recovery.ProjectionAlpha, Path.Combine(directory, "projection-alpha.png"), invert: true);

            var d = execution.Recovery.Diagnostics;
            File.WriteAllText(Path.Combine(directory, "diagnostics.json"), JsonSerializer.Serialize(new
            {
                airlight_bgr_linear = new[] { execution.Airlight.Value.V0, execution.Airlight.Value.V1, execution.Airlight.Value.V2 },
                airlight_channel_variance = execution.Airlight.ChannelVariance,
                airlight_samples_bgr_linear = execution.Airlight.Samples.Select(x => new[] { x.V0, x.V1, x.V2 }),
                diagnostics = new
                {
                    d.MeanGainParallel, d.MeanGainPerpendicular, d.MeanAlpha,
                    d.ProjectedPixelFraction, d.InvalidChannelFractionBefore,
                    d.InvalidChannelFractionAfter, d.MeanRequiredClipping
                },
                parameters
            }, new JsonSerializerOptions { WriteIndented = true }));
        }

        private static void SaveHeatmap(Mat source, string path, bool invert = false)
        {
            using var normalized = new Mat();
            CvInvoke.Normalize(source, normalized, 0, 255, NormType.MinMax, DepthType.Cv8U);
            if (invert) CvInvoke.Subtract(new ScalarArray(255), normalized, normalized);
            using var colored = new Mat();
            CvInvoke.ApplyColorMap(normalized, colored, ColorMapType.Turbo);
            CvInvoke.Imwrite(path, colored);
        }
    }
}
