using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Airlight-normalized HCV recovery on the exact same physical front-end as RGB A²CR.
    /// This is a research hypothesis and controlled coordinate-system ablation, not a claim that
    /// HSV/HCV dehazing or hue preservation are new by themselves.
    /// </summary>
    public sealed class HcvA2crMethod : IDeHazeMethod
    {
        private static readonly HashSet<string> ExcludedA2crParameters = new(StringComparer.Ordinal)
        {
            "tv", "tviter", "tvedge", CudaBackend.ParameterKey,
        };

        internal static IReadOnlyList<ParamDef> PhysicalParameters { get; } = new A2crMethod().Parameters
            .Where(definition => !ExcludedA2crParameters.Contains(definition.Key))
            .Select(definition => new ParamDef(
                definition.Key, definition.Label, definition.Min, definition.Max,
                definition.Key switch
                {
                    "tunc" => 0.5,
                    "airunc" => 0.5,
                    "noise" => 0.004,
                    _ => definition.Default,
                },
                definition.Step, definition.IsInt, definition.Log,
                definition.Search, definition.Tunable))
            .Concat(new[]
            {
                new ParamDef("family", "Fusion: равный вес prior families (0/1)", 0, 1, 0, 1,
                    isInt: true, tunable: false),
                new ParamDef("refvar", "Uncertainty: пересчитать Var(t) после refine (0/1)", 0, 1, 1, 1,
                    isInt: true, tunable: false),
                new ParamDef("hueunc", "Hue confidence: вес uncertainty(t)", 0, 20, 1),
            })
            .ToArray();

        public string Name => "HCV-A²CR (airlight-normalized, эксперимент)";

        public string Description =>
            "Точная HCV-координатная абляция A²CR в линейном RGB. Сначала X=I/A, затем " +
            "C=max(X)-min(X), V=max(X). Для scalar atmospheric model выполняются " +
            "C_X=t·C_Y, V_X-1=t(V_Y-1), а hue-sector сохраняется.\n\n" +
            "Value-offset и абсолютная chroma получают отдельные uncertainty-aware gains gV/gC. " +
            "Возле серой оси chroma gain возвращается к identity, а proposed gains проецируются " +
            "на точный per-pixel RGB-feasible многоугольник. При gV=gC=1/t результат совпадает " +
            "с классической RGB-инверсией. Новизна проверяется у совместной risk/feasibility " +
            "композиции, а не у применения HSV как такового.";

        public IReadOnlyList<ParamDef> Parameters => PhysicalParameters;

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> parameters)
        {
            using var execution = Execute(input, parameters);
            return execution.SrgbResult.Clone();
        }

        internal static HcvA2crExecution Execute(Image<Bgr, byte> input,
            IReadOnlyDictionary<string, double> p)
        {
            var scene = A2crMethod.EstimateScene(input, p);
            try
            {
                var options = CreateRecoveryOptions(p);
                var recovery = HcvA2crRecovery.Recover(scene.LinearInput, scene.Transmission,
                    scene.TransmissionVariance, scene.Airlight, options);
                var srgb = ColorSpace.ToSrgb(recovery.LinearResult);
                return new HcvA2crExecution(scene, recovery, srgb);
            }
            catch
            {
                scene.Dispose();
                throw;
            }
        }

        internal static HcvA2crRecoveryOptions CreateRecoveryOptions(IReadOnlyDictionary<string, double> p)
            => new(
                p["min"], p["noise"] * p["noise"], p["airunc"],
                p["noiseon"] >= 0.5, p["tuncon"] >= 0.5, p["auncon"] >= 0.5,
                p["feasible"] >= 0.5, (int)p["radius"], p["couple"], p["hueunc"]);
    }

    internal sealed class HcvA2crExecution : IDisposable
    {
        public A2crSceneEstimate Scene { get; }
        public HcvA2crRecoveryResult Recovery { get; }
        public Mat SrgbResult { get; }

        public HcvA2crExecution(A2crSceneEstimate scene, HcvA2crRecoveryResult recovery, Mat srgbResult)
        {
            Scene = scene;
            Recovery = recovery;
            SrgbResult = srgbResult;
        }

        public void Dispose()
        {
            SrgbResult.Dispose();
            Recovery.Dispose();
            Scene.Dispose();
        }
    }
}
