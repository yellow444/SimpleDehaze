using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Validation-selected companion of <see cref="TransScaleLaplacianMethod"/>. It keeps the same
    /// transmission, airlight and finishing pipeline, changing only the analyzed scalar (HSV V)
    /// and the multiscale basis (full-resolution Domain-Transform residual bands).
    /// </summary>
    public sealed class TransmissionAwareHsvEdgeMethod : IDeHazeMethod
    {
        private static readonly TransScaleLaplacianMethod BaseMethod = new();

        public string Name => "Transmission-aware HSV Edge Bands (эксперимент)";

        public string Description =>
            "HSV-V + edge-aware companion к Transmission-aware Laplacian. Оценка transmission, " +
            "локального atmospheric light и весь recovery одинаковы с RGB/Lab-L baseline. " +
            "Различаются только два фактора:\n\n" +
            "1. Полосы строятся по HSV Value; Hue и Saturation не фильтруются как линейные сигналы.\n" +
            "2. Вместо downsampled Gaussian/Laplacian pyramid используются полноразмерные " +
            "Domain-Transform residual bands B_l=F_{l-1}(V)-F_l(V).\n" +
            "3. Каждая полоса получает тот же transmission/scale gate: мелкие детали подавляются " +
            "в плотной дымке, крупные структуры сохраняются.\n\n" +
            "Сам Domain Transform и residual decomposition известны; исследовательский кандидат — " +
            "их training-free композиция с transmission-conditioned gate. В O-HAZE pilot вариант " +
            "улучшил baseline, но ещё не является независимым SOTA-результатом.";

        public IReadOnlyList<ParamDef> Parameters { get; } = BaseMethod.Parameters
            .Where(definition => definition.Key is not "space" and not "basis" and not "wiener" and
                not "uNoise" and not "uUnc" and not "uRadius" and not "uLimit")
            .Select(definition => new ParamDef(
                definition.Key,
                definition.Label,
                definition.Min,
                definition.Max,
                definition.Key switch
                {
                    "gFine" => 0.5,
                    "gMid" => 1.6,
                    "gCoarse" => 1.2,
                    _ => definition.Default,
                },
                definition.Step,
                definition.IsInt,
                definition.Log,
                definition.Search,
                definition.Tunable || definition.Key is "edgeS" or "edgeR"))
            .ToArray();

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> parameters)
        {
            var selected = new Dictionary<string, double>(parameters, StringComparer.Ordinal)
            {
                ["space"] = 1,
                ["basis"] = 1,
            };
            return BaseMethod.Process(input, selected);
        }
    }
}
