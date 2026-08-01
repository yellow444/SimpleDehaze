using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    public sealed class TransmissionAwareHsvUtawMethod : IDeHazeMethod
    {
        private static readonly TransScaleLaplacianMethod BaseMethod = new();

        public string Name => "Transmission-aware HSV UTAW (эксперимент)";

        public string Description =>
            "Сохраняет recovery и визуальный характер исходного Transmission-aware, но заменяет " +
            "decimated Laplacian/Domain bands на stationary B3-spline à trous по HSV-V. " +
            "Коэффициент каждой полосы ограничивают local band power, t, noise и optical-depth " +
            "расхождение CAP↔DCP; delta имеет явный energy cap против мозаичной фактуры.\n\n" +
            "Stationary wavelets известны. Исследовательская гипотеза — joint reliability/budget, " +
            "а не замена Laplacian сама по себе.";

        public IReadOnlyList<ParamDef> Parameters { get; } = BaseMethod.Parameters
            .Where(definition => definition.Key is not "space" and not "basis" and not "wiener" and
                not "edgeS" and not "edgeR")
            .Select(definition => new ParamDef(
                definition.Key, definition.Label, definition.Min, definition.Max,
                definition.Key switch
                {
                    "gFine" => 0.5,
                    "gMid" => 1.6,
                    "gCoarse" => 1.2,
                    _ => definition.Default,
                }, definition.Step, definition.IsInt, definition.Log,
                definition.Search || definition.Key is "uNoise" or "uUnc" or "uLimit",
                definition.Tunable || definition.Key is "uNoise" or "uUnc" or "uRadius" or "uLimit"))
            .ToArray();

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> parameters)
        {
            var selected = new Dictionary<string, double>(parameters, StringComparer.Ordinal)
            {
                ["space"] = 1,
                ["basis"] = 2,
            };
            return BaseMethod.Process(input, selected);
        }
    }
}
