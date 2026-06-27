using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>DCP с уточнением t полной вариацией (TV, primal-dual). См. docs/methods/more-ideas.md.</summary>
    public sealed class TvMethod : IDeHazeMethod
    {
        public string Name => "DCP - Total Variation";

        public string Description =>
            "DCP, но карта t уточняется минимизацией Total Variation:\n" +
            "    min_t  1/2||t - t||^2 + λ*TV(t),   TV(t)=Σ||∇t||.\n\n" +
            "l1-штраф на градиент сохраняет резкие границы (в отличие от l2-сглаживания).\n" +
            "Решается matrix-free алгоритмом Чамболля-Пока (только градиент/дивергенция и проекция).\n\n" +
            "Параметры: λ - гладкость; iters - итераций; ω, patch - DCP.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",  "ω - доля удаляемой дымки", 0.3, 0.95, 0.5, search: true),
            new ParamDef("patch",  "Патч тёмного канала",      1, 15, 5, 1, isInt: true),
            new ParamDef("min",    "t_min - нижний порог t",   0.01, 0.5, 0.1),
            new ParamDef("lambda", "λ - гладкость",            0.01, 2.0, 0.2, log: true, search: true),
            new ParamDef("iters",  "Итераций",                 10, 120, 40, 1, isInt: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
            => DehazeCore.Run(input, p["omega"], (int)p["patch"], p["min"],
                              (i, t) => Refiners.Tv(i, t, p["lambda"], (int)p["iters"]));
    }
}
