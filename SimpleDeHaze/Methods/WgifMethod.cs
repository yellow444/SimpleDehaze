using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>DCP с уточнением t Weighted Guided Filter (адаптивная ε, меньше ореолов). См. docs/methods/guided-filter-variants.md.</summary>
    public sealed class WgifMethod : IDeHazeMethod
    {
        public string Name => "DCP - Weighted Guided Filter";

        public string Description =>
            "DCP, но уточнение t - Weighted Guided Image Filter (Li, 2015) вместо базового guided filter.\n\n" +
            "Регуляризация ε делается пространственной: ε/Γ(x), где Γ - edge-aware вес по локальной\n" +
            "дисперсии яркости. На краях Γ велик -> эффективная ε мала -> край сохраняется (меньше halo);\n" +
            "на гладком - сильнее сглаживание. Гайд - яркость (одноканальный, быстрый).\n\n" +
            "Параметры: ω - сила; refine - радиус; ε - базовая регуляризация.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega", "ω - доля удаляемой дымки", 0.3, 0.95, 0.5, search: true),
            new ParamDef("patch", "Патч тёмного канала",      1, 15, 5, 1, isInt: true),
            new ParamDef("min",   "t_min - нижний порог t",   0.01, 0.5, 0.1),
            new ParamDef("refine","Радиус WGIF",              5, 120, 50, 1, isInt: true),
            new ParamDef("eps",   "ε - базовая регуляризация", 1e-4, 0.5, 0.01, log: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
            => DehazeCore.Run(input, p["omega"], (int)p["patch"], p["min"],
                              (i, t) => Refiners.Wgif(i, t, (int)p["refine"], p["eps"]));
    }
}
