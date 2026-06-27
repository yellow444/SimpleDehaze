using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>DCP с уточнением карты t дробным лапласианом (через ДПФ). См. docs/methods/fractional-laplacian.md.</summary>
    public sealed class FractionalMethod : IDeHazeMethod
    {
        public string Name => "DCP - Fractional Laplacian";

        public string Description =>
            "Классический DCP, но карта t сглаживается дробным лапласианом (-Δ)^α в частотной области (ДПФ).\n\n" +
            "Грубая t = 1 - ω*darkChannel(I/A). Уточнение - закрытая формула в Фурье:\n" +
            "    t(ξ) = t(ξ) / ( 1 + λ*|ξ|^{2α} )\n" +
            "О(N*logN), мало памяти. Изотропно (не edge-aware), даёт очень гладкую карту.\n\n" +
            "Параметры: α - порядок производной; λ - сила сглаживания; ω, patch - DCP.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega", "ω - доля удаляемой дымки", 0.3, 0.95, 0.5, search: true),
            new ParamDef("patch", "Патч тёмного канала",      1,   15,   5, 1, isInt: true),
            new ParamDef("min",   "t_min - нижний порог t",   0.01, 0.5, 0.1),
            new ParamDef("alpha", "α - порядок (-Δ)^α",       0.3, 2.0,  0.8, search: true),
            new ParamDef("lambda","λ - сила сглаживания",     1,   300,  30, log: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
            => DehazeCore.Run(input, p["omega"], (int)p["patch"], p["min"],
                              (i, t) => Refiners.Fractional(i, t, p["alpha"], p["lambda"]));
    }
}
