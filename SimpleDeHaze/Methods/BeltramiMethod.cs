using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>DCP с уточнением t геометрической (Beltrami-style) анизотропной диффузией. См. docs/methods/beltrami-flow.md.</summary>
    public sealed class BeltramiMethod : IDeHazeMethod
    {
        public string Name => "DCP - Beltrami Flow";

        public string Description =>
            "Классический DCP, но карта t уточняется анизотропной диффузией с краевым торможением по гайду\n" +
            "(в духе потока Бельтрами / Перона-Малика):\n" +
            "    ∂t/∂τ = c(|∇I|)*Δt,   c = exp(-|∇I|^2/κ^2)\n" +
            "Диффузия сильна на однородных областях и тормозится на цветовых границах ->\n" +
            "края сохраняются, цвет/небо стабильнее. Итеративно (N шагов).\n\n" +
            "Параметры: iters - число шагов; κ - порог края; ω, patch - DCP.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega", "ω - доля удаляемой дымки", 0.3, 0.95, 0.5, search: true),
            new ParamDef("patch", "Патч тёмного канала",      1,   15,   5, 1, isInt: true),
            new ParamDef("min",   "t_min - нижний порог t",   0.01, 0.5, 0.1),
            new ParamDef("iters", "Итераций диффузии",        5,   60,   20, 1, isInt: true, search: true),
            new ParamDef("kappa", "κ - порог края",           0.01, 0.5, 0.1, log: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
            => DehazeCore.Run(input, p["omega"], (int)p["patch"], p["min"],
                              (i, t) => Refiners.Beltrami(i, t, (int)p["iters"], p["kappa"]));
    }
}
