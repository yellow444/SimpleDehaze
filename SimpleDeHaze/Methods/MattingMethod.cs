using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>DCP с глобальным уточнением t в стиле matting-Laplacian (WLS). См. docs/methods/laplacian-matting.md.</summary>
    public sealed class MattingMethod : IDeHazeMethod
    {
        public string Name => "DCP - Matting Laplacian (WLS)";

        public string Description =>
            "Классический DCP с глобальным edge-preserving уточнением t (семейство матового лапласиана).\n\n" +
            "Минимизируется энергия:  E(t) = Σ (t-t)^2 + λ*Σ w*(∇t)^2 ,\n" +
            "где веса w малы на яркостных границах кадра. Решается матрично-свободно (взвешенный Якоби),\n" +
            "полное разрешение, память O(N) - практичная альтернатива разреженным матрицам NxN.\n\n" +
            "Полное разрешение, итеративный решатель (на больших кадрах медленнее).\n" +
            "Параметры: λ - гладкость; iters - итераций решателя; ω, patch - DCP.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega", "ω - доля удаляемой дымки", 0.3, 0.95, 0.5, search: true),
            new ParamDef("patch", "Патч тёмного канала",      1,   15,   5, 1, isInt: true),
            new ParamDef("min",   "t_min - нижний порог t",   0.01, 0.5, 0.1),
            new ParamDef("lambda","λ - гладкость",            1,   300,  40, log: true, search: true),
            new ParamDef("iters", "Итераций решателя",        5,   80,   30, 1, isInt: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
            => DehazeCore.Run(input, p["omega"], (int)p["patch"], p["min"],
                              (i, t) => Refiners.Wls(i, t, p["lambda"], (int)p["iters"]));
    }
}
