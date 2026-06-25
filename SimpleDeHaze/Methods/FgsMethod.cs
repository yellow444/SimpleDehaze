using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>DCP с уточнением t Fast Global Smoother (быстрая WLS-аппроксимация). См. docs/methods/more-ideas.md.</summary>
    public sealed class FgsMethod : IDeHazeMethod
    {
        public string Name => "DCP - Fast Global Smoother";

        public string Description =>
            "DCP, но карта t уточняется Fast Global Smoother (Min et al., 2014) - быстрый\n" +
            "аппроксиматор WLS/edge-aware сглаживания через последовательность 1D-задач.\n" +
            "Ближе к WLS-качеству, чем guided filter, но проще и быстрее полного sparse-solve.\n\n" +
            "Параметры: λ - гладкость; σ_color - чувствительность к краям; ω, patch - DCP.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",  "ω - доля удаляемой дымки", 0.3, 0.95, 0.5, search: true),
            new ParamDef("patch",  "Патч тёмного канала",      1, 15, 5, 1, isInt: true),
            new ParamDef("min",    "t_min - нижний порог t",   0.01, 0.5, 0.1),
            new ParamDef("lambda", "λ - гладкость",            50, 4000, 600, log: true, search: true),
            new ParamDef("sigmaC", "σ_color - края (0..255)",  5, 150, 30),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
            => DehazeCore.Run(input, p["omega"], (int)p["patch"], p["min"], (i, t) =>
            {
                var dst = new Mat();
                using var guide8 = new Mat();
                i.ConvertTo(guide8, DepthType.Cv8U, 255.0);   // FGS требует 8U-гайд
                XImgprocInvoke.FastGlobalSmootherFilter(guide8, t, dst, p["lambda"], p["sigmaC"], 0.25, 3);
                return dst;
            });
    }
}
