using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>DCP с уточнением t Domain Transform фильтром (edge-aware, O(N)). См. docs/methods/more-ideas.md.</summary>
    public sealed class DomainTransformMethod : IDeHazeMethod
    {
        public string Name => "DCP - Domain Transform";

        public string Description =>
            "DCP, но карта t уточняется Domain Transform фильтром (Gastal & Oliveira, 2011) -\n" +
            "edge-aware сглаживание за O(N) через 'геодезическое' 1D-преобразование координат,\n" +
            "применяемое сепарабельно по строкам и столбцам. Очень быстро, годится для видео.\n\n" +
            "Параметры: σ_s - пространственный масштаб; σ_r - цветовой; ω, patch - DCP.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",  "ω - доля удаляемой дымки", 0.3, 0.95, 0.5, search: true),
            new ParamDef("patch",  "Патч тёмного канала",      1, 15, 5, 1, isInt: true),
            new ParamDef("min",    "t_min - нижний порог t",   0.01, 0.5, 0.1),
            new ParamDef("sigmaS", "σ_s - пространственный",   10, 120, 40, search: true),
            new ParamDef("sigmaR", "σ_r - цветовой",           0.05, 1.0, 0.2),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
            => DehazeCore.Run(input, p["omega"], (int)p["patch"], p["min"], (i, t) =>
            {
                var dst = new Mat();
                XImgprocInvoke.DtFilter(i, t, dst, p["sigmaS"], p["sigmaR"], DtFilterType.NC, 3);
                return dst;
            });
    }
}
