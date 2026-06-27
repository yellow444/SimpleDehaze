using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>DCP с уточнением t анизотропной диффузией (Beltrami), итерации на GPU (CUDA).</summary>
    public sealed class BeltramiGpuMethod : IDeHazeMethod
    {
        public string Name => "DCP - Beltrami Flow (GPU)";

        public string Description =>
            "То же, что 'DCP - Beltrami Flow', но итерации диффузии выполняются на GPU (CUDA):\n" +
            "ядро DCP - на CPU, карта t уточняется на видеокарте.\n\n" +
            "∂t/∂τ = c(|∇I|)*Δt,  c = exp(-|∇I|^2/κ^2). Каждая итерация поэлементная ->\n" +
            "хорошо параллелится на GPU.\n\n" +
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
            => GpuCore.Run(input, p["omega"], (int)p["patch"], p["min"],
                           (gi, gt) => GpuRefiners.BeltramiCore(gi, gt, (int)p["iters"], p["kappa"]));
    }
}
