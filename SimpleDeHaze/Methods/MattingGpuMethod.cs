using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>DCP с глобальным уточнением t (WLS / matting-Laplacian), итерации на GPU (CUDA).</summary>
    public sealed class MattingGpuMethod : IDeHazeMethod
    {
        public string Name => "DCP - Matting WLS (GPU)";

        public string Description =>
            "То же, что 'DCP - Matting Laplacian (WLS)', но итеративный решатель выполняется на GPU (CUDA):\n" +
            "ядро DCP считается на CPU, карта t загружается в GpuMat и уточняется на видеокарте.\n\n" +
            "E(t) = Σ (t-t)^2 + λ*Σ w*(∇t)^2 ,  взвешенный Якоби; каждая итерация - поэлементная,\n" +
            "поэтому отлично ускоряется параллелизмом GPU.\n\n" +
            "Параметры: λ - гладкость; iters - итераций; ω, patch - DCP.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega", "ω - доля удаляемой дымки", 0.3, 0.95, 0.5, search: true),
            new ParamDef("patch", "Патч тёмного канала",      1,   15,   5, 1, isInt: true),
            new ParamDef("min",   "t_min - нижний порог t",   0.01, 0.5, 0.1),
            new ParamDef("lambda","λ - гладкость",            1,   300,  40, log: true, search: true),
            new ParamDef("iters", "Итераций решателя",        5,   80,   30, 1, isInt: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
            => GpuCore.Run(input, p["omega"], (int)p["patch"], p["min"],
                           (gi, gt) => GpuRefiners.WlsCore(gi, gt, p["lambda"], (int)p["iters"]));
    }
}
