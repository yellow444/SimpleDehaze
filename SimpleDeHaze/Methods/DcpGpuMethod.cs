using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>Dark Channel Prior на GPU (CUDA) - обёртка над <see cref="DeHazeGPU"/>.</summary>
    public sealed class DcpGpuMethod : IDeHazeMethod
    {
        public string Name => "Dark Channel Prior (GPU - CUDA)";

        public string Description =>
            "Тот же Dark Channel Prior (He, 2009), но все стадии - на GPU (CUDA, GpuMat).\n" +
            "На крупных кадрах заметно быстрее CPU-версии.\n\n" +
            "Шаги:\n" +
            "1. A - атмосферный свет по тёмному каналу светлой зоны (quad-decomposition).\n" +
            "2. t_c = clip( 1 - exp(-β*A_c / min I_c) ).\n" +
            "3. Уточнение Guided Filter (ручная реализация на GpuMat, портирование кода He).\n" +
            "4. J_c = (I_c - A_c) / max(t_c, t_min) + A_c.\n" +
            "5. Восстановление тона: глобальные авто-уровни по яркости возвращают контраст,\n" +
            "   'съеденный' per-channel восстановлением (цвет не трогаем).\n\n" +
            "Параметры: β - сила; patch - окно; refine/ε - Guided Filter; tone - сила авто-уровней.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("beta",   "β - сила удаления дымки",   0.05, 2.0,  0.5, search: true),
            new ParamDef("patch",  "Патч тёмного канала",       1,    15,   3,   1, isInt: true, search: true),
            new ParamDef("decomp", "Размер quad-декомпозиции",  2,    64,   8,   1, isInt: true),
            new ParamDef("min",    "t_min - нижний порог t",    0.005, 0.5, 0.05),
            new ParamDef("percen", "Доля ярких пикселей для A", 0.01, 1.0,  0.1),
            new ParamDef("refine", "Радиус Guided Filter",      3,    150,  60,  1, isInt: true),
            new ParamDef("eps",    "ε - регуляризация GF",      1e-5, 1e-2, 1e-3, log: true),
            new ParamDef("tone",   "Восстановление тона (контраст)", 0.0, 1.0, 0.6),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            using var m = new DeHazeGPU();
            using var raw = m.RemoveHaze(input.Clone(), debug: false,
                beta: (float)p["beta"], patchDarkChannel: (int)p["patch"], decompositionSize: (int)p["decomp"],
                min: (float)p["min"], percen: (float)p["percen"], refineSize: (int)p["refine"], eps: p["eps"]);
            return DehazeCore.RestoreTone(raw, p["tone"]);
        }
    }
}
