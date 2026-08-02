using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Историческая (legacy) CPU-ветка проекта - обёртка над <see cref="DeHazeCPU"/>.
    /// ЭТО НЕ канонический DCP: атмосферный свет берётся по тёмному каналу, но карта пропускания
    /// строится по ПОКАНАЛЬНОМУ локальному минимуму и экспоненциальной формуле. Эталонный baseline -
    /// <see cref="CanonicalDcpMethod"/>; сравнивать новые приоры нужно с ним.
    /// </summary>
    public sealed class DcpCpuMethod : IDeHazeMethod
    {
        public string Name => "Legacy поканальный (не канонический DCP)";

        public string Description =>
            "Историческая ветка проекта (класс DeHazeCPU) - сохранена для воспроизводимости старых\n" +
            "результатов и статьи 2024 года. ЭТО НЕ канонический DCP: эталон - «DCP канонический (He 2009)».\n\n" +
            "Отличия от канонического DCP:\n" +
            "• минимум берётся ПО КАЖДОМУ каналу отдельно (m_c = min_Ω I_c), а не min по B,G,R;\n" +
            "• трансмиссия экспоненциальная и поканальная: t_c = clip(1 - exp(-β·A_c/m_c));\n" +
            "• A ищется внутри самой светлой области, найденной quad-декомпозицией;\n" +
            "• в конце применяются авто-уровни по яркости (косметика), поэтому для научного сравнения\n" +
            "  параметр tone нужно ставить в 0.\n\n" +
            "Шаги:\n" +
            "1. quad-декомпозиция → самая светлая область → тёмный канал → A;\n" +
            "2. m_c = min_Ω I_c (поканально);\n" +
            "3. t_c = clip(1 - exp(-β·A_c/m_c));\n" +
            "4. уточнение Guided Filter;\n" +
            "5. J_c = (I_c - A_c)/max(t_c, t_min) + A_c;\n" +
            "6. tone - авто-уровни по L (выключаемая косметика).\n\n" +
            "Параметры: β - сила; patch - окно; refine/ε - Guided Filter; tone - сила авто-уровней.\n" +
            "Постоянный параметр «Вычисление» вручную выбирает CPU или полный CUDA/GpuMat pipeline.";

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
            CudaBackend.ModeParameter,
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            if (CudaBackend.IsRequested(p))
            {
                CudaBackend.RequireAvailable();
                using var gpu = new DeHazeGPU();
                using var gpuRaw = gpu.RemoveHaze(input.Clone(), debug: false,
                    beta: (float)p["beta"], patchDarkChannel: (int)p["patch"], decompositionSize: (int)p["decomp"],
                    min: (float)p["min"], percen: (float)p["percen"], refineSize: (int)p["refine"], eps: p["eps"]);
                return DehazeCore.RestoreTone(gpuRaw, p["tone"]);
            }

            using var m = new DeHazeCPU();
            // классический per-channel DCP затемняет/площит (контраст < входа) - возвращаем тон авто-уровнями
            using var raw = m.RemoveHaze(input.Clone(), debug: false,
                beta: (float)p["beta"], patchDarkChannel: (int)p["patch"], decompositionSize: (int)p["decomp"],
                min: (float)p["min"], percen: (float)p["percen"], refineSize: (int)p["refine"], eps: p["eps"]);
            return DehazeCore.RestoreTone(raw, p["tone"]);
        }
    }
}
