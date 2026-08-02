using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Канонический Dark Channel Prior (He, Sun, Tang, CVPR 2009 / TPAMI 2011) - ЭТАЛОННЫЙ baseline
    /// для сравнений. Здесь намеренно нет ничего, кроме самой физической модели:
    /// нет восстановления тона, нет ограничителя цветности, нет sky-mask, нет chroma-safe recovery.
    ///
    /// Именно с этим методом (а не с legacy-веткой <see cref="DeHazeCPU"/>) нужно сравнивать новые
    /// приоры: legacy считает поканальный минимум и экспоненциальную t, то есть это не DCP.
    ///
    /// Порядок операций канонический: сначала минимум по каналам, затем минимум по окну.
    /// </summary>
    public sealed class CanonicalDcpMethod : IDeHazeMethod
    {
        public string Name => "DCP канонический (He 2009, baseline)";

        public string Description =>
            "Эталонная реализация Dark Channel Prior без какой-либо косметики - для честных сравнений.\n\n" +
            "1. dark(x) = min_Ω min_c I_c(y)  (сначала по каналам, потом по окну);\n" +
            "2. A - среднее по top-k пикселям тёмного канала (гистограммный порог, O(N));\n" +
            "3. t = 1 - ω·min_Ω min_c (I_c/A_c);\n" +
            "4. уточнение Guided Filter по исходному кадру;\n" +
            "5. J_c = (I_c - A_c)/max(t, t_min) + A_c - стандартная инверсия.\n\n" +
            "Параметр «линейный радианс» включает физически корректный конвейер:\n" +
            "sRGB → линейное пространство → вся физика → обратно в sRGB. Модель I = J·t + A·(1-t)\n" +
            "линейна по световому сигналу, поэтому в sRGB оценка t систематически занижена\n" +
            "(см. docs/research/physics-linear-spectral.md). Это даёт ablation A/B из аудита:\n" +
            "linear=0 - как раньше, linear=1 - корректно; ω при этом нужно перекалибровать.\n\n" +
            "Никакой пост-обработки: всё, что видно на выходе - результат физической модели.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",  "ω - доля удаляемой дымки",   0.30, 0.99,  0.95, search: true),
            new ParamDef("patch",  "Радиус окна тёмного канала", 1,    15,    7,    1, isInt: true, search: true),
            new ParamDef("top",    "Доля пикселей для A, %",     0.01, 2.0,   0.1),
            new ParamDef("min",    "t_min - нижний порог t",     0.01, 0.5,   0.1),
            new ParamDef("refine", "Радиус Guided Filter",       3,    150,   60,   1, isInt: true),
            new ParamDef("eps",    "ε - регуляризация GF",       1e-5, 1e-2,  1e-3, log: true),
            new ParamDef("linear", "Линейный радианс (0/1)",     0,    1,     0,    1, isInt: true, tunable: false),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"], top = p["top"] / 100.0, tmin = p["min"], eps = p["eps"];
            int patch = (int)p["patch"], refine = (int)p["refine"];
            bool linear = p["linear"] >= 0.5;

            using var I = ColorSpace.Normalize(input, linear);
            using var dark = DehazeCore.DarkChannel(I, patch);
            var A = DehazeCore.Atmospheric(I, dark, top);

            using var tRaw = DehazeCore.RawTransmission(I, A, omega, patch);
            using var t = new Mat();
            XImgprocInvoke.GuidedFilter(I, tRaw, t, refine, eps);
            CvInvoke.PatchNaNs(t, tmin);
            DehazeCore.Clamp01(t);

            // chromaFloor = 0 -> хрома и яркость делятся на один и тот же t, то есть ровно
            // J_c = (I_c - A_c)/max(t,t_min) + A_c без защиты цвета.
            using var J = DehazeCore.Recover(I, t, A, tmin, chromaFloor: 0.0);
            return ColorSpace.Encode(J, linear);
        }
    }
}
