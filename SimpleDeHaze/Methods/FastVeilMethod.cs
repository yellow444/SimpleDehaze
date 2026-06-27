using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Быстрая вуаль (скорость + контуры). Новый метод: см. docs/methods/fast-veil.md.
    ///
    /// Цель - 'хотя бы контуры, очень быстро': никаких итераций, ДПФ и Guided Filter. Карта пропускания
    /// сглаживается прореживающим Гауссом (FastGaussian) вместо дорогого edge-aware фильтра, восстановление
    /// с защитой цвета - поканально (O(N)), затем быстрые авто-уровни возвращают контраст/контуры.
    /// Учитывает атмосферу (A_c) и яркость зон (гейт неба).
    /// </summary>
    public sealed class FastVeilMethod : IDeHazeMethod
    {
        public string Name => "Быстрая вуаль (контуры, скорость)";

        public string Description =>
            "Скорость: всё пиксельно/один быстрый Гаусс - ни итераций, ни ДПФ, ни Guided Filter.\n\n" +
            "Шаги:\n" +
            "1. A_c - атмосферный свет (ярчайшие в тёмном канале).\n" +
            "2. t = 1 - ω*min_Ω(I_c/A_c); сглаживание FastGaussian (прореживание - быстро).\n" +
            "3. Небо: маска (ярко*малонасыщенно) поднимает t - не пережигаем пересветы.\n" +
            "4. Восстановление J_c с защитой цвета (яркость/хрома раздельно), поканально.\n" +
            "5. Быстрые авто-уровни по яркости -> контраст/контуры.\n\n" +
            "Параметры: ω, patch - дымка; sigma - сглаживание t; tone - авто-уровни (контуры).";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega", "ω - доля удаляемой дымки", 0.3, 0.95, 0.7, search: true),
            new ParamDef("patch", "Патч тёмного канала",      1, 15, 5, 1, isInt: true),
            new ParamDef("min",   "t_min - нижний порог t",   0.01, 0.5, 0.1),
            new ParamDef("sigma", "σ сглаживания t",          5, 80, 25, 1, isInt: true),
            new ParamDef("tone",  "Авто-уровни (контуры)",    0.0, 1.0, 0.5),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"], tmin = p["min"], tone = p["tone"], sigma = p["sigma"];
            int patch = (int)p["patch"];

            using var I = DehazeCore.Normalize(input);
            using var dark = DehazeCore.DarkChannel(I, patch);
            var A = DehazeCore.Atmospheric(I, dark, 0.001);

            using var tRaw = DehazeCore.RawTransmission(I, A, omega, patch);   // 1 - ω*dark(I/A)
            using var t = DehazeCore.FastGaussian(tRaw, sigma);               // быстрое сглаживание (вместо Guided Filter)
            using (var sky = DehazeCore.SkyMask(I)) DehazeCore.RaiseInSky(t, sky, 0.7);
            DehazeCore.Clamp01(t);

            using var J = DehazeCore.Recover(I, t, A, tmin);                  // защита цвета (яркость/хрома раздельно)
            return DehazeCore.RestoreTone(J, tone);                          // авто-уровни -> контуры
        }
    }
}
