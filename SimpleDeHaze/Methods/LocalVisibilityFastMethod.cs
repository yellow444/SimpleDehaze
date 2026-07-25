using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>Быстрый локальный режим: локальная вуаль + Лапласиан для контуров.</summary>
    public sealed class LocalVisibilityFastMethod : IDeHazeMethod
    {
        public string Name => "Локальная видимость FAST (Лапласиан)";

        public string Description =>
            "Быстрый режим для неоднородной дымки/дыма/снега: локально оценивает вуаль, затем вытаскивает контуры через Лапласиан.\n\n" +
            "Идея: не искать одну силу дымки на весь кадр. Карта t строится локально, потом яркость и контуры усиливаются там,\n" +
            "где в исходнике остался слабый структурный сигнал. Это режим для роботизированного зрения и быстрого перебора параметров.\n\n" +
            "Формула: спектральная t — поканальные оценки t_c = 1 − ω·min_Ω(I_c/A_c), веса по локальному контрасту;\n" +
            "edge-aware уточнение; J = (I − A)/max(t, t_min) + A; затем Lab-CLAHE + Лапласиан/уншарп усиливают контуры.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",   "omega - доля удаляемой вуали", 0.30, 0.97, 0.84, search: true),
            new ParamDef("patch",   "Патч локальной вуали",         1,    15,   4,    1, isInt: true),
            new ParamDef("min",     "t_min - нижний порог t",       0.02, 0.45, 0.06, search: true),
            new ParamDef("chroma",  "chromaFloor - защита цвета",   0.10, 0.90, 0.44),
            new ParamDef("refine",  "Сглаживание карты t",          0,    70,   16,   1, isInt: true),
            new ParamDef("tsky",    "t_sky - белый дым/снег",       0.45, 0.96, 0.72),
            new ParamDef("clip",    "CLAHE контраст",               0.5,  8.0,  3.2,  search: true),
            new ParamDef("sat",     "Цветовой вибранс",             0.0,  1.0,  0.28, search: true),
            new ParamDef("detail",  "Микроконтраст L",              0.0,  0.7,  0.06),
            new ParamDef("lap",     "Лапласиан-контуры",            0.0,  2.8,  0.60, search: true),
            new ParamDef("unsharp", "Вклад контурной детали",       0.0,  2.2,  0.52),
            new ParamDef("smooth",  "Подавление мелкого шума",       0.0,  8.0,  4.6),
            new ParamDef("color",   "Масштаб цветности",            0.75, 1.70, 1.18),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
            => LocalVisibilityCore.Run(input, p, highQuality: false);
    }
}
