using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>Медленный качественный локальный режим: upscale + FGS + Лапласиан.</summary>
    public sealed class LocalVisibilityQualityMethod : IDeHazeMethod
    {
        public string Name => "Локальная видимость HQ (upscale+Лапласиан)";

        public string Description =>
            "Медленный качественный режим для плотной неоднородной вуали. Он увеличивает изображение, сглаживает пиксельную сетку,\n" +
            "строит локальную карту пропускания на увеличенной сетке, уточняет её FGS-фильтром и усиливает контуры Лапласианом.\n\n" +
            "Это не нейросеть: только локальная физическая модель, масштабирование, сглаживание, Лапласиан, CLAHE и bilateral.\n" +
            "Используй, когда нужно вытащить слабые предметы и можно заплатить временем.\n\n" +
            "Формула: как FAST, но на увеличенной ×scale сетке: спектральная t → FGS-уточнение →\n" +
            "J = (I − A)/max(t, t_min) + A → CLAHE + Лапласиан + bilateral → уменьшение (Inter.Area).";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",   "omega - доля удаляемой вуали", 0.30, 0.98, 0.86, search: true),
            new ParamDef("patch",   "Патч локальной вуали",         1,    15,   5,    1, isInt: true),
            new ParamDef("min",     "t_min - нижний порог t",       0.02, 0.45, 0.055, search: true),
            new ParamDef("chroma",  "chromaFloor - защита цвета",   0.10, 0.90, 0.48),
            new ParamDef("refine",  "FGS sigmaColor",               5,    140,  48,   1, isInt: true),
            new ParamDef("tsky",    "t_sky - белый дым/снег",       0.45, 0.97, 0.74),
            new ParamDef("clip",    "CLAHE контраст",               0.5,  8.5,  3.7,  search: true),
            new ParamDef("sat",     "Цветовой вибранс",             0.0,  1.0,  0.32, search: true),
            new ParamDef("detail",  "Микроконтраст L",              0.0,  0.8,  0.10),
            new ParamDef("lap",     "Лапласиан-контуры",            0.0,  3.2,  0.82, search: true),
            new ParamDef("unsharp", "Вклад контурной детали",       0.0,  2.6,  0.72),
            new ParamDef("smooth",  "Bilateral после усиления",      0.0,  10.0, 6.2),
            new ParamDef("color",   "Масштаб цветности",            0.75, 1.75, 1.22),
            new ParamDef("scale",   "Upscale перед поиском",        1,    3,    2,    1, isInt: true, tunable: false),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
            => LocalVisibilityCore.Run(input, p, highQuality: true);
    }
}
