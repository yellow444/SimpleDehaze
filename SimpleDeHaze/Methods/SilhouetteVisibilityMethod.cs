using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// A photo-like visibility mode for very dense haze: quieter than VisibilityBoost,
    /// but still more contrasty than natural restoration.
    /// </summary>
    public sealed class SilhouetteVisibilityMethod : IDeHazeMethod
    {
        public string Name => "Изображение видимости (полутона+цвет)";

        public string Description =>
            "Фото-подобный режим для очень плотной вуали: не бинарная карта и не черная маска.\n\n" +
            "Цель - получить серо-цветное изображение с полутонами, контурами и умеренным цветом,\n" +
            "не превращая туман в снежную шумовую кашу.\n\n" +
            "Шаги:\n" +
            "1. Спектральная оценка вуали и атмосферного света.\n" +
            "2. FGS уточняет карту t по крупным границам.\n" +
            "3. Recover вытаскивает видимость, но chromaFloor защищает цвет от выжигания.\n" +
            "4. Мягкий Lab CLAHE возвращает контраст и полутона.\n" +
            "5. Ограничение цветности и bilateral-сглаживание подавляют мелкий дымовой шум.\n\n" +
            "Формула: спектральная t = 1 − ω·min_c min_Ω(I_c/A_c) → FGS-уточнение по крупным границам;\n" +
            "J = (I − A)/max(t, t_min) + A с защитой хромы (÷max(t, chromaFloor)) — полутона без шумовой каши.\n\n" +
            "Если нужен более агрессивный, шумный режим - используй 'Усиление видимости'.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",  "omega - доля удаляемой вуали", 0.30, 0.95, 0.80, search: true),
            new ParamDef("patch",  "Патч min/контраста",           1,    15,   5,    1, isInt: true),
            new ParamDef("min",    "t_min - нижний порог t",       0.02, 0.45, 0.06),
            new ParamDef("chroma", "chromaFloor - защита цвета",   0.12, 0.90, 0.45),
            new ParamDef("refine", "FGS sigmaColor",               5,    120,  22,   1, isInt: true),
            new ParamDef("tsky",   "t_sky - защита белых зон",     0.50, 0.95, 0.72),
            new ParamDef("clip",   "Lab CLAHE контраст",           1.0,  7.0,  3.4,  search: true),
            new ParamDef("tiles",  "CLAHE сетка",                  2,    16,   8,    1, isInt: true),
            new ParamDef("sat",    "Вибранс цвета",                0.0,  0.9,  0.42, search: true),
            new ParamDef("detail", "Микроконтраст L",              0.0,  0.6,  0.06),
            new ParamDef("color",  "Потолок усиления цветности",    1.0,  1.6,  1.38),
            new ParamDef("smooth", "Подавление мелкого шума",       0.0,  10.0, 7.0),
            new ParamDef("context","Вернуть исходный контекст",      0.0,  0.45, 0.08),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"], tmin = p["min"], chroma = p["chroma"], tSky = p["tsky"];
            double clip = p["clip"], sat = p["sat"], detail = p["detail"], color = p["color"];
            double smooth = p["smooth"], context = p["context"];
            int patch = (int)p["patch"], refine = (int)p["refine"], tiles = (int)p["tiles"];

            using var i01 = DehazeCore.Normalize(input);
            using var tRaw = DehazeCore.SpectralTransmission(i01, omega, patch, tSky, out var a);

            using var t = new Mat();
            using (var guide8 = new Mat())
            {
                i01.ConvertTo(guide8, DepthType.Cv8U, 255.0);
                XImgprocInvoke.FastGlobalSmootherFilter(guide8, tRaw, t, 520, refine, 0.25, 3);
            }
            DehazeCore.Clamp01(t);

            using var recovered = DehazeCore.Recover(i01, t, a, tmin, chroma);
            using var toned = DehazeCore.LabEnhance(recovered, clip, tiles, sat, detail);
            using var limited = DehazeCore.LimitColorfulness(toned, input.Mat, color);
            using var quiet = DehazeCore.BilateralDenoise(limited, smooth);

            if (context <= 1e-6)
                return quiet.Clone();

            using var contextTone = DehazeCore.RestoreToneFast(i01, 0.65, 0.01, 1.35);
            using var contextSoft = DehazeCore.LabEnhance(contextTone, 1.4, 8, 0.04, 0.02);
            var result = new Mat();
            CvInvoke.AddWeighted(quiet, 1.0 - context, contextSoft, context, 0.0, result);
            return DeHazeCPU.Clip(result);
        }
    }
}
