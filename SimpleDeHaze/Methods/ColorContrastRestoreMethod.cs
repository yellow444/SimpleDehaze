using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Натуральное восстановление: убрать атмосферную вуаль без "выкручивания" цвета.
    /// Цель - высокая близость к GT: аккуратный цвет, контролируемый контраст и минимум пересвета.
    /// </summary>
    public sealed class ColorContrastRestoreMethod : IDeHazeMethod
    {
        public string Name => "Восстановление цвета/контраста (натурально)";

        public string Description =>
            "Режим восстановления, а не усиления: правдоподобные цвета и контраст без агрессивного HDR-вида.\n\n" +
            "Шаги:\n" +
            "1. DCP оценивает атмосферный свет A и карту t.\n" +
            "2. Яркие малонасыщенные зоны получают более высокий t: небо/снег/дым не выжигаются.\n" +
            "3. Fast Global Smoother уточняет t по краям.\n" +
            "4. Recover разделяет яркость и хрому: дымку убираем, цвет не перенасыщаем.\n" +
            "5. Мягкая Lab-коррекция возвращает контраст и цвет, затем ставится потолок цветности.\n\n" +
            "Формула: t = 1 − ω·min_c min_Ω(I_c/A_c), в ярких малонасыщ. зонах t ← max(t, t_sky), уточн. FGS;\n" +
            "J = (I − A)/max(t, t_min) + A, хрома делится на max(t, chromaFloor).\n\n" +
            "Используй, когда нужна наиболее честная реконструкция сцены.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",  "omega - доля удаляемой дымки", 0.3, 0.9, 0.62, search: true),
            new ParamDef("patch",  "Патч тёмного канала",          1,   15,  5,    1, isInt: true),
            new ParamDef("min",    "t_min - нижний порог t",       0.02, 0.4, 0.12),
            new ParamDef("chroma", "chromaFloor - защита цвета",   0.2, 0.9, 0.55),
            new ParamDef("refine", "FGS sigmaColor",               5,   120, 38,   1, isInt: true),
            new ParamDef("tsky",   "t_sky - защита неба/снега",    0.55, 0.95, 0.74),
            new ParamDef("clip",   "Lab CLAHE контраст",           0.0, 4.0, 2.0, search: true),
            new ParamDef("tiles",  "CLAHE сетка",                  2,   16,  8,    1, isInt: true),
            new ParamDef("sat",    "Вибранс цвета",                0.0, 0.45, 0.12),
            new ParamDef("detail", "Микроконтраст L",              0.0, 0.35, 0.08),
            new ParamDef("color",  "Потолок усиления цветности",    1.0, 1.35, 1.18),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"], tmin = p["min"], chroma = p["chroma"], tSky = p["tsky"];
            double clip = p["clip"], sat = p["sat"], detail = p["detail"], color = p["color"];
            int patch = (int)p["patch"], refine = (int)p["refine"], tiles = (int)p["tiles"];

            using var I = DehazeCore.Normalize(input);
            using var tRaw = DehazeCore.SpectralTransmission(I, omega, patch, tSky, out var A);

            using var t = new Mat();
            using (var guide8 = new Mat())
            {
                I.ConvertTo(guide8, DepthType.Cv8U, 255.0);
                XImgprocInvoke.FastGlobalSmootherFilter(guide8, tRaw, t, 450, refine, 0.25, 3);
            }
            DehazeCore.Clamp01(t);

            // structure-confidence: восстанавливаем цвет за дымкой там, где есть реальная текстура
            using var conf = ContourOps.FractalRichness(I, 3, 17);
            using var recovered = DehazeCore.Recover(I, t, A, tmin, chroma, conf);
            using var polished = DehazeCore.LabEnhance(recovered, clip, tiles, sat, detail);
            using var hmAll = LocalHazeCore.HazeDensity(t);   // потолок цветности ослабляем по плотности дымки
            return DehazeCore.LimitColorfulness(polished, input.Mat, color, CvInvoke.Mean(hmAll).V0);
        }
    }
}
