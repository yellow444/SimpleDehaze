using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Усиление видимости: агрессивнее вытаскивает границы, цвет и локальный контраст.
    /// Это не самый "верный" цвет к GT, а режим для оператора/роботизированного зрения.
    /// </summary>
    public sealed class VisibilityBoostMethod : IDeHazeMethod
    {
        public string Name => "Усиление видимости (контраст+цвет)";

        public string Description =>
            "Режим полезной видимости: лучше видеть границы/объекты в тумане, дыме, снегу и воздушной засветке.\n\n" +
            "Шаги:\n" +
            "1. A_c - атмосферный свет.\n" +
            "2. Спектральная карта t: каждый канал даёт свою оценку, веса берутся из локального контраста.\n" +
            "3. Небо/снег/белый дым защищаются через t_sky, затем FGS уточняет карту t.\n" +
            "4. Recover с умеренной защитой цвета.\n" +
            "5. Lab CLAHE + микроконтраст + вибранс дают читаемые контуры и цветовые различия.\n\n" +
            "Формула: спектральная t = 1 − ω·min_c min_Ω(I_c/A_c) (веса по локальному контрасту, floor t_sky в небе)\n" +
            "→ FGS; J = (I − A)/max(t, t_min) + A; затем Lab-CLAHE + микроконтраст + вибранс.\n\n" +
            "Это намеренно не GT-восстановление: цвета и контраст могут быть сильнее оригинала,\n" +
            "если так картинка становится разборчивее и приятнее глазу.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",  "omega - доля удаляемой дымки", 0.3, 0.95, 0.82, search: true),
            new ParamDef("patch",  "Патч min/контраста",           1,   15,   5,    1, isInt: true),
            new ParamDef("min",    "t_min - нижний порог t",       0.02, 0.45, 0.06),
            new ParamDef("chroma", "chromaFloor - защита цвета",   0.12, 0.8,  0.32),
            new ParamDef("refine", "FGS sigmaColor",               5,    120,  22,   1, isInt: true),
            new ParamDef("tsky",   "t_sky - защита белых зон",     0.5,  0.95, 0.62),
            new ParamDef("clip",   "Lab CLAHE контраст",           1.0,  7.0,  5.8,  search: true),
            new ParamDef("tiles",  "CLAHE сетка",                  2,    16,   8,    1, isInt: true),
            new ParamDef("sat",    "Вибранс цвета",                0.0,  0.9,  0.68, search: true),
            new ParamDef("detail", "Микроконтраст L",              0.0,  0.6,  0.26),
            new ParamDef("color",  "Потолок усиления цветности",    1.05, 1.6,  1.55),
            new ParamDef("smooth", "Сглаживание шума",             0.0,  6.0,  4.0),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"], tmin = p["min"], chroma = p["chroma"], tSky = p["tsky"];
            double clip = p["clip"], sat = p["sat"], detail = p["detail"], color = p["color"], smooth = p["smooth"];
            int patch = (int)p["patch"], refine = (int)p["refine"], tiles = (int)p["tiles"];

            using var I = DehazeCore.Normalize(input);
            using var tFused = DehazeCore.SpectralTransmission(I, omega, patch, tSky, out var A);

            using var t = new Mat();
            using (var guide8 = new Mat())
            {
                I.ConvertTo(guide8, DepthType.Cv8U, 255.0);
                XImgprocInvoke.FastGlobalSmootherFilter(guide8, tFused, t, 550, refine, 0.25, 3);
            }
            DehazeCore.Clamp01(t);

            using var recovered = DehazeCore.Recover(I, t, A, tmin, chroma);
            using var boosted = DehazeCore.LabEnhance(recovered, clip, tiles, sat, detail);
            using var limited = DehazeCore.LimitColorfulness(boosted, input.Mat, color);
            return DehazeCore.BilateralDenoise(limited, smooth);
        }

    }
}
