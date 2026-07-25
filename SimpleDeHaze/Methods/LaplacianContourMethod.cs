using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Многомасштабные контуры через Лапласиан-пирамиду (OpenCV pyrDown/pyrUp). Яркость раскладывается на
    /// полосы разного масштаба; мелкую полосу (шум) держим слабо, средние и крупные контуры/силуэты
    /// усиливаем - так контуры «держатся» на меньшем и большем масштабе одновременно. Опционально сначала
    /// прогоняется лёгкий дехейз (CAP+), потом контуры - это уже мини-цепочка из двух алгоритмов.
    /// </summary>
    public sealed class LaplacianContourMethod : IDeHazeMethod
    {
        public string Name => "Многомасштабные контуры (Лапласиан-пирамида)";

        public string Description =>
            "Контуры на нескольких масштабах сразу - идея Лапласиан-пирамиды (OpenCV pyrDown/pyrUp).\n\n" +
            "1. (опц.) лёгкий дехейз CAP+ как база.\n" +
            "2. Яркость L -> гауссова пирамида -> полосы L_i = G_i - up(G_{i+1}).\n" +
            "3. Усиление полос: мелкая (шум) - слабо, средние/крупные (контуры/силуэты) - сильнее.\n" +
            "4. Сборка пирамиды обратно + мягкий тон/вибранс.\n\n" +
            "Параметры: уровней пирамиды, усиление мелких/средних/крупных полос. Цвет (a,b) не трогаем.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("predehaze", "Сначала дехейз (CAP+): 0/1", 0,   1,    1,    1, isInt: true, tunable: false),
            new ParamDef("levels",  "Уровней пирамиды",            3,    6,    5,    1, isInt: true),
            new ParamDef("gFine",   "Мелкие детали (шум) - слабо", 0.0,  1.5,  0.45),
            new ParamDef("gMid",    "Средние контуры",             0.0,  3.0,  1.8,  search: true),
            new ParamDef("gCoarse", "Крупные силуэты",             0.0,  2.5,  1.4),
            new ParamDef("sat",     "Вибранс цвета",               0.0,  0.8,  0.25, search: true),
            new ParamDef("tone",    "Возврат тона (растяжение L)", 0.0,  1.0,  0.45),
            new ParamDef("color",   "Потолок усиления цветности",  1.05, 1.7,  1.40),
            new ParamDef("smooth",  "Шумоподавление",             0.0,  6.0,  0.5),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            bool predehaze = p["predehaze"] >= 0.5;
            int levels = (int)p["levels"];
            double gFine = p["gFine"], gMid = p["gMid"], gCoarse = p["gCoarse"];
            double sat = p["sat"], tone = p["tone"], color = p["color"], smooth = p["smooth"];

            using var base01 = predehaze
                ? new CapLocalMethod().Process(input, new CapLocalMethod().Parameters.ToDictionary(x => x.Key, x => x.Default))
                : Normalize(input);

            // structure-confidence по ОРИГИНАЛУ (не по пред-дехейзу — там уже усилен шум): гейтуем полосы,
            // чтобы плоский туман не хрустел, а реальные кромки держались.
            using var inN = Normalize(input);
            using var structMap = ContourOps.FractalRichness(inN, 3, 17);
            using var contoured = ContourOps.MultiScaleLaplacian(base01, levels, gFine, gMid, gCoarse, structMap, 0.06);
            using var boosted = DehazeCore.LabEnhance(contoured, 0.0, 8, sat, 0.0);
            using var toned = DehazeCore.RestoreTone(boosted, tone, 0.01);
            using var limited = DehazeCore.LimitColorfulness(toned, input.Mat, color);
            return smooth > 0.01 ? DehazeCore.BilateralDenoise(limited, smooth) : DeHazeCPU.Clip(limited.Clone());
        }

        private static Mat Normalize(Image<Bgr, byte> img)
        {
            var m = new Mat();
            img.Mat.ConvertTo(m, DepthType.Cv32F, 1.0 / 255.0);
            return m;
        }
    }
}
