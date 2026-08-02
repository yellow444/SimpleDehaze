using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Локально-адаптивная дымка, вариант «объекты» - цель: чтобы все предметы были видны и чётко,
    /// включая слабый угол кадра (дерево в правом верхнем углу 09_outdoor). Сильнее поднимает ω в
    /// плотных зонах, добавляет локальный контраст (CLAHE) и микроконтраст/резкость.
    /// Для авто-подбора выбирайте цель «Объекты/контуры».
    /// </summary>
    public sealed class LocalHazeObjectsMethod : IDeHazeMethod
    {
        public string Name => "Локальная дымка - объекты (чётко)";

        public string Description =>
            "Пространственно-неоднородная дымка с упором на ЧИТАЕМОСТЬ объектов по всему кадру.\n\n" +
            "Движок LocalHazeCore: поле A(x), локально-адаптивная ω(x) (плотный угол чистится\n" +
            "сильнее), краевой FGS-уточнитель t - границы остаются резкими.\n\n" +
            "Финиш: Lab CLAHE + микроконтраст/резкость, ограничение цветности, лёгкое\n" +
            "шумоподавление. Цвет может уходить от GT - приоритет у различимости контуров.\n\n" +
            "Формула: J = (I − A(x))/max(t, t_min) + A(x) (t локально-адаптивна, ББ по цвету дымки);\n" +
            "финиш = денойз → цвет ×(1 + 0.25·(1−t)·структура) → CLAHE + гейт-резкость (структура·t) → потолок цветности.\n\n" +
            "Совет: для авто-подбора берите цель «Объекты/контуры».";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",   "ω - базовая доля удаляемой дымки", 0.5,  0.95, 0.86, search: true),
            new ParamDef("ogain",   "Прирост ω в плотных зонах",        0.0,  0.8,  0.50, search: true),
            new ParamDef("airMix",  "Сила съёма сплошной вуали (глоб. A)", 0.0, 1.0, 0.75, search: true),
            new ParamDef("patch",   "Патч тёмного канала",              1,    15,   5,    1, isInt: true),
            new ParamDef("aRadius", "Окно поля A(x)",                   20,   200,  90,   1, isInt: true),
            new ParamDef("min",     "t_min - нижний порог t",           0.03, 0.4,  0.07),
            new ParamDef("chroma",  "chromaFloor - защита цвета",       0.12, 0.8,  0.34),
            new ParamDef("wb",      "Баланс белого (убрать цвет вуали)", 0.0, 1.5, 0.7, search: true),
            new ParamDef("refine",  "FGS sigmaColor",                   5,    120,  24,   1, isInt: true),
            new ParamDef("tsky",    "t_sky - защита белых зон",         0.5,  0.95, 0.66),
            new ParamDef("knee",    "Колено мягких светов",             0.7,  1.0,  0.86),
            new ParamDef("denoise", "Денойз плотных зон (по 1-t)",      0.0,  1.0,  0.45, search: true),
            new ParamDef("quality", "Качество: 0=быстро, 1=HQ",        0,    1,    0,    1, isInt: true, tunable: false),
            new ParamDef("scale",   "HQ: апскейл ×",                    1,    3,    2,    1, isInt: true, tunable: false),
            new ParamDef("contour", "HQ: многомасштабные контуры",      0.0,  1.5,  0.75),
            new ParamDef("clip",    "Lab CLAHE контраст",               1.0,  7.0,  3.4,  search: true),
            new ParamDef("tiles",   "CLAHE сетка",                      2,    16,   8,    1, isInt: true),
            new ParamDef("sat",     "Вибранс цвета",                    0.0,  0.9,  0.32, search: true),
            new ParamDef("detail",  "Микроконтраст/резкость L",         0.0,  0.7,  0.26, search: true),
            new ParamDef("colorRestore", "Восстановление цвета за дымкой", 0.0, 1.0, 0.25, search: true),
            new ParamDef("color",   "Потолок усиления цветности",       1.05, 1.7,  1.5),
            new ParamDef("smooth",  "Шумоподавление (до усиления)",    0.0,  6.0,  3.5),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
            => LocalHazeCore.Run(input, p, ObjectsFinish);

        // STRUCTURE-профиль: сильное проявление контуров + СКРОМНОЕ восстановление цвета (чтобы не спорил
        // с контуром). Деталь гейтуется structure·transmission (анти-хруст на ровных зонах), цвет —
        // тоносохраняющий по маске (1-t)·conf, потолок цветности ослаблен по плотности дымки.
        private static Mat ObjectsFinish(Mat recovered01, Mat inputBgr8, Mat t, Mat conf, IReadOnlyDictionary<string, double> p)
        {
            double G(string k, double d) => p.TryGetValue(k, out var v) ? v : d;
            double clip = G("clip", 3.4), sat = G("sat", 0.32), detail = G("detail", 0.26);
            double color = G("color", 1.5), smooth = G("smooth", 3.5), colorRestore = G("colorRestore", 0.25);
            int tiles = (int)G("tiles", 8);

            // сначала глушим усиленный шум плотных зон, потом тянем цвет/контур
            using var quiet = DehazeCore.BilateralDenoise(recovered01, smooth);
            using var hazeMask = LocalHazeCore.HazeDensity(t);
            double meanHaze = CvInvoke.Mean(hazeMask).V0;
            using var colored = DehazeCore.ScaleChromaByMask(quiet, hazeMask, colorRestore, conf);
            using var gate = LocalHazeCore.DetailGate(conf, t);
            using var boosted = DehazeCore.LabEnhance(colored, clip, tiles, sat, detail, gate);
            return DehazeCore.LimitColorfulness(boosted, inputBgr8, color, meanHaze);
        }
    }
}
