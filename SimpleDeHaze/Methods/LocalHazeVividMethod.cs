using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Локально-адаптивная дымка, вариант «сочно» - яркая, насыщенная, контрастная картинка для глаза.
    /// Тот же движок LocalHazeCore, но финиш сильнее тянет контраст и цвет (с потолком цветности и
    /// мягким коленом светов против пересвета). Для авто-подбора выбирайте цель «Сочно/контраст».
    /// </summary>
    public sealed class LocalHazeVividMethod : IDeHazeMethod
    {
        public string Name => "Локальная дымка - сочно";

        public string Description =>
            "Пространственно-неоднородная дымка с упором на СОЧНУЮ картинку для глаза.\n\n" +
            "Движок LocalHazeCore: поле A(x), локально-адаптивная ω(x), краевой FGS-уточнитель t.\n\n" +
            "Финиш: сильнее Lab CLAHE + повышенный вибранс, потолок цветности против «кислотности»,\n" +
            "мягкое колено светов (knee) гасит пересвет. Яркая, контрастная, насыщенная.\n\n" +
            "Формула: J = (I − A(x))/max(t, t_min) + A(x); финиш = цвет ×(1 + 0.7·(1−t)·структура)\n" +
            "(в Lab, оттенок сохранён) → сильный CLAHE + вибранс → потолок цветности (ослаблен по плотности дымки).\n\n" +
            "Совет: для авто-подбора берите цель «Сочно/контраст».";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",   "ω - базовая доля удаляемой дымки", 0.5,  0.95, 0.85, search: true),
            new ParamDef("ogain",   "Прирост ω в плотных зонах",        0.0,  0.8,  0.45, search: true),
            new ParamDef("airMix",  "Сила съёма сплошной вуали (глоб. A)", 0.0, 1.0, 0.75, search: true),
            new ParamDef("patch",   "Патч тёмного канала",              1,    15,   5,    1, isInt: true),
            new ParamDef("aRadius", "Окно поля A(x)",                   20,   200,  90,   1, isInt: true),
            new ParamDef("min",     "t_min - нижний порог t",           0.03, 0.4,  0.07),
            new ParamDef("chroma",  "chromaFloor - защита цвета",       0.12, 0.8,  0.32),
            new ParamDef("wb",      "Баланс белого (убрать цвет вуали)", 0.0, 1.5, 0.7, search: true),
            new ParamDef("refine",  "FGS sigmaColor",                   5,    120,  22,   1, isInt: true),
            new ParamDef("tsky",    "t_sky - защита белых зон",         0.5,  0.95, 0.64),
            new ParamDef("knee",    "Колено мягких светов",             0.7,  1.0,  0.88),
            new ParamDef("denoise", "Денойз плотных зон (по 1-t)",      0.0,  1.0,  0.40, search: true),
            new ParamDef("quality", "Качество: 0=быстро, 1=HQ",        0,    1,    0,    1, isInt: true, tunable: false),
            new ParamDef("scale",   "HQ: апскейл ×",                    1,    3,    2,    1, isInt: true, tunable: false),
            new ParamDef("contour", "HQ: многомасштабные контуры",      0.0,  1.5,  0.62),
            new ParamDef("clip",    "Lab CLAHE контраст",               1.0,  7.0,  5.6,  search: true),
            new ParamDef("tiles",   "CLAHE сетка",                      2,    16,   8,    1, isInt: true),
            new ParamDef("sat",     "Вибранс цвета",                    0.0,  0.9,  0.72, search: true),
            new ParamDef("detail",  "Микроконтраст L",                  0.0,  0.7,  0.24),
            new ParamDef("colorRestore", "Восстановление цвета за дымкой", 0.0, 1.2, 0.70, search: true),
            new ParamDef("color",   "Потолок усиления цветности",       1.05, 1.8,  1.6),
            new ParamDef("smooth",  "Шумоподавление (до усиления)",    0.0,  6.0,  3.0),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
            => LocalHazeCore.Run(input, p, VividFinish);

        // VIVID-профиль: акцент на ЦВЕТ. Сильное тоносохраняющее восстановление хромы по (1-t)·conf
        // (не gray-world, оттенок сохранён), потолок цветности ослаблен по плотности дымки — сочно, но без «кислоты».
        private static Mat VividFinish(Mat recovered01, Mat inputBgr8, Mat t, Mat conf, IReadOnlyDictionary<string, double> p)
        {
            double G(string k, double d) => p.TryGetValue(k, out var v) ? v : d;
            double clip = G("clip", 5.6), sat = G("sat", 0.72), detail = G("detail", 0.24);
            double color = G("color", 1.6), smooth = G("smooth", 3.0), colorRestore = G("colorRestore", 0.70);
            int tiles = (int)G("tiles", 8);

            // денойз до усиления: иначе CLAHE+вибранс делают зерно из шума плотных зон
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
