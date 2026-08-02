using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Локально-адаптивная дымка, вариант «баланс» - характер по умолчанию: НАТУРАЛЬНЫЙ цвет за дымкой
    /// возвращён и объекты читаемы, но без агрессивной резкости/HDR. Тот же движок LocalHazeCore, что и
    /// «точность/объекты/сочно», но финиш умеренный: тоносохраняющее восстановление цвета по плотности
    /// дымки (гейт по фрактальной структуре - не красит плоский туман), лёгкий CLAHE + мягкая
    /// гейтованная резкость (анти-хруст), возврат тона и потолок цветности, ослабленный по плотности дымки.
    /// Это «объекты без пережима» + честный цвет: зелень/коричневый сцены возвращаются, картинка спокойная.
    /// </summary>
    public sealed class LocalHazeBalanceMethod : IDeHazeMethod
    {
        public string Name => "Локальная дымка - баланс (цвет + объекты)";

        public string Description =>
            "Пространственно-неоднородная дымка, БАЛАНС: настоящий цвет за дымкой + читаемые объекты, без пережима.\n\n" +
            "Движок LocalHazeCore: поле A(x), локально-адаптивная ω(x), краевой FGS-уточнитель t,\n" +
            "structure-confidence (фрактальная насыщенность) управляет и цветом, и резкостью.\n\n" +
            "Финиш: тоносохраняющее восстановление хромы по (1-t)·conf (оттенок не уплывает, плоский\n" +
            "туман не красится), лёгкий CLAHE + мягкая ГЕЙТОВАННАЯ резкость (нет хруста на ровных зонах),\n" +
            "возврат тона, потолок цветности ослаблен по плотности дымки. Спокойная натуральная картинка.\n\n" +
            "Формула: J восстановлен LocalHazeCore; хрома в Lab a/b × (1 + colorRestore·(1-t)·conf);\n" +
            "деталь × conf·smoothstep(t); потолок цветности × (1 + 0.6·средняя_плотность_дымки).";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",   "ω - базовая доля удаляемой дымки", 0.5,  0.95, 0.82, search: true),
            new ParamDef("ogain",   "Прирост ω в плотных зонах",        0.0,  0.8,  0.42, search: true),
            new ParamDef("airMix",  "Сила съёма сплошной вуали (глоб. A)", 0.0, 1.0, 0.75, search: true),
            new ParamDef("patch",   "Патч тёмного канала",              1,    15,   5,    1, isInt: true),
            new ParamDef("aRadius", "Окно поля A(x)",                   20,   200,  90,   1, isInt: true),
            new ParamDef("min",     "t_min - нижний порог t",           0.03, 0.4,  0.08),
            new ParamDef("chroma",  "chromaFloor - защита цвета",       0.12, 0.8,  0.34),
            new ParamDef("refine",  "FGS sigmaColor",                   5,    120,  24,   1, isInt: true),
            new ParamDef("tsky",    "t_sky - защита белых зон",         0.5,  0.95, 0.68),
            new ParamDef("knee",    "Колено мягких светов",             0.7,  1.0,  0.90),
            new ParamDef("denoise", "Денойз плотных зон (по 1-t)",      0.0,  1.0,  0.45),
            new ParamDef("quality", "Качество: 0=быстро, 1=HQ",        0,    1,    0,    1, isInt: true, tunable: false),
            new ParamDef("scale",   "HQ: апскейл ×",                    1,    3,    2,    1, isInt: true, tunable: false),
            new ParamDef("contour", "HQ: многомасштабные контуры",      0.0,  1.5,  0.45),
            new ParamDef("wb",      "Баланс белого (убрать цвет вуали)", 0.0, 1.5, 0.7, search: true),
            new ParamDef("colorRestore", "Восстановление цвета за дымкой", 0.0, 1.0, 0.35, search: true),
            new ParamDef("clip",    "Lab CLAHE контраст",               0.0,  4.0,  2.0,  search: true),
            new ParamDef("tiles",   "CLAHE сетка",                      2,    16,   8,    1, isInt: true),
            new ParamDef("sat",     "Вибранс цвета",                    0.0,  0.8,  0.30),
            new ParamDef("detail",  "Микроконтраст/резкость L",         0.0,  0.5,  0.12),
            new ParamDef("tone",    "Возврат тона (растяжение L)",      0.0,  1.0,  0.50, search: true),
            new ParamDef("color",   "Потолок усиления цветности",       1.05, 1.7,  1.35),
            new ParamDef("smooth",  "Шумоподавление (после)",          0.0,  4.0,  1.0),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
            => LocalHazeCore.Run(input, p, BalanceFinish);

        // BALANCE-профиль (по умолчанию): натуральный цвет + читаемые объекты, без пережима.
        private static Mat BalanceFinish(Mat recovered01, Mat inputBgr8, Mat t, Mat conf, IReadOnlyDictionary<string, double> p)
        {
            double G(string k, double d) => p.TryGetValue(k, out var v) ? v : d;
            double colorRestore = G("colorRestore", 0.35), clip = G("clip", 2.0), sat = G("sat", 0.30);
            double detail = G("detail", 0.12), tone = G("tone", 0.50), color = G("color", 1.35), smooth = G("smooth", 1.0);
            int tiles = (int)G("tiles", 8);

            using var hazeMask = LocalHazeCore.HazeDensity(t);
            double meanHaze = CvInvoke.Mean(hazeMask).V0;
            using var colored = DehazeCore.ScaleChromaByMask(recovered01, hazeMask, colorRestore, conf);
            using var toned = DehazeCore.RestoreTone(colored, tone, 0.01);
            using var gate = LocalHazeCore.DetailGate(conf, t);
            using var boosted = DehazeCore.LabEnhance(toned, clip, tiles, sat, detail, gate);
            using var limited = DehazeCore.LimitColorfulness(boosted, inputBgr8, color, meanHaze);
            return DehazeCore.BilateralDenoise(limited, smooth);
        }
    }
}
