using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Локально-адаптивная дымка, вариант «грубый показ» - для зон ОЧЕНЬ густой дымки, где тонкая
    /// деталь утоплена в шуме (SNR&lt;1), а структура объектов сохраняется только на грубом масштабе
    /// (контуры ~10-40 px). Вместо тонкой резкости (которая там даёт лишь шум) выполняется
    /// масштабно-избирательное восстановление: по ВХОДУ строится грубая карта контуров (пулинг
    /// усредняет шум вниз → вычитание локального airlight → нормировка) и впечатывается в яркость
    /// результата с весом «мало сигнала». Так из плотной дымки проявляются грубые силуэты деревьев.
    ///
    /// На 09_outdoor (правый верхний угол): корреляция такой реконструкции с эталоном ~0.65 -
    /// проявляются ель слева и массы кроны; тонкий ствол - на грани (контурный намёк, не сплошной).
    /// Это не «выдумывание» деталей: грубая структура там статистически реальна (доказано нулевыми
    /// базлайнами), но тонкую деталь из-под шума вернуть нельзя.
    /// </summary>
    public sealed class LocalHazeCoarseRevealMethod : IDeHazeMethod
    {
        public string Name => "Локальная дымка - грубый показ (контуры в дымке)";

        public string Description =>
            "Для ОЧЕНЬ плотной дымки, где обычное усиление даёт только шум.\n\n" +
            "Идея: в зонах густой дымки сигнал деревьев сохраняется лишь на ГРУБОМ масштабе.\n" +
            "Шаги: восстановление LocalHazeCore → по входу строим грубую карту контуров\n" +
            "(пулинг усредняет шум, вычитаем локальный airlight, нормируем) → впечатываем контуры\n" +
            "в яркость с весом 'мало сигнала' (там, где локальный контраст входа мал).\n\n" +
            "Параметры 'Грубый показ': пулинг (усреднение шума), вычитание airlight (σ), сила контуров,\n" +
            "порог/ширина зоны 'мало сигнала'. Тонкую деталь из-под шума не вернуть - честно показываем\n" +
            "грубые силуэты.\n\n" +
            "Формула: J = (I − A(x))/max(t, t_min) + A(x); в плотных зонах впечатываем грубые контуры\n" +
            "входа (пулинг → −airlight → нормировка) с весом маски «мало сигнала» (локальный σ входа мал).\n\n" +
            "Совет: для авто-подбора берите цель «Объекты/контуры».";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",   "ω - базовая доля удаляемой дымки", 0.5,  0.95, 0.85, search: true),
            new ParamDef("ogain",   "Прирост ω в плотных зонах",        0.0,  0.8,  0.30, search: true),
            new ParamDef("airMix",  "Сила съёма сплошной вуали (глоб. A)", 0.0, 1.0, 0.75, search: true),
            new ParamDef("patch",   "Патч тёмного канала",              1,    15,   5,    1, isInt: true),
            new ParamDef("aRadius", "Окно поля A(x)",                   20,   200,  110,  1, isInt: true),
            new ParamDef("min",     "t_min - нижний порог t",           0.03, 0.4,  0.12),
            new ParamDef("chroma",  "chromaFloor - защита цвета",       0.12, 0.8,  0.35),
            new ParamDef("wb",      "Баланс белого (убрать цвет вуали)", 0.0, 1.5, 0.55, search: true),
            new ParamDef("refine",  "FGS sigmaColor",                   5,    120,  90,   1, isInt: true),
            new ParamDef("tsky",    "t_sky - защита белых зон",         0.5,  0.95, 0.70),
            new ParamDef("knee",    "Колено мягких светов",             0.7,  1.0,  0.90),
            new ParamDef("denoise", "Денойз плотных зон (по 1-t)",      0.0,  1.0,  0.85),
            new ParamDef("pool",    "Грубый показ: пулинг (усреднение шума)", 2, 10, 5, 1, isInt: true),
            new ParamDef("csub",    "Грубый показ: вычитание airlight σ", 6,  40,   18),
            new ParamDef("cgain",   "Грубый показ: сила контуров",      0.0,  1.5,  0.80, search: true),
            new ParamDef("cthr",    "Грубый показ: порог 'мало сигнала' σ", 0.005, 0.08, 0.030),
            new ParamDef("cband",   "Грубый показ: ширина порога",      0.005, 0.05, 0.020),
            new ParamDef("tone",    "Возврат тона (растяжение L)",      0.0,  1.0,  0.40),
            new ParamDef("sat",     "Насыщенность",                     0.0,  0.6,  0.20),
            new ParamDef("smooth",  "Шумоподавление",                  0.0,  6.0,  4.0),
            new ParamDef("quality", "Качество: 0=быстро, 1=HQ",        0,    1,    0,    1, isInt: true, tunable: false),
            new ParamDef("scale",   "HQ: апскейл ×",                    1,    3,    2,    1, isInt: true, tunable: false),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
            => LocalHazeCore.Run(input, p, CoarseRevealFinish);

        private static Mat CoarseRevealFinish(Mat recovered01, Mat inputBgr8, Mat t, Mat conf, IReadOnlyDictionary<string, double> p)
        {
            double G(string k, double d) => p.TryGetValue(k, out var v) ? v : d;
            double smooth = G("smooth", 4.0), cgain = G("cgain", 0.80), tone = G("tone", 0.40), sat = G("sat", 0.20);
            int pool = (int)G("pool", 5);
            double csub = G("csub", 18.0), cthr = G("cthr", 0.030), cband = G("cband", 0.020);

            using var quiet = DehazeCore.BilateralDenoise(recovered01, smooth);
            using var revealed = LocalHazeCore.CoarseReveal(quiet, inputBgr8, pool, csub, cgain, cthr, cband);
            using var toned = DehazeCore.RestoreTone(revealed, tone, 0.01);
            return Math.Abs(sat) > 1e-3 ? DehazeCore.ScaleChroma(toned, 1.0 + sat) : toned.Clone();
        }
    }
}
