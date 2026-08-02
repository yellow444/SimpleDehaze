using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Локально-адаптивная дымка, вариант «точность» - цель максимальные совмещённые PSNR/MSE/SSIM
    /// против эталона. Чистит умеренно, тон возвращает перцентильным растяжением яркости (без
    /// раскраски), цвет почти не трогает - чтобы структура и тон были близки к GT.
    /// Для авто-подбора выбирайте цель «GT PSNR/SSIM/ΔE/цвет (эталон)».
    /// </summary>
    public sealed class LocalHazeFidelityMethod : IDeHazeMethod
    {
        public string Name => "Локальная дымка - точность (PSNR/MSE/SSIM)";

        public string Description =>
            "Пространственно-неоднородная дымка с упором на ВЕРНОСТЬ эталону (PSNR/MSE/SSIM).\n\n" +
            "Движок LocalHazeCore: поле атмосферного света A(x), локально-адаптивная сила ω(x)\n" +
            "(гуще вуаль - сильнее чистим, небо/белый дым защищены), краевой FGS-уточнитель t.\n\n" +
            "Финиш: умеренное восстановление + перцентильное растяжение яркости (RestoreTone),\n" +
            "минимум усиления цвета, лёгкое шумоподавление. Не «сочно», а близко к GT.\n\n" +
            "Формула: t = 1 − ω(x)·DC(I/A(x)), уточн. FGS; J = (I − A(x))/max(t, t_min) + A(x),\n" +
            "хрома ÷ max(t, chromaFloor); баланс белого по цвету дымки A(x); финиш = RestoreTone(L) + денойз.\n\n" +
            "Совет: для авто-подбора берите цель «GT PSNR/SSIM/ΔE/цвет (эталон)».";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",   "ω - базовая доля удаляемой дымки", 0.5,  0.95, 0.78, search: true),
            new ParamDef("ogain",   "Прирост ω в плотных зонах",        0.0,  0.7,  0.30, search: true),
            new ParamDef("airMix",  "Сила съёма сплошной вуали (глоб. A)", 0.0, 1.0, 0.62, search: true),
            new ParamDef("patch",   "Патч тёмного канала",              1,    15,   5,    1, isInt: true),
            new ParamDef("aRadius", "Окно поля A(x)",                   20,   200,  90,   1, isInt: true),
            new ParamDef("min",     "t_min - нижний порог t",           0.03, 0.4,  0.10, search: true),
            new ParamDef("chroma",  "chromaFloor - защита цвета",       0.12, 0.8,  0.40),
            new ParamDef("wb",      "Баланс белого (убрать цвет вуали)", 0.0, 1.5, 0.55, search: true),
            new ParamDef("refine",  "FGS sigmaColor",                   5,    120,  24,   1, isInt: true),
            new ParamDef("tsky",    "t_sky - защита белых зон",         0.5,  0.95, 0.72),
            new ParamDef("knee",    "Колено мягких светов",             0.7,  1.0,  0.95),
            new ParamDef("denoise", "Денойз плотных зон (по 1-t)",      0.0,  1.0,  0.25),
            new ParamDef("quality", "Качество: 0=быстро, 1=HQ",        0,    1,    0,    1, isInt: true, tunable: false),
            new ParamDef("scale",   "HQ: апскейл ×",                    1,    3,    2,    1, isInt: true, tunable: false),
            new ParamDef("contour", "HQ: многомасштабные контуры",      0.0,  1.5,  0.30),
            new ParamDef("tone",    "Возврат тона (растяжение L)",      0.0,  1.0,  0.55, search: true),
            new ParamDef("smooth",  "Шумоподавление",                  0.0,  4.0,  1.0),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
            => LocalHazeCore.Run(input, p, FidelityFinish);

        // Референс-финиш: держим близость к GT (тон/структура), цвет специально не форсируем. t/conf не используем.
        private static Mat FidelityFinish(Mat recovered01, Mat inputBgr8, Mat t, Mat conf, IReadOnlyDictionary<string, double> p)
        {
            double tone = p.TryGetValue("tone", out var tv) ? tv : 0.55;
            double smooth = p.TryGetValue("smooth", out var sv) ? sv : 1.0;
            using var toned = DehazeCore.RestoreTone(recovered01, tone, 0.01);
            return DehazeCore.BilateralDenoise(toned, smooth);
        }
    }
}
