using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Экспериментальный HSV/CAP-конвейер с картой многомасштабной шероховатости.
    /// Карта используется только как confidence наличия структуры сцены. Это не оценка плотности дымки
    /// и не доказанная локальная фрактальная размерность: такой claim запрещён в NOVELTY.md.
    ///
    /// Пайплайн: HSV Color Attenuation Prior → пропускание t; фрактальная насыщенность R(x); гейт детали
    /// G(x)=R(x)·gate(t) (текстура И тонкая дымка = деталь реальна); восстановление с локальным A(x);
    /// затем усиление Лапласовых полос, где мелкие полосы усиливаются пропорционально G(x), а крупные
    /// контуры — везде. Финал: баланс белого + мягкий тон.
    /// </summary>
    public sealed class FractalHsvMethod : IDeHazeMethod
    {
        public string Name => "HSV + многомасштабная шероховатость (эксперимент)";

        public string Description =>
            "Экспериментальный метод: карта шероховатости задаёт confidence структуры сцены.\n" +
            "Она не измеряет плотность дымки и не заявляется фрактальной размерностью.\n\n" +
            "1. HSV Color Attenuation Prior: d=θ0+θ1·V+θ2·S, t=exp(-β·d), min-фильтр + Guided.\n" +
            "2. Шероховатость R(x)=1-H: прежние 2 масштаба либо 5 масштабов + R² для ablation:\n" +
            "   велика на согласованной многомасштабной текстуре, мала на гладких областях.\n" +
            "3. Гейт детали G = R·gate(t): усиливаем деталь ТОЛЬКО где текстура И тонкая дымка;\n" +
            "   в гладкой дымке/шуме — не трогаем. Крупные контуры усиливаем везде.\n" +
            "4. Восстановление с локальным A(x) + защита хромы, Лапласиан-реконструкция по G(x).\n" +
            "5. Баланс белого + мягкий тон/вибранс.\n\n" +
            "rough=1 — пятишкальная оценка с R²; rough=0 — прежняя двухмасштабная эвристика для ablation.";

        private const double Theta0 = 0.121779, Theta1 = 0.959710, Theta2 = -0.780245;

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("beta",    "β - Color Attenuation",        0.1,  3.0,  1.3,  search: true),
            new ParamDef("depth",   "Адаптивная глубина (чистка плотных зон)", 0.0, 2.5, 0.9, search: true),
            new ParamDef("clahe",   "Локальный контраст (проявить вуаль)", 0.0, 5.0, 1.6),
            new ParamDef("rmin",    "Радиус min-фильтра глубины",   1,    25,   7,    1, isInt: true),
            new ParamDef("rguide",  "Радиус Guided Filter",         5,    120,  50,   1, isInt: true),
            new ParamDef("eps",     "ε - регуляризация GF",         1e-5, 1e-2, 1e-3, log: true),
            new ParamDef("patch",   "Патч для A(x)",                1,    15,   5,    1, isInt: true),
            new ParamDef("aRadius", "Окно поля A(x)",               20,   200,  90,   1, isInt: true),
            new ParamDef("airMix",  "Сила дехейза (глобальность A)", 0.0, 1.0,  0.85, search: true),
            new ParamDef("min",     "t_min - нижний порог t",       0.02, 0.4,  0.08),
            new ParamDef("chroma",  "chromaFloor - защита цвета",   0.12, 0.8,  0.22),
            new ParamDef("colorRestore", "Восстановление цвета (по плотности дымки)", 0.0, 1.5, 0.7, search: true),
            new ParamDef("kSmall",  "Фрактал: малое окно",          3,    9,    3,    1, isInt: true),
            new ParamDef("kLarge",  "Фрактал: большое окно",        9,    31,   17,   1, isInt: true),
            new ParamDef("rough",   "Шероховатость: 0=2 масштаба, 1=МНК по 5", 0, 1, 0, 1, isInt: true, tunable: false),
            new ParamDef("tGlo",    "Гейт t: ниже = плотно",        0.0,  0.6,  0.12),
            new ParamDef("tGhi",    "Гейт t: выше = тонко",         0.1,  0.9,  0.45),
            new ParamDef("levels",  "Уровней пирамиды",             3,    6,    5,    1, isInt: true),
            new ParamDef("gFine",   "Мелкая деталь (по фракталу)",  0.0,  1.8,  0.8),
            new ParamDef("gMid",    "Средние контуры",              0.0,  3.0,  1.9,  search: true),
            new ParamDef("gCoarse", "Крупные силуэты (везде)",      0.0,  2.5,  1.5),
            new ParamDef("wb",      "Баланс белого (убрать цвет вуали)", 0.0, 1.5, 0.85, search: true),
            new ParamDef("sat",     "Вибранс цвета",                0.0,  0.8,  0.28, search: true),
            new ParamDef("tone",    "Возврат тона (растяжение L)",  0.0,  1.0,  0.45),
            new ParamDef("color",   "Потолок усиления цветности",   1.05, 1.7,  1.45),
            new ParamDef("smooth",  "Шумоподавление",              0.0,  6.0,  0.4),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            float beta = (float)p["beta"];
            int rMin = (int)p["rmin"], rGuide = (int)p["rguide"], patch = (int)p["patch"], aRadius = (int)p["aRadius"];
            int kSmall = (int)p["kSmall"], kLarge = (int)p["kLarge"], levels = (int)p["levels"];
            double eps = p["eps"], tmin = p["min"], chroma = p["chroma"], tGlo = p["tGlo"], tGhi = p["tGhi"], airMix = p["airMix"];
            double depthGain = p["depth"], clahe = p["clahe"], colorRestore = p["colorRestore"];
            double gFine = p["gFine"], gMid = p["gMid"], gCoarse = p["gCoarse"];
            double wb = p["wb"], sat = p["sat"], tone = p["tone"], color = p["color"], smooth = p["smooth"];

            using var I = DehazeCore.Normalize(input);

            // --- HSV Color Attenuation Prior -> t ---
            using var hsv = new Mat();
            CvInvoke.CvtColor(I, hsv, ColorConversion.Bgr2Hsv);
            var hsvCh = hsv.Split();
            using var S = hsvCh[1];
            using var V = hsvCh[2];
            hsvCh[0].Dispose();

            using var d = new Mat();
            CvInvoke.AddWeighted(V, Theta1, S, Theta2, Theta0, d, DepthType.Cv32F);
            using (var elem = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(2 * rMin + 1, 2 * rMin + 1), new Point(-1, -1)))
                CvInvoke.Erode(d, d, elem, new Point(-1, -1), 1, BorderType.Reflect101, default);
            using var dRef = new Mat();
            XImgprocInvoke.GuidedFilter(V, d, dRef, rGuide, eps);
            // адаптивная глубина d' = d·(1+depth·d): в плотных зонах (большая d) чистим сильнее
            using var dAdj = new Mat();
            if (depthGain > 1e-4)
            {
                using var dd = new Mat(); CvInvoke.Multiply(dRef, dRef, dd);
                CvInvoke.AddWeighted(dRef, 1.0, dd, depthGain, 0.0, dAdj);
            }
            else dRef.CopyTo(dAdj);
            using var t = new Mat();
            using (var nb = new Mat()) { dAdj.ConvertTo(nb, DepthType.Cv32F, -beta); CvInvoke.Exp(nb, t); }
            DehazeCore.Clamp01(t);

            // --- фрактальная насыщенность и гейт детали G = R·gate(t) ---
            Mat richMap;
            if (p.TryGetValue("rough", out var roughMode) && roughMode >= 0.5)
            {
                richMap = ContourOps.MultiscaleRoughness(I, out var r2);
                using (r2) CvInvoke.Multiply(richMap, r2, richMap);
                DehazeCore.Clamp01(richMap);
            }
            else richMap = ContourOps.FractalRichness(I, kSmall, kLarge);
            using var rich = richMap;
            double span = Math.Max(1e-3, tGhi - tGlo);
            using var tGate = new Mat();
            t.ConvertTo(tGate, DepthType.Cv32F, 1.0 / span, -tGlo / span);
            DehazeCore.Clamp01(tGate);
            using var gate = new Mat();
            CvInvoke.Multiply(rich, tGate, gate);
            DehazeCore.Clamp01(gate);

            // глобальный атмосферный свет = ЦВЕТ ДЫМКИ (нужен и для силы дехейза, и для баланса белого)
            using var dark = DehazeCore.DarkChannel(I, patch);
            var gA = DehazeCore.Atmospheric(I, dark, 0.001);

            // --- атмосферный свет: локальное поле, смешанное с ГЛОБАЛЬНЫМ. В однородной плотной дымке
            //     локальное A(x)≈I → вуаль не убирается; подмешивание глобального A (airMix→1) реально
            //     снимает пелену (сила дехейза). ---
            var aField = LocalHazeCore.AirlightField(I, patch, aRadius);
            if (airMix > 1e-3)
            {
                double[] ga = { gA.V0, gA.V1, gA.V2 };
                for (int c = 0; c < 3; c++)
                    aField[c].ConvertTo(aField[c], DepthType.Cv32F, 1.0 - airMix, ga[c] * airMix);   // A·(1-mix)+globalA·mix
            }
            using var recovered = LocalHazeCore.RecoverLocal(I, t, aField, tmin, chroma, rich);   // цвет за дымкой по фрактальной структуре
            using var J01 = DeHazeCPU.Clip(recovered.Clone());
            // баланс белого по ЛОКАЛЬНОМУ цвету дымки A(x), гейт по насыщенности вуали: убирает синий/серый
            // налёт именно там, где вуаль цветная, не трогая настоящий цвет сцены (адаптивнее глобального ББ).
            LocalHazeCore.LocalAirlightWhiteBalance(J01, aField, t, wb);
            foreach (var a in aField) a.Dispose();

            // --- восстановление цвета: где дымка была гуще (малое t), там вернуть больше хромы ---
            Mat colored;
            if (colorRestore > 1e-3)
            {
                using var hazeMask = new Mat();
                t.ConvertTo(hazeMask, DepthType.Cv32F, -1.0, 1.0);   // 1 - t = плотность дымки
                DehazeCore.Clamp01(hazeMask);
                colored = DehazeCore.ScaleChromaByMask(J01, hazeMask, colorRestore, rich);   // hue-faithful, гейт по фракталу
            }
            else colored = J01.Clone();

            // --- Лапласиан-реконструкция: мелкие полосы по гейту G, крупные - везде ---
            using var enhanced = ContourOps.TransmissionScaleLaplacian(colored, gate, levels, gFine, gMid, gCoarse, 0.0, 1.0, rich, 0.08);
            colored.Dispose();

            // --- финал: цвет уже сбалансирован локальным ББ выше; здесь CLAHE + тон ---
            using var boosted = DehazeCore.LabEnhance(enhanced, clahe, 8, sat, 0.0);   // CLAHE проявляет структуру в пелене
            using var toned = DehazeCore.RestoreTone(boosted, tone, 0.01);
            using var hmAll = LocalHazeCore.HazeDensity(t);   // потолок цветности ослабляем по плотности дымки
            using var limited = DehazeCore.LimitColorfulness(toned, input.Mat, color, CvInvoke.Mean(hmAll).V0);
            return smooth > 0.01 ? DehazeCore.BilateralDenoise(limited, smooth) : DeHazeCPU.Clip(limited.Clone());
        }

    }
}
