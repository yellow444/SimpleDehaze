using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Экспериментальная transmission-aware Laplacian-реконструкция. Общий принцип многомасштабного
    /// dehazing с Gaussian/Laplacian pyramids известен (см. NOVELTY.md и arXiv:2111.05700).
    /// Здесь исследуется конкретный гейт полос по локальному t и измеренной мощности шума:
    ///   • мелкие детали усиливаем только там, где дымка тонкая (t велико → деталь реальна), и гасим
    ///     в плотной дымке (t мало → инверсия сильнее усиливает шум);
    ///   • крупные контуры/силуэты усиливаем везде (на грубом масштабе t-зависимость исчезает).
    ///
    /// Пропускание t оценивается СРАЗУ ДВУМЯ приорами (Color Attenuation Prior + Dark Channel) и
    /// уточняется Guided-фильтром; восстановление — с локальным полем атмосферного света и защитой
    /// хромы; финал — баланс белого + мягкий тон. Так контуры и детали держатся ровно там, где они
    /// физически восстановимы, а шум плотных зон не раздувается.
    /// </summary>
    public sealed class TransScaleLaplacianMethod : IDeHazeMethod
    {
        public string Name => "Transmission-aware Laplacian (эксперимент)";

        public string Description =>
            "Эксперимент: усиление многомасштабных полос по локальному пропусканию t(x), масштабу и шуму.\n" +
            "Общие Laplacian/Gaussian и edge-aware разложения известны; проверяемое отличие — " +
            "конкретный transmission/scale gate.\n\n" +
            "1. t из ДВУХ приоров: Color Attenuation Prior (exp(-β·d)) и Dark Channel (1-ω·DC), берём min,\n" +
            "   уточняем Guided-фильтром.\n" +
            "2. Восстановление с локальным полем A(x) и защитой хромы.\n" +
            "3. По умолчанию Laplacian-пирамида Lab-L; переключатели space/basis дают HSV-V, " +
            "полноразмерные edge-aware или stationary à trous bands (CPU/CUDA). Мелкие полосы усиливаем ТОЛЬКО где t велико,\n" +
            "   гасим где t мало (плотная дымка = шум); крупные контуры усиливаем везде.\n" +
            "4. Баланс белого + мягкий тон/вибранс.\n\n" +
            "Параметры t_lo/t_hi — где переключается «деталь ↔ шум» по пропусканию; gFine/gMid/gCoarse —\n" +
            "усиление мелких/средних/крупных полос. Деталь и контуры держатся там, где физически восстановимы.";

        private const double Theta0 = 0.121779, Theta1 = 0.959710, Theta2 = -0.780245;

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("beta",    "β - Color Attenuation",        0.1,  3.0,  1.2,  search: true),
            new ParamDef("omega",   "ω - Dark Channel",             0.5,  1.0,  0.90),
            new ParamDef("patch",   "Патч тёмного канала",          1,    15,   5,    1, isInt: true),
            new ParamDef("aRadius", "Окно поля A(x)",               20,   200,  90,   1, isInt: true),
            new ParamDef("airMix",  "Сила дехейза (глобальность A)", 0.0, 1.0,  0.85, search: true),
            new ParamDef("depth",   "Адаптивная глубина (чистка плотных зон)", 0.0, 2.5, 0.8, search: true),
            new ParamDef("min",     "t_min - нижний порог t",       0.02, 0.4,  0.07),
            new ParamDef("chroma",  "chromaFloor - защита цвета",   0.12, 0.8,  0.40),
            new ParamDef("rguide",  "Радиус Guided Filter",         5,    120,  50,   1, isInt: true),
            new ParamDef("eps",     "ε - регуляризация GF",         1e-5, 1e-2, 1e-3, log: true),
            new ParamDef("levels",  "Уровней пирамиды",             3,    6,    5,    1, isInt: true),
            new ParamDef("gFine",   "Мелкие детали (где тонкая дымка)", 0.0, 1.5, 0.5),
            new ParamDef("gMid",    "Средние контуры",              0.0,  3.0,  1.9,  search: true),
            new ParamDef("gCoarse", "Крупные силуэты (везде)",      0.0,  2.5,  1.5),
            new ParamDef("tLo",     "t_lo: ниже = плотно (гасим деталь)", 0.0, 0.6, 0.12),
            new ParamDef("tHi",     "t_hi: выше = тонко (полная деталь)", 0.1, 0.9, 0.45),
            new ParamDef("wb",      "Баланс белого (убрать цвет вуали)", 0.0, 1.5, 0.85, search: true),
            new ParamDef("sat",     "Вибранс цвета",                0.0,  0.8,  0.28, search: true),
            new ParamDef("tone",    "Возврат тона (растяжение L)",  0.0,  1.0,  0.45),
            new ParamDef("color",   "Потолок усиления цветности",   1.05, 1.7,  1.45),
            new ParamDef("smooth",  "Шумоподавление",              0.0,  6.0,  0.4),
            new ParamDef("wiener",  "Gate: 0=smoothstep, 1=модель шума", 0, 1, 0, 1, isInt: true, tunable: false),
            new ParamDef("rough",   "Шероховатость: 0=2 масштаба, 1=МНК по 5", 0, 1, 0, 1, isInt: true, tunable: false),
            new ParamDef("space",   "Полосы: 0=Lab-L, 1=HSV-V",    0, 1, 0, 1, isInt: true, tunable: false),
            new ParamDef("basis",   "Базис: 0=Laplacian, 1=edge, 2=UTAW CPU, 3=UTAW GPU", 0, 3, 0, 1, isInt: true, tunable: false),
            new ParamDef("edgeS",   "Edge bands: базовый spatial scale", 4, 48, 12, tunable: false),
            new ParamDef("edgeR",   "Edge bands: range scale",    0.03, 0.5, 0.18, tunable: false),
            new ParamDef("uNoise",  "UTAW: σ шума",                 0, 0.05, 0.004, tunable: false),
            new ParamDef("uUnc",    "UTAW: штраф CAP↔DCP",          0, 10, 1.0, tunable: false),
            new ParamDef("uRadius", "UTAW: радиус энергии",         0, 12, 3, 1, isInt: true, tunable: false),
            new ParamDef("uLimit",  "UTAW: предел delta полосы", 0.005, 0.15, 0.04, tunable: false),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            float beta = (float)p["beta"], omega = (float)p["omega"];
            int patch = (int)p["patch"], aRadius = (int)p["aRadius"], rGuide = (int)p["rguide"], levels = (int)p["levels"];
            double tmin = p["min"], chroma = p["chroma"], eps = p["eps"], airMix = p["airMix"], depthGain = p["depth"];
            double gFine = p["gFine"], gMid = p["gMid"], gCoarse = p["gCoarse"], tLo = p["tLo"], tHi = p["tHi"];
            double wb = p["wb"], sat = p["sat"], tone = p["tone"], color = p["color"], smooth = p["smooth"];

            bool wiener = p.TryGetValue("wiener", out var wv) && wv >= 0.5;
            bool roughRobust = p.TryGetValue("rough", out var rv) && rv >= 0.5;
            bool hsvValue = p.TryGetValue("space", out var sv) && sv >= 0.5;
            int basisMode = p.TryGetValue("basis", out var bv) ? Math.Clamp((int)Math.Round(bv), 0, 3) : 0;

            using var I = DehazeCore.Normalize(input);
            // structure-confidence: цвет и полосы гейтуем по ней. rough=1 - многомасштабная оценка с
            // МНК-наклоном, домноженная на R² (доверие); rough=0 - прежняя двухмасштабная.
            Mat rich;
            if (roughRobust)
            {
                rich = ContourOps.MultiscaleRoughness(I, out var r2);
                using (r2) CvInvoke.Multiply(rich, r2, rich);
                DehazeCore.Clamp01(rich);
            }
            else rich = ContourOps.FractalRichness(I, 3, 17);
            using var richOwned = rich;

            // --- пропускание из ДВУХ приоров ---
            using var hsv = new Mat();
            CvInvoke.CvtColor(I, hsv, ColorConversion.Bgr2Hsv);
            var hsvCh = hsv.Split();
            using var S = hsvCh[1];
            using var V = hsvCh[2];
            hsvCh[0].Dispose();

            using var dCap = new Mat();
            CvInvoke.AddWeighted(V, Theta1, S, Theta2, Theta0, dCap, DepthType.Cv32F);
            // адаптивная глубина d' = d·(1+depth·d): плотные зоны чистим сильнее
            using var dAdj = new Mat();
            if (depthGain > 1e-4)
            {
                using var dd = new Mat(); CvInvoke.Multiply(dCap, dCap, dd);
                CvInvoke.AddWeighted(dCap, 1.0, dd, depthGain, 0.0, dAdj);
            }
            else dCap.CopyTo(dAdj);
            using var tCap = new Mat();
            using (var nb = new Mat()) { dAdj.ConvertTo(nb, DepthType.Cv32F, -beta); CvInvoke.Exp(nb, tCap); }   // exp(-β·d')

            using var dark = DehazeCore.DarkChannel(I, patch);
            var A0 = DehazeCore.Atmospheric(I, dark, 0.001);
            using var tDcp = DehazeCore.RawTransmission(I, A0, omega, patch);                                    // 1-ω·DC

            using var tFused = new Mat();
            CvInvoke.Min(tCap, tDcp, tFused);                                                                    // консервативно: больше дымки
            using var tRef = new Mat();
            XImgprocInvoke.GuidedFilter(V, tFused, tRef, rGuide, eps);
            DehazeCore.Clamp01(tRef);

            // --- восстановление: локальное поле A(x), подмешанное к ГЛОБАЛЬНОМУ A (airMix→1 = сильнее
            //     снимается однородная вуаль; локальное A(x)≈I на ровной дымке само её не убирает) ---
            var aField = LocalHazeCore.AirlightField(I, patch, aRadius);
            if (airMix > 1e-3)
            {
                double[] ga = { A0.V0, A0.V1, A0.V2 };
                for (int c = 0; c < 3; c++)
                    aField[c].ConvertTo(aField[c], DepthType.Cv32F, 1.0 - airMix, ga[c] * airMix);   // A·(1-mix)+globalA·mix
            }
            using var recovered = LocalHazeCore.RecoverLocal(I, tRef, aField, tmin, chroma, rich);   // цвет за дымкой по структуре
            using var J01 = DeHazeCPU.Clip(recovered.Clone());
            LocalHazeCore.LocalAirlightWhiteBalance(J01, aField, tRef, wb);   // локальный ББ: убрать цвет вуали адаптивно
            foreach (var a in aField) a.Dispose();

            // --- транс-масштабная Лапласиан-реконструкция (ядро метода) ---
            // wiener=1: коэффициент полосы выводится из модели шума S/(S+σ²/t²) - без ручных t_lo/t_hi.
            // Шум оценивается по ЯРКОСТИ ВХОДА, то есть до усиления делением на t.
            Mat enhanced;
            if (basisMode is 2 or 3)
            {
                using var capSafe = tCap.Clone(); using var dcpSafe = tDcp.Clone();
                DehazeCore.Clamp(capSafe, tmin, 1); DehazeCore.Clamp(dcpSafe, tmin, 1);
                CvInvoke.Log(capSafe, capSafe); CvInvoke.Log(dcpSafe, dcpSafe);
                using var sigmaDepth = new Mat(); CvInvoke.AbsDiff(capSafe, dcpSafe, sigmaDepth);
                if (basisMode == 3 && hsvValue && GpuStationaryAtrous.IsAvailable)
                    enhanced = GpuStationaryAtrous.TransmissionAtrousHsv(J01, tRef, sigmaDepth,
                        Math.Clamp(levels - 1, 2, 5), gFine, gMid, gCoarse, tLo, tHi,
                        p["uNoise"], p["uUnc"], (int)p["uRadius"], p["uLimit"]);
                else
                    enhanced = ContourOps.TransmissionAtrousBands(J01, tRef, sigmaDepth, Math.Clamp(levels - 1, 2, 5),
                        gFine, gMid, gCoarse, tLo, tHi, hsvValue, p["uNoise"], p["uUnc"],
                        (int)p["uRadius"], p["uLimit"]);
            }
            else if (basisMode == 1)
                enhanced = ContourOps.TransmissionEdgeAwareBands(J01, tRef, Math.Clamp(levels - 2, 2, 4),
                    gFine, gMid, gCoarse, tLo, tHi, p["edgeS"], p["edgeR"], hsvValue, rich, 0.08);
            else if (hsvValue)
                enhanced = ContourOps.TransmissionScaleHsvValue(J01, tRef, levels,
                    gFine, gMid, gCoarse, tLo, tHi, rich, 0.08);
            else if (wiener)
                enhanced = ContourOps.WienerScaleLaplacian(J01, tRef, levels, gFine, gMid, gCoarse, I, tmin);
            else enhanced = ContourOps.TransmissionScaleLaplacian(J01, tRef, levels,
                    gFine, gMid, gCoarse, tLo, tHi, rich, 0.08);
            using var enhancedOwned = enhanced;

            // --- финал: цвет уже сбалансирован локальным ББ выше; тон/вибранс/цвет ---
            using var boosted = DehazeCore.LabEnhance(enhancedOwned, 0.0, 8, sat, 0.0);
            using var toned = DehazeCore.RestoreTone(boosted, tone, 0.01);
            using var hmAll = LocalHazeCore.HazeDensity(tRef);   // потолок цветности ослабляем по плотности дымки
            using var limited = DehazeCore.LimitColorfulness(toned, input.Mat, color, CvInvoke.Mean(hmAll).V0);
            return smooth > 0.01 ? DehazeCore.BilateralDenoise(limited, smooth) : DeHazeCPU.Clip(limited.Clone());
        }

    }
}
