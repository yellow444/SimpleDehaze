using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Улучшенный Color Attenuation Prior. База — депт-приор Zhu/Mao/Wang (2015): глубина d = θ0+θ1·V+θ2·S,
    /// трансмиссия t = exp(-β·d), J = (I-A)/t + A. Он хорошо проявляет КОНТУРЫ (деревья сквозь дымку),
    /// потому что это лёгкое физическое восстановление без тяжёлой пост-обработки, которая контуры съедает.
    ///
    /// Что улучшено (минимально, чтобы НЕ потерять контуры CAP):
    ///  • <b>адаптивная глубина</b> d' = d·(1+gain·d) — в плотных зонах (большая d) чистим сильнее, так
    ///    остаточная вуаль над деревьями уходит и контуры читаются ярче, тонкие зоны не пережигаются;
    ///  • <b>баланс белого</b> (gray-world) — убирает характерную для CAP синеву, цвет ближе к натуральному;
    ///  • мягкий тон + вибранс + лёгкая резкость, БЕЗ CLAHE и без сильного денойза.
    /// </summary>
    public sealed class CapLocalMethod : IDeHazeMethod
    {
        public string Name => "Color Attenuation+ (адаптивная глубина, баланс белого)";

        public string Description =>
            "Улучшенный Color Attenuation Prior (Zhu/Mao/Wang 2015).\n\n" +
            "Шаги:\n" +
            "1. HSV: глубина d = 0.122 + 0.96·V - 0.78·S (дымка ↑V, ↓S → ↑глубина).\n" +
            "2. min-фильтр + Guided Filter (гайд — V).\n" +
            "3. Адаптивная глубина d' = d·(1 + depthGain·d): плотные зоны чистятся сильнее.\n" +
            "4. t = exp(-β·d'), t = max(t, t_min); J = (I - A)/t + A.\n" +
            "5. Баланс белого (gray-world) — убрать синеву CAP.\n" +
            "6. Мягкий тон + вибранс + лёгкая резкость (без CLAHE/тяжёлого денойза — берегём контуры).\n\n" +
            "Цель: контуры как у CAP, но чище от вуали, без синевы и читаемее.";

        private const double Theta0 = 0.121779, Theta1 = 0.959710, Theta2 = -0.780245;

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("beta",    "β - коэф. рассеяния",          0.1,  3.0,  1.5,  search: true),
            new ParamDef("depth",   "Адаптивная глубина (чистка плотных)", 0.0, 2.5, 0.7, search: true),
            new ParamDef("rmin",    "Радиус min-фильтра",           1,    25,   7,    1, isInt: true),
            new ParamDef("rguide",  "Радиус Guided Filter",         5,    120,  40,   1, isInt: true),
            new ParamDef("eps",     "ε - регуляризация GF",         1e-5, 1e-2, 1e-3, log: true),
            new ParamDef("min",     "t_min - нижний порог t",       0.02, 0.4,  0.07),
            new ParamDef("patch",   "Патч для оценки A",            1,    15,   5,    1, isInt: true),
            new ParamDef("wb",      "Баланс белого (убрать синеву)", 0.0, 1.0,  0.6),
            new ParamDef("detail",  "Резкость (микроконтраст)",     0.0,  0.7,  0.12),
            new ParamDef("sat",     "Вибранс цвета",                0.0,  0.8,  0.30, search: true),
            new ParamDef("tone",    "Возврат тона (растяжение L)",  0.0,  1.0,  0.50),
            new ParamDef("color",   "Потолок усиления цветности",   1.05, 1.7,  1.45),
            new ParamDef("smooth",  "Шумоподавление",              0.0,  6.0,  0.6),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            float beta = (float)p["beta"];
            double depthGain = p["depth"];
            int rMin = (int)p["rmin"], rGuide = (int)p["rguide"], patch = (int)p["patch"];
            double eps = p["eps"], tmin = p["min"], wb = p["wb"];
            double detail = p["detail"], sat = p["sat"], tone = p["tone"], color = p["color"], smooth = p["smooth"];
            float tMin = (float)tmin;

            using var I = new Mat();
            input.Mat.ConvertTo(I, DepthType.Cv32F, 1.0 / 255.0);

            // --- глубина по Color Attenuation Prior ---
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

            // --- адаптивная глубина: d' = d·(1 + depthGain·d) - плотные зоны чистим сильнее ---
            using var dAdj = new Mat();
            if (depthGain > 1e-4)
            {
                using var dd = new Mat(); CvInvoke.Multiply(dRef, dRef, dd);   // d²
                CvInvoke.AddWeighted(dRef, 1.0, dd, depthGain, 0.0, dAdj);      // d + gain·d²
            }
            else dRef.CopyTo(dAdj);

            // --- трансмиссия t = exp(-β·d'), t = max(t, t_min) ---
            using var t = new Mat();
            using (var negBetaD = new Mat()) { dAdj.ConvertTo(negBetaD, DepthType.Cv32F, -beta); CvInvoke.Exp(negBetaD, t); }
            using (var tm = new Mat(t.Size, DepthType.Cv32F, 1)) { tm.SetTo(new MCvScalar(tMin)); CvInvoke.Max(t, tm, t); }

            // --- глобальный атмосферный свет + классическое CAP-восстановление (даёт контраст) ---
            MCvScalar A = AtmosphericByDepth(I, dRef, 0.001);
            using var J = RecoverGlobal(I, t, A);

            // --- баланс белого (gray-world) -> убрать синеву CAP ---
            using var balanced = WhiteBalance(J, A, wb);
            using var bal01 = DeHazeCPU.Clip(balanced.Clone());

            // --- лёгкая косметика: вибранс + микроконтраст (БЕЗ CLAHE), тон, потолок цвета ---
            using var boosted = DehazeCore.LabEnhance(bal01, 0.0, 8, sat, detail);
            using var toned = DehazeCore.RestoreTone(boosted, tone, 0.01);
            using var limited = DehazeCore.LimitColorfulness(toned, input.Mat, color);
            return smooth > 0.01 ? DehazeCore.BilateralDenoise(limited, smooth) : DeHazeCPU.Clip(limited.Clone());
        }

        private static Mat RecoverGlobal(Mat i01, Mat t, MCvScalar a)
        {
            var src = i01.Split();
            double[] av = { a.V0, a.V1, a.V2 };
            using var outv = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                var jc = new Mat();
                CvInvoke.Subtract(src[c], new ScalarArray(av[c]), jc);
                CvInvoke.Divide(jc, t, jc);
                CvInvoke.Add(jc, new ScalarArray(av[c]), jc);
                outv.Push(jc); jc.Dispose(); src[c].Dispose();
            }
            var J = new Mat();
            CvInvoke.Merge(outv, J);
            return J;
        }

        /// <summary>
        /// Баланс белого по ЦВЕТУ ДЫМКИ A (не gray-world): усиление канала = mean(A)/A_c. Убирает налёт
        /// тумана, но не тянет средний цвет кадра к серому - настоящий цвет сцены (зелень) сохраняется.
        /// </summary>
        private static Mat WhiteBalance(Mat bgr01, MCvScalar a, double strength)
        {
            if (strength <= 1e-3) return bgr01.Clone();
            double meanA = (a.V0 + a.V1 + a.V2) / 3.0;
            double[] ac = { a.V0, a.V1, a.V2 };
            var ch = bgr01.Split();
            using var outv = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                double gain = Math.Clamp(meanA / (ac[c] + 1e-6), 0.7, 1.5);
                CvInvoke.Multiply(ch[c], new ScalarArray(1.0 + strength * (gain - 1.0)), ch[c]);
                outv.Push(ch[c]);
            }
            var res = new Mat();
            CvInvoke.Merge(outv, res);
            foreach (var c in ch) c.Dispose();
            return res;
        }

        private static MCvScalar AtmosphericByDepth(Mat i01, Mat depth, double topPercent)
        {
            int n = depth.Rows * depth.Cols;
            var dData = new float[n]; depth.CopyTo(dData);
            var ch = i01.Split();
            var b = new float[n]; var g = new float[n]; var r = new float[n];
            ch[0].CopyTo(b); ch[1].CopyTo(g); ch[2].CopyTo(r);
            foreach (var c in ch) c.Dispose();
            int k = Math.Max(1, (int)(n * topPercent));
            var idx = Enumerable.Range(0, n).OrderByDescending(i => dData[i]).Take(k).ToArray();
            double sb = 0, sg = 0, sr = 0;
            foreach (var i in idx) { sb += b[i]; sg += g[i]; sr += r[i]; }
            return new MCvScalar(sb / k, sg / k, sr / k);
        }
    }
}
