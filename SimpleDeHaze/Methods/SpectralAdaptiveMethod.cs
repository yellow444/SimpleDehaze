using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Спектрально-адаптивное восстановление (качество). Новый метод: см. docs/methods/spectral-adaptive.md.
    ///
    /// Идея: оценить пропускание t НЕ по одному 'тёмному' каналу, а по каждому каналу отдельно и
    /// слить их с весом локального контраста - там, где канал 'засвечен' (вуаль дымки подняла его
    /// к A_c, контраст низкий), его вклад мал, а где канал держит структуру - велик. Плюс яркостный
    /// гейт для неба, восстановление с защитой цвета и финальное усиление в HSV (CLAHE по V -
    /// контуры; буст S - цветность).
    /// </summary>
    public sealed class SpectralAdaptiveMethod : IDeHazeMethod
    {
        public string Name => "Спектрально-адаптивный (качество)";

        public string Description =>
            "Качество: контуры + цвет по максимуму. Учитывает атмосферу, яркость зон и спектр.\n\n" +
            "Формулы:\n" +
            "1. A_c - атмосферный свет (ярчайшие в тёмном канале).\n" +
            "2. По каждому каналу: t_c = 1 - ω*min_Ω(I_c/A_c); вес w_c = локальное СКО I_c\n" +
            "   (мало там, где канал 'засвечен' вуалью). Слияние: t = Σ_c w_c*t_c / Σ_c w_c.\n" +
            "3. Небо: маска (ярко*малонасыщенно) поднимает t -> меньше дехейзим пересветы.\n" +
            "4. Уточнение t через FGS; восстановление J с защитой цвета (яркость/хрома раздельно).\n" +
            "5. Lab: CLAHE по L (контуры) + вибранс a,b (цветность естественнее HSV-S).\n" +
            "6. Декуплинг: дехейзим агрессивно (контуры по максимуму), а цвет держим\n" +
            "   глобальным потолком x1.25 (живо, но без перенасыщения).\n\n" +
            "Параметры: ω, patch - дымка; refine - FGS; clip, tiles - CLAHE; sat - вибранс.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega", "ω - доля удаляемой дымки",     0.3, 0.95, 0.72, search: true),
            new ParamDef("patch", "Патч (окно min/контраста)",    1, 15, 5, 1, isInt: true),
            new ParamDef("min",   "t_min - нижний порог t",       0.01, 0.5, 0.1),
            new ParamDef("refine","Сглаживание t (FGS, σ края)",  5, 120, 30, 1, isInt: true),
            new ParamDef("clip",  "CLAHE: контраст (контуры)",    1.0, 6.0, 2.5, search: true),
            new ParamDef("tiles", "CLAHE: сетка тайлов",          2, 16, 8, 1, isInt: true),
            new ParamDef("sat",   "Усиление насыщенности",        0.0, 1.0, 0.2, search: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"], tmin = p["min"], satBoost = p["sat"], clip = p["clip"];
            int patch = (int)p["patch"], refine = (int)p["refine"], tiles = Math.Max(2, (int)p["tiles"]);

            using var I = DehazeCore.Normalize(input);
            using var dark = DehazeCore.DarkChannel(I, patch);
            var A = DehazeCore.Atmospheric(I, dark, 0.001);
            double[] av = { A.V0, A.V1, A.V2 };

            var ch = I.Split();
            var ks = new Size(2 * patch + 1, 2 * patch + 1);
            var anc = new Point(-1, -1);
            using var elem = CvInvoke.GetStructuringElement(ElementShape.Rectangle, ks, anc);

            // --- спектрально-адаптивное t: слияние поканальных t_c с весом локального контраста ---
            using var wsum = new Mat(I.Size, DepthType.Cv32F, 1); wsum.SetTo(new MCvScalar(0));
            using var twsum = new Mat(I.Size, DepthType.Cv32F, 1); twsum.SetTo(new MCvScalar(0));
            for (int c = 0; c < 3; c++)
            {
                using var norm = new Mat(); CvInvoke.Divide(ch[c], new ScalarArray(av[c]), norm);       // I_c/A_c
                using var lmin = new Mat(); CvInvoke.Erode(norm, lmin, elem, anc, 1, BorderType.Reflect101, default);
                using var tc = new Mat(); lmin.ConvertTo(tc, DepthType.Cv32F, -omega, 1.0);             // t_c = 1 - ω*min

                using var mean = new Mat(); CvInvoke.Blur(ch[c], mean, ks, anc);
                using var mean2 = new Mat(); using (var sq = new Mat()) { CvInvoke.Multiply(ch[c], ch[c], sq); CvInvoke.Blur(sq, mean2, ks, anc); }
                using var wc = new Mat(); using (var m2 = new Mat()) { CvInvoke.Multiply(mean, mean, m2); CvInvoke.Subtract(mean2, m2, wc); }   // var
                DehazeCore.Clamp01(wc);                                                                  // var >= 0 (и <=1 - безразлично)
                CvInvoke.Sqrt(wc, wc);                                                                   // σ = локальный контраст
                CvInvoke.Add(wc, new ScalarArray(1e-3), wc);                                             // + ε
                // меньше доверия 'засвеченным' зонам канала (яркий I_c -> вуаль/клиппинг): w_c *= 1 - 0.8*clamp((I_c-0.85)/0.15)
                using (var anti = new Mat())
                {
                    mean.ConvertTo(anti, DepthType.Cv32F, 1.0 / 0.15, -0.85 / 0.15);
                    DehazeCore.Clamp01(anti);
                    anti.ConvertTo(anti, DepthType.Cv32F, -0.8, 1.0);                                    // 1 -> 0.2 по мере засветки
                    CvInvoke.Multiply(wc, anti, wc);
                }

                using (var wt = new Mat()) { CvInvoke.Multiply(wc, tc, wt); CvInvoke.Add(twsum, wt, twsum); }
                CvInvoke.Add(wsum, wc, wsum);
            }
            using var tFused = new Mat(); CvInvoke.Divide(twsum, wsum, tFused);                          // Σw*t / Σw

            // --- яркостный гейт неба: ярко+малонасыщенно => дехейзим слабее ---
            using (var sky = DehazeCore.SkyMask(I)) DehazeCore.RaiseInSky(tFused, sky, 0.7);
            DehazeCore.Clamp01(tFused);

            // --- edge-aware уточнение (FGS - ближе к WLS, чище края, чем guided filter) ---
            using var t = new Mat();
            using (var guide8 = new Mat()) { I.ConvertTo(guide8, DepthType.Cv8U, 255.0); XImgprocInvoke.FastGlobalSmootherFilter(guide8, tFused, t, 500, refine, 0.25, 3); }
            using var J = DehazeCore.Recover(I, t, A, tmin);
            foreach (var c in ch) c.Dispose();

            // --- #2 Lab: CLAHE по L (контуры) + ВИБРАНС a,b (цвет естественнее HSV-S, перцептивно ровнее) ---
            using var J8 = new Mat(); J.ConvertTo(J8, DepthType.Cv8U, 255.0);
            using var lab = new Mat(); CvInvoke.CvtColor(J8, lab, ColorConversion.Bgr2Lab);
            var lc = lab.Split();   // L, a, b (8U; a,b центрированы на 128)
            CvInvoke.CLAHE(lc[0], clip, new Size(tiles, tiles), lc[0]);                                  // CLAHE по L -> контуры
            using (var af = new Mat()) using (var bf = new Mat())
            {
                lc[1].ConvertTo(af, DepthType.Cv32F, 1.0, -128.0);   // a - 128
                lc[2].ConvertTo(bf, DepthType.Cv32F, 1.0, -128.0);   // b - 128
                using var chroma = new Mat();
                using (var a2 = new Mat()) using (var b2 = new Mat()) { CvInvoke.Multiply(af, af, a2); CvInvoke.Multiply(bf, bf, b2); CvInvoke.Add(a2, b2, chroma); }
                CvInvoke.Sqrt(chroma, chroma);                                                            // |хрома|
                using var factor = new Mat(); chroma.ConvertTo(factor, DepthType.Cv32F, -1.0 / 128.0, 1.0); DehazeCore.Clamp01(factor);
                factor.ConvertTo(factor, DepthType.Cv32F, satBoost, 1.0);                                 // 1 + sat*(1 - |хрома|/128): больше малонасыщенным, бережём вивид
                CvInvoke.Multiply(af, factor, af); CvInvoke.Multiply(bf, factor, bf);
                af.ConvertTo(lc[1], DepthType.Cv8U, 1.0, 128.0); bf.ConvertTo(lc[2], DepthType.Cv8U, 1.0, 128.0);
            }
            using (var v = new VectorOfMat(lc)) CvInvoke.Merge(v, lab);
            foreach (var x in lc) x.Dispose();

            using var outBgr = new Mat(); CvInvoke.CvtColor(lab, outBgr, ColorConversion.Lab2Bgr);
            var res = new Mat(); outBgr.ConvertTo(res, DepthType.Cv32F, 1.0 / 255.0);

            // ДЕКУПЛИНГ цвета и дехейза: дехейзим агрессивно (большая ω -> контуры), цвет держим потолком ~x1.25
            using (var res8 = new Mat())
            {
                res.ConvertTo(res8, DepthType.Cv8U, 255.0);
                double ratio = Metrics.Colorfulness(res8) / (Metrics.Colorfulness(input.Mat) + 1e-6);
                if (ratio > 1.25)
                {
                    double scale = 1.25 / ratio;
                    using var gray = new Mat(); CvInvoke.CvtColor(res, gray, ColorConversion.Bgr2Gray);
                    using var gray3 = new Mat(); using (var v = new VectorOfMat()) { v.Push(gray); v.Push(gray); v.Push(gray); CvInvoke.Merge(v, gray3); }
                    using var chroma = new Mat(); CvInvoke.Subtract(res, gray3, chroma); chroma.ConvertTo(chroma, DepthType.Cv32F, scale);
                    CvInvoke.Add(gray3, chroma, res);   // res = gray + scale*(res - gray)
                }
            }
            return res;
        }
    }
}
