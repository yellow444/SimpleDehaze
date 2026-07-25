using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Общий локально-адаптивный движок для пространственно-неоднородной дымки (дым/туман/снег,
    /// у которых плотность сильно меняется по кадру). В отличие от классического DCP с одной
    /// константой A и одной силой ω, здесь:
    ///   * атмосферный свет - гладкое поле A_c(x) = blur(q·I_c)/blur(q), где доверие
    ///     q = D²·(1-S)^1.5·flat велико в ярких, малонасыщенных и гладких зонах (небо/вуаль)
    ///     и мало на структурных объектах;
    ///   * сила удаления дымки ω(x) поднимается там, где локально гуще остаточная вуаль
    ///     (большой тёмный канал I/A), и НЕ поднимается в небе/белом дыме;
    ///   * границы объектов сохраняет краевой FGS-уточнитель карты t.
    /// Так плотный угол вычищается сильнее, а тонкие зоны не выжигаются - объекты
    /// (например дерево в правом верхнем углу 09_outdoor) восстанавливаются заметно лучше.
    ///
    /// Косметика (тон/контраст/цвет) вынесена в делегат <see cref="Finish"/> - три метода-пресета
    /// (точность / объекты / сочно) отличаются только им и дефолтами параметров.
    /// </summary>
    internal static class LocalHazeCore
    {
        /// <summary>
        /// Финишная косметика варианта: получает восстановленный BGR float[0,1], оригинал (8U), карту
        /// пропускания t и structure-confidence conf (фрактальная насыщенность) — обе одного размера с
        /// recovered01. t/conf ЗАИМСТВОВАНЫ (владеет Run) — не диспозить и не менять на месте.
        /// </summary>
        public delegate Mat Finish(Mat recovered01, Mat inputBgr8, Mat t, Mat conf, IReadOnlyDictionary<string, double> p);

        /// <summary>
        /// Гейт резкости/цвета для финишей: gate = conf · smoothstep(t; 0.10..0.40). Велик там, где ЕСТЬ
        /// структура (conf) И дымка тонкая (t велико) — там резкость/деталь реальна; в плотной/плоской
        /// дымке ≈0 (анти-хруст). Возвращает новый Mat [0,1].
        /// </summary>
        /// <summary>Плотность дымки = clamp01(1 - t). Новый Mat [0,1] (вызывающий освобождает).</summary>
        public static Mat HazeDensity(Mat t)
        {
            var m = new Mat();
            t.ConvertTo(m, DepthType.Cv32F, -1.0, 1.0);
            DehazeCore.Clamp01(m);
            return m;
        }

        public static Mat DetailGate(Mat conf, Mat t)
        {
            const double lo = 0.10, span = 0.30;
            var g = new Mat();
            t.ConvertTo(g, DepthType.Cv32F, 1.0 / span, -lo / span);
            DehazeCore.Clamp01(g);
            using (var s2 = new Mat()) { CvInvoke.Multiply(g, g, s2); using var s3 = new Mat(); CvInvoke.Multiply(s2, g, s3); CvInvoke.AddWeighted(s2, 3.0, s3, -2.0, 0.0, g); }  // 3s²-2s³
            using (var c = DehazeCore.ResizeTo(conf, g.Size)) CvInvoke.Multiply(g, c, g);
            DehazeCore.Clamp01(g);
            return g;
        }

        public static Mat Run(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p, Finish finish)
        {
            double Get(string k, double def) => p.TryGetValue(k, out var v) ? v : def;

            double omega = Get("omega", 0.85), ogain = Get("ogain", 0.45), tmin = Get("min", 0.08);
            double chroma = Get("chroma", 0.35), tsky = Get("tsky", 0.7), knee = Get("knee", 0.92);
            double denoise = Get("denoise", 0.5), contour = Get("contour", 0.0);

            // Два уровня качества. HQ (quality>=0.5): кадр апскейлится (Lanczos+сглаживание), весь пайплайн
            // и многомасштабное усиление контуров считаются на увеличенной сетке, затем результат
            // ужимается обратно. Это в разы дороже (×scale² пикселей), но вытаскивает больше контуров/
            // объектов в умеренной дымке. Fast (quality<0.5): прежний быстрый путь без апскейла.
            bool hq = Get("quality", 0.0) >= 0.5;
            int scale = hq ? Math.Clamp((int)Get("scale", 2), 1, 3) : 1;
            int patch = (int)Get("patch", 5), aR = (int)Get("aRadius", 90), refine = (int)Get("refine", 24);
            if (hq) { aR *= scale; refine = Math.Max(refine, refine * scale / 2 + refine); }

            var anc = new Point(-1, -1);
            using var work = scale <= 1 ? input.Clone() : Upscale(input, scale);
            using var I = DehazeCore.Normalize(work);

            // --- доверие к airlight: q = D² · (1-S)^1.5 · flat ---
            using var D = DehazeCore.DarkChannel(I, patch);
            using var S = DehazeCore.Saturation(I);
            using var flat = DehazeCore.Flatness(I, 2.0);     // ~1 на гладком, ~0 на краях

            using var q = new Mat();
            CvInvoke.Multiply(D, D, q);
            using (var oneMinusS = new Mat())
            {
                S.ConvertTo(oneMinusS, DepthType.Cv32F, -1.0, 1.0);
                using var sm = new Mat();
                CvInvoke.Pow(oneMinusS, 1.5, sm);
                CvInvoke.Multiply(q, sm, q);
            }
            CvInvoke.Multiply(q, flat, q);
            CvInvoke.Add(q, new ScalarArray(1e-4), q);

            // --- поле атмосферного света A_c(x) = blur(q·I_c)/blur(q) ---
            var ks = new Size(2 * aR + 1, 2 * aR + 1);
            using var denA = new Mat();
            CvInvoke.Blur(q, denA, ks, anc);
            CvInvoke.Add(denA, new ScalarArray(1e-6), denA);

            var ch = I.Split();
            var aField = new Mat[3];
            for (int c = 0; c < 3; c++)
            {
                using var qi = new Mat();
                CvInvoke.Multiply(q, ch[c], qi);
                aField[c] = new Mat();
                CvInvoke.Blur(qi, aField[c], ks, anc);
                CvInvoke.Divide(aField[c], denA, aField[c]);
                using var floor = new Mat(aField[c].Size, DepthType.Cv32F, 1);
                floor.SetTo(new MCvScalar(1e-2));
                CvInvoke.Max(aField[c], floor, aField[c]);   // защита от деления на ~0
            }

            // подмешать ГЛОБАЛЬНЫЙ атмосферный свет A в поле A(x): на однородной ПЛОТНОЙ дымке локальное
            // A(x)≈I, поэтому (I−A)≈0 и вуаль почти не снимается; глобальное A (airMix→1) реально убирает
            // сплошную пелену. Так «Локальная дымка» дехейзит не слабее фрактального (у которого airMix есть).
            double airMix = Get("airMix", 0.75);
            if (airMix > 1e-3)
            {
                var gA = DehazeCore.Atmospheric(I, D, 0.001);
                double[] ga = { gA.V0, gA.V1, gA.V2 };
                for (int c = 0; c < 3; c++)
                    aField[c].ConvertTo(aField[c], DepthType.Cv32F, 1.0 - airMix, ga[c] * airMix);   // A(x)·(1−mix)+A_глоб·mix
            }

            // --- тёмный канал нормированного I/A(x) = локальная плотность вуали ---
            using var norm = new Mat();
            using (var vn = new VectorOfMat())
            {
                for (int c = 0; c < 3; c++) { using var nc = new Mat(); CvInvoke.Divide(ch[c], aField[c], nc); vn.Push(nc); }
                CvInvoke.Merge(vn, norm);
            }
            using var dcA = DehazeCore.DarkChannel(norm, patch);
            DehazeCore.Clamp01(dcA);

            // защита неба/белого дыма, но НЕ структурных ярких объектов (у них flat≈0)
            using var protect = DehazeCore.SkyMask(I);   // bright·lowsat
            CvInvoke.Multiply(protect, flat, protect);
            DehazeCore.Clamp01(protect);

            // --- локально-адаптивная ω(x) = clamp(ω + ogain·density·(1-protect)) ---
            using var density = DehazeCore.FastGaussian(dcA, Math.Max(8.0, aR / 4.0));
            DehazeCore.Clamp01(density);
            using var protectInv = new Mat();
            protect.ConvertTo(protectInv, DepthType.Cv32F, -1.0, 1.0);   // 1 - protect
            using var omegaMap = new Mat();
            CvInvoke.Multiply(density, protectInv, omegaMap);
            CvInvoke.Multiply(omegaMap, new ScalarArray(ogain), omegaMap);
            CvInvoke.Add(omegaMap, new ScalarArray(omega), omegaMap);
            DehazeCore.Clamp(omegaMap, 0.30, 0.98);

            using var veil = new Mat();
            CvInvoke.Multiply(omegaMap, dcA, veil);            // ω(x)·dc
            using var tRaw = new Mat();
            veil.ConvertTo(tRaw, DepthType.Cv32F, -1.0, 1.0);  // t = 1 - ω(x)·dc
            DehazeCore.RaiseInSky(tRaw, protect, tsky);        // поднять t в защищённых зонах
            DehazeCore.Clamp01(tRaw);

            // --- краевой уточнитель карты t (резкие границы объектов без гало) ---
            using var t = new Mat();
            using (var guide8 = new Mat())
            {
                I.ConvertTo(guide8, DepthType.Cv8U, 255.0);
                XImgprocInvoke.FastGlobalSmootherFilter(guide8, tRaw, t, 550, Math.Max(5, refine), 0.25, 3);
            }
            DehazeCore.Clamp01(t);

            // --- structure-confidence (фрактальная насыщенность): считаем ОДИН раз, отдаём в восстановление
            //     цвета (per-pixel chroma-floor) и в финиш (гейт резкости/цвета) - чтобы не пересчитывать ---
            using var conf = ContourOps.FractalRichness(I, 3, 17);

            // --- восстановление с локальным A(x) и защитой хромы, мягкий roll-off светов ---
            using var recovered = RecoverLocal(I, t, aField, tmin, chroma, conf);
            using var recovered01 = DeHazeCPU.Clip(recovered.Clone());
            // баланс белого по ЛОКАЛЬНОМУ цвету дымки A(x), гейт по плотности (1-t): в плотной дымке
            // (где J≈A - цветной налёт вуали, напр. синева) убираем цветовой каст к серому, в чистых зонах
            // не трогаем настоящий цвет сцены. Это чинит «стена красная → сине-серая».
            LocalAirlightWhiteBalance(recovered01, aField, t, Get("wb", 0.6));
            HazeGuidedDenoise(recovered01, t, denoise);   // глушим усиленный шум именно в плотных зонах

            foreach (var c in ch) c.Dispose();
            foreach (var a in aField) a.Dispose();

            using var finished = DeHazeCPU.Clip(finish(recovered01, work.Mat, t, conf, p).Clone());

            // HQ: многомасштабное (Лапласиан/DoG) усиление контуров - вытаскиваем силуэты объектов на
            // всех масштабах, давим полосу шума. Работает на увеличенной сетке -> ловит и слабую структуру.
            if (hq && contour > 1e-3)
                MultiScaleContourBoost(finished, contour, denoise);

            // мягкое колено светов - на ФИНАЛЕ: финишные CLAHE/вибранс/контраст сами поднимают света,
            // поэтому колено до финиша не спасало от пересвета. Здесь оно реально гасит клип.
            if (knee < 0.999)
                SoftHighlightBgr(finished, knee);
            using var clipped = DeHazeCPU.Clip(finished.Clone());

            if (scale <= 1)
                return clipped.Clone();
            var down = new Mat();
            CvInvoke.Resize(clipped, down, input.Size, 0, 0, Inter.Area);
            return DeHazeCPU.Clip(down);
        }

        /// <summary>Апскейл (Lanczos + лёгкое сглаживание) для HQ-режима: больше сетки под многомасштабный анализ.</summary>
        private static Image<Bgr, byte> Upscale(Image<Bgr, byte> input, int scale)
        {
            var up = new Mat();
            var size = new Size(input.Width * scale, input.Height * scale);
            CvInvoke.Resize(input.Mat, up, size, 0, 0, Inter.Lanczos4);
            CvInvoke.GaussianBlur(up, up, new Size(0, 0), 0.5);
            return up.ToImage<Bgr, byte>();
        }

        /// <summary>
        /// Многомасштабное усиление контуров по яркости L (Lab). Раскладываем L на полосы масштаб-пространства
        /// Гаусса (DoG): самая мелкая полоса - шум (давим её через noiseSuppress), полосы силуэтов (fine/mid/
        /// coarse) усиливаем с ростом contour. Так проявляются контуры и формы объектов на всех масштабах,
        /// включая слабые в умеренной дымке. На месте.
        /// </summary>
        private static void MultiScaleContourBoost(Mat bgr01, double contour, double noiseSuppress)
        {
            using var bgr8 = new Mat();
            bgr01.ConvertTo(bgr8, DepthType.Cv8U, 255.0);
            using var lab = new Mat();
            CvInvoke.CvtColor(bgr8, lab, ColorConversion.Bgr2Lab);
            var chl = lab.Split();

            using var L = new Mat();
            chl[0].ConvertTo(L, DepthType.Cv32F, 1.0 / 255.0);
            var z = new Size(0, 0);
            using var g1 = new Mat(); CvInvoke.GaussianBlur(L, g1, z, 1.2);
            using var g2 = new Mat(); CvInvoke.GaussianBlur(L, g2, z, 3.0);
            using var g3 = new Mat(); CvInvoke.GaussianBlur(L, g3, z, 8.0);
            using var g4 = new Mat(); CvInvoke.GaussianBlur(L, g4, z, 20.0);

            using var bNoise = new Mat(); CvInvoke.Subtract(L, g1, bNoise);    // мельчайшая полоса = шум
            using var bFine = new Mat(); CvInvoke.Subtract(g1, g2, bFine);
            using var bMid = new Mat(); CvInvoke.Subtract(g2, g3, bMid);
            using var bCoarse = new Mat(); CvInvoke.Subtract(g3, g4, bCoarse); // широкий силуэт

            double gNoise = Math.Max(0.0, 1.0 - 1.3 * noiseSuppress);
            double gFine = 1.0 + 0.8 * contour;
            double gMid = 1.0 + 1.8 * contour;
            double gCoarse = 1.0 + 1.2 * contour;

            using var Lp = g4.Clone();   // база - самый гладкий уровень
            CvInvoke.AddWeighted(Lp, 1.0, bNoise, gNoise, 0.0, Lp);
            CvInvoke.AddWeighted(Lp, 1.0, bFine, gFine, 0.0, Lp);
            CvInvoke.AddWeighted(Lp, 1.0, bMid, gMid, 0.0, Lp);
            CvInvoke.AddWeighted(Lp, 1.0, bCoarse, gCoarse, 0.0, Lp);
            DehazeCore.Clamp01(Lp);
            Lp.ConvertTo(chl[0], DepthType.Cv8U, 255.0);

            using (var v = new VectorOfMat(chl)) CvInvoke.Merge(v, lab);
            foreach (var c in chl) c.Dispose();
            using var out8 = new Mat();
            CvInvoke.CvtColor(lab, out8, ColorConversion.Lab2Bgr);
            out8.ConvertTo(bgr01, DepthType.Cv32F, 1.0 / 255.0);
        }

        /// <summary>
        /// «Грубый показ»: масштабно-избирательное восстановление структуры в зонах густой дымки, где
        /// тонкая деталь утоплена в шуме (SNR&lt;1), а сигнал живёт только на грубом масштабе. По ВХОДУ
        /// (не по шумному результату) строим грубую карту контуров: пулинг (усредняет шум вниз) → краевое
        /// сглаживание → вычитание локального airlight → нормировка → апскейл. Затем впечатываем эти контуры
        /// в яркость L результата, ВЗВЕШИВАЯ маской 'мало сигнала' (локальный σ входа мал) - так контуры
        /// деревьев проявляются именно там, где обычное усиление дало бы только шум. Доказано на 09_outdoor:
        /// корреляция такой реконструкции с эталоном в правом-верхнем углу ~0.65.
        /// Возвращает новый BGR float[0,1].
        /// </summary>
        public static Mat CoarseReveal(Mat recovered01, Mat inputBgr8, int pool, double airSub, double gain, double thr, double band)
        {
            if (gain <= 1e-3) return recovered01.Clone();
            using var detail = CoarseRevealDetail(inputBgr8, pool, airSub);   // ~[-0.5,0.5], размер входа
            using var low = LowSignalMask(inputBgr8, thr, band);             // [0,1]
            using var w = new Mat();
            CvInvoke.Multiply(low, new ScalarArray(gain), w);
            DehazeCore.Clamp(w, 0.0, 1.5);
            using var imprint = new Mat();
            CvInvoke.Multiply(detail, w, imprint);
            CvInvoke.Multiply(imprint, new ScalarArray(140.0), imprint);     // амплитуда в единицах L (0..255)

            using var bgr8 = new Mat(); recovered01.ConvertTo(bgr8, DepthType.Cv8U, 255.0);
            using var lab = new Mat(); CvInvoke.CvtColor(bgr8, lab, ColorConversion.Bgr2Lab);
            var ch = lab.Split();
            using (var Lf = new Mat())
            {
                ch[0].ConvertTo(Lf, DepthType.Cv32F);
                CvInvoke.Add(Lf, imprint, Lf);
                DehazeCore.Clamp(Lf, 0.0, 255.0);
                Lf.ConvertTo(ch[0], DepthType.Cv8U);
            }
            using (var v = new VectorOfMat(ch)) CvInvoke.Merge(v, lab);
            foreach (var c in ch) c.Dispose();
            using var outBgr = new Mat(); CvInvoke.CvtColor(lab, outBgr, ColorConversion.Lab2Bgr);
            var res = new Mat(); outBgr.ConvertTo(res, DepthType.Cv32F, 1.0 / 255.0);
            return res;
        }

        /// <summary>Грубая карта контуров из ВХОДА: пулинг (усреднение шума) → вычитание локального airlight → нормировка. 1 канал ~[-0.5,0.5], размер входа.</summary>
        private static Mat CoarseRevealDetail(Mat inputBgr8, int pool, double airSub)
        {
            pool = Math.Max(2, pool);
            using var gray = new Mat(); CvInvoke.CvtColor(inputBgr8, gray, ColorConversion.Bgr2Gray);
            var szSmall = new Size(Math.Max(8, inputBgr8.Cols / pool), Math.Max(8, inputBgr8.Rows / pool));
            using var small8 = new Mat(); CvInvoke.Resize(gray, small8, szSmall, 0, 0, Inter.Area);
            using var smB = new Mat(); CvInvoke.BilateralFilter(small8, smB, 7, 30.0, 7.0, BorderType.Reflect101);
            using var small = new Mat(); smB.ConvertTo(small, DepthType.Cv32F, 1.0 / 255.0);
            using var air = new Mat(); CvInvoke.GaussianBlur(small, air, new Size(0, 0), airSub);
            var detail = new Mat(); CvInvoke.Subtract(small, air, detail);
            CvInvoke.GaussianBlur(detail, detail, new Size(0, 0), 1.2);       // убрать остаточный мелкий шум
            MCvScalar mean = default, std = default; CvInvoke.MeanStdDev(detail, ref mean, ref std);
            double s = std.V0 > 1e-6 ? 0.5 / (2.0 * std.V0) : 0.0;            // ±2σ -> ±0.5
            detail.ConvertTo(detail, DepthType.Cv32F, s, -mean.V0 * s);
            DehazeCore.Clamp(detail, -0.5, 0.5);
            using (detail) { var up = new Mat(); CvInvoke.Resize(detail, up, inputBgr8.Size, 0, 0, Inter.Cubic); return up; }
        }

        /// <summary>Маска 'мало сигнала' ∈[0,1]: велика где локальный σ ВХОДА мал (густая дымка). low=clamp((thr-σ)/band), σ в долях [0,1].</summary>
        private static Mat LowSignalMask(Mat inputBgr8, double thr, double band)
        {
            using var gray = new Mat(); CvInvoke.CvtColor(inputBgr8, gray, ColorConversion.Bgr2Gray);
            using var g = new Mat(); gray.ConvertTo(g, DepthType.Cv32F, 1.0 / 255.0);
            var ks = new Size(9, 9); var anc = new Point(-1, -1);
            using var mean = new Mat(); CvInvoke.Blur(g, mean, ks, anc);
            using var mean2 = new Mat();
            using (var sq = new Mat()) { CvInvoke.Multiply(g, g, sq); CvInvoke.Blur(sq, mean2, ks, anc); }
            using var var0 = new Mat();
            using (var m2 = new Mat()) { CvInvoke.Multiply(mean, mean, m2); CvInvoke.Subtract(mean2, m2, var0); }
            DehazeCore.Clamp01(var0);
            using var sd = new Mat(); CvInvoke.Sqrt(var0, sd);
            var low = new Mat();
            sd.ConvertTo(low, DepthType.Cv32F, -1.0 / band, thr / band);      // (thr - σ)/band
            DehazeCore.Clamp01(low);
            return low;
        }

        /// <summary>
        /// Поле атмосферного света A_c(x) = blur(q·I_c)/blur(q), доверие q = D²·(1-S)^1.5·flat
        /// (велико в ярких, малонасыщенных, гладких зонах = небо/вуаль; мало на структуре). Большой
        /// <paramref name="radius"/> -> почти глобальный A; малый -> локально-адаптивный (под цвет дымки
        /// в каждой зоне). Возвращает 3 канала [B,G,R]; вызывающий освобождает их.
        /// </summary>
        public static Mat[] AirlightField(Mat i01, int patch, int radius)
        {
            var anc = new Point(-1, -1);
            using var D = DehazeCore.DarkChannel(i01, patch);
            using var S = DehazeCore.Saturation(i01);
            using var flat = DehazeCore.Flatness(i01, 2.0);
            using var q = new Mat();
            CvInvoke.Multiply(D, D, q);
            using (var oneMinusS = new Mat())
            {
                S.ConvertTo(oneMinusS, DepthType.Cv32F, -1.0, 1.0);
                using var sm = new Mat(); CvInvoke.Pow(oneMinusS, 1.5, sm);
                CvInvoke.Multiply(q, sm, q);
            }
            CvInvoke.Multiply(q, flat, q);
            CvInvoke.Add(q, new ScalarArray(1e-4), q);

            var ks = new Size(2 * radius + 1, 2 * radius + 1);
            using var denA = new Mat();
            CvInvoke.Blur(q, denA, ks, anc);
            CvInvoke.Add(denA, new ScalarArray(1e-6), denA);

            var ch = i01.Split();
            var aField = new Mat[3];
            for (int c = 0; c < 3; c++)
            {
                using var qi = new Mat(); CvInvoke.Multiply(q, ch[c], qi);
                aField[c] = new Mat(); CvInvoke.Blur(qi, aField[c], ks, anc);
                CvInvoke.Divide(aField[c], denA, aField[c]);
                using var floor = new Mat(aField[c].Size, DepthType.Cv32F, 1); floor.SetTo(new MCvScalar(1e-2));
                CvInvoke.Max(aField[c], floor, aField[c]);
            }
            foreach (var c in ch) c.Dispose();
            return aField;
        }

        /// <summary>
        /// Восстановление с пространственным полем A_c(x). Как <see cref="DehazeCore.Recover"/>, но A -
        /// карта: (I-A) раскладываем на ахроматическую часть d=mean_c(I_c-A_c) и хрому δ_c, делим яркость
        /// на max(t,t_min), а хрому на max(t,chromaFloor) (слабее) - цвет не выжигается при малом t.
        /// Возвращает НЕзажатый Mat (клип/roll-off делает вызывающий).
        /// </summary>
        public static Mat RecoverLocal(Mat i01, Mat tSingle, Mat[] aField, double tmin, double chromaFloor, Mat? conf = null)
        {
            using var tLum = new Mat();
            using (var tm = new Mat(tSingle.Size, DepthType.Cv32F, 1)) { tm.SetTo(new MCvScalar(tmin)); CvInvoke.Max(tSingle, tm, tLum); }
            using var tChroma = BuildChromaTransmission(tSingle, tmin, chromaFloor, conf);

            var ch = i01.Split();
            var d = new Mat[3];
            for (int c = 0; c < 3; c++) { d[c] = new Mat(); CvInvoke.Subtract(ch[c], aField[c], d[c]); ch[c].Dispose(); }

            using var dbar = new Mat();
            CvInvoke.Add(d[0], d[1], dbar); CvInvoke.Add(dbar, d[2], dbar);
            dbar.ConvertTo(dbar, DepthType.Cv32F, 1.0 / 3.0);
            using var lumPart = new Mat(); CvInvoke.Divide(dbar, tLum, lumPart);

            using var outv = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                using var delta = new Mat(); CvInvoke.Subtract(d[c], dbar, delta);
                var jc = new Mat(); CvInvoke.Divide(delta, tChroma, jc);
                CvInvoke.Add(jc, lumPart, jc);
                CvInvoke.Add(jc, aField[c], jc);
                outv.Push(jc); jc.Dispose();
                d[c].Dispose();
            }
            var J = new Mat();
            CvInvoke.Merge(outv, J);
            return J;
        }

        /// <summary>
        /// Хроматически-якорное восстановление. Локальное A(x) задаёт только среднюю яркость airlight,
        /// а его нулевая по сумме хроматическая часть смешивается с устойчивой глобальной хроматичностью
        /// A_g. Это не позволяет широкому цветному объекту попасть в локальное A(x) и затем быть ошибочно
        /// нейтрализованным как «цвет тумана».
        ///
        /// Для centered RGB C(X)=X-mean_c(X) точная модель даёт
        /// C(J)=[C(I)-(1-t)C(A)]/t. В плотной дымке C(I) предварительно смешивается с крупномасштабной
        /// версией: широкая цветовая область сохраняется, а поканальный шум усредняется.
        /// Возвращает незажатый BGR float в том же (желательно линейном) пространстве, что и вход.
        /// </summary>
        public static Mat RecoverChromaticAnchor(Mat i01, Mat tSingle, Mat[] aField, MCvScalar globalAirlight,
            double tmin, double chromaFloor, double globalAnchor, double coarseSigma, double coarseMix, Mat? conf = null)
        {
            globalAnchor = Math.Clamp(globalAnchor, 0.0, 1.0);
            coarseSigma = Math.Max(0.0, coarseSigma);
            coarseMix = Math.Clamp(coarseMix, 0.0, 1.0);

            using var tLum = new Mat();
            using (var tm = new Mat(tSingle.Size, DepthType.Cv32F, 1)) { tm.SetTo(new MCvScalar(tmin)); CvInvoke.Max(tSingle, tm, tLum); }
            using var tChroma = BuildChromaTransmission(tSingle, tmin, chromaFloor, conf);

            var input = i01.Split();
            using var inputMean = new Mat();
            CvInvoke.Add(input[0], input[1], inputMean); CvInvoke.Add(inputMean, input[2], inputMean);
            inputMean.ConvertTo(inputMean, DepthType.Cv32F, 1.0 / 3.0);

            using var airMean = new Mat();
            CvInvoke.Add(aField[0], aField[1], airMean); CvInvoke.Add(airMean, aField[2], airMean);
            airMean.ConvertTo(airMean, DepthType.Cv32F, 1.0 / 3.0);

            using var meanResidual = new Mat(); CvInvoke.Subtract(inputMean, airMean, meanResidual);
            using var recoveredMean = new Mat(); CvInvoke.Divide(meanResidual, tLum, recoveredMean); CvInvoke.Add(recoveredMean, airMean, recoveredMean);

            Mat[]? coarse = null;
            Mat? coarseMean = null;
            if (coarseSigma > 1e-3 && coarseMix > 1e-3)
            {
                coarse = new Mat[3];
                for (int c = 0; c < 3; c++)
                {
                    coarse[c] = new Mat();
                    CvInvoke.GaussianBlur(input[c], coarse[c], new Size(0, 0), coarseSigma, coarseSigma, BorderType.Reflect101);
                }
                coarseMean = new Mat();
                CvInvoke.Add(coarse[0], coarse[1], coarseMean); CvInvoke.Add(coarseMean, coarse[2], coarseMean);
                coarseMean.ConvertTo(coarseMean, DepthType.Cv32F, 1.0 / 3.0);
            }

            using var coarseWeight = new Mat();
            tSingle.ConvertTo(coarseWeight, DepthType.Cv32F, -1.0, 1.0);
            DehazeCore.Clamp01(coarseWeight);
            CvInvoke.Multiply(coarseWeight, coarseWeight, coarseWeight);
            CvInvoke.Multiply(coarseWeight, new ScalarArray(coarseMix), coarseWeight);

            double globalMean = (globalAirlight.V0 + globalAirlight.V1 + globalAirlight.V2) / 3.0;
            double[] globalCentered =
            {
                globalAirlight.V0 - globalMean,
                globalAirlight.V1 - globalMean,
                globalAirlight.V2 - globalMean,
            };
            using var oneMinusT = new Mat();
            tSingle.ConvertTo(oneMinusT, DepthType.Cv32F, -1.0, 1.0);
            DehazeCore.Clamp01(oneMinusT);

            using var output = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                using var observedChroma = new Mat(); CvInvoke.Subtract(input[c], inputMean, observedChroma);
                if (coarse != null && coarseMean != null)
                {
                    using var coarseChroma = new Mat(); CvInvoke.Subtract(coarse[c], coarseMean, coarseChroma);
                    using var diff = new Mat(); CvInvoke.Subtract(coarseChroma, observedChroma, diff);
                    CvInvoke.Multiply(diff, coarseWeight, diff);
                    CvInvoke.Add(observedChroma, diff, observedChroma);
                }

                using var localCentered = new Mat(); CvInvoke.Subtract(aField[c], airMean, localCentered);
                using var anchoredAirChroma = new Mat();
                localCentered.ConvertTo(anchoredAirChroma, DepthType.Cv32F, 1.0 - globalAnchor,
                    globalAnchor * globalCentered[c]);
                using var hazeChroma = new Mat(); CvInvoke.Multiply(anchoredAirChroma, oneMinusT, hazeChroma);
                using var cleanChroma = new Mat(); CvInvoke.Subtract(observedChroma, hazeChroma, cleanChroma);
                CvInvoke.Divide(cleanChroma, tChroma, cleanChroma);

                var channel = new Mat(); CvInvoke.Add(recoveredMean, cleanChroma, channel);
                output.Push(channel); channel.Dispose();
            }

            foreach (var channel in input) channel.Dispose();
            if (coarse != null) foreach (var channel in coarse) channel.Dispose();
            coarseMean?.Dispose();
            var result = new Mat(); CvInvoke.Merge(output, result); return result;
        }

        private static Mat BuildChromaTransmission(Mat tSingle, double tmin, double chromaFloor, Mat? conf)
        {
            double cf = Math.Max(tmin, chromaFloor);
            var tChroma = new Mat();
            if (conf != null && chromaFloor > tmin)
            {
                // Per-pixel floor спускается к t_min там, где плотная дымка И есть надёжная структура.
                using var confR = DehazeCore.ResizeTo(conf, tSingle.Size);
                using var density = new Mat();
                tSingle.ConvertTo(density, DepthType.Cv32F, -1.0, 1.0);
                DehazeCore.Clamp01(density);
                CvInvoke.Multiply(density, confR, density);
                using var cfMap = new Mat();
                density.ConvertTo(cfMap, DepthType.Cv32F, -(chromaFloor - tmin), chromaFloor);
                double hardFloor = Math.Max(0.10, tmin);
                using (var lo = new Mat(cfMap.Size, DepthType.Cv32F, 1)) { lo.SetTo(new MCvScalar(hardFloor)); CvInvoke.Max(cfMap, lo, cfMap); }
                CvInvoke.Max(tSingle, cfMap, tChroma);
            }
            else if (cf > tmin)
            {
                using var floor = new Mat(tSingle.Size, DepthType.Cv32F, 1);
                floor.SetTo(new MCvScalar(cf));
                CvInvoke.Max(tSingle, floor, tChroma);
            }
            else
            {
                using var floor = new Mat(tSingle.Size, DepthType.Cv32F, 1);
                floor.SetTo(new MCvScalar(tmin));
                CvInvoke.Max(tSingle, floor, tChroma);
            }
            return tChroma;
        }

        /// <summary>
        /// Шумоподавление, привязанное к плотности дымки: вес w = strength·(1-t). Там, где пропускание t
        /// мало (густая дымка), деление на малый t раздуло шум - его и глушим краевым (билатеральным)
        /// фильтром; где дымка тонкая (t≈1), детали остаются нетронутыми. На месте.
        /// </summary>
        private static void HazeGuidedDenoise(Mat bgr01, Mat t, double strength)
        {
            if (strength <= 1e-3) return;

            using var bgr8 = new Mat();
            bgr01.ConvertTo(bgr8, DepthType.Cv8U, 255.0);
            using var sm8 = new Mat();
            CvInvoke.BilateralFilter(bgr8, sm8, 7, 60.0, 6.0, BorderType.Reflect101);
            using var smooth = new Mat();
            sm8.ConvertTo(smooth, DepthType.Cv32F, 1.0 / 255.0);

            // вес strength·(1-t)²: кадр задымлён целиком, линейный (1-t) сгладил бы всю структуру и уронил
            // SSIM; квадрат оставляет денойз только в самых плотных зонах (t→0).
            using var w = new Mat();
            t.ConvertTo(w, DepthType.Cv32F, -1.0, 1.0);             // 1 - t
            DehazeCore.Clamp01(w);
            CvInvoke.Multiply(w, w, w);                             // (1 - t)²
            CvInvoke.Multiply(w, new ScalarArray(strength), w);
            DehazeCore.Clamp01(w);
            using var w3 = new Mat();
            using (var v = new VectorOfMat()) { v.Push(w); v.Push(w); v.Push(w); CvInvoke.Merge(v, w3); }

            using var diff = new Mat();
            CvInvoke.Subtract(smooth, bgr01, diff);   // (smooth - bgr) · w  ->  bgr + w·(smooth-bgr)
            CvInvoke.Multiply(diff, w3, diff);
            CvInvoke.Add(bgr01, diff, bgr01);
        }

        /// <summary>
        /// Баланс белого по ЛОКАЛЬНОМУ полю дымки A(x): усиление канала gain_c(x)=Abar(x)/A_c(x), где
        /// Abar=mean_c A. Убирает ЦВЕТОВОЙ каст вуали (её оттенок, напр. синеву) к серому — но только по
        /// весу плотности дымки w=strength·(1-t): в плотной вуали (где J≈A) каст снимается, в чистых зонах
        /// (t→1) настоящий цвет сцены не трогается. gain зажат [0.55,1.8]. На месте.
        /// </summary>
        public static void LocalAirlightWhiteBalance(Mat bgr01, Mat[] aField, Mat t, double strength)
        {
            if (strength <= 1e-3) return;
            using var abar = new Mat();
            CvInvoke.Add(aField[0], aField[1], abar); CvInvoke.Add(abar, aField[2], abar);
            abar.ConvertTo(abar, DepthType.Cv32F, 1.0 / 3.0);

            // вес = strength · насыщенность локального airlight: правим ТОЛЬКО там, где вуаль ЦВЕТНАЯ
            // (напр. синева), а не по (1-t) — иначе защищённая «как небо» зона (высокий t) не корректируется.
            using var amax = new Mat(); CvInvoke.Max(aField[0], aField[1], amax); CvInvoke.Max(amax, aField[2], amax);
            using var amin = new Mat(); CvInvoke.Min(aField[0], aField[1], amin); CvInvoke.Min(amin, aField[2], amin);
            using var asat = new Mat();
            using (var den = new Mat()) { CvInvoke.Add(amax, new ScalarArray(1e-3), den); CvInvoke.Subtract(amax, amin, asat); CvInvoke.Divide(asat, den, asat); }   // (max-min)/max
            using var w = new Mat();
            asat.ConvertTo(w, DepthType.Cv32F, strength / 0.12, 0.0);   // насыщенность вуали ~0.12 → полный вес
            DehazeCore.Clamp(w, 0.0, strength);

            var ch = bgr01.Split();
            using var outv = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                using var gain = new Mat();
                CvInvoke.Divide(abar, aField[c], gain);        // Abar / A_c
                DehazeCore.Clamp(gain, 0.55, 1.8);
                // factor = 1 + w·(gain-1)
                using var factor = new Mat();
                CvInvoke.Subtract(gain, new ScalarArray(1.0), factor);
                CvInvoke.Multiply(factor, w, factor);
                CvInvoke.Add(factor, new ScalarArray(1.0), factor);
                CvInvoke.Multiply(ch[c], factor, ch[c]);
                DehazeCore.Clamp01(ch[c]);                      // одноканально (Clamp строит 1-канальный порог)
                outv.Push(ch[c]);
            }
            CvInvoke.Merge(outv, bgr01);
            foreach (var c in ch) c.Dispose();
        }

        /// <summary>Мягкое колено светов по каждому каналу BGR (на месте): вместо жёсткого клипа на 1.</summary>
        private static void SoftHighlightBgr(Mat bgr01, double knee)
        {
            var ch = bgr01.Split();
            using var outv = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                DehazeCore.SoftHighlight(ch[c], knee);
                outv.Push(ch[c]);
            }
            CvInvoke.Merge(outv, bgr01);
            foreach (var c in ch) c.Dispose();
        }
    }
}
