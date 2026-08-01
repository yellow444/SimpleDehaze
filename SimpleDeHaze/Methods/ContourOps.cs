using System.Collections.Generic;
using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>Операции над контурами/деталями: многомасштабная Лапласиан-пирамида (OpenCV pyrDown/pyrUp).</summary>
    internal static class ContourOps
    {
        /// <summary>
        /// Усиление контуров по Лапласиан-пирамиде: яркость L раскладывается на полосы разного масштаба
        /// (L_i = G_i - up(G_{i+1})). Каждую полосу усиливаем своим коэффициентом - мелкие детали держим
        /// слабо (это шум), средние и крупные контуры/силуэты усиливаем сильнее, затем пирамида собирается
        /// обратно. Это и есть «держим контуры на меньшем и большем масштабе». Цвет (a,b) не трогаем.
        /// </summary>
        public static Mat MultiScaleLaplacian(Mat bgr01, int levels, double gFine, double gMid, double gCoarse,
            Mat? structMap = null, double clipC = 0.0)
        {
            levels = System.Math.Clamp(levels, 2, 7);

            using var bgr8 = new Mat();
            bgr01.ConvertTo(bgr8, DepthType.Cv8U, 255.0);
            using var lab = new Mat();
            CvInvoke.CvtColor(bgr8, lab, ColorConversion.Bgr2Lab);
            var chl = lab.Split();
            using var L = new Mat();
            chl[0].ConvertTo(L, DepthType.Cv32F, 1.0 / 255.0);

            // гауссова пирамида
            var gauss = new List<Mat> { L.Clone() };
            for (int i = 1; i < levels; i++)
            {
                var dn = new Mat();
                CvInvoke.PyrDown(gauss[i - 1], dn);
                gauss.Add(dn);
            }

            // лапласовы полосы L_i = G_i - up(G_{i+1})
            var lap = new List<Mat>();
            for (int i = 0; i < levels - 1; i++)
            {
                var up = Up(gauss[i + 1], gauss[i].Size);
                var l = new Mat();
                CvInvoke.Subtract(gauss[i], up, l);
                up.Dispose();
                lap.Add(l);
            }

            double Gain(int i)
            {
                if (i == 0) return gFine;                                   // мельчайшая полоса = шум
                double f = (double)(i - 1) / System.Math.Max(1, levels - 2);
                return gMid + f * (gCoarse - gMid);                         // от средних к крупным
            }

            // собрать обратно с усилением полос
            var cur = gauss[levels - 1].Clone();                            // низкочастотная база
            for (int i = levels - 2; i >= 0; i--)
            {
                using var up = Up(cur, gauss[i].Size);
                var nc = new Mat();
                if (structMap == null)
                {
                    CvInvoke.AddWeighted(up, 1.0, lap[i], Gain(i), 0.0, nc);    // up + gain·L_i (без гейта)
                }
                else
                {
                    // effGain(x) = 1 + (Gain(i)-1)·struct(x): на плоском тумане (struct≈0) полосу почти не
                    // усиливаем (нет хруста), на реальной текстуре — полное усиление. Клип полосы = анти-шум.
                    using var sm = new Mat();
                    if (!structMap.Size.Equals(lap[i].Size)) CvInvoke.Resize(structMap, sm, lap[i].Size); else structMap.CopyTo(sm);
                    sm.ConvertTo(sm, DepthType.Cv32F, Gain(i) - 1.0, 1.0);      // 1 + (g-1)·struct
                    using var contrib = new Mat();
                    CvInvoke.Multiply(lap[i], sm, contrib);                     // L_i · effGain
                    if (clipC > 1e-6) DehazeCore.Clamp(contrib, -clipC, clipC);
                    CvInvoke.Add(up, contrib, nc);
                }
                cur.Dispose();
                cur = nc;
            }
            DehazeCore.Clamp01(cur);
            cur.ConvertTo(chl[0], DepthType.Cv8U, 255.0);
            cur.Dispose();

            foreach (var g in gauss) g.Dispose();
            foreach (var l in lap) l.Dispose();

            using (var v = new VectorOfMat(chl)) CvInvoke.Merge(v, lab);
            foreach (var c in chl) c.Dispose();
            using var out8 = new Mat();
            CvInvoke.CvtColor(lab, out8, ColorConversion.Lab2Bgr);
            var res = new Mat();
            out8.ConvertTo(res, DepthType.Cv32F, 1.0 / 255.0);
            return res;
        }

        /// <summary>
        /// Экспериментальная транс-масштабная Лапласиан-реконструкция. Усиление каждой полосы зависит от
        /// ЛОКАЛЬНОГО ПРОПУСКАНИЯ t(x) И масштаба: мелкие детали усиливаем только там, где дымка тонкая
        /// (t велико → деталь реальна), и гасим в плотной дымке (t мало → там это шум); крупные контуры/
        /// силуэты усиливаем везде (на грубом масштабе t-зависимость исчезает). Это эвристический
        /// smoothstep-gate; измеренная модель шума для него не заявляется. Цвет (a,b) не трогаем.
        ///
        /// gate_ℓ(x) = smoothstep(t_ℓ(x); tLo,tHi)·(1-sf) + sf,  sf = ℓ/(levels-2)  (0 у мелких, 1 у крупных)
        /// усиление полосы = baseGain(ℓ)·gate_ℓ(x).
        /// </summary>
        public static Mat TransmissionScaleLaplacian(Mat bgr01, Mat tMap, int levels,
            double gFine, double gMid, double gCoarse, double tLo, double tHi,
            Mat? richMap = null, double bandC = 0.0)
        {
            levels = System.Math.Clamp(levels, 2, 7);
            double span = System.Math.Max(1e-3, tHi - tLo);
            const double coarseWeight = 0.8;

            using var bgr8 = new Mat();
            bgr01.ConvertTo(bgr8, DepthType.Cv8U, 255.0);
            using var lab = new Mat();
            CvInvoke.CvtColor(bgr8, lab, ColorConversion.Bgr2Lab);
            var chl = lab.Split();
            using var L = new Mat();
            chl[0].ConvertTo(L, DepthType.Cv32F, 1.0 / 255.0);

            // пирамиды: яркость L, полосы Лапласа, и пропускание t (downsample синхронно)
            var gauss = new List<Mat> { L.Clone() };
            var tp = new List<Mat> { tMap.Clone() };
            for (int i = 1; i < levels; i++)
            {
                var dn = new Mat(); CvInvoke.PyrDown(gauss[i - 1], dn); gauss.Add(dn);
                var td = new Mat(); CvInvoke.PyrDown(tp[i - 1], td); tp.Add(td);
            }
            var lap = new List<Mat>();
            for (int i = 0; i < levels - 1; i++)
            {
                var up = Up(gauss[i + 1], gauss[i].Size);
                var l = new Mat(); CvInvoke.Subtract(gauss[i], up, l); up.Dispose(); lap.Add(l);
            }

            double BaseGain(int i)
            {
                if (i == 0) return gFine;
                double f = (double)(i - 1) / System.Math.Max(1, levels - 2);
                return gMid + f * (gCoarse - gMid);
            }

            var cur = gauss[levels - 1].Clone();
            for (int i = levels - 2; i >= 0; i--)
            {
                using var up = Up(cur, gauss[i].Size);
                double bg = BaseGain(i);
                double sf = (double)i / System.Math.Max(1, levels - 2);   // 0 мелкие .. 1 крупные

                // gate = smoothstep(t)·(1-sf) + sf
                using var ss = new Mat();
                tp[i].ConvertTo(ss, DepthType.Cv32F, 1.0 / span, -tLo / span);   // (t - tLo)/span
                DehazeCore.Clamp01(ss);
                using (var s2 = new Mat()) { CvInvoke.Multiply(ss, ss, s2); using var s3 = new Mat(); CvInvoke.Multiply(s2, ss, s3); CvInvoke.AddWeighted(s2, 3.0, s3, -2.0, 0.0, ss); }  // 3s²-2s³
                using var gate = new Mat();
                ss.ConvertTo(gate, DepthType.Cv32F, 1.0 - sf, sf);
                if (!gate.Size.Equals(lap[i].Size)) CvInvoke.Resize(gate, gate, lap[i].Size);

                // richness-floor: на КРУПНЫХ полосах (sf→1) даём проявиться реальной структуре даже при
                // малом t (чистый t-гейт её бы задавил как «шум») — там, где richMap высок. Мелкие полосы
                // (sf→0) не поднимаем: при малом t это действительно шум.
                if (richMap != null)
                {
                    using var rr = new Mat();
                    if (!richMap.Size.Equals(lap[i].Size)) CvInvoke.Resize(richMap, rr, lap[i].Size); else richMap.CopyTo(rr);
                    rr.ConvertTo(rr, DepthType.Cv32F, coarseWeight * sf, 0.0);   // rich·coarseWeight·sf
                    CvInvoke.Max(gate, rr, gate);
                }

                using var contrib = new Mat();
                CvInvoke.Multiply(lap[i], gate, contrib);           // L_ℓ · gate
                if (bandC > 1e-6) DehazeCore.Clamp(contrib, -bandC, bandC);   // ограничить амплитуду полосы (анти-шум)
                using var scaled = new Mat();
                contrib.ConvertTo(scaled, DepthType.Cv32F, bg);     // · baseGain
                var nc = new Mat();
                CvInvoke.Add(up, scaled, nc);
                cur.Dispose(); cur = nc;
            }
            DehazeCore.Clamp01(cur);
            cur.ConvertTo(chl[0], DepthType.Cv8U, 255.0);
            cur.Dispose();

            foreach (var g in gauss) g.Dispose();
            foreach (var t in tp) t.Dispose();
            foreach (var l in lap) l.Dispose();

            using (var v = new VectorOfMat(chl)) CvInvoke.Merge(v, lab);
            foreach (var c in chl) c.Dispose();
            using var out8 = new Mat();
            CvInvoke.CvtColor(lab, out8, ColorConversion.Lab2Bgr);
            var res = new Mat();
            out8.ConvertTo(res, DepthType.Cv32F, 1.0 / 255.0);
            return res;
        }

        /// <summary>
        /// HSV-V control variant of the transmission-aware pyramid. H and S are copied unchanged;
        /// only V is decomposed. This deliberately avoids treating circular Hue as a scalar band.
        /// </summary>
        public static Mat TransmissionScaleHsvValue(Mat bgr01, Mat tMap, int levels,
            double gFine, double gMid, double gCoarse, double tLo, double tHi,
            Mat? richMap = null, double bandC = 0.0)
        {
            using var hsv = new Mat();
            CvInvoke.CvtColor(bgr01, hsv, ColorConversion.Bgr2Hsv);
            var channels = hsv.Split();
            using var enhancedValue = TransmissionPyramidScalar(channels[2], tMap, levels,
                gFine, gMid, gCoarse, tLo, tHi, richMap, bandC);
            enhancedValue.CopyTo(channels[2]);

            using (var vector = new VectorOfMat(channels)) CvInvoke.Merge(vector, hsv);
            foreach (var channel in channels) channel.Dispose();
            var result = new Mat();
            CvInvoke.CvtColor(hsv, result, ColorConversion.Hsv2Bgr);
            DehazeCore.Clamp01(result);
            return result;
        }

        /// <summary>
        /// Full-resolution edge-aware residual stack. Unlike a Gaussian/Laplacian pyramid it does
        /// not downsample: each band is the difference between two Domain-Transform smoothings.
        /// The same transmission/scale gate is applied to each band. This is an experimental basis,
        /// not a claim that edge-aware or residual decompositions themselves are new.
        /// </summary>
        public static Mat TransmissionEdgeAwareBands(Mat bgr01, Mat tMap, int bands,
            double gFine, double gMid, double gCoarse, double tLo, double tHi,
            double spatialScale, double rangeScale, bool hsvValue,
            Mat? richMap = null, double bandC = 0.0)
        {
            bands = Math.Clamp(bands, 2, 4);
            spatialScale = Math.Clamp(spatialScale, 2, 96);
            rangeScale = Math.Clamp(rangeScale, 0.01, 1.0);

            if (hsvValue)
            {
                using var hsv = new Mat();
                CvInvoke.CvtColor(bgr01, hsv, ColorConversion.Bgr2Hsv);
                var channels = hsv.Split();
                using var enhanced = TransmissionEdgeBandScalar(channels[2], bgr01, tMap, bands,
                    gFine, gMid, gCoarse, tLo, tHi, spatialScale, rangeScale, richMap, bandC);
                enhanced.CopyTo(channels[2]);
                using (var vector = new VectorOfMat(channels)) CvInvoke.Merge(vector, hsv);
                foreach (var channel in channels) channel.Dispose();
                var result = new Mat();
                CvInvoke.CvtColor(hsv, result, ColorConversion.Hsv2Bgr);
                DehazeCore.Clamp01(result);
                return result;
            }

            // RGB companion: build bands in linear luminance and add the luminance delta equally
            // to linear RGB. This preserves opponent-channel differences better than three
            // independent RGB pyramids and keeps the comparison to HSV-V one-dimensional.
            using var linear = ColorSpace.ToLinear(bgr01);
            using var luminance = ColorSpace.Luminance(linear);
            using var enhancedLuminance = TransmissionEdgeBandScalar(luminance, bgr01, tMap, bands,
                gFine, gMid, gCoarse, tLo, tHi, spatialScale, rangeScale, richMap, bandC);
            using var delta = new Mat();
            CvInvoke.Subtract(enhancedLuminance, luminance, delta);
            var rgb = linear.Split();
            foreach (var channel in rgb)
            {
                CvInvoke.Add(channel, delta, channel);
                DehazeCore.Clamp01(channel);
            }
            using var adjustedLinear = new Mat();
            using (var vector = new VectorOfMat(rgb)) CvInvoke.Merge(vector, adjustedLinear);
            foreach (var channel in rgb) channel.Dispose();
            var srgb = ColorSpace.ToSrgb(adjustedLinear);
            DehazeCore.Clamp01(srgb);
            return srgb;
        }

        internal static double TransmissionBandGate(double transmission, double tLo, double tHi,
            double scaleFraction, double richness = 0.0)
        {
            double span = Math.Max(1e-12, tHi - tLo);
            double s = Math.Clamp((transmission - tLo) / span, 0, 1);
            double smooth = s * s * (3 - 2 * s);
            double sf = Math.Clamp(scaleFraction, 0, 1);
            double gate = smooth * (1 - sf) + sf;
            return Math.Max(gate, Math.Clamp(richness, 0, 1) * 0.8 * sf);
        }

        private static Mat TransmissionPyramidScalar(Mat signal, Mat tMap, int levels,
            double gFine, double gMid, double gCoarse, double tLo, double tHi,
            Mat? richMap, double bandC)
        {
            levels = Math.Clamp(levels, 2, 7);
            double span = Math.Max(1e-3, tHi - tLo);
            const double coarseWeight = 0.8;
            var gauss = new List<Mat> { signal.Clone() };
            var tp = new List<Mat> { tMap.Clone() };
            for (int i = 1; i < levels; i++)
            {
                var down = new Mat(); CvInvoke.PyrDown(gauss[i - 1], down); gauss.Add(down);
                var transmission = new Mat(); CvInvoke.PyrDown(tp[i - 1], transmission); tp.Add(transmission);
            }

            var residuals = new List<Mat>();
            for (int i = 0; i < levels - 1; i++)
            {
                using var up = Up(gauss[i + 1], gauss[i].Size);
                var residual = new Mat(); CvInvoke.Subtract(gauss[i], up, residual); residuals.Add(residual);
            }

            double Gain(int i)
            {
                if (i == 0) return gFine;
                double f = (double)(i - 1) / Math.Max(1, levels - 2);
                return gMid + f * (gCoarse - gMid);
            }

            var current = gauss[levels - 1].Clone();
            for (int i = levels - 2; i >= 0; i--)
            {
                using var up = Up(current, gauss[i].Size);
                double sf = (double)i / Math.Max(1, levels - 2);
                using var normalized = new Mat();
                tp[i].ConvertTo(normalized, DepthType.Cv32F, 1.0 / span, -tLo / span);
                DehazeCore.Clamp01(normalized);
                using (var square = new Mat())
                {
                    CvInvoke.Multiply(normalized, normalized, square);
                    using var cube = new Mat(); CvInvoke.Multiply(square, normalized, cube);
                    CvInvoke.AddWeighted(square, 3.0, cube, -2.0, 0.0, normalized);
                }
                using var gate = new Mat();
                normalized.ConvertTo(gate, DepthType.Cv32F, 1.0 - sf, sf);
                if (richMap != null)
                {
                    using var richness = new Mat();
                    if (richMap.Size != gate.Size) CvInvoke.Resize(richMap, richness, gate.Size); else richMap.CopyTo(richness);
                    richness.ConvertTo(richness, DepthType.Cv32F, coarseWeight * sf);
                    CvInvoke.Max(gate, richness, gate);
                }
                using var contribution = new Mat(); CvInvoke.Multiply(residuals[i], gate, contribution);
                if (bandC > 1e-6) DehazeCore.Clamp(contribution, -bandC, bandC);
                using var scaled = new Mat(); contribution.ConvertTo(scaled, DepthType.Cv32F, Gain(i));
                var next = new Mat(); CvInvoke.Add(up, scaled, next);
                current.Dispose(); current = next;
            }

            DehazeCore.Clamp01(current);
            foreach (var item in gauss) item.Dispose();
            foreach (var item in tp) item.Dispose();
            foreach (var item in residuals) item.Dispose();
            return current;
        }

        private static Mat TransmissionEdgeBandScalar(Mat signal, Mat guide, Mat tMap, int bands,
            double gFine, double gMid, double gCoarse, double tLo, double tHi,
            double spatialScale, double rangeScale, Mat? richMap, double bandC)
        {
            var smooth = new List<Mat>();
            for (int i = 0; i < bands; i++)
            {
                var filtered = new Mat();
                XImgprocInvoke.DtFilter(guide, signal, filtered,
                    spatialScale * Math.Pow(2, i), rangeScale, DtFilterType.NC, 2);
                smooth.Add(filtered);
            }

            var residuals = new List<Mat>();
            Mat previous = signal;
            foreach (var filtered in smooth)
            {
                var residual = new Mat(); CvInvoke.Subtract(previous, filtered, residual); residuals.Add(residual);
                previous = filtered;
            }

            double Gain(int i)
            {
                if (i == 0) return gFine;
                double f = (double)(i - 1) / Math.Max(1, bands - 2);
                return gMid + f * (gCoarse - gMid);
            }

            var current = smooth[^1].Clone();
            double span = Math.Max(1e-3, tHi - tLo);
            for (int i = 0; i < bands; i++)
            {
                double sf = (double)i / Math.Max(1, bands - 1);
                using var normalized = new Mat();
                tMap.ConvertTo(normalized, DepthType.Cv32F, 1.0 / span, -tLo / span);
                DehazeCore.Clamp01(normalized);
                using (var square = new Mat())
                {
                    CvInvoke.Multiply(normalized, normalized, square);
                    using var cube = new Mat(); CvInvoke.Multiply(square, normalized, cube);
                    CvInvoke.AddWeighted(square, 3.0, cube, -2.0, 0.0, normalized);
                }
                using var gate = new Mat(); normalized.ConvertTo(gate, DepthType.Cv32F, 1.0 - sf, sf);
                if (richMap != null)
                {
                    using var richness = new Mat();
                    if (richMap.Size != gate.Size) CvInvoke.Resize(richMap, richness, gate.Size); else richMap.CopyTo(richness);
                    richness.ConvertTo(richness, DepthType.Cv32F, 0.8 * sf);
                    CvInvoke.Max(gate, richness, gate);
                }
                using var contribution = new Mat(); CvInvoke.Multiply(residuals[i], gate, contribution);
                if (bandC > 1e-6) DehazeCore.Clamp(contribution, -bandC, bandC);
                using var scaled = new Mat(); contribution.ConvertTo(scaled, DepthType.Cv32F, Gain(i));
                var next = new Mat(); CvInvoke.Add(current, scaled, next);
                current.Dispose(); current = next;
            }

            DehazeCore.Clamp01(current);
            foreach (var item in smooth) item.Dispose();
            foreach (var item in residuals) item.Dispose();
            return current;
        }

        /// <summary>
        /// Усиление полос, выведенное из МОДЕЛИ ШУМА (винеровский коэффициент), а не подобранное
        /// вручную. Восстановление делит на t, поэтому дисперсия шума растёт как 1/t²:
        ///
        ///     g_ℓ(x) = S_ℓ(x) / ( S_ℓ(x) + σ_ℓ² / t(x)² ),
        ///
        /// где σ_ℓ - СКО шума полосы ℓ, оценённое по ВХОДНОМУ кадру робастной статистикой
        /// σ = 1.4826·median|L_ℓ| (стандартная оценка Донохо: в лапласовой полосе большинство
        /// коэффициентов - шум), а S_ℓ(x) = max(0, локальная мощность полосы - σ_ℓ²/t²) - оценка
        /// мощности сигнала. Там, где сигнал ниже усиленного шума, коэффициент сам стремится к нулю;
        /// где сигнал уверенно выше - к единице. Ручных порогов t_lo/t_hi здесь нет.
        ///
        /// <paramref name="noiseRefBgrSrgb"/> - ВХОДНОЙ (задымлённый) sRGB-кадр в [0,1]; именно по его
        /// линейной яркости оценивается шум ДО усиления. Если null, оценка берётся по самому результату
        /// (тогда σ уже включает усиление и модель становится приблизительной).
        /// </summary>
        public static Mat WienerScaleLaplacian(Mat bgrSrgb01, Mat tMap, int levels,
            double gFine, double gMid, double gCoarse, Mat? noiseRefBgrSrgb, double tFloor = 0.05)
        {
            levels = System.Math.Clamp(levels, 2, 7);

            // Вся пирамидальная физика ведётся в float linear RGB. Прежний путь через 8-bit Lab
            // квантовал полосы и применял нелинейную L* к модели, линейной по световому сигналу.
            using var linear = ColorSpace.ToLinear(bgrSrgb01);
            using var L = ColorSpace.Luminance(linear);
            using var refL = BuildNoiseReference(noiseRefBgrSrgb, L);

            var gauss = new List<Mat> { L.Clone() };
            var refG = new List<Mat> { refL.Clone() };
            var tp = new List<Mat> { tMap.Clone() };
            for (int i = 1; i < levels; i++)
            {
                var dn = new Mat(); CvInvoke.PyrDown(gauss[i - 1], dn); gauss.Add(dn);
                var rn = new Mat(); CvInvoke.PyrDown(refG[i - 1], rn); refG.Add(rn);
                var td = new Mat(); CvInvoke.PyrDown(tp[i - 1], td); tp.Add(td);
            }

            var lap = new List<Mat>();
            var refLap = new List<Mat>();
            for (int i = 0; i < levels - 1; i++)
            {
                using (var up = Up(gauss[i + 1], gauss[i].Size))
                { var l = new Mat(); CvInvoke.Subtract(gauss[i], up, l); lap.Add(l); }
                using (var up = Up(refG[i + 1], refG[i].Size))
                { var l = new Mat(); CvInvoke.Subtract(refG[i], up, l); refLap.Add(l); }
            }

            double BaseGain(int i)
            {
                if (i == 0) return gFine;
                double f = (double)(i - 1) / System.Math.Max(1, levels - 2);
                return gMid + f * (gCoarse - gMid);
            }

            var cur = gauss[levels - 1].Clone();
            for (int i = levels - 2; i >= 0; i--)
            {
                using var up = Up(cur, gauss[i].Size);
                double sigma = RobustSigma(refLap[i]);                     // шум полосы во ВХОДЕ
                double sigma2 = sigma * sigma;

                // локальная мощность полосы в результате
                using var power = new Mat();
                using (var sq = new Mat())
                {
                    CvInvoke.Multiply(lap[i], lap[i], sq);
                    CvInvoke.Blur(sq, power, new Size(7, 7), new Point(-1, -1));
                }

                // шум, усиленный делением на t: σ²/t²
                using var tLev = DehazeCore.ResizeTo(tp[i], lap[i].Size);
                using var tSafe = new Mat();
                using (var f = new Mat(tLev.Size, DepthType.Cv32F, 1)) { f.SetTo(new MCvScalar(tFloor)); CvInvoke.Max(tLev, f, tSafe); }
                using var noise = new Mat();
                CvInvoke.Multiply(tSafe, tSafe, noise);                     // t²
                using (var num = new Mat(noise.Size, DepthType.Cv32F, 1))
                {
                    num.SetTo(new MCvScalar(sigma2));
                    CvInvoke.Divide(num, noise, noise);                    // σ²/t²
                }

                using var signal = new Mat();                              // S = max(0, power - noise)
                CvInvoke.Subtract(power, noise, signal);
                using (var z = new Mat(signal.Size, DepthType.Cv32F, 1)) { z.SetTo(new MCvScalar(0)); CvInvoke.Max(signal, z, signal); }

                using var gate = new Mat();                                // S/(S+noise)
                using (var den = new Mat())
                {
                    CvInvoke.Add(signal, noise, den);
                    CvInvoke.Add(den, new ScalarArray(1e-12), den);
                    CvInvoke.Divide(signal, den, gate);
                }
                DehazeCore.Clamp01(gate);

                using var contrib = new Mat();
                CvInvoke.Multiply(lap[i], gate, contrib);
                using var scaled = new Mat();
                contrib.ConvertTo(scaled, DepthType.Cv32F, BaseGain(i));
                var nc = new Mat();
                CvInvoke.Add(up, scaled, nc);
                cur.Dispose(); cur = nc;
            }

            DehazeCore.Clamp01(cur);

            // Меняем только линейную яркость: одинаковая добавка delta к B/G/R сохраняет
            // Y, потому что веса яркости суммируются в единицу. Затем возвращаемся в sRGB.
            using var deltaY = new Mat();
            CvInvoke.Subtract(cur, L, deltaY);
            var rgb = linear.Split();
            for (int c = 0; c < rgb.Length; c++) CvInvoke.Add(rgb[c], deltaY, rgb[c]);
            using var merged = new Mat();
            using (var v = new VectorOfMat(rgb)) CvInvoke.Merge(v, merged);
            foreach (var c in rgb) c.Dispose();
            DehazeCore.Clamp01(merged);
            var result = ColorSpace.ToSrgb(merged);
            cur.Dispose();

            foreach (var g in gauss) g.Dispose();
            foreach (var g in refG) g.Dispose();
            foreach (var t in tp) t.Dispose();
            foreach (var l in lap) l.Dispose();
            foreach (var l in refLap) l.Dispose();

            return result;
        }

        private static Mat BuildNoiseReference(Mat? bgrSrgb, Mat fallbackL)
        {
            if (bgrSrgb == null) return fallbackL.Clone();
            if (bgrSrgb.NumberOfChannels == 3)
            {
                using var resized = DehazeCore.ResizeTo(bgrSrgb, fallbackL.Size);
                using var linear = ColorSpace.ToLinear(resized);
                return ColorSpace.Luminance(linear);
            }
            return DehazeCore.ResizeTo(bgrSrgb, fallbackL.Size);
        }

        /// <summary>Робастная оценка СКО шума полосы: σ = 1.4826·median|x| (Донохо).</summary>
        private static double RobustSigma(Mat band)
        {
            int n = band.Rows * band.Cols;
            var data = new float[n];
            band.CopyTo(data);
            for (int i = 0; i < n; i++) data[i] = System.Math.Abs(data[i]);
            Array.Sort(data);
            double median = n % 2 == 1 ? data[n / 2] : 0.5 * (data[n / 2 - 1] + data[n / 2]);
            return System.Math.Max(1e-5, 1.4826 * median);
        }

        /// <summary>
        /// Робастная многомасштабная оценка шероховатости: σ(k) считается на 5 логарифмически
        /// расставленных окнах, наклон H = d ln σ / d ln k находится МНК, и вместе с ним возвращается
        /// R² - мера того, насколько зависимость вообще степенная.
        ///
        /// Это замена двухмасштабной оценке <see cref="FractalRichness"/>, у которой нет ни регрессии,
        /// ни доверия. ВАЖНО: даже так это НЕ оценка фрактальной размерности - для неё нужны
        /// structure functions по приращениям и калибровка на fBm с известным H; здесь величина
        /// используется только как «есть ли многомасштабная структура», и называть её фрактальной
        /// в публикации нельзя.
        ///
        /// Возвращает roughness = clamp01(1-H); в <paramref name="r2"/> - карта доверия [0,1].
        /// </summary>
        public static Mat MultiscaleRoughness(Mat bgr01, out Mat r2, int[]? windows = null)
        {
            windows ??= new[] { 3, 5, 9, 17, 33 };
            using var gray = new Mat();
            CvInvoke.CvtColor(bgr01, gray, ColorConversion.Bgr2Gray);

            int k = windows.Length;
            var y = new Mat[k];                                   // y_i = ln σ_i
            var x = new double[k];
            for (int i = 0; i < k; i++)
            {
                using var sd = LocalStd(gray, windows[i]);
                y[i] = new Mat();
                CvInvoke.Add(sd, new ScalarArray(1e-4), y[i]);
                CvInvoke.Log(y[i], y[i]);
                x[i] = System.Math.Log(windows[i]);
            }

            double xbar = x.Average();
            double sxx = x.Sum(v => (v - xbar) * (v - xbar));

            using var ybar = new Mat(y[0].Size, DepthType.Cv32F, 1);
            ybar.SetTo(new MCvScalar(0));
            foreach (var m in y) CvInvoke.Add(ybar, m, ybar);
            ybar.ConvertTo(ybar, DepthType.Cv32F, 1.0 / k);

            var H = new Mat(y[0].Size, DepthType.Cv32F, 1);       // наклон
            H.SetTo(new MCvScalar(0));
            for (int i = 0; i < k; i++)
                using (var term = new Mat())
                {
                    y[i].ConvertTo(term, DepthType.Cv32F, (x[i] - xbar) / sxx);
                    CvInvoke.Add(H, term, H);
                }

            // R² = 1 - SS_res/SS_tot
            using var ssRes = new Mat(y[0].Size, DepthType.Cv32F, 1); ssRes.SetTo(new MCvScalar(0));
            using var ssTot = new Mat(y[0].Size, DepthType.Cv32F, 1); ssTot.SetTo(new MCvScalar(0));
            for (int i = 0; i < k; i++)
            {
                using var pred = new Mat();                       // ȳ + H·(x_i - x̄)
                H.ConvertTo(pred, DepthType.Cv32F, x[i] - xbar);
                CvInvoke.Add(pred, ybar, pred);
                using var dev = new Mat(); CvInvoke.Subtract(y[i], pred, dev);
                using (var sq = new Mat()) { CvInvoke.Multiply(dev, dev, sq); CvInvoke.Add(ssRes, sq, ssRes); }
                using var dt = new Mat(); CvInvoke.Subtract(y[i], ybar, dt);
                using (var sq = new Mat()) { CvInvoke.Multiply(dt, dt, sq); CvInvoke.Add(ssTot, sq, ssTot); }
            }
            r2 = new Mat();
            using (var den = new Mat())
            {
                CvInvoke.Add(ssTot, new ScalarArray(1e-8), den);
                CvInvoke.Divide(ssRes, den, r2);
            }
            r2.ConvertTo(r2, DepthType.Cv32F, -1.0, 1.0);
            DehazeCore.Clamp01(r2);

            var rough = new Mat();
            H.ConvertTo(rough, DepthType.Cv32F, -1.0, 1.0);       // 1 - H
            DehazeCore.Clamp01(rough);
            CvInvoke.GaussianBlur(rough, rough, new Size(0, 0), 2.0);
            DehazeCore.Clamp01(rough);

            H.Dispose();
            foreach (var m in y) m.Dispose();
            return rough;
        }

        /// <summary>
        /// Карта «есть ли многомасштабная структура» ∈[0,1] по масштабированию дисперсии (двухмасштабная).
        /// Локальный σ растёт с окном как r^H: у гладких поверхностей H велик, у текстурных - мал.
        /// H = log(σ_large/σ_small)/log(kLarge/kSmall), результат = 1-H.
        ///
        /// ЭТО НЕ ОЦЕНКА ФРАКТАЛЬНОЙ РАЗМЕРНОСТИ. Здесь всего два масштаба, нет регрессии наклона,
        /// нет меры качества степенной зависимости, используются дисперсии значений в окнах, а не
        /// structure functions приращений, и нет калибровки на процессах с известным H. Величина
        /// пригодна только как эвристическая уверенность «в этом месте есть структура сцены»;
        /// называть её фрактальной в публикации нельзя.
        ///
        /// Кроме того, в идеальной локальной модели I = t·J + (1-t)·A имеем σ_r(I) = t·σ_r(J), значит
        /// ОТНОШЕНИЕ σ_large/σ_small от плотности дымки не зависит вовсе - показатель меняется лишь
        /// из-за пространственной неоднородности t, размытия, шума и границ объектов. Поэтому трактовать
        /// его как меру плотности дымки некорректно.
        ///
        /// Более обоснованный вариант - <see cref="MultiscaleRoughness"/> (5 масштабов, МНК-наклон, R²).
        /// </summary>
        public static Mat FractalRichness(Mat bgr01, int kSmall, int kLarge)
        {
            using var gray = new Mat();
            CvInvoke.CvtColor(bgr01, gray, ColorConversion.Bgr2Gray);
            using var sSmall = LocalStd(gray, kSmall);
            using var sLarge = LocalStd(gray, kLarge);

            using var ratio = new Mat();
            using (var a = new Mat()) using (var b = new Mat())
            {
                CvInvoke.Add(sLarge, new ScalarArray(1e-4), a);
                CvInvoke.Add(sSmall, new ScalarArray(1e-4), b);
                CvInvoke.Divide(a, b, ratio);                      // σ_large/σ_small
            }
            using var H = new Mat();
            CvInvoke.Log(ratio, H);                                 // ln(ratio)
            double denom = System.Math.Log(System.Math.Max(1.5, (double)kLarge / System.Math.Max(1, kSmall)));
            H.ConvertTo(H, DepthType.Cv32F, 1.0 / denom);           // H = ln(ratio)/ln(scaleRatio)
            DehazeCore.Clamp01(H);

            var rich = new Mat();
            H.ConvertTo(rich, DepthType.Cv32F, -1.0, 1.0);          // 1 - H
            DehazeCore.Clamp01(rich);
            CvInvoke.GaussianBlur(rich, rich, new Size(0, 0), 2.0); // сгладить оценку
            DehazeCore.Clamp01(rich);
            return rich;
        }

        private static Mat LocalStd(Mat gray32, int k)
        {
            k = System.Math.Max(3, k | 1);
            var ks = new Size(k, k); var anc = new Point(-1, -1);
            using var mean = new Mat(); CvInvoke.Blur(gray32, mean, ks, anc);
            using var mean2 = new Mat();
            using (var sq = new Mat()) { CvInvoke.Multiply(gray32, gray32, sq); CvInvoke.Blur(sq, mean2, ks, anc); }
            var var0 = new Mat();
            using (var m2 = new Mat()) { CvInvoke.Multiply(mean, mean, m2); CvInvoke.Subtract(mean2, m2, var0); }
            DehazeCore.Clamp01(var0);
            CvInvoke.Sqrt(var0, var0);
            return var0;
        }

        /// <summary>pyrUp с подгонкой под точный размер (pyrUp может дать ±1 px на нечётных сторонах).</summary>
        private static Mat Up(Mat src, Size target)
        {
            var up = new Mat();
            CvInvoke.PyrUp(src, up);
            if (!up.Size.Equals(target))
                CvInvoke.Resize(up, up, target);
            return up;
        }
    }
}
