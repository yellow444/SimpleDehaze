using System.Drawing;
using System.Text;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Метрики качества дехейзинга - и с эталоном, и без него.
    ///
    /// Главная идея: эталон <c>hazefree/</c> снят отдельно (другая экспозиция/баланс белого),
    /// поэтому 'сырой' PSNR награждает совпадение по тону, а не реальное удаление дымки.
    /// Поэтому считаем ещё и <b>совмещённые</b> PSNR/SSIM: перед сравнением подгоняем результат
    /// к эталону поканальным аффинным преобразованием (усиление+сдвиг) - это убирает глобальную
    /// разницу экспозиции/ББ и оставляет в метрике именно структуру и относительный цвет.
    /// Плюс набор без-эталонных метрик (убрана ли дымка, контраст, грани, пересвет, насыщенность).
    /// </summary>
    public static class Metrics
    {
        public readonly record struct Report(
            bool HasRef,
            double Psnr, double PsnrAligned,
            double Ssim, double SsimAligned,
            double HazeRemoved,   // доля снижения dark-channel: 1 = дымки не осталось
            double ContrastGain,  // x, std(результат)/std(вход)
            double EdgeGain,      // x, средний градиент результат/вход
            double ClipPct,       // % пересвеченных/заваленных пикселей
            double ColorRatio,    // x, насыщенность результат/вход (>>1 - перенасыщено)
            double Score)         // 0..100, без-эталонная сводная оценка
        {
            /// <summary>Многострочный отчёт для панели метрик.</summary>
            public string Format()
            {
                var sb = new StringBuilder();
                if (HasRef)
                {
                    sb.AppendLine($"PSNR:  {Psnr,6:F2} дБ   ->  совмещ. {PsnrAligned,6:F2} дБ");
                    sb.AppendLine($"SSIM:  {Ssim,6:F3}      ->  совмещ. {SsimAligned,6:F3}");
                    sb.AppendLine("('совмещ.' = после выравнивания экспозиции/ББ под эталон;");
                    sb.AppendLine(" честнее 'сырого' PSNR, см. ниже)");
                }
                else
                {
                    sb.AppendLine("Эталон не загружен - только без-эталонные метрики.");
                }
                sb.AppendLine();
                sb.AppendLine($"Дымка убрана (dark-channel): {HazeRemoved * 100,4:F0} %");
                sb.AppendLine($"Контраст:  x{ContrastGain:F2}     Грани:  x{EdgeGain:F2}");
                sb.AppendLine($"Пересвет/завал:  {ClipPct:F1} %");
                sb.AppendLine($"Насыщенность:  x{ColorRatio:F2}" + (ColorRatio > 1.5 ? "  (перенасыщено!)" : ""));
                sb.AppendLine();
                sb.AppendLine($"Сводная оценка (без эталона):  {Score:F0} / 100");
                return sb.ToString();
            }

            /// <summary>Короткая приписка к строке статуса.</summary>
            public string StatusSuffix()
            {
                string refPart = HasRef ? $"  -  PSNR {Psnr:F1}/совмещ.{PsnrAligned:F1} дБ - SSIM {SsimAligned:F2}" : "";
                return $"{refPart}  -  оценка {Score:F0}/100  (дымка{HazeRemoved * 100:F0}% пересвет {ClipPct:F0}% цвет x{ColorRatio:F1})";
            }
        }

        // ---------- публичные точки входа ----------

        /// <summary>PSNR результата (BGR float [0,1]) против эталона (BGR 8U), дБ. Больше = ближе к эталону.</summary>
        public static double Psnr(Mat resultFloat01, Mat gt8)
        {
            using var r8 = new Mat(); resultFloat01.ConvertTo(r8, DepthType.Cv8U, 255.0);
            using var g8 = ResizeTo(gt8, r8.Size);
            return Psnr8(r8, g8);
        }

        /// <summary>Полный набор метрик. <paramref name="gt8"/> может быть null (тогда только без-эталонные).</summary>
        public static Report Evaluate(Mat resultFloat01, Mat? gt8, Mat input8)
        {
            using var r8full = new Mat(); resultFloat01.ConvertTo(r8full, DepthType.Cv8U, 255.0);
            using var r8 = Down(r8full, 1024);          // метрики на уменьшенной копии - быстро и устойчиво
            Size sz = r8.Size;
            using var inp = ResizeTo(input8, sz);

            using var grR = Gray(r8);
            using var grI = Gray(inp);
            double contrast = Std(grR) / (Std(grI) + 1e-6);
            double edge = MeanGrad(grR) / (MeanGrad(grI) + 1e-6);

            double dcI = DarkChannelMean(inp), dcR = DarkChannelMean(r8);
            double haze = dcI > 1e-6 ? Math.Clamp((dcI - dcR) / dcI, -1, 1) : 0;

            double clip = ClipFraction(r8);
            double color = Colorfulness(r8) / (Colorfulness(inp) + 1e-6);
            double score = NoRef01(haze, edge, contrast, clip, color) * 100.0;

            double psnr = double.NaN, psnrA = double.NaN, ssim = double.NaN, ssimA = double.NaN;
            bool hasRef = gt8 != null;
            if (gt8 != null)
            {
                using var g8 = ResizeTo(gt8, sz);
                using var grG = Gray(g8);
                psnr = Psnr8(r8, g8);
                ssim = Ssim(grR, grG);
                using var aligned = AlignExposure(r8, g8);
                using var grA = Gray(aligned);
                psnrA = Psnr8(aligned, g8);
                ssimA = Ssim(grA, grG);
            }
            return new Report(hasRef, psnr, psnrA, ssim, ssimA, haze, contrast, edge, clip * 100.0, color, score);
        }

        /// <summary>Без-эталонная сводная оценка результата (для авто-подбора). 0..100, больше = лучше.</summary>
        public static double NoRefScore(Mat resultFloat01, Mat input8) => Evaluate(resultFloat01, null, input8).Score;

        // ---------- сводная без-эталонная оценка ----------

        /// <summary>
        /// Композитная оценка 0..1: награждает удаление дымки + детали/контраст,
        /// штрафует пересвет и перенасыщение (главные источники 'отвратительного' результата).
        /// </summary>
        private static double NoRef01(double hazeRemoved, double edgeRatio, double contrastRatio, double clipFrac, double colorRatio)
        {
            // польза. Дымка с насыщением: sqrt -> убирать сверх ~50% даёт всё меньше прибавки
            // (иначе оценка тянет к 'выкручено по максимуму' - пересатур/перетемнение без выигрыша в верности).
            double haze = Math.Sqrt(Math.Clamp(hazeRemoved, 0, 1));
            double detail = Math.Clamp((edgeRatio - 1.0) / 1.5, 0, 1);     // прирост граней (x2.5 -> 1)
            double contrast = Math.Clamp(contrastRatio - 1.0, 0, 1);       // прирост контраста

            // штрафы за артефакты - строже и симметрично по цвету
            double clipPen = Math.Clamp(clipFrac / 0.05, 0, 1);            // 5% пересвета -> полный штраф
            double overSat = Math.Clamp((colorRatio - 1.25) / 0.45, 0, 1); // перенасыщение (>x1.25)
            double underSat = Math.Clamp((0.92 - colorRatio) / 0.30, 0, 1);// обесцвечивание (<x0.92)

            double good = 0.45 * haze + 0.35 * detail + 0.20 * contrast;
            double pen = 0.45 * clipPen + 0.45 * overSat + 0.20 * underSat;

            // отдельный множитель за СИЛЬНЫЙ пересвет: >5% завала всё сильнее 'гасит' оценку
            // (иначе 6% и 18% пересвета штрафуются одинаково и 'выжигатель' сидит в середине).
            double severe = Math.Clamp((clipFrac - 0.05) / 0.15, 0, 1);  // 5% -> 0, 20% -> 1
            return Math.Clamp((good - pen) * (1.0 - 0.85 * severe), 0, 1);
        }

        // ---------- метрики ----------

        private static double Psnr8(Mat a8, Mat b8)
        {
            using var diff = new Mat(); CvInvoke.AbsDiff(a8, b8, diff);
            using var d32 = new Mat(); diff.ConvertTo(d32, DepthType.Cv32F);
            CvInvoke.Multiply(d32, d32, d32);
            var m = CvInvoke.Mean(d32);
            double mse = (m.V0 + m.V1 + m.V2) / 3.0;
            return mse < 1e-9 ? 99 : 10 * Math.Log10(255.0 * 255.0 / mse);
        }

        /// <summary>Одно-масштабный SSIM по яркости (окно Гаусса 11x11), [0,1].</summary>
        private static double Ssim(Mat a8, Mat b8)
        {
            const double c1 = 6.5025, c2 = 58.5225;     // (0.01*255)^2, (0.03*255)^2
            var win = new Size(11, 11); const double sg = 1.5;
            using var a = new Mat(); a8.ConvertTo(a, DepthType.Cv32F);
            using var b = new Mat(); b8.ConvertTo(b, DepthType.Cv32F);

            using var mu1 = new Mat(); CvInvoke.GaussianBlur(a, mu1, win, sg);
            using var mu2 = new Mat(); CvInvoke.GaussianBlur(b, mu2, win, sg);
            using var mu1_2 = Mul(mu1, mu1);
            using var mu2_2 = Mul(mu2, mu2);
            using var mu1mu2 = Mul(mu1, mu2);

            using var aa = Mul(a, a); using var bb = Mul(b, b); using var ab = Mul(a, b);
            using var s1 = new Mat(); CvInvoke.GaussianBlur(aa, s1, win, sg); CvInvoke.Subtract(s1, mu1_2, s1);
            using var s2 = new Mat(); CvInvoke.GaussianBlur(bb, s2, win, sg); CvInvoke.Subtract(s2, mu2_2, s2);
            using var s12 = new Mat(); CvInvoke.GaussianBlur(ab, s12, win, sg); CvInvoke.Subtract(s12, mu1mu2, s12);

            using var n1 = new Mat(); mu1mu2.ConvertTo(n1, DepthType.Cv32F, 2.0, c1);    // 2*μ1μ2 + C1
            using var n2 = new Mat(); s12.ConvertTo(n2, DepthType.Cv32F, 2.0, c2);       // 2*σ12 + C2
            using var d1 = new Mat(); CvInvoke.Add(mu1_2, mu2_2, d1); d1.ConvertTo(d1, DepthType.Cv32F, 1.0, c1);
            using var d2 = new Mat(); CvInvoke.Add(s1, s2, d2); d2.ConvertTo(d2, DepthType.Cv32F, 1.0, c2);

            using var num = Mul(n1, n2);
            using var den = Mul(d1, d2);
            using var map = new Mat(); CvInvoke.Divide(num, den, map);
            return CvInvoke.Mean(map).V0;
        }

        /// <summary>Поканальное аффинное совмещение результата с эталоном (убирает разницу экспозиции/ББ).</summary>
        private static Mat AlignExposure(Mat result8, Mat gt8)
        {
            using var rf = new Mat(); result8.ConvertTo(rf, DepthType.Cv32F);
            using var gf = new Mat(); gt8.ConvertTo(gf, DepthType.Cv32F);
            var mr = CvInvoke.Mean(rf); var mg = CvInvoke.Mean(gf);
            using var rr = new Mat(); CvInvoke.Multiply(rf, rf, rr); var err = CvInvoke.Mean(rr);
            using var rg = new Mat(); CvInvoke.Multiply(rf, gf, rg); var erg = CvInvoke.Mean(rg);

            double[] mrv = { mr.V0, mr.V1, mr.V2 }, mgv = { mg.V0, mg.V1, mg.V2 };
            double[] errv = { err.V0, err.V1, err.V2 }, ergv = { erg.V0, erg.V1, erg.V2 };
            double[] a = new double[3], bb = new double[3];
            for (int c = 0; c < 3; c++)
            {
                double varr = errv[c] - mrv[c] * mrv[c];
                double cov = ergv[c] - mrv[c] * mgv[c];
                a[c] = cov / (varr + 1e-6);
                bb[c] = mgv[c] - a[c] * mrv[c];
            }
            using var aMat = new Mat(rf.Size, DepthType.Cv32F, 3); aMat.SetTo(new MCvScalar(a[0], a[1], a[2]));
            using var bMat = new Mat(rf.Size, DepthType.Cv32F, 3); bMat.SetTo(new MCvScalar(bb[0], bb[1], bb[2]));
            using var alignedf = new Mat(); CvInvoke.Multiply(rf, aMat, alignedf); CvInvoke.Add(alignedf, bMat, alignedf);
            var aligned8 = new Mat(); alignedf.ConvertTo(aligned8, DepthType.Cv8U);   // saturate -> [0,255]
            return aligned8;
        }

        /// <summary>Средняя величина dark-channel (0..255). Меньше -> меньше дымки.</summary>
        private static double DarkChannelMean(Mat bgr8)
        {
            using var minc = MinChannel(bgr8);
            int k = Math.Max(3, (Math.Min(bgr8.Rows, bgr8.Cols) / 100) | 1);
            using var se = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(k, k), new Point(-1, -1));
            using var dark = new Mat();
            CvInvoke.Erode(minc, dark, se, new Point(-1, -1), 1, BorderType.Replicate, default);
            return CvInvoke.Mean(dark).V0;
        }

        private static double ClipFraction(Mat bgr8)
        {
            using var vm = new VectorOfMat(); CvInvoke.Split(bgr8, vm);
            using var b = vm[0]; using var g = vm[1]; using var r = vm[2];
            using var maxc = new Mat(); CvInvoke.Max(b, g, maxc); CvInvoke.Max(maxc, r, maxc);
            using var minc = new Mat(); CvInvoke.Min(b, g, minc); CvInvoke.Min(minc, r, minc);
            using var hi = new Mat(); CvInvoke.Threshold(maxc, hi, 253, 255, ThresholdType.Binary);
            using var lo = new Mat(); CvInvoke.Threshold(minc, lo, 1, 255, ThresholdType.BinaryInv);
            int n = bgr8.Rows * bgr8.Cols;
            int c = CvInvoke.CountNonZero(hi) + CvInvoke.CountNonZero(lo);
            return Math.Min(1.0, c / (double)n);
        }

        /// <summary>Колоритность Хаслера-Зюсструнка (больше = насыщеннее).</summary>
        public static double Colorfulness(Mat bgr8)
        {
            using var vm = new VectorOfMat(); CvInvoke.Split(bgr8, vm);
            using var b8 = vm[0]; using var g8 = vm[1]; using var r8 = vm[2];
            using var B = new Mat(); b8.ConvertTo(B, DepthType.Cv32F);
            using var G = new Mat(); g8.ConvertTo(G, DepthType.Cv32F);
            using var R = new Mat(); r8.ConvertTo(R, DepthType.Cv32F);

            using var rg = new Mat(); CvInvoke.Subtract(R, G, rg);                 // R - G
            using var rpg = new Mat(); CvInvoke.Add(R, G, rpg);
            using var yb = new Mat(); rpg.ConvertTo(yb, DepthType.Cv32F, 0.5, 0.0); CvInvoke.Subtract(yb, B, yb); // 1/2(R+G) - B

            MCvScalar mrg = default, srg = default, myb = default, syb = default;
            CvInvoke.MeanStdDev(rg, ref mrg, ref srg);
            CvInvoke.MeanStdDev(yb, ref myb, ref syb);
            double stdRoot = Math.Sqrt(srg.V0 * srg.V0 + syb.V0 * syb.V0);
            double meanRoot = Math.Sqrt(mrg.V0 * mrg.V0 + myb.V0 * myb.V0);
            return stdRoot + 0.3 * meanRoot;
        }

        // ---------- мелкие помощники ----------

        private static Mat MinChannel(Mat bgr8)
        {
            using var vm = new VectorOfMat(); CvInvoke.Split(bgr8, vm);
            using var b = vm[0]; using var g = vm[1]; using var r = vm[2];
            var minc = new Mat(); CvInvoke.Min(b, g, minc); CvInvoke.Min(minc, r, minc);
            return minc;
        }

        private static double MeanGrad(Mat gray8)
        {
            using var gx = new Mat(); CvInvoke.Sobel(gray8, gx, DepthType.Cv32F, 1, 0, 3);
            using var gy = new Mat(); CvInvoke.Sobel(gray8, gy, DepthType.Cv32F, 0, 1, 3);
            using var mag = new Mat();
            using (var gx2 = Mul(gx, gx)) { using var gy2 = Mul(gy, gy); CvInvoke.Add(gx2, gy2, mag); }
            CvInvoke.Sqrt(mag, mag);
            return CvInvoke.Mean(mag).V0;
        }

        private static double Std(Mat gray8)
        {
            MCvScalar m = default, s = default;
            CvInvoke.MeanStdDev(gray8, ref m, ref s);
            return s.V0;
        }

        private static Mat Gray(Mat bgr8)
        {
            var g = new Mat(); CvInvoke.CvtColor(bgr8, g, ColorConversion.Bgr2Gray);
            return g;
        }

        private static Mat Mul(Mat a, Mat b)
        {
            var o = new Mat(); CvInvoke.Multiply(a, b, o);
            return o;
        }

        private static Mat Down(Mat m, int maxDim)
        {
            int w = m.Cols, h = m.Rows;
            double s = Math.Min(1.0, (double)maxDim / Math.Max(w, h));
            var o = new Mat();
            if (s >= 1.0) m.CopyTo(o);
            else CvInvoke.Resize(m, o, new Size(Math.Max(1, (int)(w * s)), Math.Max(1, (int)(h * s))), 0, 0, Inter.Area);
            return o;
        }

        private static Mat ResizeTo(Mat m, Size sz)
        {
            var o = new Mat();
            if (m.Size.Equals(sz)) m.CopyTo(o);
            else CvInvoke.Resize(m, o, sz, 0, 0, Inter.Area);
            return o;
        }
    }
}
