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
        internal readonly record struct ChromaticFidelityReport(
            double Score, double WeightedRelativeError, int ChromaticPixels,
            double ChromaticPixelFraction, double ChromaThreshold);

        internal readonly record struct LocalChromaExpansionReport(
            double MeanExcess, double P95Excess, double ExplodedPixelFraction);

        public readonly record struct Report(
            bool HasRef,
            double Psnr, double PsnrAligned,
            double Mse, double MseAligned,
            double Ssim, double SsimAligned,
            double Ciede2000, double Ciede2000Aligned,
            double NaturalnessDev,   // СОБСТВЕННАЯ эвристика, НЕ NIQE: отклонение статистик от «типичного» кадра
            double ArtifactDev,      // СОБСТВЕННАЯ эвристика, НЕ BRISQUE: контраст/резкость/цвет/клип/блочность
            double FlatNoiseRatio,   // усиление шума на плоских участках, result/input (1 = не усилен)
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
                    sb.AppendLine("ОСНОВНЫЕ (без подгонки под эталон):");
                    sb.AppendLine($"  PSNR:  {Psnr,6:F2} дБ    SSIM: {Ssim,6:F3}");
                    sb.AppendLine($"  MSE:   {Mse,6:F1}       CIEDE2000: {Ciede2000,6:F2}  (меньше лучше)");
                    sb.AppendLine("ДИАГНОСТИЧЕСКИЕ (после поканального аффинного совмещения с эталоном):");
                    sb.AppendLine($"  PSNR:  {PsnrAligned,6:F2} дБ    SSIM: {SsimAligned,6:F3}");
                    sb.AppendLine($"  MSE:   {MseAligned,6:F1}       CIEDE2000: {Ciede2000Aligned,6:F2}");
                    sb.AppendLine("  ВНИМАНИЕ: совмещение использует эталон для правки результата,");
                    sb.AppendLine("  поэтому эти числа НЕЛЬЗЯ подавать как основной результат.");
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
                sb.AppendLine($"Шум на плоских зонах:  x{FlatNoiseRatio:F2}  (1 = не усилен)");
                sb.AppendLine($"Собств. оценки (НЕ NIQE/BRISQUE):  натуральность {NaturalnessDev:F1}   артефакты {ArtifactDev:F1}  (меньше лучше)");
                sb.AppendLine();
                sb.AppendLine($"Сводная оценка (без эталона):  {Score:F0} / 100");
                return sb.ToString();
            }

            /// <summary>Короткая приписка к строке статуса.</summary>
            public string StatusSuffix()
            {
                string refPart = HasRef ? $"  -  PSNR {Psnr:F1} дБ - SSIM {Ssim:F2} - CIEDE {Ciede2000:F1}  (совмещ. {PsnrAligned:F1}/{SsimAligned:F2})" : "";
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

        /// <summary>MSE результата (BGR float [0,1]) против эталона (BGR 8U). Меньше = ближе к эталону.</summary>
        public static double Mse(Mat resultFloat01, Mat gt8)
        {
            using var r8 = new Mat(); resultFloat01.ConvertTo(r8, DepthType.Cv8U, 255.0);
            using var g8 = ResizeTo(gt8, r8.Size);
            return Mse8(r8, g8);
        }

        /// <summary>
        /// Максимальная сторона копии, на которой считаются метрики. По умолчанию 1024 (быстро и
        /// устойчиво), но для публикуемых чисел ставьте int.MaxValue: PSNR/SSIM зависят от масштаба.
        /// Меняется ключом --evalfull в headless-режимах.
        /// </summary>
        public static int EvalMaxSide { get; set; } = 1024;

        /// <summary>Полный набор метрик. <paramref name="gt8"/> может быть null (тогда только без-эталонные).</summary>
        public static Report Evaluate(Mat resultFloat01, Mat? gt8, Mat input8)
        {
            using var r8full = new Mat(); resultFloat01.ConvertTo(r8full, DepthType.Cv8U, 255.0);
            using var r8 = Down(r8full, EvalMaxSide);   // см. EvalMaxSide: для публикации - полный размер
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
            double naturalnessDev = NaturalnessDeviation(r8);
            double artifactDev = ArtifactDeviation(r8);
            double flatNoise = FlatNoise(r8) / (FlatNoise(inp) + 1e-6);
            double score = NoRef01(haze, edge, contrast, clip, color) * 100.0;

            double psnr = double.NaN, psnrA = double.NaN, mse = double.NaN, mseA = double.NaN, ssim = double.NaN, ssimA = double.NaN;
            double ciede = double.NaN, ciedeA = double.NaN;
            bool hasRef = gt8 != null;
            if (gt8 != null)
            {
                using var g8 = ResizeTo(gt8, sz);
                using var grG = Gray(g8);
                mse = Mse8(r8, g8);
                psnr = Psnr8(r8, g8);
                ssim = Ssim(grR, grG);
                ciede = Ciede2000(r8, g8);
                using var aligned = AlignExposure(r8, g8);
                using var grA = Gray(aligned);
                mseA = Mse8(aligned, g8);
                psnrA = Psnr8(aligned, g8);
                ssimA = Ssim(grA, grG);
                ciedeA = Ciede2000(aligned, g8);
            }
            return new Report(hasRef, psnr, psnrA, mse, mseA, ssim, ssimA, ciede, ciedeA,
                naturalnessDev, artifactDev, flatNoise, haze, contrast, edge, clip * 100.0, color, score);
        }

        /// <summary>
        /// Точность вектора Lab (a*, b*) на пикселях, где GT не ахроматичен. Обычная средняя ΔE
        /// почти не замечает маленькую цветовую таблицу или локальный цветной объект на серой сцене;
        /// здесь оценивается верхний дециль C*_ab(GT), но не ниже C*=4, а вес равен C*². Поэтому
        /// серый фон не разбавляет небольшие цветные объекты. 100 = точное совпадение, 0 = полная
        /// потеря либо разворот значимого цветового вектора.
        /// </summary>
        internal static ChromaticFidelityReport ChromaticFidelity(Mat resultFloat01, Mat gt8)
        {
            using var resultBytesFull = new Mat();
            resultFloat01.ConvertTo(resultBytesFull, DepthType.Cv8U, 255.0);
            using var resultBytes = Down(resultBytesFull, 560);
            using var truthBytes = ResizeTo(gt8, resultBytes.Size);
            using var resultLab = new Mat();
            using var truthLab = new Mat();
            CvInvoke.CvtColor(resultBytes, resultLab, ColorConversion.Bgr2Lab);
            CvInvoke.CvtColor(truthBytes, truthLab, ColorConversion.Bgr2Lab);

            int pixels = resultLab.Rows * resultLab.Cols;
            var result = new byte[pixels * 3];
            var truth = new byte[pixels * 3];
            resultLab.CopyTo(result);
            truthLab.CopyTo(truth);
            var truthChromaValues = new double[pixels];
            for (int pixel = 0, offset = 0; pixel < pixels; pixel++, offset += 3)
            {
                double truthA = truth[offset + 1] - 128.0;
                double truthB = truth[offset + 2] - 128.0;
                truthChromaValues[pixel] = Math.Sqrt(truthA * truthA + truthB * truthB);
            }
            var orderedChroma = (double[])truthChromaValues.Clone();
            Array.Sort(orderedChroma);
            int percentileIndex = Math.Clamp((int)Math.Floor(0.90 * Math.Max(0, pixels - 1)), 0,
                Math.Max(0, pixels - 1));
            double chromaThreshold = pixels > 0 ? Math.Max(4.0, orderedChroma[percentileIndex]) : 4.0;

            double weightedError = 0.0;
            double weightSum = 0.0;
            int chromatic = 0;
            for (int pixel = 0, offset = 0; offset < result.Length; pixel++, offset += 3)
            {
                double resultA = result[offset + 1] - 128.0;
                double resultB = result[offset + 2] - 128.0;
                double truthA = truth[offset + 1] - 128.0;
                double truthB = truth[offset + 2] - 128.0;
                double truthChroma = truthChromaValues[pixel];
                if (truthChroma + 1e-12 < chromaThreshold) continue;

                double da = resultA - truthA;
                double db = resultB - truthB;
                double relativeError = Math.Min(1.0, Math.Sqrt(da * da + db * db) /
                    Math.Max(4.0, truthChroma));
                double weight = Math.Min(2500.0, truthChroma * truthChroma);
                weightedError += weight * relativeError;
                weightSum += weight;
                chromatic++;
            }

            // У полностью ахроматичного GT нет цветового сигнала, который можно потерять.
            double meanError = weightSum > 0.0 ? weightedError / weightSum : 0.0;
            return new ChromaticFidelityReport(
                100.0 * (1.0 - Math.Clamp(meanError, 0.0, 1.0)),
                meanError,
                chromatic,
                chromatic / (double)Math.Max(1, pixels),
                chromaThreshold);
        }

        /// <summary>
        /// Безэталонный заслон локального перенасыщения. Допускается C*(result) до 1.5*C*(input)+8;
        /// всё сверх этого считается избытком. Перцентиль и доля нужны, потому что средняя
        /// насыщенность всего кадра скрывает небольшие радужные пятна.
        /// </summary>
        internal static LocalChromaExpansionReport LocalChromaExpansion(Mat resultFloat01, Mat input8)
        {
            using var resultBytesFull = new Mat();
            resultFloat01.ConvertTo(resultBytesFull, DepthType.Cv8U, 255.0);
            using var resultBytes = Down(resultBytesFull, 560);
            using var inputBytes = ResizeTo(input8, resultBytes.Size);
            using var resultLab = new Mat();
            using var inputLab = new Mat();
            CvInvoke.CvtColor(resultBytes, resultLab, ColorConversion.Bgr2Lab);
            CvInvoke.CvtColor(inputBytes, inputLab, ColorConversion.Bgr2Lab);

            int pixels = resultLab.Rows * resultLab.Cols;
            var result = new byte[pixels * 3];
            var input = new byte[pixels * 3];
            resultLab.CopyTo(result);
            inputLab.CopyTo(input);
            var excesses = new double[pixels];
            double excessSum = 0.0;
            int exploded = 0;
            for (int pixel = 0, offset = 0; pixel < pixels; pixel++, offset += 3)
            {
                double resultA = result[offset + 1] - 128.0;
                double resultB = result[offset + 2] - 128.0;
                double inputA = input[offset + 1] - 128.0;
                double inputB = input[offset + 2] - 128.0;
                double resultChroma = Math.Sqrt(resultA * resultA + resultB * resultB);
                double inputChroma = Math.Sqrt(inputA * inputA + inputB * inputB);
                double excess = Math.Max(0.0, resultChroma - 1.5 * inputChroma - 8.0);
                excesses[pixel] = excess;
                excessSum += excess;
                if (excess > 4.0) exploded++;
            }
            Array.Sort(excesses);
            int percentileIndex = Math.Clamp((int)Math.Floor(0.95 * Math.Max(0, pixels - 1)), 0,
                Math.Max(0, pixels - 1));
            double p95 = pixels > 0 ? excesses[percentileIndex] : 0.0;
            return new LocalChromaExpansionReport(
                excessSum / Math.Max(1, pixels),
                p95,
                exploded / (double)Math.Max(1, pixels));
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
            double mse = Mse8(a8, b8);
            return mse < 1e-9 ? 99 : 10 * Math.Log10(255.0 * 255.0 / mse);
        }

        private static double Mse8(Mat a8, Mat b8)
        {
            using var diff = new Mat(); CvInvoke.AbsDiff(a8, b8, diff);
            using var d32 = new Mat(); diff.ConvertTo(d32, DepthType.Cv32F);
            CvInvoke.Multiply(d32, d32, d32);
            var m = CvInvoke.Mean(d32);
            return (m.V0 + m.V1 + m.V2) / 3.0;
        }

        /// <summary>Средняя цветовая ошибка CIEDE2000 между двумя BGR 8U изображениями. Меньше = лучше.</summary>
        public static double Ciede2000(Mat a8, Mat b8)
        {
            using var aa = ResizeTo(a8, b8.Size);
            using var lab1 = new Mat();
            using var lab2 = new Mat();
            CvInvoke.CvtColor(aa, lab1, ColorConversion.Bgr2Lab);
            CvInvoke.CvtColor(b8, lab2, ColorConversion.Bgr2Lab);

            int n = lab1.Rows * lab1.Cols;
            var x = new byte[n * 3];
            var y = new byte[n * 3];
            lab1.CopyTo(x);
            lab2.CopyTo(y);

            double sum = 0;
            for (int i = 0, o = 0; i < n; i++, o += 3)
            {
                double l1 = x[o] * (100.0 / 255.0);
                double a1 = x[o + 1] - 128.0;
                double b1 = x[o + 2] - 128.0;
                double l2 = y[o] * (100.0 / 255.0);
                double a2 = y[o + 1] - 128.0;
                double b2 = y[o + 2] - 128.0;
                sum += DeltaE2000(l1, a1, b1, l2, a2, b2);
            }
            return sum / Math.Max(1, n);
        }

        internal static double DeltaE2000(double l1, double a1, double b1, double l2, double a2, double b2)
        {
            const double pow25To7 = 6103515625.0; // 25^7
            double c1 = Math.Sqrt(a1 * a1 + b1 * b1);
            double c2 = Math.Sqrt(a2 * a2 + b2 * b2);
            double cBar = (c1 + c2) * 0.5;
            double cBar7 = Math.Pow(cBar, 7.0);
            double g = 0.5 * (1.0 - Math.Sqrt(cBar7 / (cBar7 + pow25To7)));
            double a1p = (1.0 + g) * a1;
            double a2p = (1.0 + g) * a2;
            double c1p = Math.Sqrt(a1p * a1p + b1 * b1);
            double c2p = Math.Sqrt(a2p * a2p + b2 * b2);

            double h1p = HueDeg(b1, a1p);
            double h2p = HueDeg(b2, a2p);
            double dLp = l2 - l1;
            double dCp = c2p - c1p;
            double dhp = 0.0;
            if (c1p * c2p > 1e-12)
            {
                dhp = h2p - h1p;
                if (dhp > 180.0) dhp -= 360.0;
                else if (dhp < -180.0) dhp += 360.0;
            }
            double dHp = 2.0 * Math.Sqrt(c1p * c2p) * Math.Sin(DegToRad(dhp * 0.5));

            double lBarP = (l1 + l2) * 0.5;
            double cBarP = (c1p + c2p) * 0.5;
            double hBarP;
            if (c1p * c2p <= 1e-12) hBarP = h1p + h2p;
            else if (Math.Abs(h1p - h2p) <= 180.0) hBarP = (h1p + h2p) * 0.5;
            else if (h1p + h2p < 360.0) hBarP = (h1p + h2p + 360.0) * 0.5;
            else hBarP = (h1p + h2p - 360.0) * 0.5;

            double t = 1.0
                - 0.17 * Math.Cos(DegToRad(hBarP - 30.0))
                + 0.24 * Math.Cos(DegToRad(2.0 * hBarP))
                + 0.32 * Math.Cos(DegToRad(3.0 * hBarP + 6.0))
                - 0.20 * Math.Cos(DegToRad(4.0 * hBarP - 63.0));
            double dTheta = 30.0 * Math.Exp(-Math.Pow((hBarP - 275.0) / 25.0, 2.0));
            double cBarP7 = Math.Pow(cBarP, 7.0);
            double rC = 2.0 * Math.Sqrt(cBarP7 / (cBarP7 + pow25To7));
            double sL = 1.0 + (0.015 * Math.Pow(lBarP - 50.0, 2.0)) / Math.Sqrt(20.0 + Math.Pow(lBarP - 50.0, 2.0));
            double sC = 1.0 + 0.045 * cBarP;
            double sH = 1.0 + 0.015 * cBarP * t;
            double rT = -Math.Sin(DegToRad(2.0 * dTheta)) * rC;

            double lTerm = dLp / sL;
            double cTerm = dCp / sC;
            double hTerm = dHp / sH;
            return Math.Sqrt(Math.Max(0.0, lTerm * lTerm + cTerm * cTerm + hTerm * hTerm + rT * cTerm * hTerm));
        }

        private static double HueDeg(double b, double a)
        {
            double h = Math.Atan2(b, a) * 180.0 / Math.PI;
            return h < 0.0 ? h + 360.0 : h;
        }

        private static double DegToRad(double deg) => deg * Math.PI / 180.0;

        /// <summary>
        /// Оценка уровня шума на ПЛОСКИХ участках: медиана локального σ по 10 % самых гладких окон.
        /// Деление на t усиливает шум как 1/t, поэтому отношение (результат/вход) прямо показывает,
        /// во сколько раз метод раздул шум там, где сигнала нет.
        /// </summary>
        private static double FlatNoise(Mat bgr8)
        {
            using var gray = Gray(bgr8);
            using var g = new Mat(); gray.ConvertTo(g, DepthType.Cv32F, 1.0 / 255.0);
            var ks = new Size(9, 9); var anc = new Point(-1, -1);
            using var mean = new Mat(); CvInvoke.Blur(g, mean, ks, anc);
            using var mean2 = new Mat();
            using (var sq = new Mat()) { CvInvoke.Multiply(g, g, sq); CvInvoke.Blur(sq, mean2, ks, anc); }
            using var v = new Mat();
            using (var m2 = new Mat()) { CvInvoke.Multiply(mean, mean, m2); CvInvoke.Subtract(mean2, m2, v); }
            using var sd = new Mat();
            using (var z = new Mat(v.Size, DepthType.Cv32F, 1)) { z.SetTo(new MCvScalar(0)); CvInvoke.Max(v, z, sd); }
            CvInvoke.Sqrt(sd, sd);

            int n = sd.Rows * sd.Cols;
            var data = new float[n];
            sd.CopyTo(data);
            Array.Sort(data);
            int lo = Math.Max(1, (int)(n * 0.05)), hi = Math.Max(lo + 1, (int)(n * 0.10));
            double sum = 0;
            for (int i = lo; i < hi; i++) sum += data[i];
            return sum / (hi - lo);
        }

        /// <summary>
        /// СОБСТВЕННАЯ эвристика, НЕ NIQE. Настоящий NIQE требует опубликованной MVG-модели,
        /// обученной на корпусе естественных изображений; здесь считается лишь отклонение простых
        /// статистик (яркость, разброс, градиент, клиппинг) от эмпирически «типичных» значений.
        /// Сравнивать эти числа с публикуемыми NIQE НЕЛЬЗЯ.
        /// </summary>
        private static double NaturalnessDeviation(Mat bgr8)
        {
            using var gray = Gray(bgr8);
            MCvScalar mean = default, std = default;
            CvInvoke.MeanStdDev(gray, ref mean, ref std);
            double mu = mean.V0 / 255.0;
            double sigma = std.V0 / 255.0;
            double grad = MeanGrad(gray) / 255.0;
            double clip = ClipFraction(bgr8);

            double meanPen = Math.Clamp(Math.Abs(mu - 0.50) / 0.28, 0, 1);
            double stdPen = Math.Clamp(Math.Abs(sigma - 0.22) / 0.22, 0, 1);
            double gradPen = Math.Clamp(Math.Abs(grad - 0.10) / 0.18, 0, 1);
            double clipPen = Math.Clamp(clip / 0.05, 0, 1);
            return 100.0 * (0.30 * meanPen + 0.30 * stdPen + 0.20 * gradPen + 0.20 * clipPen);
        }

        /// <summary>
        /// СОБСТВЕННАЯ эвристика, НЕ BRISQUE. Настоящий BRISQUE - обученный на MOS регрессор по
        /// MSCN-признакам; здесь просто взвешенные отклонения контраста, резкости, цветности,
        /// клиппинга и блочности. Сравнивать с публикуемыми BRISQUE НЕЛЬЗЯ.
        /// </summary>
        private static double ArtifactDeviation(Mat bgr8)
        {
            using var gray = Gray(bgr8);
            double contrast = Std(gray) / 64.0;
            double grad = MeanGrad(gray) / 64.0;
            double clip = ClipFraction(bgr8);
            double color = Colorfulness(bgr8) / 45.0;
            double block = Blockiness(gray);

            double contrastPen = Math.Clamp(Math.Abs(contrast - 1.0) / 1.2, 0, 1);
            double sharpPen = Math.Clamp(Math.Abs(grad - 1.0) / 1.5, 0, 1);
            double colorPen = Math.Clamp(Math.Abs(color - 1.0) / 1.6, 0, 1);
            double clipPen = Math.Clamp(clip / 0.05, 0, 1);
            double blockPen = Math.Clamp(block / 12.0, 0, 1);
            return 100.0 * (0.22 * contrastPen + 0.22 * sharpPen + 0.18 * colorPen + 0.23 * clipPen + 0.15 * blockPen);
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

        private static double Blockiness(Mat gray8)
        {
            int w = gray8.Cols, h = gray8.Rows;
            if (w < 16 || h < 16) return 0;

            var data = new byte[w * h];
            gray8.CopyTo(data);

            double boundary = 0;
            int boundaryCount = 0;
            double interior = 0;
            int interiorCount = 0;

            for (int y = 0; y < h; y++)
            {
                int row = y * w;
                for (int x = 1; x < w; x++)
                {
                    double d = Math.Abs(data[row + x] - data[row + x - 1]);
                    if (x % 8 == 0) { boundary += d; boundaryCount++; }
                    else { interior += d; interiorCount++; }
                }
            }

            for (int y = 1; y < h; y++)
            {
                int row = y * w;
                int prev = (y - 1) * w;
                for (int x = 0; x < w; x++)
                {
                    double d = Math.Abs(data[row + x] - data[prev + x]);
                    if (y % 8 == 0) { boundary += d; boundaryCount++; }
                    else { interior += d; interiorCount++; }
                }
            }

            double b = boundary / Math.Max(1, boundaryCount);
            double i = interior / Math.Max(1, interiorCount);
            return Math.Max(0, b - i);
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
