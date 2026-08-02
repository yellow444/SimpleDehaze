using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Общий 'классический' конвейер Dark Channel Prior (одна карта t) с подключаемым уточнителем.
    /// Используется методами-вариантами (Fractional, Beltrami, MST, Matting), которые отличаются
    /// только способом уточнения карты пропускания t.
    /// </summary>
    internal static class DehazeCore
    {
        /// <summary>BGR byte -> BGR float [0,1].</summary>
        public static Mat Normalize(Image<Bgr, byte> img)
        {
            var m = new Mat();
            img.Mat.ConvertTo(m, DepthType.Cv32F, 1.0 / 255.0);
            return m;
        }

        /// <summary>Тёмный канал: min по B,G,R + эрозия окном patch.</summary>
        public static Mat DarkChannel(Mat i01, int patch)
        {
            var ch = i01.Split();
            var dc = new Mat();
            CvInvoke.Min(ch[0], ch[1], dc);
            CvInvoke.Min(dc, ch[2], dc);
            foreach (var c in ch) c.Dispose();
            int k = Math.Max(1, patch);
            using var elem = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(2 * k + 1, 2 * k + 1), new Point(-1, -1));
            CvInvoke.Erode(dc, dc, elem, new Point(-1, -1), 1, BorderType.Reflect101, default);
            return dc;
        }

        /// <summary>Атмосферный свет: среднее BGR по топ-доле самых ярких пикселей тёмного канала.</summary>
        public static MCvScalar Atmospheric(Mat i01, Mat dark, double topPercent)
        {
            int n = dark.Rows * dark.Cols;
            var d = new float[n]; dark.CopyTo(d);
            var ch = i01.Split();
            var b = new float[n]; var g = new float[n]; var r = new float[n];
            ch[0].CopyTo(b); ch[1].CopyTo(g); ch[2].CopyTo(r);
            foreach (var c in ch) c.Dispose();
            int k = Math.Max(1, (int)(n * topPercent));
            // top-k по тёмному каналу без полной сортировки: гистограммный порог, O(n) (d ∈ [0,1])
            const int B = 256;
            var hist = new int[B + 1];
            for (int i = 0; i < n; i++)
            {
                int bin = (int)(d[i] * B);
                hist[bin < 0 ? 0 : (bin > B ? B : bin)]++;
            }
            int need = k, thrBin = 0;
            for (int bin = B; bin >= 0; bin--) { need -= hist[bin]; if (need <= 0) { thrBin = bin; break; } }
            float thr = thrBin / (float)B;

            double sb = 0, sg = 0, sr = 0; int cnt = 0;
            for (int i = 0; i < n; i++)
                if (d[i] >= thr) { sb += b[i]; sg += g[i]; sr += r[i]; cnt++; }
            if (cnt == 0) cnt = 1;
            return new MCvScalar(sb / cnt, sg / cnt, sr / cnt);
        }

        /// <summary>Грубая карта пропускания: t = 1 - ω*darkChannel(I ./ A) (один канал).</summary>
        public static Mat RawTransmission(Mat i01, MCvScalar a, double omega, int patch)
        {
            var ch = i01.Split();
            double[] av = { a.V0, a.V1, a.V2 };
            using var norm = new Mat();
            using (var vec = new VectorOfMat())
            {
                for (int c = 0; c < 3; c++)
                {
                    var nc = new Mat();
                    CvInvoke.Divide(ch[c], new ScalarArray(av[c]), nc);
                    vec.Push(nc);
                    nc.Dispose();
                    ch[c].Dispose();
                }
                CvInvoke.Merge(vec, norm);
            }
            using var dc = DarkChannel(norm, patch);
            var t = new Mat();
            dc.ConvertTo(t, DepthType.Cv32F, -omega, 1.0);   // t = -ω*dc + 1
            return t;
        }

        /// <summary>
        /// Спектрально-адаптивная карта t: каждый канал даёт свою dark-channel оценку, а вклад канала
        /// взвешивается локальным контрастом и понижается в засвеченных областях.
        /// </summary>
        public static Mat SpectralTransmission(Mat i01, double omega, int patch, double tSky, out MCvScalar atmospheric)
        {
            using var dark = DarkChannel(i01, patch);
            atmospheric = Atmospheric(i01, dark, 0.001);
            double[] av = { atmospheric.V0, atmospheric.V1, atmospheric.V2 };

            var ch = i01.Split();
            var ks = new Size(2 * patch + 1, 2 * patch + 1);
            var anc = new Point(-1, -1);
            using var elem = CvInvoke.GetStructuringElement(ElementShape.Rectangle, ks, anc);

            using var wsum = new Mat(i01.Size, DepthType.Cv32F, 1);
            using var twsum = new Mat(i01.Size, DepthType.Cv32F, 1);
            wsum.SetTo(new MCvScalar(0));
            twsum.SetTo(new MCvScalar(0));

            for (int c = 0; c < 3; c++)
            {
                using var norm = new Mat();
                CvInvoke.Divide(ch[c], new ScalarArray(av[c]), norm);
                using var lmin = new Mat();
                CvInvoke.Erode(norm, lmin, elem, anc, 1, BorderType.Reflect101, default);
                using var tc = new Mat();
                lmin.ConvertTo(tc, DepthType.Cv32F, -omega, 1.0);

                using var mean = new Mat();
                CvInvoke.Blur(ch[c], mean, ks, anc);
                using var mean2 = new Mat();
                using (var sq = new Mat())
                {
                    CvInvoke.Multiply(ch[c], ch[c], sq);
                    CvInvoke.Blur(sq, mean2, ks, anc);
                }

                using var wc = new Mat();
                using (var m2 = new Mat())
                {
                    CvInvoke.Multiply(mean, mean, m2);
                    CvInvoke.Subtract(mean2, m2, wc);
                }
                Clamp01(wc);
                CvInvoke.Sqrt(wc, wc);
                CvInvoke.Add(wc, new ScalarArray(1e-3), wc);

                using (var anti = new Mat())
                {
                    mean.ConvertTo(anti, DepthType.Cv32F, 1.0 / 0.15, -0.85 / 0.15);
                    Clamp01(anti);
                    anti.ConvertTo(anti, DepthType.Cv32F, -0.8, 1.0);
                    CvInvoke.Multiply(wc, anti, wc);
                }

                using var wt = new Mat();
                CvInvoke.Multiply(wc, tc, wt);
                CvInvoke.Add(twsum, wt, twsum);
                CvInvoke.Add(wsum, wc, wsum);
            }

            foreach (var c in ch) c.Dispose();

            var t = new Mat();
            CvInvoke.Divide(twsum, wsum, t);
            using (var sky = SkyMask(i01))
                RaiseInSky(t, sky, tSky);
            Clamp01(t);
            return t;
        }

        /// <summary>
        /// Восстановление с защитой от перенасыщения. Раскладываем (I-A) на ахроматическую часть d
        /// (среднее по каналам) и хрому δ_c = (I_c-A_c) - d:
        ///     J_c = A_c + d/max(t,t_min) + δ_c/max(t, chromaFloor)
        /// Яркость (вуаль) убирается полностью (делитель t_min), а хрома усиливается слабее
        /// (делитель chromaFloor >= t_min) - поэтому при большой ω (малом t) цвет не 'выжигается'.
        /// chromaFloor <= t_min -> обычный DCP (δ и d делятся на один t). Клип [0,1].
        /// </summary>
        public static Mat Recover(Mat i01, Mat tSingle, MCvScalar a, double tmin, double chromaFloor = 0.4, Mat? conf = null)
        {
            using var tLum = new Mat();
            using (var tm = new Mat(tSingle.Size, DepthType.Cv32F, 1)) { tm.SetTo(new MCvScalar(tmin)); CvInvoke.Max(tSingle, tm, tLum); }

            double cf = Math.Max(tmin, chromaFloor);
            using var tChroma = new Mat();
            if (conf != null && chromaFloor > tmin)
            {
                // Per-pixel хрома-floor: спускается к t_min там, где плотная дымка (1-t велико) И есть
                // структура (conf велико) → настоящий цвет за дымкой возвращается полностью; на плоском
                // шуме (conf≈0) floor остаётся высоким и безопасным. cfMap = chromaFloor-(chromaFloor-tmin)·(1-t)·conf.
                using var confR = ResizeTo(conf, tSingle.Size);
                using var density = new Mat();
                tSingle.ConvertTo(density, DepthType.Cv32F, -1.0, 1.0);   // 1 - t
                Clamp01(density);
                CvInvoke.Multiply(density, confR, density);               // (1-t)·conf
                using var cfMap = new Mat();
                density.ConvertTo(cfMap, DepthType.Cv32F, -(chromaFloor - tmin), chromaFloor);
                double hardFloor = Math.Max(0.10, tmin);                  // жёсткий пол: хрома не делится сильнее яркости
                using (var lo = new Mat(cfMap.Size, DepthType.Cv32F, 1)) { lo.SetTo(new MCvScalar(hardFloor)); CvInvoke.Max(cfMap, lo, cfMap); }
                CvInvoke.Max(tSingle, cfMap, tChroma);
            }
            else if (cf > tmin)
            {
                using var tcm = new Mat(tSingle.Size, DepthType.Cv32F, 1);
                tcm.SetTo(new MCvScalar(cf));
                CvInvoke.Max(tSingle, tcm, tChroma);
            }
            else tLum.CopyTo(tChroma);   // хрома делится на тот же t -> классический DCP

            var ch = i01.Split();
            double[] av = { a.V0, a.V1, a.V2 };
            var d = new Mat[3];
            for (int c = 0; c < 3; c++) { d[c] = new Mat(); CvInvoke.Subtract(ch[c], new ScalarArray(av[c]), d[c]); ch[c].Dispose(); }

            using var dbar = new Mat();                                   // ахроматическая часть = mean_c (I_c-A_c)
            CvInvoke.Add(d[0], d[1], dbar); CvInvoke.Add(dbar, d[2], dbar);
            dbar.ConvertTo(dbar, DepthType.Cv32F, 1.0 / 3.0);
            using var lumPart = new Mat(); CvInvoke.Divide(dbar, tLum, lumPart);   // d / t_lum

            using var outv = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                using var delta = new Mat(); CvInvoke.Subtract(d[c], dbar, delta); // хрома δ_c
                var jc = new Mat(); CvInvoke.Divide(delta, tChroma, jc);           // δ_c / t_chroma (слабее)
                CvInvoke.Add(jc, lumPart, jc);
                CvInvoke.Add(jc, new ScalarArray(av[c]), jc);
                outv.Push(jc); jc.Dispose();
                d[c].Dispose();
            }
            using var J = new Mat();
            CvInvoke.Merge(outv, J);
            return DeHazeCPU.Clip(J.Clone());
        }

        /// <summary>
        /// Нижняя граница пропускания для СТАНДАРТНОЙ инверсии J_c = (I_c-A_c)/t + A_c.
        /// Из условия 0 &lt;= J_c &lt;= 1 следует
        ///     t &gt;= max_c max( (I_c-A_c)/(1-A_c), (A_c-I_c)/A_c ).
        /// Это известный boundary constraint: Meng, Wang, Duan, Xiang, Pan,
        /// "Efficient Image Dehazing with Boundary Constraint and Contextual Regularization", ICCV 2013
        /// (у авторов границы радианса C0/C1 настраиваемые, здесь взяты 0 и 1). Метод в репозитории
        /// использует эту границу как готовый результат, а не как собственный вклад.
        /// </summary>
        public static Mat BoundaryConstraint(Mat i01, MCvScalar a)
        {
            var ch = i01.Split();
            double[] av = { a.V0, a.V1, a.V2 };
            var bound = new Mat(i01.Size, DepthType.Cv32F, 1);
            bound.SetTo(new MCvScalar(0));

            for (int c = 0; c < 3; c++)
            {
                using var hi = new Mat();                                     // (I_c - A_c)/(1 - A_c)
                ch[c].ConvertTo(hi, DepthType.Cv32F, 1.0 / Math.Max(1e-4, 1.0 - av[c]), -av[c] / Math.Max(1e-4, 1.0 - av[c]));
                using var lo = new Mat();                                     // (A_c - I_c)/A_c
                ch[c].ConvertTo(lo, DepthType.Cv32F, -1.0 / Math.Max(1e-4, av[c]), av[c] / Math.Max(1e-4, av[c]));
                CvInvoke.Max(bound, hi, bound);
                CvInvoke.Max(bound, lo, bound);
                ch[c].Dispose();
            }
            Clamp01(bound);
            return bound;
        }

        /// <summary>
        /// Нижняя граница пропускания для ФАКТИЧЕСКОГО chroma-safe восстановления
        ///     J_c = A_c + d̄/max(t,m) + δ_c/max(t,q),   d̄ = mean_k(I_k-A_k),  δ_c = (I_c-A_c) - d̄,
        /// где m = t_min, q = chromaFloor >= m. Обычный boundary constraint к этой формуле НЕ применим:
        /// он выведен для деления всего (I-A) на один t.
        ///
        /// Разбор по трём областям (t - искомое пропускание):
        ///  • t >= q: обе части делятся на t, формула сводится к стандартной инверсии
        ///            => t >= t_box (см. <see cref="BoundaryConstraint"/>), и дополнительно t >= q;
        ///  • m &lt;= t &lt; q: хрома делится на КОНСТАНТУ q, значит e_c = A_c + δ_c/q фиксировано, и
        ///            J_c(t) = e_c + d̄/t монотонна по t. Отсюда для каждого канала:
        ///              d̄ > 0 (J убывает): J&lt;=1 ⇔ t >= d̄/(1-e_c) (если e_c&lt;1, иначе недостижимо);
        ///                                   J>=0 ⇔ t &lt;= d̄/(-e_c), но только если e_c &lt; 0;
        ///              d̄ &lt; 0 (J растёт):  J>=0 ⇔ t >= |d̄|/e_c   (если e_c>0, иначе недостижимо);
        ///                                   J&lt;=1 ⇔ t &lt;= |d̄|/(e_c-1), но только если e_c > 1;
        ///              d̄ = 0:              допустимо ⇔ 0 &lt;= e_c &lt;= 1 при любом t.
        ///            То есть допустимое множество здесь - ОТРЕЗОК [tLo, tHi], а не луч: при e_c вне
        ///            [0,1] слишком большое t тоже нарушает границы куба.
        ///  • t &lt; m: оба знаменателя - константы, J_c от t не зависит; но так как m &lt;= tLo не требуется
        ///            (при u=m модуль d̄/u только меньше), найденная граница остаётся достаточной.
        ///
        /// Итог: если отрезок второй области непуст и его левый конец лежит ниже q - это и есть
        /// минимальное допустимое t; иначе допустимость достижима только в первой области,
        /// т.е. t >= max(t_box, q). Отсюда следствие: chroma-safe восстановление обычно требует
        /// МЕНЬШЕГО t, чем стандартная инверсия, и проекция на t_box для него избыточно консервативна.
        ///
        /// Корректность проверяется в --mathtest перебором случайных наборов (I, A, m, q).
        /// </summary>
        public static Mat ChromaSafeLowerBound(Mat i01, MCvScalar a, double tmin, double chromaFloor)
        {
            double q = Math.Max(tmin, chromaFloor);
            if (q <= tmin + 1e-9) return BoundaryConstraint(i01, a);   // хрома делится на тот же t

            int rows = i01.Rows, cols = i01.Cols, n = rows * cols;
            var src = new float[n * 3];
            i01.CopyTo(src);
            var dst = new float[n];
            double[] av = { a.V0, a.V1, a.V2 };

            System.Threading.Tasks.Parallel.For(0, rows, y =>
            {
                int row = y * cols;
                for (int x = 0; x < cols; x++)
                {
                    int i = row + x, p = i * 3;
                    dst[i] = (float)ChromaSafeAt(src[p], src[p + 1], src[p + 2], av, tmin, q);
                }
            });

            return MatFromFloats(dst, rows, cols);
        }

        /// <summary>Скалярное ядро <see cref="ChromaSafeLowerBound"/> для одного пикселя (B,G,R).</summary>
        public static double ChromaSafeAt(double b, double g, double r, double[] av, double tmin, double q)
        {
            const double Big = 1e6, Eps = 1e-6;
            double d0 = b - av[0], d1 = g - av[1], d2 = r - av[2];
            double dbar = (d0 + d1 + d2) / 3.0;

            double tLo = 0, tHi = Big;
            bool feasible = true;

            for (int c = 0; c < 3; c++)
            {
                double dc = c == 0 ? d0 : c == 1 ? d1 : d2;
                double e = av[c] + (dc - dbar) / q;          // e_c = A_c + δ_c/q

                if (dbar > Eps)                               // J_c(t) убывает по t
                {
                    if (e < 1.0 - Eps) tLo = Math.Max(tLo, dbar / (1.0 - e));
                    else { feasible = false; break; }
                    if (e < -Eps) tHi = Math.Min(tHi, dbar / -e);
                }
                else if (dbar < -Eps)                         // J_c(t) растёт по t
                {
                    if (e > Eps) tLo = Math.Max(tLo, -dbar / e);
                    else { feasible = false; break; }
                    if (e > 1.0 + Eps) tHi = Math.Min(tHi, -dbar / (e - 1.0));
                }
                else if (e < -Eps || e > 1.0 + Eps)           // d̄ = 0: t ни на что не влияет
                {
                    feasible = false; break;
                }
            }

            // Яркостный знаменатель на самом деле u = max(t, m), поэтому в отрезок [tLo,tHi] должно
            // попадать именно u. Если tLo < m, то u = m при любом выборе t - и тогда допустимость
            // требует m <= tHi. Без этой проверки граница «работает» лишь там, где tLo >= m.
            double uLo = Math.Max(tLo, tmin);
            if (feasible && uLo <= tHi && tLo < q)
                return Math.Clamp(tLo, 0, 1);

            // иначе - первая область: t >= max(t_box, q)
            double box = 0;
            for (int c = 0; c < 3; c++)
            {
                double ic = c == 0 ? b : c == 1 ? g : r;
                box = Math.Max(box, (ic - av[c]) / Math.Max(1e-4, 1.0 - av[c]));
                box = Math.Max(box, (av[c] - ic) / Math.Max(1e-4, av[c]));
            }
            return Math.Clamp(Math.Max(box, q), 0, 1);
        }

        /// <summary>
        /// Доля пикселей, где карта <paramref name="t"/> лежит НИЖЕ допустимой границы
        /// <paramref name="lowerBound"/>, то есть восстановление гарантированно выйдет за RGB-куб
        /// и будет срезано клиппингом. Диагностическая метрика для статьи (0 = нарушений нет).
        /// </summary>
        public static double ViolationRate(Mat t, Mat lowerBound)
        {
            using var b = ResizeTo(lowerBound, t.Size);
            using var diff = new Mat();
            CvInvoke.Subtract(b, t, diff);                     // >0 там, где t < bound
            using var mask = new Mat();
            using (var z = new Mat(diff.Size, DepthType.Cv32F, 1))
            {
                z.SetTo(new MCvScalar(1e-4));
                CvInvoke.Compare(diff, z, mask, CmpType.GreaterThan);
            }
            return CvInvoke.CountNonZero(mask) / (double)Math.Max(1, t.Rows * t.Cols);
        }

        /// <summary>Классический DCP с подключаемым уточнителем refine(I, t_raw) -> t_refined.</summary>
        public static Mat Run(Image<Bgr, byte> img, double omega, int patch, double tmin, Func<Mat, Mat, Mat> refine, double chromaFloor = 0.4)
        {
            using var I = Normalize(img);
            using var dark = DarkChannel(I, patch);
            var a = Atmospheric(I, dark, 0.001);
            using var tRaw = RawTransmission(I, a, omega, patch);
            using var tRef = refine(I, tRaw);
            return Recover(I, tRef, a, tmin, chromaFloor);
        }

        /// <summary>
        /// Глобальная авто-коррекция тона: линейно растягивает яркость L (Lab) по перцентилям
        /// [pct, 1-pct]. Возвращает контраст/яркость, 'съеденные' затемнением классического DCP
        /// (per-channel восстановление часто даёт std(результата) ниже входа). Цвет a,b не трогает.
        /// <paramref name="strength"/> ∈ [0,1] - доля смешивания с исходным (0 = выкл, 1 = полностью).
        /// </summary>
        public static Mat RestoreTone(Mat bgr01, double strength, double pct = 0.01)
        {
            if (strength <= 1e-3) return bgr01.Clone();

            using var i8 = new Mat(); bgr01.ConvertTo(i8, DepthType.Cv8U, 255.0);
            using var lab = new Mat(); CvInvoke.CvtColor(i8, lab, ColorConversion.Bgr2Lab);
            var ch = lab.Split();

            int n = ch[0].Rows * ch[0].Cols;
            var L = new byte[n]; ch[0].CopyTo(L);
            var hist = new int[256];
            for (int i = 0; i < n; i++) hist[L[i]]++;
            int need = Math.Max(1, (int)(n * pct));
            int lo = 0, hi = 255, acc = 0;
            for (int b = 0; b < 256; b++) { acc += hist[b]; if (acc >= need) { lo = b; break; } }
            acc = 0;
            for (int b = 255; b >= 0; b--) { acc += hist[b]; if (acc >= need) { hi = b; break; } }
            if (hi - lo < 8) { lo = 0; hi = 255; }   // защита от вырожденного диапазона

            double scale = 255.0 / (hi - lo);
            ch[0].ConvertTo(ch[0], DepthType.Cv8U, scale, -lo * scale);   // L -> (L-lo)*255/(hi-lo)
            using (var v = new VectorOfMat(ch)) CvInvoke.Merge(v, lab);
            foreach (var c in ch) c.Dispose();

            using var outBgr = new Mat(); CvInvoke.CvtColor(lab, outBgr, ColorConversion.Lab2Bgr);
            using var stretched = new Mat(); outBgr.ConvertTo(stretched, DepthType.Cv32F, 1.0 / 255.0);

            if (strength >= 0.999) return stretched.Clone();
            var res = new Mat();
            CvInvoke.AddWeighted(bgr01, 1.0 - strength, stretched, strength, 0.0, res);
            return res;
        }

        /// <summary>
        /// Гаусс с большим σ через прореживание: при σ&gt;20 размываем уменьшенную копию и возвращаем
        /// обратно. Размытие большим σ - низкочастотное, поэтому разница незаметна, а скорость кратно выше
        /// (прямой Гаусс σ=130 на полном кадре очень дорог). Возвращает новый Mat.
        /// </summary>
        public static Mat FastGaussian(Mat src, double sigma)
        {
            var o = new Mat();
            if (sigma <= 20) { CvInvoke.GaussianBlur(src, o, new Size(0, 0), sigma); return o; }
            int f = Math.Max(1, (int)(sigma / 8.0));
            int w = Math.Max(1, src.Cols / f), h = Math.Max(1, src.Rows / f);
            using var small = new Mat();
            CvInvoke.Resize(src, small, new Size(w, h), 0, 0, Inter.Area);
            CvInvoke.GaussianBlur(small, small, new Size(0, 0), sigma / f);
            CvInvoke.Resize(small, o, src.Size, 0, 0, Inter.Linear);
            return o;
        }

        /// <summary>
        /// Мягкий roll-off светов (одноканальный float): значения выше колена <paramref name="knee"/>
        /// плавно сжимаются в [knee, 1) через 1-exp, вместо жёсткого клипа на 1 (который 'выжигает' света).
        /// Меняет <paramref name="y"/> на месте; значения ниже колена не трогает.
        /// </summary>
        public static void SoftHighlight(Mat y, double knee)
        {
            double span = 1.0 - knee;
            using var excess = new Mat(); CvInvoke.Subtract(y, new ScalarArray(knee), excess);
            using (var z = new Mat(y.Size, DepthType.Cv32F, 1)) { z.SetTo(new MCvScalar(0)); CvInvoke.Max(excess, z, excess); }  // max(y-k,0)
            CvInvoke.Multiply(excess, new ScalarArray(-1.0 / span), excess);
            CvInvoke.Exp(excess, excess);                                          // exp(-(y-k)/span)
            using var roll = new Mat();
            CvInvoke.Multiply(excess, new ScalarArray(-span), roll); CvInvoke.Add(roll, new ScalarArray(span), roll);  // span*(1-exp)
            using (var kn = new Mat(y.Size, DepthType.Cv32F, 1)) { kn.SetTo(new MCvScalar(knee)); CvInvoke.Min(y, kn, y); }   // min(y,k)
            CvInvoke.Add(y, roll, y);                                              // y' = min(y,k) + roll
        }

        /// <summary>
        /// Зажать значения Mat в [min,max] на месте. Число каналов берётся у самого Mat: раньше порог
        /// строился всегда одноканальным, и на трёхканальном входе OpenCV падал с
        /// «neither array op array nor array op scalar».
        /// </summary>
        public static void Clamp(Mat m, double min, double max)
        {
            int ch = m.NumberOfChannels;
            using (var z = new Mat(m.Size, DepthType.Cv32F, ch)) { z.SetTo(new MCvScalar(min, min, min, min)); CvInvoke.Max(m, z, m); }
            using (var o = new Mat(m.Size, DepthType.Cv32F, ch)) { o.SetTo(new MCvScalar(max, max, max, max)); CvInvoke.Min(m, o, m); }
        }

        /// <summary>Зажать значения Mat в [0,1] на месте.</summary>
        public static void Clamp01(Mat m) => Clamp(m, 0, 1);

        /// <summary>
        /// Быстрая цветосохраняющая тон-коррекция: строит перцентильные авто-уровни по серой яркости и
        /// применяет их как мультипликативный gain к BGR. Это дешевле Lab-конверсии и достаточно для
        /// контурного/роботического режима, где важнее стабильная геометрия, чем перцептивная полировка.
        /// </summary>
        public static Mat RestoreToneFast(Mat bgr01, double strength, double pct = 0.01, double maxGain = 1.45)
        {
            if (strength <= 1e-3) return bgr01.Clone();

            using var bgr8 = new Mat();
            bgr01.ConvertTo(bgr8, DepthType.Cv8U, 255.0);
            using var gray8 = new Mat();
            CvInvoke.CvtColor(bgr8, gray8, ColorConversion.Bgr2Gray);

            int n = gray8.Rows * gray8.Cols;
            var y8 = new byte[n];
            gray8.CopyTo(y8);
            var hist = new int[256];
            for (int i = 0; i < n; i++) hist[y8[i]]++;

            int need = Math.Max(1, (int)(n * pct));
            int lo = 0, hi = 255, acc = 0;
            for (int b = 0; b < 256; b++) { acc += hist[b]; if (acc >= need) { lo = b; break; } }
            acc = 0;
            for (int b = 255; b >= 0; b--) { acc += hist[b]; if (acc >= need) { hi = b; break; } }
            if (hi - lo < 8) return bgr01.Clone();

            using var gray = new Mat();
            CvInvoke.CvtColor(bgr01, gray, ColorConversion.Bgr2Gray);
            using var stretched = new Mat();
            gray.ConvertTo(stretched, DepthType.Cv32F, 255.0 / (hi - lo), -lo / (double)(hi - lo));
            Clamp01(stretched);

            using var denom = new Mat();
            CvInvoke.Add(gray, new ScalarArray(1e-4), denom);
            using var gain = new Mat();
            CvInvoke.Divide(stretched, denom, gain);
            if (strength < 0.999)
                gain.ConvertTo(gain, DepthType.Cv32F, strength, 1.0 - strength);
            Clamp(gain, 0.50, maxGain);

            var ch = bgr01.Split();
            using var outv = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                CvInvoke.Multiply(ch[c], gain, ch[c]);
                outv.Push(ch[c]);
            }
            using var merged = new Mat();
            CvInvoke.Merge(outv, merged);
            foreach (var c in ch) c.Dispose();
            return DeHazeCPU.Clip(merged.Clone());
        }

        /// <summary>
        /// Перцептивное усиление в Lab: CLAHE по L, лёгкий unsharp по L и вибранс a/b.
        /// Цветность усиливается сильнее у малонасыщенных зон и слабее у уже ярких цветов.
        /// </summary>
        public static Mat LabEnhance(Mat bgr01, double clip, int tiles, double vibrance, double detail = 0.0, Mat? gate = null)
        {
            using var i8 = new Mat();
            bgr01.ConvertTo(i8, DepthType.Cv8U, 255.0);
            using var lab = new Mat();
            CvInvoke.CvtColor(i8, lab, ColorConversion.Bgr2Lab);
            var ch = lab.Split();

            int grid = Math.Max(2, tiles);
            if (clip > 1e-3)
                CvInvoke.CLAHE(ch[0], clip, new Size(grid, grid), ch[0]);

            if (detail > 1e-3)
            {
                if (gate == null)
                {
                    using var blur = new Mat();
                    CvInvoke.GaussianBlur(ch[0], blur, new Size(0, 0), 1.0);
                    CvInvoke.AddWeighted(ch[0], 1.0 + detail, blur, -detail, 0.0, ch[0]);
                }
                else
                {
                    // Пространственно-гейтованная резкость: остаток high=L-blur ограничиваем по амплитуде
                    // (анти-хруст) и умножаем на gate (structure·transmission), затем добавляем. Так на
                    // ровных/плотных зонах (gate≈0) микроконтраст не раздувает шум, а на реальных кромках работает.
                    using var Lf = new Mat(); ch[0].ConvertTo(Lf, DepthType.Cv32F);   // 0..255
                    using var blur = new Mat(); CvInvoke.GaussianBlur(Lf, blur, new Size(0, 0), 1.0);
                    using var high = new Mat(); CvInvoke.Subtract(Lf, blur, high);
                    Clamp(high, -12.0, 12.0);                                          // ±12 L-единиц
                    using var g = ResizeTo(gate, high.Size);
                    CvInvoke.Multiply(high, g, high);
                    using (var add = new Mat()) { high.ConvertTo(add, DepthType.Cv32F, detail); CvInvoke.Add(Lf, add, Lf); }
                    Clamp(Lf, 0.0, 255.0);
                    Lf.ConvertTo(ch[0], DepthType.Cv8U);
                }
            }

            if (Math.Abs(vibrance) > 1e-3)
            {
                using var af = new Mat();
                using var bf = new Mat();
                ch[1].ConvertTo(af, DepthType.Cv32F, 1.0, -128.0);
                ch[2].ConvertTo(bf, DepthType.Cv32F, 1.0, -128.0);
                using var chroma = new Mat();
                using (var a2 = new Mat())
                using (var b2 = new Mat())
                {
                    CvInvoke.Multiply(af, af, a2);
                    CvInvoke.Multiply(bf, bf, b2);
                    CvInvoke.Add(a2, b2, chroma);
                }
                CvInvoke.Sqrt(chroma, chroma);

                using var factor = new Mat();
                chroma.ConvertTo(factor, DepthType.Cv32F, -1.0 / 128.0, 1.0);
                Clamp01(factor);
                factor.ConvertTo(factor, DepthType.Cv32F, vibrance, 1.0);
                Clamp(factor, 0.5, 1.8);

                CvInvoke.Multiply(af, factor, af);
                CvInvoke.Multiply(bf, factor, bf);
                af.ConvertTo(ch[1], DepthType.Cv8U, 1.0, 128.0);
                bf.ConvertTo(ch[2], DepthType.Cv8U, 1.0, 128.0);
            }

            using (var v = new VectorOfMat(ch))
                CvInvoke.Merge(v, lab);
            foreach (var c in ch) c.Dispose();

            using var outBgr = new Mat();
            CvInvoke.CvtColor(lab, outBgr, ColorConversion.Lab2Bgr);
            var res = new Mat();
            outBgr.ConvertTo(res, DepthType.Cv32F, 1.0 / 255.0);
            return res;
        }

        /// <summary>
        /// Ограничить усиление цветности результата относительно входа, не меняя яркостную структуру:
        /// result = gray + scale*(result-gray), если colorfulness(result)/colorfulness(input) выше maxRatio.
        /// </summary>
        public static Mat LimitColorfulness(Mat bgr01, Mat inputBgr8, double maxRatio, double meanHazeDensity = 0.0)
        {
            var result = bgr01.Clone();
            using var result8 = new Mat();
            result.ConvertTo(result8, DepthType.Cv8U, 255.0);

            // Потолок меряется относительно ВХОДА, а у задымлённого входа цветность занижена. Ослабляем
            // потолок пропорционально измеренной плотности дымки, чтобы честно восстановленный цвет за
            // дымкой не гасился в серый; на чистых кадрах (meanHazeDensity≈0) поведение прежнее.
            double effMaxRatio = Math.Min(2.2, maxRatio * (1.0 + 0.6 * meanHazeDensity));
            double ratio = Metrics.Colorfulness(result8) / (Metrics.Colorfulness(inputBgr8) + 1e-6);
            if (ratio <= effMaxRatio)
                return result;

            double scale = effMaxRatio / ratio;
            using var gray = new Mat();
            CvInvoke.CvtColor(result, gray, ColorConversion.Bgr2Gray);
            using var gray3 = new Mat();
            using (var channels = new VectorOfMat())
            {
                channels.Push(gray);
                channels.Push(gray);
                channels.Push(gray);
                CvInvoke.Merge(channels, gray3);
            }

            using var chroma = new Mat();
            CvInvoke.Subtract(result, gray3, chroma);
            chroma.ConvertTo(chroma, DepthType.Cv32F, scale);
            CvInvoke.Add(gray3, chroma, result);
            return DeHazeCPU.Clip(result);
        }

        /// <summary>Быстро приглушить/усилить хрому без оценки метрики цветности: result = gray + scale*(result-gray).</summary>
        public static Mat ScaleChroma(Mat bgr01, double scale)
        {
            if (Math.Abs(scale - 1.0) <= 1e-3)
                return bgr01.Clone();

            var result = bgr01.Clone();
            using var gray = new Mat();
            CvInvoke.CvtColor(result, gray, ColorConversion.Bgr2Gray);
            using var gray3 = new Mat();
            using (var channels = new VectorOfMat())
            {
                channels.Push(gray);
                channels.Push(gray);
                channels.Push(gray);
                CvInvoke.Merge(channels, gray3);
            }

            using var chroma = new Mat();
            CvInvoke.Subtract(result, gray3, chroma);
            chroma.ConvertTo(chroma, DepthType.Cv32F, scale);
            CvInvoke.Add(gray3, chroma, result);
            return DeHazeCPU.Clip(result);
        }

        /// <summary>
        /// Восстановление цвета за дымкой, ТОНОСОХРАНЯЮЩЕЕ: усиление насыщенности идёт в Lab a/b, то есть
        /// растёт только МОДУЛЬ хромы, а направление (оттенок = atan2(b,a)) не меняется — зелень не уплывает
        /// в жёлтый/циан. Вес per-pixel w = clamp01(mask·conf): mask = плотность дымки (1-t) возвращает
        /// цвет именно в задымлённых зонах, а structure-confidence <paramref name="conf"/> (фрактальная
        /// насыщенность) НЕ даёт красить плоский туман/шум фантомным цветом. Потолок по запасу (127/|chroma|)
        /// защищает уже насыщенные пиксели от постеризации. conf=null → гейт только по маске. Клип [0,1].
        /// </summary>
        public static Mat ScaleChromaByMask(Mat bgr01, Mat mask01, double gain, Mat? conf = null)
        {
            if (gain <= 1e-3) return bgr01.Clone();

            using var i8 = new Mat();
            bgr01.ConvertTo(i8, DepthType.Cv8U, 255.0);
            using var lab = new Mat();
            CvInvoke.CvtColor(i8, lab, ColorConversion.Bgr2Lab);
            var ch = lab.Split();   // L, a, b (8U)

            // вес w = clamp01(mask · (conf ?? 1))
            using var w = ResizeTo(mask01, bgr01.Size);
            if (conf != null) { using var cf = ResizeTo(conf, bgr01.Size); CvInvoke.Multiply(w, cf, w); }
            Clamp01(w);

            using var af = new Mat(); using var bf = new Mat();
            ch[1].ConvertTo(af, DepthType.Cv32F, 1.0, -128.0);   // a' = a-128
            ch[2].ConvertTo(bf, DepthType.Cv32F, 1.0, -128.0);   // b' = b-128

            // модуль хромы для потолка по запасу
            using var chroma = new Mat();
            using (var a2 = new Mat()) using (var b2 = new Mat())
            {
                CvInvoke.Multiply(af, af, a2);
                CvInvoke.Multiply(bf, bf, b2);
                CvInvoke.Add(a2, b2, chroma);
            }
            CvInvoke.Sqrt(chroma, chroma);
            CvInvoke.Add(chroma, new ScalarArray(1e-3), chroma);

            // factor = 1 + gain·w, но не больше 127/|chroma| (чтобы |a'|,|b'| ≤ 127 — без постеризации)
            using var factor = new Mat();
            w.ConvertTo(factor, DepthType.Cv32F, gain, 1.0);
            using (var num = new Mat(chroma.Size, DepthType.Cv32F, 1))
            {
                num.SetTo(new MCvScalar(127.0));
                using var cap = new Mat();
                CvInvoke.Divide(num, chroma, cap);
                CvInvoke.Min(factor, cap, factor);
            }

            CvInvoke.Multiply(af, factor, af);
            CvInvoke.Multiply(bf, factor, bf);
            af.ConvertTo(ch[1], DepthType.Cv8U, 1.0, 128.0);
            bf.ConvertTo(ch[2], DepthType.Cv8U, 1.0, 128.0);

            // лёгкий медианный фильтр по хроме a,b: убирает ЦВЕТНОЙ спекл, усиленный вместе с сигналом,
            // не трогая яркость L и не размывая структуру (у зрения низкая острота по цвету — почти бесплатно).
            CvInvoke.MedianBlur(ch[1], ch[1], 3);
            CvInvoke.MedianBlur(ch[2], ch[2], 3);

            using (var v = new VectorOfMat(ch)) CvInvoke.Merge(v, lab);
            foreach (var c in ch) c.Dispose();
            using var outBgr = new Mat();
            CvInvoke.CvtColor(lab, outBgr, ColorConversion.Lab2Bgr);
            var result = new Mat();
            outBgr.ConvertTo(result, DepthType.Cv32F, 1.0 / 255.0);
            return DeHazeCPU.Clip(result);
        }

        /// <summary>Копия/ресайз карты под нужный размер (возвращает новый Mat; вызывающий освобождает).</summary>
        public static Mat ResizeTo(Mat m, Size size)
        {
            var r = new Mat();
            if (m.Size.Equals(size)) m.CopyTo(r);
            else CvInvoke.Resize(m, r, size);
            return r;
        }

        public static Mat BilateralDenoise(Mat bgr01, double strength)
        {
            if (strength <= 0.01)
                return bgr01.Clone();

            using var bgr8 = new Mat();
            bgr01.ConvertTo(bgr8, DepthType.Cv8U, 255.0);
            using var filtered8 = new Mat();
            int d = Math.Max(3, ((int)Math.Round(strength) * 2 + 1) | 1);
            double sigmaColor = 12.0 + strength * 8.0;
            double sigmaSpace = 2.0 + strength * 1.4;
            CvInvoke.BilateralFilter(bgr8, filtered8, d, sigmaColor, sigmaSpace, BorderType.Reflect101);

            var result = new Mat();
            filtered8.ConvertTo(result, DepthType.Cv32F, 1.0 / 255.0);
            return result;
        }

        /// <summary>
        /// Маска 'неба/пересвета' ∈ [0,1]: велика там, где ярко (V=max_c I_c велико) и малонасыщенно
        /// (S низкое) - типичные зоны, где dark-channel ломается. Используется, чтобы дехейзить их слабее.
        /// </summary>
        public static Mat SkyMask(Mat i01)
        {
            var ch = i01.Split();
            using var V = new Mat(); CvInvoke.Max(ch[0], ch[1], V); CvInvoke.Max(V, ch[2], V);
            using var mn = new Mat(); CvInvoke.Min(ch[0], ch[1], mn); CvInvoke.Min(mn, ch[2], mn);
            foreach (var c in ch) c.Dispose();
            using var S = new Mat();
            using (var d = new Mat()) using (var ve = new Mat()) { CvInvoke.Subtract(V, mn, d); CvInvoke.Add(V, new ScalarArray(1e-6), ve); CvInvoke.Divide(d, ve, S); }   // S=(V-min)/V
            using var bright = new Mat(); V.ConvertTo(bright, DepthType.Cv32F, 1.0 / 0.3, -0.6 / 0.3); Clamp01(bright);   // ярко: clamp((V-0.6)/0.3)
            using var lowsat = new Mat(); S.ConvertTo(lowsat, DepthType.Cv32F, -1.0 / 0.2, 0.25 / 0.2); Clamp01(lowsat);  // мало S: clamp((0.25-S)/0.2)
            var sky = new Mat(); CvInvoke.Multiply(bright, lowsat, sky);
            return sky;
        }

        /// <summary>Поднять карту t к <paramref name="tSky"/> в зонах <paramref name="sky"/>: t <- t + sky*(tSky - t).</summary>
        public static void RaiseInSky(Mat t, Mat sky, double tSky)
        {
            using var diff = new Mat(); t.ConvertTo(diff, DepthType.Cv32F, -1.0, tSky);   // tSky - t
            using var add = new Mat(); CvInvoke.Multiply(sky, diff, add);
            CvInvoke.Add(t, add, t);
        }

        /// <summary>Минимум по каналам B,G,R (без эрозии) -> 1 канал.</summary>
        public static Mat MinChannel(Mat i01)
        {
            var ch = i01.Split();
            var m = new Mat();
            CvInvoke.Min(ch[0], ch[1], m);
            CvInvoke.Min(m, ch[2], m);
            foreach (var c in ch) c.Dispose();
            return m;
        }

        /// <summary>Поканальное деление I_c / A_c -> 3 канала.</summary>
        public static Mat NormByA(Mat i01, MCvScalar a)
        {
            var ch = i01.Split();
            double[] av = { a.V0, a.V1, a.V2 };
            using var vec = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                var nc = new Mat();
                CvInvoke.Divide(ch[c], new ScalarArray(av[c]), nc);
                vec.Push(nc); nc.Dispose(); ch[c].Dispose();
            }
            var norm = new Mat();
            CvInvoke.Merge(vec, norm);
            return norm;
        }

        /// <summary>Модуль градиента яркости (Sobel), 1 канал float.</summary>
        public static Mat GradMag(Mat i01)
        {
            using var gray = new Mat();
            CvInvoke.CvtColor(i01, gray, ColorConversion.Bgr2Gray);
            using var gx = new Mat(); using var gy = new Mat();
            CvInvoke.Sobel(gray, gx, DepthType.Cv32F, 1, 0, 3);
            CvInvoke.Sobel(gray, gy, DepthType.Cv32F, 0, 1, 3);
            var mag = new Mat();
            using (var gx2 = new Mat()) { CvInvoke.Multiply(gx, gx, gx2); using var gy2 = new Mat(); CvInvoke.Multiply(gy, gy, gy2); CvInvoke.Add(gx2, gy2, mag); }
            CvInvoke.Sqrt(mag, mag);
            return mag;
        }

        /// <summary>'Плоскостность' weight = exp(-k*|∇Y|) ∈ (0,1]: ~1 на гладком, ~0 на краях.</summary>
        public static Mat Flatness(Mat i01, double k)
        {
            var w = GradMag(i01);
            CvInvoke.Multiply(w, new ScalarArray(-k), w);
            CvInvoke.Exp(w, w);
            return w;
        }

        /// <summary>Насыщенность S из HSV (float [0,1]).</summary>
        public static Mat Saturation(Mat i01)
        {
            using var hsv = new Mat();
            CvInvoke.CvtColor(i01, hsv, ColorConversion.Bgr2Hsv);
            var ch = hsv.Split();
            var s = ch[1].Clone();
            foreach (var c in ch) c.Dispose();
            return s;
        }

        /// <summary>Светлый канал: max по B,G,R + дилатация (max-фильтр) окном patch.</summary>
        public static Mat BrightChannel(Mat i01, int patch)
        {
            var ch = i01.Split();
            var b = new Mat();
            CvInvoke.Max(ch[0], ch[1], b);
            CvInvoke.Max(b, ch[2], b);
            foreach (var c in ch) c.Dispose();
            int k = Math.Max(1, patch);
            using var elem = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(2 * k + 1, 2 * k + 1), new Point(-1, -1));
            CvInvoke.Dilate(b, b, elem, new Point(-1, -1), 1, BorderType.Reflect101, default);
            return b;
        }

        /// <summary>Создать одноканальный float-Mat из массива (row-major).</summary>
        public static Mat MatFromFloats(float[] data, int rows, int cols)
        {
            var m = new Mat(rows, cols, DepthType.Cv32F, 1);
            System.Runtime.InteropServices.Marshal.Copy(data, 0, m.DataPointer, data.Length);
            return m;
        }

        /// <summary>Сдвиг карты на (dx,dy) с реплицированной границей: result(x,y) = m(x-dx, y-dy).</summary>
        public static Mat Shift(Mat m, int dx, int dy)
        {
            using var b = new Mat();
            CvInvoke.CopyMakeBorder(m, b, 1, 1, 1, 1, BorderType.Replicate);
            return new Mat(b, new Rectangle(1 - dx, 1 - dy, m.Cols, m.Rows)).Clone();
        }
    }
}
