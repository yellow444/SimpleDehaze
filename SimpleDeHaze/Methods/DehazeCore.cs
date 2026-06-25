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
        /// Восстановление с защитой от перенасыщения. Раскладываем (I-A) на ахроматическую часть d
        /// (среднее по каналам) и хрому δ_c = (I_c-A_c) - d:
        ///     J_c = A_c + d/max(t,t_min) + δ_c/max(t, chromaFloor)
        /// Яркость (вуаль) убирается полностью (делитель t_min), а хрома усиливается слабее
        /// (делитель chromaFloor >= t_min) - поэтому при большой ω (малом t) цвет не 'выжигается'.
        /// chromaFloor <= t_min -> обычный DCP (δ и d делятся на один t). Клип [0,1].
        /// </summary>
        public static Mat Recover(Mat i01, Mat tSingle, MCvScalar a, double tmin, double chromaFloor = 0.4)
        {
            using var tLum = new Mat();
            using (var tm = new Mat(tSingle.Size, DepthType.Cv32F, 1)) { tm.SetTo(new MCvScalar(tmin)); CvInvoke.Max(tSingle, tm, tLum); }

            double cf = Math.Max(tmin, chromaFloor);
            using var tChroma = new Mat();
            if (cf > tmin)
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

        /// <summary>Зажать значения Mat в [0,1] на месте.</summary>
        public static void Clamp01(Mat m)
        {
            using (var z = new Mat(m.Size, DepthType.Cv32F, 1)) { z.SetTo(new MCvScalar(0)); CvInvoke.Max(m, z, m); }
            using (var o = new Mat(m.Size, DepthType.Cv32F, 1)) { o.SetTo(new MCvScalar(1)); CvInvoke.Min(m, o, m); }
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
