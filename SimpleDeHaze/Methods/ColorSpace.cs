using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Перевод sRGB &lt;-&gt; линейный радианс.
    ///
    /// Атмосферная модель I = J·t + A·(1-t) линейна по СВЕТОВОМУ СИГНАЛУ, а в JPEG/PNG лежит
    /// sRGB с нелинейной передаточной кривой. Если считать тёмный канал и t прямо по sRGB, то
    /// вместо dark(I/A) получается dark(I/A)^(1/2.2): деление не коммутирует с гаммой, а минимум -
    /// коммутирует. Поэтому набор пикселей для A не меняется, а вот t систематически ЗАНИЖАЕТСЯ
    /// (дымки «как будто больше»), тем сильнее, чем она тоньше. Подробный разбор и таблица ошибки -
    /// docs/research/physics-linear-spectral.md.
    ///
    /// Здесь используется точная кусочная кривая sRGB (IEC 61966-2-1), а не приближение x^2.2.
    /// </summary>
    internal static class ColorSpace
    {
        private const double ToeSrgb = 0.04045, ToeLinear = 0.0031308;
        private const double Slope = 12.92, A = 0.055, Gamma = 2.4;

        private static readonly float[] SrgbToLinearLut = BuildLut();

        private static float[] BuildLut()
        {
            var lut = new float[256];
            for (int i = 0; i < 256; i++)
            {
                double c = i / 255.0;
                lut[i] = (float)(c <= ToeSrgb ? c / Slope : Math.Pow((c + A) / (1 + A), Gamma));
            }
            return lut;
        }

        /// <summary>
        /// 8-битный BGR -&gt; линейный BGR float [0,1] через таблицу на 256 значений (точно и быстро).
        /// Прямая замена <see cref="DehazeCore.Normalize"/> для физически корректного конвейера.
        /// </summary>
        public static Mat NormalizeLinear(Image<Bgr, byte> img)
        {
            int n = img.Mat.Rows * img.Mat.Cols * 3;
            var src = new byte[n];
            img.Mat.CopyTo(src);
            var dst = new float[n];
            for (int i = 0; i < n; i++) dst[i] = SrgbToLinearLut[src[i]];

            var m = new Mat(img.Mat.Rows, img.Mat.Cols, DepthType.Cv32F, 3);
            System.Runtime.InteropServices.Marshal.Copy(dst, 0, m.DataPointer, n);
            return m;
        }

        /// <summary>
        /// Склейка двух ветвей кусочной кривой без масок: sel = 1 там, где c &lt;= toe, иначе 0
        /// (Threshold, в отличие от Compare, штатно работает с многоканальными float-массивами).
        /// Возвращает hi + (lo - hi)*sel, освобождая lo и hi.
        /// </summary>
        private static Mat Blend(Mat c, Mat lo, Mat hi, double toe)
        {
            using (lo)
            using (hi)
            {
                using var sel = new Mat();
                CvInvoke.Threshold(c, sel, toe, 1.0, ThresholdType.BinaryInv);   // 1 при c <= toe
                using var diff = new Mat();
                CvInvoke.Subtract(lo, hi, diff);
                CvInvoke.Multiply(diff, sel, diff);
                var res = new Mat();
                CvInvoke.Add(hi, diff, res);
                return res;
            }
        }

        /// <summary>sRGB float [0,1] -&gt; линейный float [0,1] (любое число каналов). Новый Mat.</summary>
        public static Mat ToLinear(Mat srgb01)
        {
            using var c = srgb01.Clone();
            DehazeCore.Clamp01(c);

            var lo = new Mat();
            c.ConvertTo(lo, DepthType.Cv32F, 1.0 / Slope);                    // c / 12.92

            var hi = new Mat();
            c.ConvertTo(hi, DepthType.Cv32F, 1.0 / (1 + A), A / (1 + A));     // (c + 0.055) / 1.055
            CvInvoke.Pow(hi, Gamma, hi);

            var res = Blend(c, lo, hi, ToeSrgb);
            DehazeCore.Clamp01(res);
            return res;
        }

        /// <summary>Линейный float [0,1] -&gt; sRGB float [0,1] (обратная кривая). Новый Mat.</summary>
        public static Mat ToSrgb(Mat linear01)
        {
            using var c = linear01.Clone();
            DehazeCore.Clamp01(c);

            var lo = new Mat();
            c.ConvertTo(lo, DepthType.Cv32F, Slope);                          // 12.92 * c

            var hi = c.Clone();
            CvInvoke.Pow(hi, 1.0 / Gamma, hi);                                // c^(1/2.4)
            hi.ConvertTo(hi, DepthType.Cv32F, 1 + A, -A);                     // 1.055*x - 0.055

            var res = Blend(c, lo, hi, ToeLinear);
            DehazeCore.Clamp01(res);
            return res;
        }

        /// <summary>
        /// Нормализация входа под выбранное рабочее пространство: <paramref name="linear"/> = true -
        /// линейный радианс (физически корректно), false - прежнее поведение (sRGB как есть).
        /// </summary>
        public static Mat Normalize(Image<Bgr, byte> img, bool linear)
            => linear ? NormalizeLinear(img) : DehazeCore.Normalize(img);

        /// <summary>Возврат результата в sRGB, если работа велась в линейном пространстве.</summary>
        public static Mat Encode(Mat result01, bool linear)
            => linear ? ToSrgb(result01) : result01.Clone();

        /// <summary>
        /// Яркость по линейным первичным sRGB: Y = 0.2126R + 0.7152G + 0.0722B.
        /// В ЛИНЕЙНОМ пространстве модель дымки замкнута относительно любой линейной комбинации
        /// каналов, поэтому Y_I = t·Y_J + (1-t)·Y_A выполняется точно (в отличие от L из Lab).
        /// Вход - BGR float. Новый одноканальный Mat.
        /// </summary>
        public static Mat Luminance(Mat bgrLinear)
        {
            var ch = bgrLinear.Split();
            var y = new Mat();
            using (var b = new Mat()) using (var g = new Mat()) using (var r = new Mat())
            {
                ch[0].ConvertTo(b, DepthType.Cv32F, 0.0722);
                ch[1].ConvertTo(g, DepthType.Cv32F, 0.7152);
                ch[2].ConvertTo(r, DepthType.Cv32F, 0.2126);
                CvInvoke.Add(b, g, y);
                CvInvoke.Add(y, r, y);
            }
            foreach (var c in ch) c.Dispose();
            return y;
        }
    }
}
