using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace SimpleDeHaze.Methods
{
    /// <summary>Многомасштабное слияние пирамид Лапласа (Ancuti, 2013). См. docs/methods/laplacian-pyramid-fusion.md.</summary>
    public sealed class PyramidFusionMethod : IDeHazeMethod
    {
        public string Name => "Multi-Scale Fusion (Laplacian Pyramid)";

        public string Description =>
            "Ancuti (2013). Дехейзинг без карты t - как слияние улучшенных версий кадра на разных масштабах.\n\n" +
            "Шаги:\n" +
            "1. Два 'входа': I_wb (баланс белого, убирает оттенок дымки) и I_enh (усиление контраста).\n" +
            "2. Весовые карты (контраст/резкость), нормировка Σ=1.\n" +
            "3. Пирамиды Лапласа входов и Гаусса весов, послойное смешивание:\n" +
            "       F_l = Σ_k G_l{w_k} - L_l{I_k}\n" +
            "4. Сборка результата из пирамиды.\n\n" +
            "Только свёртки/пирамиды, без матриц. Параметры: levels - число уровней; gain - усиление контраста.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("levels", "Уровней пирамиды", 3, 6, 5, 1, isInt: true),
            new ParamDef("gain",   "Усиление контраста", 1.0, 3.0, 1.8, search: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            int levels = Math.Max(2, (int)p["levels"]);
            double gain = p["gain"];

            using var I = DehazeCore.Normalize(input);
            using var i1 = WhiteBalance(I);          // де-тинт: убираем цветной оттенок дымки
            using var i2 = Enhance(i1, gain);        // видимость: гамма-затемнение вуали + контраст ОТ i1
            using var w1 = Weight(i1);
            using var w2 = Weight(i2);

            // нормировка весов: w_k / (w1 + w2 + eps)
            using var sum = new Mat();
            CvInvoke.Add(w1, w2, sum);
            CvInvoke.Add(sum, new ScalarArray(1e-6), sum);
            using var w1n = new Mat(); CvInvoke.Divide(w1, sum, w1n);
            using var w2n = new Mat(); CvInvoke.Divide(w2, sum, w2n);

            var lp1 = LaplacianPyramid(i1, levels);
            var lp2 = LaplacianPyramid(i2, levels);
            var gw1 = GaussianPyramid(w1n, levels);
            var gw2 = GaussianPyramid(w2n, levels);

            var blended = new List<Mat>();
            for (int l = 0; l < levels; l++)
            {
                using var a = Mul3(lp1[l], gw1[l]);
                using var bb = Mul3(lp2[l], gw2[l]);
                var f = new Mat(); CvInvoke.Add(a, bb, f);
                blended.Add(f);
            }

            using var collapsed = Collapse(blended);
            using var saturated = RestoreSaturation(collapsed, 1.25);   // gray-world WB обесцвечивает - возвращаем насыщенность

            DisposeAll(lp1); DisposeAll(lp2); DisposeAll(gw1); DisposeAll(gw2); DisposeAll(blended);
            return DeHazeCPU.Clip(saturated.Clone());
        }

        /// <summary>Восстановление насыщенности вокруг яркости: J = Y + s*(J - Y).</summary>
        private static Mat RestoreSaturation(Mat bgr, double s)
        {
            using var gray = new Mat(); CvInvoke.CvtColor(bgr, gray, ColorConversion.Bgr2Gray);
            using var gray3 = new Mat();
            using (var v = new VectorOfMat()) { v.Push(gray); v.Push(gray); v.Push(gray); CvInvoke.Merge(v, gray3); }
            var o = new Mat();
            CvInvoke.Subtract(bgr, gray3, o);          // хроматическая часть
            o.ConvertTo(o, DepthType.Cv32F, s);        // усиливаем
            CvInvoke.Add(o, gray3, o);                 // обратно к яркости
            return o;
        }

        private static Mat WhiteBalance(Mat i01)
        {
            var ch = i01.Split();
            double[] m = { CvInvoke.Mean(ch[0]).V0, CvInvoke.Mean(ch[1]).V0, CvInvoke.Mean(ch[2]).V0 };
            double avg = (m[0] + m[1] + m[2]) / 3.0;
            using var outv = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                var o = new Mat();
                ch[c].ConvertTo(o, DepthType.Cv32F, avg / Math.Max(m[c], 1e-6));
                outv.Push(o); o.Dispose(); ch[c].Dispose();
            }
            using var wb = new Mat(); CvInvoke.Merge(outv, wb);
            return DeHazeCPU.Clip(wb.Clone());
        }

        /// <summary>Второй вход 'видимость': гамма-затемнение (убирает молочную вуаль -> падает dark-channel) + контраст.</summary>
        private static Mat Enhance(Mat i01, double gain)
        {
            using var g = new Mat();
            CvInvoke.Pow(i01, 1.6, g);                                       // гамма >1 затемняет вуаль дымки
            var o = new Mat();
            g.ConvertTo(o, DepthType.Cv32F, gain, 0.5 * (1.0 - gain));       // (g-0.5)*gain + 0.5
            return DeHazeCPU.Clip(o);
        }

        /// <summary>Вес слияния: резкость (|Laplacian|) + насыщенность (красочные пиксели не 'вымываются' в серость).</summary>
        private static Mat Weight(Mat input)
        {
            using var gray = new Mat(); CvInvoke.CvtColor(input, gray, ColorConversion.Bgr2Gray);
            using var lap = new Mat(); CvInvoke.Laplacian(gray, lap, DepthType.Cv32F);
            var w = new Mat();
            using (var l2 = new Mat()) { CvInvoke.Multiply(lap, lap, l2); CvInvoke.Sqrt(l2, w); }   // |Laplacian| - резкость
            using (var sat = DehazeCore.Saturation(input)) CvInvoke.Add(w, sat, w);                  // + насыщенность
            CvInvoke.Add(w, new ScalarArray(0.1), w);   // ненулевой минимум
            return w;
        }

        private static List<Mat> GaussianPyramid(Mat img, int levels)
        {
            var gp = new List<Mat> { img.Clone() };
            for (int l = 1; l < levels; l++) { var d = new Mat(); CvInvoke.PyrDown(gp[l - 1], d); gp.Add(d); }
            return gp;
        }

        private static List<Mat> LaplacianPyramid(Mat img, int levels)
        {
            var gp = GaussianPyramid(img, levels);
            var lp = new List<Mat>();
            for (int l = 0; l < levels - 1; l++)
            {
                using var up = UpTo(gp[l + 1], gp[l].Size);
                var lap = new Mat(); CvInvoke.Subtract(gp[l], up, lap); lp.Add(lap);
            }
            lp.Add(gp[levels - 1].Clone());
            DisposeAll(gp);
            return lp;
        }

        private static Mat Collapse(List<Mat> lp)
        {
            var cur = lp[lp.Count - 1].Clone();
            for (int l = lp.Count - 2; l >= 0; l--)
            {
                using var up = UpTo(cur, lp[l].Size);
                var nxt = new Mat(); CvInvoke.Add(up, lp[l], nxt);
                cur.Dispose(); cur = nxt;
            }
            return cur;
        }

        /// <summary>PyrUp (~x2) с приведением к точному целевому размеру (нечётные уровни).</summary>
        private static Mat UpTo(Mat src, Size target)
        {
            var up = new Mat();
            CvInvoke.PyrUp(src, up);
            if (up.Width != target.Width || up.Height != target.Height)
            {
                var r = new Mat();
                CvInvoke.Resize(up, r, target, 0, 0, Inter.Linear);
                up.Dispose();
                return r;
            }
            return up;
        }

        private static Mat Mul3(Mat lap3, Mat w1)
        {
            using var w3 = new Mat();
            using (var v = new VectorOfMat()) { v.Push(w1); v.Push(w1); v.Push(w1); CvInvoke.Merge(v, w3); }
            var o = new Mat(); CvInvoke.Multiply(lap3, w3, o);
            return o;
        }

        private static void DisposeAll(List<Mat> list) { foreach (var m in list) m.Dispose(); }
    }
}
