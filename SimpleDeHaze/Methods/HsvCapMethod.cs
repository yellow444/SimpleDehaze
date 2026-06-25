using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// DCP через HSV - Color Attenuation Prior (Zhu et al., 2015): глубина d = θ0 + θ1*V + θ2*S,
    /// трансмиссия t = exp(-β*d). См. docs/DCP/dcp-hsv.md.
    /// </summary>
    public sealed class HsvCapMethod : IDeHazeMethod
    {
        public string Name => "Color Attenuation Prior (HSV)";

        public string Description =>
            "Color Attenuation Prior (Zhu, Mao, Wang, 2015). DCP в пространстве HSV.\n\n" +
            "Идея: дымка повышает яркость V и снижает насыщенность S, поэтому глубина растёт с (V - S).\n\n" +
            "Шаги:\n" +
            "1. BGR -> HSV, берём S и V.\n" +
            "2. Глубина d = 0.1218 + 0.9597*V - 0.7802*S.\n" +
            "3. min-фильтр (снять текстуру) + Guided Filter (гайд - V).\n" +
            "4. t = exp(-β*d), затем t = max(t, t_min).\n" +
            "5. J = (I - A) / t + A.\n\n" +
            "Мягче к небу и белым объектам, тон (H) не трогаем - цвет стабильнее.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("beta",   "β - коэф. рассеяния",      0.1,  3.0,  1.0, search: true),
            new ParamDef("rmin",   "Радиус min-фильтра",       1,    25,   7,  1, isInt: true),
            new ParamDef("rguide", "Радиус Guided Filter",     5,    120,  60, 1, isInt: true),
            new ParamDef("eps",    "ε - регуляризация GF",     1e-5, 1e-2, 1e-3, log: true),
            new ParamDef("min",    "t_min - нижний порог t",   0.01, 0.5,  0.1),
        };

        // коэффициенты Color Attenuation Prior (Zhu, Mao, Wang, 2015)
        private const double Theta0 = 0.121779, Theta1 = 0.959710, Theta2 = -0.780245;

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            float beta = (float)p["beta"];
            int rMin = (int)p["rmin"];
            int rGuide = (int)p["rguide"];
            double eps = p["eps"];
            float tMin = (float)p["min"];

            // нормализованный BGR в [0,1]
            using var I = new Mat();
            input.Mat.ConvertTo(I, DepthType.Cv32F, 1.0 / 255.0);

            // BGR -> HSV (для float: H в [0,360), S,V в [0,1])
            using var hsv = new Mat();
            CvInvoke.CvtColor(I, hsv, ColorConversion.Bgr2Hsv);
            var hsvCh = hsv.Split();   // [0]=H, [1]=S, [2]=V
            using var S = hsvCh[1];
            using var V = hsvCh[2];
            hsvCh[0].Dispose();

            // глубина d = θ0 + θ1*V + θ2*S
            using var d = new Mat();
            CvInvoke.AddWeighted(V, Theta1, S, Theta2, Theta0, d, DepthType.Cv32F);

            // снять текстуру: локальный минимум (эрозия)
            using var elem = CvInvoke.GetStructuringElement(
                ElementShape.Rectangle, new System.Drawing.Size(2 * rMin + 1, 2 * rMin + 1), new System.Drawing.Point(-1, -1));
            CvInvoke.Erode(d, d, elem, new System.Drawing.Point(-1, -1), 1, BorderType.Reflect101, default);

            // уточнить карту глубины направляющим фильтром (гайд - яркость V)
            using var dRef = new Mat();
            XImgprocInvoke.GuidedFilter(V, d, dRef, rGuide, eps);

            // трансмиссия t = exp(-beta * d), затем t = max(t, tMin)
            using var t = new Mat();
            using (var negBetaD = new Mat())
            {
                dRef.ConvertTo(negBetaD, DepthType.Cv32F, -beta);
                CvInvoke.Exp(negBetaD, t);
            }
            using (var tm = new Mat(t.Size, DepthType.Cv32F, 1))
            {
                tm.SetTo(new MCvScalar(tMin));
                CvInvoke.Max(t, tm, t);
            }

            // атмосферный свет: ярчайшие пиксели I среди самых 'глубоких' (макс. dRef)
            MCvScalar A = AtmosphericLightByDepth(I, dRef, 0.001);

            // восстановление: J_c = (I_c - A_c)/t + A_c
            var src = I.Split();
            double[] a = { A.V0, A.V1, A.V2 };
            using var outCh = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                var jc = new Mat();
                CvInvoke.Subtract(src[c], new ScalarArray(a[c]), jc);
                CvInvoke.Divide(jc, t, jc);
                CvInvoke.Add(jc, new ScalarArray(a[c]), jc);
                outCh.Push(jc);
                jc.Dispose();
                src[c].Dispose();
            }
            using var J = new Mat();
            CvInvoke.Merge(outCh, J);
            return DeHazeCPU.Clip(J.Clone());
        }

        private static MCvScalar AtmosphericLightByDepth(Mat i01, Mat depth, double topPercent)
        {
            int n = depth.Rows * depth.Cols;
            var dData = new float[n];
            depth.CopyTo(dData);

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
