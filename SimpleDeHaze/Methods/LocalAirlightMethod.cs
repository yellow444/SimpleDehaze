using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>DCP со spatially-varying атмосферным светом A(x). См. docs/methods/local-airlight-field.md.</summary>
    public sealed class LocalAirlightMethod : IDeHazeMethod
    {
        public string Name => "DCP - Local Airlight Field";

        public string Description =>
            "DCP с гладким полем атмосферного света A(x) вместо одной константы - для неравномерного\n" +
            "неба/боковой засветки.\n\n" +
            "Доверие к airlight q = D^2*(1-S)^1.5*exp(-k*|∇Y|) (светло в тёмном канале, малонасыщенно,\n" +
            "гладко). Поле A_c(x) = blur(q*I_c)/blur(q) большим окном. Дальше DCP по I/A(x) и\n" +
            "восстановление J = (I - A(x))/max(t,t_min) + A(x).\n\n" +
            "Параметры: ω - сила; patch; aRadius - окно поля A; refine/ε.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",   "ω - доля удаляемой дымки", 0.3, 0.95, 0.5, search: true),
            new ParamDef("patch",   "Патч тёмного канала",      1, 15, 5, 1, isInt: true),
            new ParamDef("aRadius", "Окно поля A(x)",           20, 200, 80, 1, isInt: true),
            new ParamDef("min",     "t_min - нижний порог t",   0.01, 0.5, 0.1),
            new ParamDef("refine",  "Радиус Guided Filter",     5, 120, 50, 1, isInt: true),
            new ParamDef("eps",     "ε - регуляризация GF",     1e-5, 1e-2, 1e-3, log: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"], tmin = p["min"]; int patch = (int)p["patch"], aR = (int)p["aRadius"], refine = (int)p["refine"]; double eps = p["eps"];
            var anc = new Point(-1, -1);

            using var I = DehazeCore.Normalize(input);
            using var D = DehazeCore.DarkChannel(I, patch);
            using var S = DehazeCore.Saturation(I);
            using var flat = DehazeCore.Flatness(I, 2.0);

            // q = D^2 - (1-S)^1.5 - flat
            using var q = new Mat();
            CvInvoke.Multiply(D, D, q);
            using (var oneMinusS = new Mat()) { S.ConvertTo(oneMinusS, DepthType.Cv32F, -1.0, 1.0); using var sm = new Mat(); CvInvoke.Pow(oneMinusS, 1.5, sm); CvInvoke.Multiply(q, sm, q); }
            CvInvoke.Multiply(q, flat, q);
            CvInvoke.Add(q, new ScalarArray(1e-4), q);

            var ks = new Size(2 * aR + 1, 2 * aR + 1);
            using var denA = new Mat(); CvInvoke.Blur(q, denA, ks, anc);

            // поле A_c(x) = blur(q*I_c)/blur(q)
            var ch = I.Split();
            var aField = new Mat[3];
            for (int c = 0; c < 3; c++)
            {
                using var qi = new Mat(); CvInvoke.Multiply(q, ch[c], qi);
                aField[c] = new Mat(); CvInvoke.Blur(qi, aField[c], ks, anc);
                CvInvoke.Divide(aField[c], denA, aField[c]);
            }

            // norm_c = I_c / A_c(x)  -> тёмный канал -> t = 1 - ω*D_A
            using var norm = new Mat();
            using (var vn = new VectorOfMat())
            {
                for (int c = 0; c < 3; c++) { using var nc = new Mat(); CvInvoke.Divide(ch[c], aField[c], nc); vn.Push(nc); }
                CvInvoke.Merge(vn, norm);
            }
            using var dA = DehazeCore.DarkChannel(norm, patch);
            using var tRaw = new Mat(); dA.ConvertTo(tRaw, DepthType.Cv32F, -omega, 1.0);
            using var t = new Mat(); XImgprocInvoke.GuidedFilter(I, tRaw, t, refine, eps);
            using var tc = new Mat();
            using (var tm = new Mat(t.Size, DepthType.Cv32F, 1)) { tm.SetTo(new MCvScalar(tmin)); CvInvoke.Max(t, tm, tc); }

            // J_c = (I_c - A_c(x)) / max(t,t_min) + A_c(x)
            using var outv = new VectorOfMat();
            for (int c = 0; c < 3; c++)
            {
                var jc = new Mat();
                CvInvoke.Subtract(ch[c], aField[c], jc);
                CvInvoke.Divide(jc, tc, jc);
                CvInvoke.Add(jc, aField[c], jc);
                outv.Push(jc); jc.Dispose();
            }
            using var J = new Mat(); CvInvoke.Merge(outv, J);
            foreach (var c in ch) c.Dispose();
            foreach (var a in aField) a.Dispose();
            return DeHazeCPU.Clip(J.Clone());
        }
    }
}
