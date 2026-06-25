using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>Adaptive Soft Dark Channel: мягкий минимум (soft-min) вместо жёсткого. См. docs/methods/adaptive-soft-dark-channel.md.</summary>
    public sealed class AdaptiveSoftDcpMethod : IDeHazeMethod
    {
        public string Name => "DCP - Adaptive Soft Dark Channel";

        public string Description =>
            "DCP, но жёсткий min по окну заменён soft-min (Больцман-взвешенное среднее):\n" +
            "    D(x) = Σ_y w_y I_min(y) / Σ_y w_y,   w_y = exp(-I_min(y)/τ).\n\n" +
            "Малое τ -> почти обычный min; большее τ -> устойчивее к шуму/JPEG/одиночным тёмным пикселям,\n" +
            "карта t менее пятнистая. Реализовано через box-фильтр числителя/знаменателя.\n\n" +
            "Параметры: ω - сила; τ - мягкость min; patch - окно; refine/ε - Guided Filter.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega", "ω - доля удаляемой дымки", 0.3, 0.95, 0.5, search: true),
            new ParamDef("patch", "Патч окна",                1, 15, 7, 1, isInt: true),
            new ParamDef("tau",   "τ - мягкость min",         0.01, 0.3, 0.05, log: true, search: true),
            new ParamDef("min",   "t_min - нижний порог t",   0.01, 0.5, 0.1),
            new ParamDef("refine","Радиус Guided Filter",     5, 120, 50, 1, isInt: true),
            new ParamDef("eps",   "ε - регуляризация GF",     1e-5, 1e-2, 1e-3, log: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"], tau = p["tau"], tmin = p["min"];
            int patch = (int)p["patch"], refine = (int)p["refine"]; double eps = p["eps"];

            using var I = DehazeCore.Normalize(input);
            using var darkA = DehazeCore.DarkChannel(I, patch);
            var A = DehazeCore.Atmospheric(I, darkA, 0.001);
            using var norm = DehazeCore.NormByA(I, A);
            using var Imin = DehazeCore.MinChannel(norm);

            // soft-min по окну: D = blur(w*Imin)/blur(w), w = exp(-Imin/τ)
            using var w = new Mat(); Imin.ConvertTo(w, DepthType.Cv32F, -1.0 / tau); CvInvoke.Exp(w, w);
            using var wi = new Mat(); CvInvoke.Multiply(w, Imin, wi);
            var ks = new Size(2 * patch + 1, 2 * patch + 1);
            using var numB = new Mat(); CvInvoke.Blur(wi, numB, ks, new Point(-1, -1));
            using var denB = new Mat(); CvInvoke.Blur(w, denB, ks, new Point(-1, -1));
            CvInvoke.Add(denB, new ScalarArray(1e-6), denB);
            using var D = new Mat(); CvInvoke.Divide(numB, denB, D);

            using var t = new Mat(); D.ConvertTo(t, DepthType.Cv32F, -omega, 1.0);
            using var tRef = new Mat(); XImgprocInvoke.GuidedFilter(I, t, tRef, refine, eps);
            return DehazeCore.Recover(I, tRef, A, tmin);
        }
    }
}
