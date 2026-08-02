using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>Multi-Scale DCP Fusion: тёмный канал на двух радиусах, смешивание по краям. См. docs/methods/multiscale-dcp-fusion.md.</summary>
    public sealed class MultiScaleDcpMethod : IDeHazeMethod
    {
        public string Name => "DCP - Multi-Scale Fusion";

        public string Description =>
            "DCP с несколькими радиусами окна тёмного канала вместо одного `patch`.\n\n" +
            "Малый радиус держит тонкие детали, но шумит; большой - стабильнее, но даёт ореолы.\n" +
            "Считаем t для малого и большого радиуса и смешиваем по 'плоскостности':\n" +
            "    t = t_small + wFlat*(t_large - t_small),  wFlat = exp(-k*|∇Y|)\n" +
            "На краях (wFlat->0) берём малый радиус (резко), на гладком (wFlat->1) - большой.\n\n" +
            "Параметры: ω - сила; rSmall/rLarge - радиусы; refine/ε - Guided Filter.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega",  "ω - доля удаляемой дымки", 0.3, 0.95, 0.5, search: true),
            new ParamDef("rSmall", "Малый радиус (детали)",    1, 8,  2, 1, isInt: true),
            new ParamDef("rLarge", "Большой радиус (гладко)",  6, 40, 18, 1, isInt: true),
            new ParamDef("min",    "t_min - нижний порог t",   0.01, 0.5, 0.1),
            new ParamDef("refine", "Радиус Guided Filter",     5, 120, 50, 1, isInt: true),
            new ParamDef("eps",    "ε - регуляризация GF",     1e-5, 1e-2, 1e-3, log: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"]; int rS = (int)p["rSmall"], rL = (int)p["rLarge"];
            double tmin = p["min"]; int refine = (int)p["refine"]; double eps = p["eps"];

            using var I = DehazeCore.Normalize(input);
            using var darkA = DehazeCore.DarkChannel(I, Math.Max(rS, 3));
            var A = DehazeCore.Atmospheric(I, darkA, 0.001);
            using var norm = DehazeCore.NormByA(I, A);

            using var Ds = DehazeCore.DarkChannel(norm, rS);
            using var Dl = DehazeCore.DarkChannel(norm, rL);
            using var tS = new Mat(); Ds.ConvertTo(tS, DepthType.Cv32F, -omega, 1.0);
            using var tL = new Mat(); Dl.ConvertTo(tL, DepthType.Cv32F, -omega, 1.0);

            using var wFlat = DehazeCore.Flatness(I, 8.0);   // ~1 на гладком -> большой радиус; ~0 на краях -> малый
            using var tRaw = new Mat();
            using (var diff = new Mat())
            {
                CvInvoke.Subtract(tL, tS, diff);
                CvInvoke.Multiply(wFlat, diff, diff);
                CvInvoke.Add(tS, diff, tRaw);                // tS + wFlat*(tL - tS)
            }

            using var t = new Mat();
            XImgprocInvoke.GuidedFilter(I, tRaw, t, refine, eps);
            return DehazeCore.Recover(I, t, A, tmin);
        }
    }
}
