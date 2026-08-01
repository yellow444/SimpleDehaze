using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    public sealed class HcvA2crUtawMethod : IDeHazeMethod
    {
        public string Name => "HCV-A²CR-UTAW (stationary wavelets, эксперимент)";

        public string Description =>
            "HCV-A²CR с undecimated à trous detail stage. Все B3-spline полосы остаются в полном " +
            "разрешении; их reliability вычисляется из локальной мощности, recovery gain, шума и " +
            "расхождения transmission priors. Усиление меняет normalized HCV Value общей добавкой " +
            "ко всем X=J/A каналам, поэтому hue/chroma direction не ломается; допустимый шаг " +
            "ограничивается RGB gamut до формирования результата.\n\n" +
            "À trous и wavelet dehazing известны. Исследовательская гипотеза — совместный " +
            "uncertainty-derived band budget поверх точного HCV-A²CR recovery.";

        public IReadOnlyList<ParamDef> Parameters { get; } = HcvA2crMethod.PhysicalParameters
            .Concat(new[]
            {
                new ParamDef("uLevels", "UTAW: уровни à trous", 1, 5, 4, 1, isInt: true, tunable: false),
                new ParamDef("uDetail", "UTAW: максимум detail boost", 0, 2, 0.6),
                new ParamDef("uUnc", "UTAW: штраф uncertainty(D)", 0, 10, 2),
                new ParamDef("uRadius", "UTAW: радиус энергии полос", 0, 12, 3, 1, isInt: true),
                new ParamDef("uLimit", "UTAW: предел delta полосы", 0.005, 0.15, 0.04),
            })
            .ToArray();

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            using var physical = HcvA2crMethod.Execute(input, p);
            using var enhanced = StationaryAtrous.EnhanceHcvValue(
                physical.Recovery.LinearResult,
                physical.Recovery.GainValue,
                physical.Scene.SigmaDepth,
                physical.Scene.Airlight,
                (int)p["uLevels"], (int)p["uRadius"], p["uDetail"], p["uUnc"],
                p["noise"] * p["noise"], p["uLimit"]);
            return ColorSpace.ToSrgb(enhanced.LinearResult);
        }
    }
}
