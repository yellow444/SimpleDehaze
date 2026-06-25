using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace SimpleDeHaze.Methods
{
    /// <summary>CLAHE - адаптивная эквализация гистограммы яркости (контрастная очистка от дымки).</summary>
    public sealed class ClaheMethod : IDeHazeMethod
    {
        public string Name => "CLAHE (адаптивная эквализация)";

        public string Description =>
            "Contrast Limited Adaptive Histogram Equalization. Не физическая модель дымки, а локальное\n" +
            "выравнивание контраста по тайлам с ограничением усиления (анти-шум).\n\n" +
            "Шаги:\n" +
            "1. BGR -> Lab, берём канал яркости L.\n" +
            "2. CLAHE к L: по сетке тайлов выравниваем гистограмму, клиппинг ограничивает усиление.\n" +
            "3. Lab -> BGR (цвет a,b не трогаем - тон сохраняется).\n\n" +
            "Быстро, надёжно, хорошо вытягивает детали в дымке. Параметры: clip - сила; tiles - сетка.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("clip",  "Ограничение контраста", 1.0, 8.0, 3.0, search: true),
            new ParamDef("tiles", "Сетка тайлов (NxN)",    2,   16,  8, 1, isInt: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double clip = p["clip"];
            int tiles = Math.Max(2, (int)p["tiles"]);

            using var lab = new Mat();
            CvInvoke.CvtColor(input, lab, ColorConversion.Bgr2Lab);
            var ch = lab.Split();
            CvInvoke.CLAHE(ch[0], clip, new Size(tiles, tiles), ch[0]);   // только яркость
            using (var v = new VectorOfMat(ch)) CvInvoke.Merge(v, lab);
            foreach (var c in ch) c.Dispose();

            using var outBgr = new Mat();
            CvInvoke.CvtColor(lab, outBgr, ColorConversion.Lab2Bgr);
            var res = new Mat();
            outBgr.ConvertTo(res, DepthType.Cv32F, 1.0 / 255.0);
            return res;
        }
    }
}
