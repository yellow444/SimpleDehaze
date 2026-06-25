using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Гибрид: физическое удаление вуали дымки (DCP с защитой цвета) + цвет-сохраняющий
    /// локальный контраст (CLAHE по яркости). Бенчмарк показывал, что чистый CLAHE выигрывает
    /// 'естественностью', но он не убирает дымку физически; а DCP убирает, но выглядит площе.
    /// Гибрид берёт сильные стороны обоих. См. docs/methods/hybrid-dcp-clahe.md.
    /// </summary>
    public sealed class HybridDcpClaheMethod : IDeHazeMethod
    {
        public string Name => "DCP + CLAHE (гибрид)";

        public string Description =>
            "Лучшее из двух: сначала физически убираем вуаль дымки моделью DCP (с защитой цвета),\n" +
            "затем добавляем локальный контраст CLAHE по яркости (цвет a,b не трогаем).\n\n" +
            "Шаги:\n" +
            "1. DCP: t = 1 - ω*darkChannel(I/A), уточнение Fast Global Smoother, восстановление\n" +
            "   с защитой от перенасыщения (яркость и хрома усиливаются раздельно).\n" +
            "2. Результат -> Lab, CLAHE к каналу L (локальный контраст), обратно в BGR.\n\n" +
            "DCP даёт настоящее удаление дымки (а не только контраст, как чистый CLAHE), CLAHE -\n" +
            "естественный вид без потери цвета. Параметры: ω, patch - дымка; clip, tiles - CLAHE.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega", "ω - доля удаляемой дымки",       0.3, 0.95, 0.6, search: true),
            new ParamDef("patch", "Патч тёмного канала",            1, 15, 5, 1, isInt: true),
            new ParamDef("min",   "t_min - нижний порог t",         0.01, 0.5, 0.1),
            new ParamDef("clip",  "CLAHE: ограничение контраста",   1.0, 6.0, 2.5, search: true),
            new ParamDef("tiles", "CLAHE: сетка тайлов (NxN)",      2, 16, 8, 1, isInt: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            // 1. DCP с уточнением t (FGS) и защитой цвета (chromaFloor по умолчанию в Recover)
            using var dcp = DehazeCore.Run(input, p["omega"], (int)p["patch"], p["min"], (i, t) =>
            {
                var dst = new Mat();
                using var guide8 = new Mat();
                i.ConvertTo(guide8, DepthType.Cv8U, 255.0);   // FGS требует 8U-гайд
                XImgprocInvoke.FastGlobalSmootherFilter(guide8, t, dst, 600, 30, 0.25, 3);
                return dst;
            });

            // 2. CLAHE по яркости (Lab), цвет (a,b) не трогаем
            using var dcp8 = new Mat();
            dcp.ConvertTo(dcp8, DepthType.Cv8U, 255.0);
            using var lab = new Mat();
            CvInvoke.CvtColor(dcp8, lab, ColorConversion.Bgr2Lab);
            var ch = lab.Split();
            int tiles = Math.Max(2, (int)p["tiles"]);
            CvInvoke.CLAHE(ch[0], p["clip"], new Size(tiles, tiles), ch[0]);
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
