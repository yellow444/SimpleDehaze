using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Быстрая инженерная эвристика вуали (скорость + контуры): см. docs/methods/fast-veil.md.
    ///
    /// Цель - 'хотя бы контуры, очень быстро': никаких итераций, ДПФ и Guided Filter. Карта пропускания
    /// сглаживается прореживающим Гауссом (FastGaussian) вместо дорогого edge-aware фильтра, восстановление
    /// с защитой цвета - поканально (O(N)), затем быстрые авто-уровни возвращают контраст/контуры.
    /// Учитывает атмосферу (A_c) и яркость зон (гейт неба).
    /// </summary>
    public sealed class FastVeilMethod : IDeHazeMethod
    {
        public string Name => "Быстрая вуаль (контуры, скорость)";

        public string Description =>
            "Скорость: всё пиксельно/один быстрый Гаусс - ни итераций, ни ДПФ, ни Guided Filter.\n\n" +
            "Шаги:\n" +
            "1. A_c - атмосферный свет (ярчайшие в тёмном канале).\n" +
            "2. A_c и t считаются на уменьшенной копии: t = 1 - ω*min_Ω(I_c/A_c); FastGaussian.\n" +
            "3. Небо: маска (ярко*малонасыщенно) поднимает t - не пережигаем пересветы.\n" +
            "4. Восстановление J_c с защитой цвета (яркость/хрома раздельно), поканально.\n" +
            "5. Быстрые авто-уровни по яркости без Lab -> контраст/контуры.\n\n" +
            "Параметры: ω, patch - дымка; sigma/scale - сглаживание и скорость; chroma/tone/gain/color - контуры без пересата.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega", "ω - доля удаляемой дымки", 0.3, 0.95, 0.7, search: true),
            new ParamDef("patch", "Патч тёмного канала",      1, 15, 5, 1, isInt: true),
            new ParamDef("min",   "t_min - нижний порог t",   0.01, 0.5, 0.1),
            new ParamDef("chroma","chromaFloor - защита цвета", 0.1, 0.9, 0.4),
            new ParamDef("sigma", "σ сглаживания t",          5, 80, 25, 1, isInt: true),
            new ParamDef("scale", "Прореживание t/A (скорость)", 1, 6, 3, 1, isInt: true, tunable: false),
            new ParamDef("tone",  "Авто-уровни (контуры)",    0.0, 1.0, 0.5),
            new ParamDef("gain",  "Потолок gain тона",        1.0, 1.8, 1.35),
            new ParamDef("color", "Доля цветности после tone", 0.5, 1.0, 0.76),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            double omega = p["omega"], tmin = p["min"], chroma = p["chroma"], tone = p["tone"], sigma = p["sigma"];
            double gain = p["gain"], color = p["color"];
            int patch = (int)p["patch"], scale = (int)p["scale"];

            using var I = DehazeCore.Normalize(input);
            using var t = EstimateTransmissionFast(I, omega, patch, sigma, scale, out var A);

            using var J = DehazeCore.Recover(I, t, A, tmin, chroma);          // защита цвета (яркость/хрома раздельно)
            using var toned = DehazeCore.RestoreToneFast(J, tone, maxGain: gain); // быстрые авто-уровни -> контуры
            return DehazeCore.ScaleChroma(toned, color);
        }

        private static Mat EstimateTransmissionFast(Mat i01, double omega, int patch, double sigma, int scale, out MCvScalar atmospheric)
        {
            int s = Math.Max(1, scale);
            bool smallEnough = s <= 1 || i01.Cols / s < 48 || i01.Rows / s < 48;
            using var work = new Mat();
            if (smallEnough)
            {
                i01.CopyTo(work);
                s = 1;
            }
            else
            {
                var sz = new Size(Math.Max(1, i01.Cols / s), Math.Max(1, i01.Rows / s));
                CvInvoke.Resize(i01, work, sz, 0, 0, Inter.Area);
            }

            int pSmall = Math.Max(1, (int)Math.Round(patch / (double)s));
            double sigmaSmall = Math.Max(1.0, sigma / s);

            using var dark = DehazeCore.DarkChannel(work, pSmall);
            atmospheric = DehazeCore.Atmospheric(work, dark, 0.001);

            using var tRaw = DehazeCore.RawTransmission(work, atmospheric, omega, pSmall);
            using var tSmall = DehazeCore.FastGaussian(tRaw, sigmaSmall);
            using (var sky = DehazeCore.SkyMask(work))
                DehazeCore.RaiseInSky(tSmall, sky, 0.7);
            DehazeCore.Clamp01(tSmall);

            if (s == 1)
                return tSmall.Clone();

            var t = new Mat();
            CvInvoke.Resize(tSmall, t, i01.Size, 0, 0, Inter.Linear);
            DehazeCore.Clamp01(t);
            return t;
        }
    }
}
