using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>Haze-Lines / Color Cube (Berman, 2016) - попиксельная векторная геометрия. См. docs/methods/color-cube-projection.md.</summary>
    public sealed class ColorCubeMethod : IDeHazeMethod
    {
        public string Name => "Haze-Lines (Color Cube)";

        public string Description =>
            "Berman (2016). Из модели I = t*J + (1-t)*A следует, что I-A = t*(J-A):\n" +
            "пиксели одного 'чистого' цвета лежат на луче из A (haze-line).\n\n" +
            "Шаги:\n" +
            "1. A - атмосферный свет (по тёмному каналу).\n" +
            "2. Группируем пиксели по направлению (I-A)/||I-A|| (бины на сфере).\n" +
            "3. На каждой линии самый дальний пиксель почти без дымки: r_max = max||I-A||.\n" +
            "4. t = ||I-A|| / r_max, лёгкая регуляризация Guided Filter.\n" +
            "5. J = (I - A)/max(t, t_min) + A.\n\n" +
            "Попиксельно, без окон/матриц. Параметры: K - число бинов направлений; refine/ε - сглаживание t.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega", "ω - доля удаляемой дымки",    0.3, 0.95, 0.5, search: true),
            new ParamDef("patch", "Патч тёмного канала (для A)", 1, 15, 5, 1, isInt: true),
            new ParamDef("min",   "t_min - нижний порог t",      0.01, 0.5, 0.1),
            new ParamDef("kbins", "K - бинов направления/ось",   8, 64, 24, 1, isInt: true),
            new ParamDef("refine","Радиус Guided Filter",        5, 120, 40, 1, isInt: true),
            new ParamDef("eps",   "ε - регуляризация GF",        1e-5, 1e-2, 1e-3, log: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            int patch = (int)p["patch"];
            double omega = p["omega"];
            double tmin = p["min"];
            int K = Math.Max(4, (int)p["kbins"]);
            int refine = (int)p["refine"];
            double eps = p["eps"];

            using var I = DehazeCore.Normalize(input);
            using var dark = DehazeCore.DarkChannel(I, patch);
            var A = DehazeCore.Atmospheric(I, dark, 0.001);

            int W = I.Cols, H = I.Rows, n = W * H;
            var ch = I.Split();
            var b = new float[n]; var g = new float[n]; var r = new float[n];
            ch[0].CopyTo(b); ch[1].CopyTo(g); ch[2].CopyTo(r);
            foreach (var c in ch) c.Dispose();

            var radius = new float[n]; var bin = new int[n];
            var rmax = new float[K * K * K];
            for (int i = 0; i < n; i++)
            {
                float rb = b[i] - (float)A.V0, rg = g[i] - (float)A.V1, rr = r[i] - (float)A.V2;
                float rad = MathF.Sqrt(rb * rb + rg * rg + rr * rr);
                radius[i] = rad;
                if (rad < 1e-6f) { bin[i] = -1; continue; }
                int qx = Math.Clamp((int)((rb / rad + 1f) * 0.5f * K), 0, K - 1);
                int qy = Math.Clamp((int)((rg / rad + 1f) * 0.5f * K), 0, K - 1);
                int qz = Math.Clamp((int)((rr / rad + 1f) * 0.5f * K), 0, K - 1);
                int bi = (qx * K + qy) * K + qz;
                bin[i] = bi;
                if (rad > rmax[bi]) rmax[bi] = rad;
            }

            var t = new float[n];
            for (int i = 0; i < n; i++)
            {
                if (bin[i] < 0) { t[i] = 1f; continue; }
                float rm = rmax[bin[i]];
                float raw = rm > 1e-6f ? radius[i] / rm : 1f;
                float gentle = 1f - (float)omega * (1f - raw);     // ослабление, как ω в DCP
                t[i] = Math.Clamp(gentle, (float)tmin, 1f);
            }

            using var tRaw = DehazeCore.MatFromFloats(t, H, W);
            using var tRef = new Mat();
            XImgprocInvoke.GuidedFilter(I, tRaw, tRef, refine, eps);
            return DehazeCore.Recover(I, tRef, A, tmin);
        }
    }
}
