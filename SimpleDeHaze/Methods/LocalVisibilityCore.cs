using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    internal static class LocalVisibilityCore
    {
        public static Mat Run(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p, bool highQuality)
        {
            int scale = highQuality ? Math.Max(1, (int)p["scale"]) : 1;
            using var work = scale <= 1 ? input.Clone() : Upscale(input, scale, highQuality);
            using var i01 = DehazeCore.Normalize(work);

            double omega = p["omega"], tmin = p["min"], chroma = p["chroma"], tSky = p["tsky"];
            int patch = (int)p["patch"], refine = (int)p["refine"];

            using var tRaw = DehazeCore.SpectralTransmission(i01, omega, patch, tSky, out var a);
            using var t = RefineTransmission(i01, tRaw, refine, highQuality);
            using var recovered = DehazeCore.Recover(i01, t, a, tmin, chroma);

            double clip = p["clip"], sat = p["sat"], detail = p["detail"];
            using var toned = DehazeCore.LabEnhance(recovered, clip, 8, sat, detail);
            using var edges = LaplacianBoost(toned, p["lap"], p["unsharp"], highQuality);
            using var colored = DehazeCore.LimitColorfulness(edges, input.Mat, p["color"]);
            using var quiet = DehazeCore.BilateralDenoise(colored, p["smooth"]);

            if (scale <= 1)
                return quiet.Clone();

            var down = new Mat();
            CvInvoke.Resize(quiet, down, input.Size, 0, 0, Inter.Area);
            return DeHazeCPU.Clip(down);
        }

        private static Image<Bgr, byte> Upscale(Image<Bgr, byte> input, int scale, bool highQuality)
        {
            var up = new Mat();
            var size = new System.Drawing.Size(input.Width * scale, input.Height * scale);
            CvInvoke.Resize(input.Mat, up, size, 0, 0, highQuality ? Inter.Lanczos4 : Inter.Nearest);
            if (highQuality)
                CvInvoke.GaussianBlur(up, up, new System.Drawing.Size(0, 0), 0.45);
            return up.ToImage<Bgr, byte>();
        }

        private static Mat RefineTransmission(Mat guide01, Mat tRaw, int refine, bool highQuality)
        {
            var t = new Mat();
            if (highQuality)
            {
                using var guide8 = new Mat();
                guide01.ConvertTo(guide8, DepthType.Cv8U, 255.0);
                XImgprocInvoke.FastGlobalSmootherFilter(guide8, tRaw, t, 680, Math.Max(5, refine), 0.25, 3);
            }
            else if (refine > 0)
            {
                double sigma = Math.Max(0.7, refine / 9.0);
                CvInvoke.GaussianBlur(tRaw, t, new System.Drawing.Size(0, 0), sigma);
            }
            else
            {
                tRaw.CopyTo(t);
            }

            DehazeCore.Clamp01(t);
            return t;
        }

        private static Mat LaplacianBoost(Mat bgr01, double lapGain, double unsharp, bool highQuality)
        {
            if (lapGain <= 1e-4 && unsharp <= 1e-4)
                return bgr01.Clone();

            using var bgr8 = new Mat();
            bgr01.ConvertTo(bgr8, DepthType.Cv8U, 255.0);
            using var lab = new Mat();
            CvInvoke.CvtColor(bgr8, lab, ColorConversion.Bgr2Lab);
            var ch = lab.Split();

            using var l01 = new Mat();
            ch[0].ConvertTo(l01, DepthType.Cv32F, 1.0 / 255.0);
            using var low = new Mat();
            CvInvoke.GaussianBlur(l01, low, new System.Drawing.Size(0, 0), highQuality ? 1.25 : 0.85);
            using var high = new Mat();
            CvInvoke.Subtract(l01, low, high);

            using var lap = new Mat();
            CvInvoke.Laplacian(l01, lap, DepthType.Cv32F, 3);
            using var zero = new Mat(lap.Size, DepthType.Cv32F, 1);
            zero.SetTo(new MCvScalar(0));
            using var mask = new Mat();
            CvInvoke.AbsDiff(lap, zero, mask);
            CvInvoke.GaussianBlur(mask, mask, new System.Drawing.Size(0, 0), highQuality ? 1.6 : 1.0);
            double min = 0, max = 0;
            System.Drawing.Point minLoc = default, maxLoc = default;
            CvInvoke.MinMaxLoc(mask, ref min, ref max, ref minLoc, ref maxLoc);
            if (max > 1e-6)
                mask.ConvertTo(mask, DepthType.Cv32F, Math.Min(8.0, lapGain / max), 0.0);
            double floor = highQuality ? 0.07 : 0.11;
            mask.ConvertTo(mask, DepthType.Cv32F, 1.0 / (1.0 - floor), -floor / (1.0 - floor));
            DehazeCore.Clamp01(mask);

            using var detail = new Mat();
            CvInvoke.Multiply(high, mask, detail);
            CvInvoke.AddWeighted(l01, 1.0, detail, unsharp, 0.0, l01);
            DehazeCore.Clamp01(l01);
            l01.ConvertTo(ch[0], DepthType.Cv8U, 255.0);

            using (var v = new VectorOfMat(ch))
                CvInvoke.Merge(v, lab);
            foreach (var c in ch) c.Dispose();

            using var out8 = new Mat();
            CvInvoke.CvtColor(lab, out8, ColorConversion.Lab2Bgr);
            var result = new Mat();
            out8.ConvertTo(result, DepthType.Cv32F, 1.0 / 255.0);
            return result;
        }
    }
}
