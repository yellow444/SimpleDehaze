using System.Diagnostics;
using System.Drawing;

using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

using Range = Emgu.CV.Structure.Range;

namespace SimpleDeHaze
{
    public class DeHazeGPU : IDisposable
    {
        private readonly Stopwatch sw = new();
        private bool _debug = false;
        private string _file = "";
        private bool disposedValue;

        public DeHazeGPU()
        {
        }

        public static GpuMat Clip(GpuMat image)
        {
            CudaInvoke.Threshold(image, image, 1, 1, ThresholdType.Trunc);
            CudaInvoke.Multiply(image, image.NumberOfChannels == 1 ? new ScalarArray(-1) : new ScalarArray(new MCvScalar(-1, -1, -1)), image);

            CudaInvoke.Threshold(image, image, 0, 0, ThresholdType.Trunc);
            CudaInvoke.Multiply(image, image.NumberOfChannels == 1 ? new ScalarArray(-1) : new ScalarArray(new MCvScalar(-1, -1, -1)), image);
            return image;
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        public Mat RemoveHaze(Image<Bgr, byte> image, bool debug = false, float beta = 0.3f, int patchDarkChannel = 3, int decompositionSize = 131, float min = 0.1f, float percen = 0.1f, int refineSize = 31, double eps = 0.02d * 0.02d, string file = "")
        {
            _debug = debug;
            _file = file;
#if RELEASE

#else
            WatchStart();
#endif
            using var srcImage = new GpuMat(image.Convert<Bgr, float>() / 255);
            using var quadDecomposition = QuadDecompositionGpu(srcImage, decompositionSize);
            using var darkChannelAtmospheric = ComputeDarkChannelPatch(quadDecomposition, patchDarkChannel);
            var atmosphericLightScalar = ComputeAtmosphericLight(quadDecomposition, darkChannelAtmospheric, percen);
            using var darkChannel = ComputeDarkChannelPatch(srcImage, patchDarkChannel);
            using var colorsChannel = ComputeColorsChannelPatch(srcImage, patchDarkChannel);
            using var transmission = EstimateTransmission(colorsChannel, atmosphericLightScalar, beta);
            using var refinedTransmission = RefineTransmission(srcImage, transmission, eps, refineSize);
            using var dehazedImage = RecoverImage(srcImage, refinedTransmission, atmosphericLightScalar, min);
#if RELEASE

#else
            WatchStop();
#endif
            return ((GpuMat)dehazedImage.Clone()).ToMat();
        }

        public void Show(string name, GpuMat image)
        {
            if (!_debug) return;
            var result = ((GpuMat)image.Clone()).ToMat();
            result *= 255;
            CvInvoke.NamedWindow($"DeHazeGPU {name} Image{result.Size}", WindowFlags.AutoSize);
            CvInvoke.Imshow($"DeHazeGPU {name} Image{result.Size}", result.ToImage<Bgr, byte>().Resize((int)(900f / result.Height * result.Width), 800, Inter.Lanczos4));
            var v = _file + $"DeHazeGPU {name} Image{result.Size}.png".Replace("{", "").Replace("}", "").Replace("=", "").Replace(",", "");
            CvInvoke.Imwrite(v, result, new KeyValuePair<ImwriteFlags, int>(ImwriteFlags.PngCompression, 0));
            CvInvoke.WaitKey();
        }

        public void Show(string name, Mat image)
        {
            if (!_debug) return;
            var result = image;
            CvInvoke.NamedWindow($"DeHazeGPU {name} Image{result.Size}", WindowFlags.AutoSize);
            CvInvoke.Imshow($"DeHazeGPU {name} Image{result.Size}", result.ToImage<Bgr, byte>().Resize((int)(900f / result.Height * result.Width), 800, Inter.Lanczos4));
            var v = _file + $"DeHazeGPU {name} Image{result.Size}.png".Replace("{", "").Replace("}", "").Replace("=", "").Replace(",", "");
            CvInvoke.Imwrite(v, result, new KeyValuePair<ImwriteFlags, int>(ImwriteFlags.PngCompression, 0));
            CvInvoke.WaitKey();
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects)
                }

                // TODO: free unmanaged resources (unmanaged objects) and override finalizer
                // TODO: set large fields to null
                disposedValue = true;
            }
        }

        private static MCvScalar ComputeAtmosphericLight(GpuMat srcImage, GpuMat darkChannel, float percen)
        {
            CudaInvoke.Multiply(srcImage, new ScalarArray(new MCvScalar(255, 255, 255)), srcImage);
            srcImage.ConvertTo(srcImage, DepthType.Cv8U);
            var channels = srcImage.Split();
            var channelsData = new byte[3][];
            for (var i = 0; i < channels.Length; i++)
            {
                channelsData[i] = new byte[channels[0].Size.Width * channels[0].Size.Height];
                channels[i].ToMat().CopyTo(channelsData[i]);
            }
            var darkChannelData = new byte[channels[0].Size.Width * channels[0].Size.Height];
            CudaInvoke.Multiply(darkChannel, new ScalarArray(255), darkChannel);
            darkChannel.ConvertTo(darkChannel, DepthType.Cv8U);
            var darkChannelByte = darkChannel.ToMat();
            darkChannelByte.CopyTo(darkChannelData);
            var sortedIndices2 = Enumerable.Range(0, darkChannelData.Length);
            var sortedIndices = sortedIndices2.OrderByDescending(i => darkChannelData[i]).ToArray();
            var topPercentage = Convert.ToInt32(darkChannelData.Length * percen);
            var numPixels = Math.Max(topPercentage, 1);
            var sumR = 0f;
            var sumG = 0f;
            var sumB = 0f;
            for (var i = 0; i < numPixels; i++)
            {
                sumB += channelsData[0][sortedIndices[i]];
                sumG += channelsData[1][sortedIndices[i]];
                sumR += channelsData[2][sortedIndices[i]];
            }
            var atmosphericLightB = (sumB / numPixels) / 255;
            var atmosphericLightG = (sumG / numPixels) / 255;
            var atmosphericLightR = (sumR / numPixels) / 255;
            darkChannelData = null;
            channelsData = null;
            return new MCvScalar(atmosphericLightB, atmosphericLightG, atmosphericLightR);
        }

        private static GpuMat Filter(GpuMat guide, GpuMat src, int r, double eps)
        {
            using var invbb = new GpuMat();
            using var invgb = new GpuMat();
            using var invgg = new GpuMat();
            using var invrb = new GpuMat();
            using var invrg = new GpuMat();
            using var invrr = new GpuMat();
            using var mean_I_b = new GpuMat();
            using var mean_I_g = new GpuMat();
            using var mean_I_r = new GpuMat();
            using var I = new GpuMat();
            if (guide.Depth == DepthType.Cv32F || guide.Depth == DepthType.Cv64F)
            {
                guide.CopyTo(I);
            }
            else
            {
                guide.ConvertTo(I, DepthType.Cv32F);
            }
            using CudaBoxFilter box_filter = new(DepthType.Cv32F, 1, DepthType.Cv32F, 1, new Size(r, r), new Point(-1, -1));
            var Ichannels = I.Split();

            box_filter.Apply(Ichannels[0], mean_I_r);
            box_filter.Apply(Ichannels[1], mean_I_g);
            box_filter.Apply(Ichannels[2], mean_I_b);
            using var var_I_rr = new GpuMat();
            using var var_I_rg = new GpuMat();
            using var var_I_rb = new GpuMat();
            using var var_I_gg = new GpuMat();
            using var var_I_gb = new GpuMat();
            using var var_I_bb = new GpuMat();
            using var working = new GpuMat();
            CudaInvoke.Multiply(Ichannels[0], Ichannels[0], working);
            box_filter.Apply(working, var_I_rr);
            CudaInvoke.Multiply(mean_I_r, mean_I_r, working);
            CudaInvoke.Subtract(var_I_rr, working, var_I_rr);
            CudaInvoke.Add(var_I_rr, new ScalarArray(eps), var_I_rr);
            CudaInvoke.Multiply(Ichannels[0], Ichannels[1], working);
            box_filter.Apply(working, var_I_rg);
            CudaInvoke.Multiply(mean_I_r, mean_I_g, working);
            CudaInvoke.Subtract(var_I_rg, working, var_I_rg);
            CudaInvoke.Multiply(Ichannels[0], Ichannels[2], working);
            box_filter.Apply(working, var_I_rb);
            CudaInvoke.Multiply(mean_I_r, mean_I_b, working);
            CudaInvoke.Subtract(var_I_rb, working, var_I_rb);
            CudaInvoke.Multiply(Ichannels[1], Ichannels[1], working);
            box_filter.Apply(working, var_I_gg);
            CudaInvoke.Multiply(mean_I_g, mean_I_g, working);
            CudaInvoke.Subtract(var_I_gg, working, var_I_gg);
            CudaInvoke.Add(var_I_gg, new ScalarArray(eps), var_I_gg);
            CudaInvoke.Multiply(Ichannels[1], Ichannels[2], working);
            box_filter.Apply(working, var_I_gb);
            CudaInvoke.Multiply(mean_I_g, mean_I_b, working);
            CudaInvoke.Subtract(var_I_gb, working, var_I_gb);
            CudaInvoke.Multiply(Ichannels[2], Ichannels[2], working);
            box_filter.Apply(working, var_I_bb);
            CudaInvoke.Multiply(mean_I_b, mean_I_b, working);
            CudaInvoke.Subtract(var_I_bb, working, var_I_bb);
            CudaInvoke.Add(var_I_bb, new ScalarArray(eps), var_I_bb);
            CudaInvoke.Multiply(var_I_gg, var_I_bb, invrr);
            CudaInvoke.Multiply(var_I_gb, var_I_gb, working);
            CudaInvoke.Subtract(invrr, working, invrr);
            CudaInvoke.Multiply(var_I_gb, var_I_rb, invrg);
            CudaInvoke.Multiply(var_I_rg, var_I_bb, working);
            CudaInvoke.Subtract(invrg, working, invrg);
            CudaInvoke.Multiply(var_I_rg, var_I_gb, invrb);
            CudaInvoke.Multiply(var_I_gg, var_I_rb, working);
            CudaInvoke.Subtract(invrb, working, invrb);
            CudaInvoke.Multiply(var_I_rr, var_I_bb, invgg);
            CudaInvoke.Multiply(var_I_rb, var_I_rb, working);
            CudaInvoke.Subtract(invgg, working, invgg);
            CudaInvoke.Multiply(var_I_rb, var_I_rg, invgb);
            CudaInvoke.Multiply(var_I_rr, var_I_gb, working);
            CudaInvoke.Subtract(invgb, working, invgb);
            CudaInvoke.Multiply(var_I_rr, var_I_gg, invbb);
            CudaInvoke.Multiply(var_I_rg, var_I_rg, working);
            CudaInvoke.Subtract(invbb, working, invbb);
            using var covDet = new GpuMat();
            CudaInvoke.Multiply(invrr, var_I_rr, var_I_rr);
            CudaInvoke.Multiply(invrg, var_I_rg, var_I_rg);
            CudaInvoke.Multiply(invrb, var_I_rb, var_I_rb);
            CudaInvoke.Add(var_I_rr, var_I_rg, covDet);
            CudaInvoke.Add(covDet, var_I_rb, covDet);
            CudaInvoke.Divide(invrr, covDet, invrr);
            CudaInvoke.Divide(invrg, covDet, invrg);
            CudaInvoke.Divide(invrb, covDet, invrb);
            CudaInvoke.Divide(invgg, covDet, invgg);
            CudaInvoke.Divide(invgb, covDet, invgb);
            CudaInvoke.Divide(invbb, covDet, invbb);
            using var result = new GpuMat();
            using var pc = new VectorOfGpuMat(((GpuMat)src.Clone()).Split());
            for (var i = 0; i < src.NumberOfChannels; i++)
            {
                using var p = (GpuMat)pc[i].Clone();
                using var mean_p = new GpuMat();
                using var mean_Ip_r = new GpuMat();
                using var mean_Ip_g = new GpuMat();
                using var mean_Ip_b = new GpuMat();
                using var cov_Ip_r = new GpuMat();
                using var cov_Ip_g = new GpuMat();
                using var cov_Ip_b = new GpuMat();
                using var a_r = new GpuMat();
                using var a_g = new GpuMat();
                using var a_b = new GpuMat();
                using var b = new GpuMat();
                box_filter.Apply(p, mean_p);
                CudaInvoke.Multiply(Ichannels[0], p, mean_Ip_r);
                CudaInvoke.Multiply(Ichannels[1], p, mean_Ip_g);
                CudaInvoke.Multiply(Ichannels[2], p, mean_Ip_b);
                box_filter.Apply(mean_Ip_r, mean_Ip_r);
                box_filter.Apply(mean_Ip_g, mean_Ip_g);
                box_filter.Apply(mean_Ip_b, mean_Ip_b);
                CudaInvoke.Multiply(mean_I_r, mean_p, cov_Ip_r);
                CudaInvoke.Subtract(mean_Ip_r, cov_Ip_r, cov_Ip_r);
                CudaInvoke.Multiply(mean_I_g, mean_p, cov_Ip_g);
                CudaInvoke.Subtract(mean_Ip_g, cov_Ip_g, cov_Ip_g);
                CudaInvoke.Multiply(mean_I_b, mean_p, cov_Ip_b);
                CudaInvoke.Subtract(mean_Ip_b, cov_Ip_b, cov_Ip_b);
                using var prod1 = new GpuMat();
                using var prod2 = new GpuMat();
                using var prod3 = new GpuMat();
                CudaInvoke.Multiply(invrr, cov_Ip_r, prod1);
                CudaInvoke.Multiply(invrg, cov_Ip_g, prod2);
                CudaInvoke.Multiply(invrb, cov_Ip_b, prod3);
                CudaInvoke.Add(prod1, prod2, prod2);
                CudaInvoke.Add(prod2, prod3, a_r);
                CudaInvoke.Multiply(invrg, cov_Ip_r, prod1);
                CudaInvoke.Multiply(invgg, cov_Ip_g, prod2);
                CudaInvoke.Multiply(invgb, cov_Ip_b, prod3);
                CudaInvoke.Add(prod1, prod2, prod2);
                CudaInvoke.Add(prod2, prod3, a_g);
                CudaInvoke.Multiply(invrb, cov_Ip_r, prod1);
                CudaInvoke.Multiply(invgb, cov_Ip_g, prod2);
                CudaInvoke.Multiply(invbb, cov_Ip_b, prod3);
                CudaInvoke.Add(prod1, prod2, prod2);
                CudaInvoke.Add(prod2, prod3, a_b);
                CudaInvoke.Multiply(a_r, mean_I_r, prod1);
                CudaInvoke.Multiply(a_g, mean_I_g, prod2);
                CudaInvoke.Multiply(a_b, mean_I_b, prod3);
                CudaInvoke.Subtract(mean_p, prod1, b);
                CudaInvoke.Subtract(b, prod2, b);
                CudaInvoke.Subtract(b, prod3, b);
                box_filter.Apply(a_r, a_r);
                box_filter.Apply(a_g, a_g);
                box_filter.Apply(a_b, a_b);
                box_filter.Apply(b, b);
                CudaInvoke.Multiply(a_r, Ichannels[0], a_r);
                CudaInvoke.Multiply(a_g, Ichannels[1], a_g);
                CudaInvoke.Multiply(a_b, Ichannels[2], a_b);
                CudaInvoke.Add(a_r, a_g, a_r);
                CudaInvoke.Add(a_r, a_b, a_r);
                CudaInvoke.Add(a_r, b, b);
                b.CopyTo(pc[i]);
            }
            CudaInvoke.Merge(pc, result);
            result.ConvertTo(result, src.Depth);
            return (GpuMat)result.Clone();
        }

        private static GpuMat GrayMat(GpuMat srcImage)
        {
            using var image = (GpuMat)srcImage.Clone();
            CudaInvoke.Multiply(image, new ScalarArray(new MCvScalar(0.114f, 0.587f, 0.299f)), image);
            using var chanels = new VectorOfGpuMat(image.Split());
            CudaInvoke.Add(chanels[0], chanels[1], chanels[0]);
            CudaInvoke.Add(chanels[0], chanels[2], chanels[0]);
            using var result = new GpuMat();
            CudaInvoke.Merge(chanels, result);
            CudaInvoke.CvtColor(result, result, ColorConversion.Bgr2Gray);
            return (GpuMat)result.Clone();
        }

        private static GpuMat GuidedFilter(GpuMat guide, GpuMat src, int r, double eps)
        {
            var result = Filter(guide, src, r, eps);
            return result;
        }

        private static GpuMat RecoverImage(GpuMat srcImage, GpuMat transmission, MCvScalar atmosphericLight, float min)
        {
            using var srcImageSplit = new VectorOfGpuMat(((GpuMat)srcImage.Clone()).Split());
            CudaInvoke.Add(srcImageSplit[0], new ScalarArray(-atmosphericLight.V0), srcImageSplit[0]);
            CudaInvoke.Add(srcImageSplit[1], new ScalarArray(-atmosphericLight.V1), srcImageSplit[1]);
            CudaInvoke.Add(srcImageSplit[2], new ScalarArray(-atmosphericLight.V2), srcImageSplit[2]);
            var transmissionSplit = transmission.Split();
            using var minMat = new GpuMat();
            transmissionSplit[0].CopyTo(minMat);
            minMat.SetTo(new MCvScalar(min));
            CudaInvoke.Max(transmissionSplit[0], minMat, transmissionSplit[0]);
            CudaInvoke.Max(transmissionSplit[1], minMat, transmissionSplit[1]);
            CudaInvoke.Max(transmissionSplit[2], minMat, transmissionSplit[2]);
            CudaInvoke.Divide(srcImageSplit[0], transmissionSplit[0], srcImageSplit[0]);
            CudaInvoke.Divide(srcImageSplit[1], transmissionSplit[1], srcImageSplit[1]);
            CudaInvoke.Divide(srcImageSplit[2], transmissionSplit[2], srcImageSplit[2]);
            CudaInvoke.Add(srcImageSplit[0], new ScalarArray(atmosphericLight.V0), srcImageSplit[0]);
            CudaInvoke.Add(srcImageSplit[1], new ScalarArray(atmosphericLight.V1), srcImageSplit[1]);
            CudaInvoke.Add(srcImageSplit[2], new ScalarArray(atmosphericLight.V2), srcImageSplit[2]);
            using var mergeImageSplit = new GpuMat();
            CudaInvoke.Merge(srcImageSplit, mergeImageSplit);
            using var result = Clip(mergeImageSplit);
            return (GpuMat)result.Clone();
        }

        private GpuMat ComputeColorsChannelPatch(GpuMat srcImage, int patch, string name = "")
        {
            using var colorsChannel = new GpuMat();
            CudaInvoke.Min(srcImage, srcImage, colorsChannel);
            using var colorsChannelSplit = new VectorOfGpuMat(((GpuMat)colorsChannel.Clone()).Split());
            var ksize = new Size(2 * patch + 1, 2 * patch + 1);
            var elem = CvInvoke.GetStructuringElement(ElementShape.Rectangle, ksize, new Point(-1, -1));
            using (var erode = new CudaMorphologyFilter(MorphOp.Erode, colorsChannel.Depth, 1, elem, new Point(-1, -1), 1))
            {
                for (var i = 0; i < colorsChannel.NumberOfChannels; i++)
                {
                    erode.Apply(colorsChannelSplit[i], colorsChannelSplit[i]);
                }
            }
            CudaInvoke.Merge(colorsChannelSplit, colorsChannel);
            Show("ComputeColorsChannelPatch" + name, colorsChannel);
            return (GpuMat)colorsChannel.Clone();
        }

        private GpuMat ComputeDarkChannelPatch(GpuMat srcImage, int patch, string name = "")
        {
            var bgrChannels = srcImage.Split();
            using var darkChannel = new GpuMat();
            CudaInvoke.Min(bgrChannels[0], bgrChannels[1], darkChannel);
            CudaInvoke.Min(darkChannel, bgrChannels[2], darkChannel);
            var ksize = new Size(2 * patch + 1, 2 * patch + 1);
            var elem = CvInvoke.GetStructuringElement(ElementShape.Rectangle, ksize, new Point(-1, -1));
            using (var erode = new CudaMorphologyFilter(MorphOp.Erode, darkChannel.Depth, 1, elem, new Point(-1, -1), 1))
            {
                erode.Apply(darkChannel, darkChannel);
            }
            Show("ComputeDarkChannel" + name, darkChannel);
            return (GpuMat)darkChannel.Clone();
        }

        private GpuMat EstimateTransmission(GpuMat colorsChannel, MCvScalar atmosphericLight, float beta)
        {
            using var colorsChannelMin = new VectorOfGpuMat(((GpuMat)colorsChannel.Clone()).Split());
            CudaInvoke.Divide(new ScalarArray(-beta * atmosphericLight.V0), colorsChannelMin[0], colorsChannelMin[0]);
            CudaInvoke.Divide(new ScalarArray(-beta * atmosphericLight.V1), colorsChannelMin[1], colorsChannelMin[1]);
            CudaInvoke.Divide(new ScalarArray(-beta * atmosphericLight.V2), colorsChannelMin[2], colorsChannelMin[2]);
            using var estimate = new GpuMat();
            CudaInvoke.Merge(colorsChannelMin, estimate);
            CudaInvoke.Exp(estimate, estimate);
            CudaInvoke.Multiply(estimate, estimate.NumberOfChannels == 1 ? new ScalarArray(-1) : new ScalarArray(new MCvScalar(-1, -1, -1)), estimate);
            CudaInvoke.Add(estimate, estimate.NumberOfChannels == 1 ? new ScalarArray(1) : new ScalarArray(new MCvScalar(1, 1, 1)), estimate);
            using var transmission = Clip(estimate);
            Show("EstimateTransmission", transmission);

            return (GpuMat)transmission.Clone();
        }

        private GpuMat QuadDecompositionGpu(GpuMat srcImage, int windowSize = 31)
        {
            using var imageOrg = new GpuMat();
            srcImage.CopyTo(imageOrg);
            while (imageOrg.Size.Width / 2 > windowSize && imageOrg.Size.Height / 2 > windowSize)
            {
                using var image = new GpuMat();
                imageOrg.CopyTo(image);
                var roi1w = new Range(0, (image.Size.Width) / 2);
                var roi1h = new Range(0, (image.Size.Height) / 2);
                var roi2w = new Range((image.Size.Width) / 2, (image.Size.Width - 1));
                var roi2h = new Range(0, (image.Size.Height) / 2);
                var roi3w = new Range((image.Size.Width) / 2, (image.Size.Width - 1));
                var roi3h = new Range((image.Size.Height) / 2, (image.Size.Height - 1));
                var roi4w = new Range(0, (image.Size.Width) / 2);
                var roi4h = new Range((image.Size.Height) / 2, (image.Size.Height - 1));
                using var imagePart1 = image.ColRange(roi1w.Start, roi1w.End).RowRange(roi1h.Start, roi1h.End);
                using var imagePart2 = image.ColRange(roi2w.Start, roi2w.End).RowRange(roi2h.Start, roi2h.End);
                using var imagePart3 = image.ColRange(roi3w.Start, roi3w.End).RowRange(roi3h.Start, roi3h.End);
                using var imagePart4 = image.ColRange(roi4w.Start, roi4w.End).RowRange(roi4h.Start, roi4h.End);
                MCvScalar meanIntensity = new();
                MCvScalar meanIntensity1 = new();
                MCvScalar meanIntensity2 = new();
                MCvScalar meanIntensity3 = new();
                MCvScalar meanIntensity4 = new();
                CudaInvoke.MeanStdDev(GrayMat(imagePart1), ref meanIntensity1, ref meanIntensity);
                CudaInvoke.MeanStdDev(GrayMat(imagePart2), ref meanIntensity2, ref meanIntensity);
                CudaInvoke.MeanStdDev(GrayMat(imagePart3), ref meanIntensity3, ref meanIntensity);
                CudaInvoke.MeanStdDev(GrayMat(imagePart4), ref meanIntensity4, ref meanIntensity);
                if (meanIntensity1.V0 >= meanIntensity2.V0 && meanIntensity1.V0 >= meanIntensity3.V0 && meanIntensity1.V0 >= meanIntensity4.V0)
                {
                    imagePart1.CopyTo(imageOrg);
                }
                else if (meanIntensity2.V0 >= meanIntensity3.V0 && meanIntensity2.V0 >= meanIntensity4.V0)
                {
                    imagePart2.CopyTo(imageOrg);
                }
                else if (meanIntensity3.V0 >= meanIntensity4.V0)
                {
                    imagePart3.CopyTo(imageOrg);
                }
                else
                {
                    imagePart4.CopyTo(imageOrg);
                }
            }
            Show("QuadDecomposition", imageOrg);
            return (GpuMat)imageOrg.Clone();
        }

        private GpuMat RefineTransmission(GpuMat srcImage, GpuMat transmission, double eps, int refineSize)
        {
            using var refinedTransmission = GuidedFilter(srcImage, transmission, refineSize, eps);
            using var result = Clip(refinedTransmission);
            Show("RefineTransmission", Clip(refinedTransmission));
            return (GpuMat)result.Clone();
        }

        private void WatchStart()
        {
            sw.Reset();
            sw.Start();
        }

        private void WatchStop()
        {
            sw.Stop();
            TimeSpan ts = sw.Elapsed;
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                ts.Hours, ts.Minutes, ts.Seconds,
                ts.Milliseconds);
            Debug.WriteLine($"GPU {elapsedTime}");
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~DeHazeGPU()
        // {
        //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        //     Dispose(disposing: false);
        // }
    }
}