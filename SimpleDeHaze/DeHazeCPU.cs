using System.Diagnostics;
using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze
{
    public class DeHazeCPU : IDisposable
    {
        private bool _debug = false;
        private string _file = "";
        private bool disposedValue;
        private readonly Stopwatch sw = new();

        public DeHazeCPU()
        {
        }

        public static Mat Clip(Mat image)
        {
            CvInvoke.Threshold(image, image, 1, 1, ThresholdType.Trunc);
            image = -1 * image;
            CvInvoke.Threshold(image, image, 0, 0, ThresholdType.Trunc);
            image = -1 * image;
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

            var srcImage = image.Convert<Bgr, float>() / 255;
            var quadDecomposition = QuadDecomposition(srcImage, decompositionSize);
            var darkChannel = ComputeDarkChannelPatch(quadDecomposition, patchDarkChannel);
            var atmosphericLightScalar = ComputeAtmosphericLight(quadDecomposition, darkChannel, percen);
            var colorsChannel = ComputeColorsChannelPatch(srcImage, patchDarkChannel);
            var transmission = EstimateTransmission(srcImage.Mat, colorsChannel, atmosphericLightScalar, beta);
            var refinedTransmission = RefineTransmission(srcImage, transmission, eps, refineSize, min);
            var dehazedImage = RecoverImage(srcImage, refinedTransmission, atmosphericLightScalar, min);
#if RELEASE

#else
            WatchStop();
#endif

            return dehazedImage;
        }

        public void Show(string name, Mat image)
        {
            if (!_debug) return;
            var result = new Mat();
            image.ConvertTo(result, DepthType.Cv8U);
            CvInvoke.NamedWindow($"DeHazeCPU {name} Image{result.Size}", WindowFlags.AutoSize);
            CvInvoke.Imshow($"DeHazeCPU {name} Image{result.Size}", result.ToImage<Bgr, byte>().Resize((int)(900f / result.Height * result.Width), 800, Inter.Lanczos4));
            var v = _file + $"DeHazeCPU {name} Image{result.Size}.png".Replace("{", "").Replace("}", "").Replace("=", "").Replace(",", "");
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

        private static MCvScalar ComputeAtmosphericLight(Image<Bgr, float> srcImage, Mat darkChannel, float percen)
        {
            var channels = (srcImage * 255).Convert<Bgr, byte>().Split();
            var channelsData = new byte[3][];
            for (var i = 0; i < channels.Length; i++)
            {
                channelsData[i] = new byte[channels[0].Width * channels[0].Height];
                channels[i].Mat.CopyTo(channelsData[i]);
            }
            var darkChannelData = new byte[channels[0].Width * channels[0].Height];
            var darkChannelByte = (darkChannel * 255).ToImage<Gray, byte>().Mat;
            darkChannelByte.CopyTo(darkChannelData);
            var sortedIndices = Enumerable.Range(0, darkChannelData.Length).OrderByDescending(i => darkChannelData[i]).ToArray();
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
            return new MCvScalar(atmosphericLightB, atmosphericLightG, atmosphericLightR);
        }

        private Mat ComputeColorsChannelPatch(Image<Bgr, float> srcImage, int patch, string name = "")
        {
            var bgrChannels = srcImage.Clone().Mat;
            var colorsChannel = new Mat();
            CvInvoke.Min(bgrChannels, bgrChannels, colorsChannel);
            CvInvoke.Erode(colorsChannel, colorsChannel, null, new Point(-1, -1), patch, BorderType.Reflect101, default);
            Show("ComputeColorsChannelPatch" + name, colorsChannel * 255);
            return colorsChannel;
        }

        private Mat ComputeDarkChannelPatch(Image<Bgr, float> srcImage, int patch, string name = "")
        {
            var bgrChannels = srcImage.Clone().Mat.Split();
            var darkChannel = new Mat();
            CvInvoke.Min(bgrChannels[0], bgrChannels[1], darkChannel);
            CvInvoke.Min(darkChannel, bgrChannels[2], darkChannel);
            CvInvoke.Erode(darkChannel, darkChannel, null, new Point(-1, -1), patch, BorderType.Reflect101, default);
            Show("ComputeDarkChannel" + name, darkChannel * 255);
            return darkChannel;
        }

        private Mat EstimateTransmission(Mat srcImage, Mat colorsChannel, MCvScalar atmosphericLight, float beta)
        {
            var estimateSplit = srcImage.Clone().Split();
            var colorsChannelMin = colorsChannel.Clone().Split();
            estimateSplit[0] = -beta * atmosphericLight.V0 / colorsChannelMin[0];
            estimateSplit[1] = -beta * atmosphericLight.V1 / colorsChannelMin[1];
            estimateSplit[2] = -beta * atmosphericLight.V2 / colorsChannelMin[2];
            var estimate = new Mat();
            CvInvoke.Merge(new VectorOfMat(estimateSplit), estimate);
            estimate.ConvertTo(estimate, DepthType.Cv32F);
            var transmission = estimate;
            CvInvoke.Exp(transmission, transmission);
            transmission = 1 - transmission;
            transmission = Clip(transmission);
            Show("EstimateTransmission", transmission * 255);
            return transmission;
        }

        private static Mat GrayMat(Mat srcImage)
        {
            var channels = srcImage.Split();
            var grayscale = 0.114f * channels[0] + 0.587f * channels[1] + 0.299f * channels[2];
            return grayscale;
        }

        private Image<Bgr, float> QuadDecomposition(Image<Bgr, float> srcImage, int windowSize = 31)
        {
            var image = srcImage.Clone();
            while (image.Width / 2 > windowSize && image.Height / 2 > windowSize)
            {
                Rectangle roi1 = new(0, 0, (image.Width) / 2, (image.Height) / 2);
                Rectangle roi2 = new((image.Width) / 2, 0, (image.Width - 1), (image.Height) / 2);
                Rectangle roi3 = new((image.Width) / 2, (image.Height) / 2, (image.Width - 1), (image.Height - 1));
                Rectangle roi4 = new(0, (image.Height) / 2, (image.Width) / 2, (image.Height - 1));
                image.ROI = roi1;
                Mat imagePart1 = image.Mat.Clone();
                image.ROI = roi2;
                Mat imagePart2 = image.Mat.Clone();
                image.ROI = roi3;
                Mat imagePart3 = image.Mat.Clone();
                image.ROI = roi4;
                Mat imagePart4 = image.Mat.Clone();
                double meanIntensity1 = CvInvoke.Mean(GrayMat(imagePart1)).V0;
                double meanIntensity2 = CvInvoke.Mean(GrayMat(imagePart2)).V0;
                double meanIntensity3 = CvInvoke.Mean(GrayMat(imagePart3)).V0;
                double meanIntensity4 = CvInvoke.Mean(GrayMat(imagePart4)).V0;
                if (meanIntensity1 >= meanIntensity2 && meanIntensity1 >= meanIntensity3 && meanIntensity1 >= meanIntensity4)
                {
                    image = imagePart1.ToImage<Bgr, float>();
                }
                else if (meanIntensity2 >= meanIntensity3 && meanIntensity2 >= meanIntensity4)
                {
                    image = imagePart2.ToImage<Bgr, float>();
                }
                else if (meanIntensity3 >= meanIntensity4)
                {
                    image = imagePart3.ToImage<Bgr, float>();
                }
                else
                {
                    image = imagePart4.ToImage<Bgr, float>();
                }
            }
            Show("QuadDecomposition", image.Mat * 255);
            return image;
        }

        private static Mat RecoverImage(Image<Bgr, float> srcImage, Mat transmission, MCvScalar atmosphericLight, float min)
        {
            var srcImageSplit = srcImage.Mat.Clone().Split();
            srcImageSplit[0] = srcImageSplit[0] - atmosphericLight.V0;
            srcImageSplit[1] = srcImageSplit[1] - atmosphericLight.V1;
            srcImageSplit[2] = srcImageSplit[2] - atmosphericLight.V2;
            CvInvoke.PatchNaNs(transmission, min);
            var transmissionSplit = transmission.Split();
            var minMat = transmissionSplit[0].Clone();
            minMat.SetTo(new MCvScalar(min));
            CvInvoke.Max(transmissionSplit[0], minMat, transmissionSplit[0]);
            CvInvoke.Max(transmissionSplit[1], minMat, transmissionSplit[1]);
            CvInvoke.Max(transmissionSplit[2], minMat, transmissionSplit[2]);
            CvInvoke.Divide(srcImageSplit[0], transmissionSplit[0], srcImageSplit[0]);
            CvInvoke.Divide(srcImageSplit[1], transmissionSplit[1], srcImageSplit[1]);
            CvInvoke.Divide(srcImageSplit[2], transmissionSplit[2], srcImageSplit[2]);
            srcImageSplit[0] = srcImageSplit[0] + atmosphericLight.V0;
            srcImageSplit[1] = srcImageSplit[1] + atmosphericLight.V1;
            srcImageSplit[2] = srcImageSplit[2] + atmosphericLight.V2;
            var result = new Mat();
            CvInvoke.Merge(new VectorOfMat(srcImageSplit), result);
            result = Clip(result);
            return result;
        }

        private Mat RefineTransmission(Image<Bgr, float> srcImage, Mat transmission, double eps, int refineSize, float min)
        {
            var refinedTransmission = new Mat();
            XImgprocInvoke.GuidedFilter(srcImage.Mat, transmission, refinedTransmission, refineSize, eps);
            CvInvoke.PatchNaNs(refinedTransmission, min);
            refinedTransmission = Clip(refinedTransmission);
            Show("RefineTransmission", refinedTransmission * 255);
            return refinedTransmission;
        }

        private void WatchStart()
        {
            sw.Restart();
            sw.Start();
        }

        private void WatchStop()
        {
            sw.Stop();
            TimeSpan ts = sw.Elapsed;
            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
                ts.Hours, ts.Minutes, ts.Seconds,
                ts.Milliseconds);
            Debug.WriteLine($"CPU {elapsedTime}");
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~DeHazeCPU()
        // {
        //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        //     Dispose(disposing: false);
        // }
    }
}