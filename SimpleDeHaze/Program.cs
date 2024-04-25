using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace SimpleDeHaze
{
    internal class Program
    {
        public static void Main(string[] args)
        {
            CvInvoke.UseOptimized = true;
            if (args != null && args.Length == 1)
            {
                var fileSrc = Path.GetFileName(args[0]);
                var path = System.Environment.CurrentDirectory;
                var file = Path.Combine(path, fileSrc);
                var fileName = Path.GetFileNameWithoutExtension(file);
                var fileCpu = Path.Combine(path, $"{fileName}_Cpu.png");
                var fileGpu = Path.Combine(path, $"{fileName}_Gpu.png");
                using var inputImage = new Image<Bgr, byte>(file);
                // Haze Image
                CvInvoke.NamedWindow("Haze Image", WindowFlags.AutoSize);
                CvInvoke.Imshow("Haze Image", inputImage.Convert<Bgr, byte>().Resize((int)(900f / inputImage.Height * inputImage.Width), 800, Inter.Lanczos4));
                CvInvoke.WaitKey();
                var patch = (int)(inputImage.Size.Height > inputImage.Size.Width ? inputImage.Size.Width * 0.01 + 1 : inputImage.Size.Height * 0.001 + 1);
                // DeHazeCPU
                DeHazeCPU deHazeCPU = new();
                var _deHazeCPU = deHazeCPU.RemoveHaze(inputImage.Clone(), debug: false, beta: 0.5f, patchDarkChannel: (int)(patch * 0.5), decompositionSize: (int)(patch * 0.5), min: 2 / 255f, percen: 0.5f, refineSize: (int)(patch * 2), eps: 0.001d / patch);
                CvInvoke.NamedWindow("DeHazeCPU Image", WindowFlags.AutoSize);
                CvInvoke.Imshow("DeHazeCPU Image", (_deHazeCPU * 255).ToImage<Bgr, byte>().Resize((int)(900f / inputImage.Height * inputImage.Width), 800, Inter.Lanczos4));
                CvInvoke.Imwrite(fileCpu, (_deHazeCPU * 255), new KeyValuePair<ImwriteFlags, int>(ImwriteFlags.PngCompression, 0));
                CvInvoke.WaitKey();
                // DeHazeGPU
                using DeHazeGPU deHazeGPU = new();
                using var _deHazeGPU = deHazeGPU.RemoveHaze(inputImage.Clone(), debug: false, beta: 0.5f, patchDarkChannel: (int)(patch * 0.5), decompositionSize: (int)(patch * 0.5), min: 2 / 255f, percen: 0.5f, refineSize: (int)(patch * 2), eps: 0.001d / patch);
                CvInvoke.NamedWindow("DeHazeGPU Image", WindowFlags.AutoSize);
                CvInvoke.Imshow("DeHazeGPU Image", (_deHazeGPU * 255).ToImage<Bgr, byte>().Resize((int)(900f / inputImage.Height * inputImage.Width), 800, Inter.Lanczos4));
                CvInvoke.Imwrite(fileGpu, (_deHazeGPU * 255), new KeyValuePair<ImwriteFlags, int>(ImwriteFlags.PngCompression, 0));
                CvInvoke.WaitKey();
            }
            else
            {
                Test();
            }
        }

        private static void Test()
        {
            var debug = false;
            var files = Directory.GetFiles(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "dataset"), "*.*");
            var filesClear = Directory.GetFiles(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "hazefree"), "*.*");
            var path = System.Environment.CurrentDirectory;
            System.IO.Directory.CreateDirectory(Path.Combine(path, "result"));
            foreach (var file in files)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                var fileName = Path.GetFileNameWithoutExtension(file);
                var fileCpu = Path.Combine(path, "result", $"{fileName}_Cpu.png");
                var fileGpu = Path.Combine(path, "result", $"{fileName}_Gpu.png");
                fileName = Path.Combine(path, "result", $"{fileName}");
                using var inputImage = new Image<Bgr, byte>(file);
                using var clearImage = new Image<Bgr, byte>(filesClear.FirstOrDefault(x => file.Replace("hazy", "GT").Replace("dataset", "hazefree").Equals(x, StringComparison.InvariantCultureIgnoreCase)));
                // Haze Image
                CvInvoke.NamedWindow("Haze Image", WindowFlags.AutoSize);
                CvInvoke.Imshow("Haze Image", inputImage.Convert<Bgr, byte>().Resize((int)(900f / inputImage.Height * inputImage.Width), 800, Inter.Lanczos4));
                // Clear Image
                CvInvoke.NamedWindow("Clear Image", WindowFlags.AutoSize);
                CvInvoke.Imshow("Clear Image", clearImage.Convert<Bgr, byte>().Resize((int)(900f / inputImage.Height * inputImage.Width), 800, Inter.Lanczos4));
                var patch = (int)(inputImage.Size.Height > inputImage.Size.Width ? inputImage.Size.Width * 0.01 + 1 : inputImage.Size.Height * 0.001 + 1);
                if (debug) CvInvoke.WaitKey();
                // DeHazeCPU
                DeHazeCPU deHazeCPU = new();
                var _deHazeCPU = deHazeCPU.RemoveHaze(inputImage.Clone(), debug: debug, beta: 0.5f, patchDarkChannel: (int)(patch * 0.5), decompositionSize: (int)(patch * 0.5), min: 2 / 255f, percen: 0.5f, refineSize: (int)(patch * 2), eps: 0.001d / patch, fileName);
                CvInvoke.NamedWindow("DeHazeCPU Image", WindowFlags.AutoSize);
                CvInvoke.Imshow("DeHazeCPU Image", (_deHazeCPU * 255).ToImage<Bgr, byte>().Resize((int)(900f / inputImage.Height * inputImage.Width), 800, Inter.Lanczos4));
                CvInvoke.Imwrite(fileCpu, _deHazeCPU * 255, new KeyValuePair<ImwriteFlags, int>(ImwriteFlags.PngCompression, 0));
                if(debug) CvInvoke.WaitKey();

                // DeHazeGPU
                using DeHazeGPU deHazeGPU = new();
                using var _deHazeGPU = deHazeGPU.RemoveHaze(inputImage.Clone(), debug: debug, beta: 0.5f, patchDarkChannel: (int)(patch * 0.5), decompositionSize: (int)(patch * 0.5), min: 2 / 255f, percen: 0.5f, refineSize: (int)(patch * 2), eps: 0.001d / patch, fileName);
                CvInvoke.NamedWindow("DeHazeGPU Image", WindowFlags.AutoSize);
                CvInvoke.Imshow("DeHazeGPU Image", (_deHazeGPU * 255).ToImage<Bgr, byte>().Resize((int)(900f / inputImage.Height * inputImage.Width), 800, Inter.Lanczos4));
                CvInvoke.Imwrite(fileGpu, _deHazeGPU * 255, new KeyValuePair<ImwriteFlags, int>(ImwriteFlags.PngCompression, 0));
                if (debug) CvInvoke.WaitKey();
            }
        }
    }
}