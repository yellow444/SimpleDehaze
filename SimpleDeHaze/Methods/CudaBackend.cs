using Emgu.CV.Cuda;

namespace SimpleDeHaze.Methods
{
    /// <summary>Единый контракт ручного выбора CPU/CUDA для методов с настоящей GPU-реализацией.</summary>
    internal static class CudaBackend
    {
        public const string ParameterKey = "cuda";

        public static bool IsAvailable
        {
            get
            {
                try { return CudaInvoke.HasCuda && CudaInvoke.GetCudaEnabledDeviceCount() > 0; }
                catch { return false; }
            }
        }

        public static ParamDef ModeParameter => new(
            ParameterKey, "Вычисление: 0=CPU, 1=CUDA", 0, 1, 0, 1,
            isInt: true, tunable: false, isEnabled: IsAvailable);

        public static bool IsRequested(IReadOnlyDictionary<string, double> parameters)
            => parameters.TryGetValue(ParameterKey, out double value) && value >= 0.5;

        public static void RequireAvailable()
        {
            if (!IsAvailable)
                throw new InvalidOperationException(
                    "CUDA недоступна в OpenCV. Переключите постоянный параметр 'Вычисление' на CPU.");
        }
    }
}
