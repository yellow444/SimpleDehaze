using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    public sealed class TransmissionAwareHsvUtawGpuMethod : IDeHazeMethod
    {
        private static readonly TransScaleLaplacianMethod BaseMethod = new();
        private static readonly TransmissionAwareHsvUtawMethod CpuMethod = new();

        public string Name => "Transmission-aware HSV UTAW (GPU CUDA, эксперимент)";

        public static bool IsCudaAvailable
        {
            get
            {
                try { return GpuStationaryAtrous.IsAvailable; }
                catch { return false; }
            }
        }

        public string Description =>
            "Тот же HSV-V stationary à trous stage и те же reliability/budget formulas, что в CPU " +
            "UTAW, но разреженные B3 filters и coefficient arithmetic выполняются на CUDA. " +
            "Общий transmission/local-airlight/recovery пока остаётся CPU, поэтому это hybrid GPU " +
            "prototype. Это строгий ручной CUDA-режим: при отсутствии CUDA выводится ошибка, " +
            "автоматического отката на CPU нет.";

        public IReadOnlyList<ParamDef> Parameters => CpuMethod.Parameters;

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> parameters)
        {
            if (!IsCudaAvailable)
                throw new InvalidOperationException(
                    "CUDA недоступна. Для расчёта без видеокарты выберите " +
                    "'Transmission-aware HSV UTAW (эксперимент)'.");

            var selected = new Dictionary<string, double>(parameters, StringComparer.Ordinal)
            {
                ["space"] = 1,
                ["basis"] = 3,
            };
            return BaseMethod.Process(input, selected);
        }
    }
}
