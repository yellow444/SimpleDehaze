using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Единый контракт метода дехейзинга. По <see cref="Parameters"/> GUI строит ползунки,
    /// <see cref="Description"/> показывает в панели описания, <see cref="Process"/> запускает расчёт.
    /// Авто-подбор параметров - общий (AutoTuner), отдельный метод не нужен.
    /// </summary>
    public interface IDeHazeMethod
    {
        /// <summary>Имя для выпадающего списка.</summary>
        string Name { get; }

        /// <summary>Краткое описание: суть, шаги алгоритма, формулы (текст с Unicode-символами).</summary>
        string Description { get; }

        /// <summary>Настраиваемые параметры метода.</summary>
        IReadOnlyList<ParamDef> Parameters { get; }

        /// <summary>Дехейзинг. Возвращает BGR float в [0,1] (GUI домножит на 255 для показа/сохранения).</summary>
        Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p);
    }
}
