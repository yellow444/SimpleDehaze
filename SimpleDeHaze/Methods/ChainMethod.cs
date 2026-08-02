using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// «Цепочка алгоритмов»: прогоняет до трёх ЛЮБЫХ методов подряд - выход одного идёт на вход следующего.
    /// Шаг задаётся НОМЕРОМ метода как в выпадающем списке («01.», «02.»…); 0 = пропустить. Саму «Цепочку»
    /// как шаг использовать нельзя (пропускается). Каждый шаг считается со своими параметрами по умолчанию;
    /// в конце общий мягкий тон/вибранс/потолок цвета.
    /// </summary>
    public sealed class ChainMethod : IDeHazeMethod
    {
        public string Name => "Цепочка алгоритмов (комбо)";

        public string Description =>
            "Несколько алгоритмов подряд (выход → вход следующего).\n\n" +
            "Шаг 1/2/3 - это НОМЕР метода как в списке методов («01.», «02.», …). 0 = пропустить.\n" +
            "Сами комбо-методы (эта «Цепочка») шагом быть не могут - пропускаются.\n\n" +
            "Примеры: 4 → 0 → 0 = один CAP+; «CAP+ → Многомасштабные контуры → CLAHE» — поставьте номера\n" +
            "этих методов из списка. Каждый шаг идёт с параметрами по умолчанию; в конце общий тон/вибранс.\n\n" +
            "Формула: результат = f_k(…f_2(f_1(I))…) — последовательная композиция выбранных методов, затем общий тон/вибранс.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("s1",      "Шаг 1: № метода (0=пропустить)", 0, 60, 4, 1, isInt: true, tunable: false),
            new ParamDef("s2",      "Шаг 2: № метода (0=пропустить)", 0, 60, 0, 1, isInt: true, tunable: false),
            new ParamDef("s3",      "Шаг 3: № метода (0=пропустить)", 0, 60, 0, 1, isInt: true, tunable: false),
            new ParamDef("sat",     "Вибранс цвета",                 0.0,  0.8,  0.20, search: true),
            new ParamDef("tone",    "Возврат тона (растяжение L)",   0.0,  1.0,  0.40),
            new ParamDef("color",   "Потолок усиления цветности",    1.05, 1.7,  1.40),
            new ParamDef("smooth",  "Шумоподавление",               0.0,  6.0,  0.5),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            int s1 = (int)p["s1"], s2 = (int)p["s2"], s3 = (int)p["s3"];
            double sat = p["sat"], tone = p["tone"], color = p["color"], smooth = p["smooth"];

            Image<Bgr, byte> cur = input;     // вход не освобождаем
            bool ownCur = false;
            Mat? last = null;
            try
            {
                foreach (int s in new[] { s1, s2, s3 })
                {
                    var outF = RunStage(s, cur);
                    if (outF == null) continue;        // пропуск/неверный номер/комбо
                    last?.Dispose();
                    last = outF;
                    var nextImg = ToImage(outF);
                    if (ownCur) cur.Dispose();
                    cur = nextImg; ownCur = true;
                }
            }
            finally { if (ownCur) cur.Dispose(); }

            using var result01 = last ?? Normalize(input);   // ни один шаг не выбран -> просто вход
            using var boosted = DehazeCore.LabEnhance(result01, 0.0, 8, sat, 0.0);
            using var toned = DehazeCore.RestoreTone(boosted, tone, 0.01);
            using var limited = DehazeCore.LimitColorfulness(toned, input.Mat, color);
            return smooth > 0.01 ? DehazeCore.BilateralDenoise(limited, smooth) : DeHazeCPU.Clip(limited.Clone());
        }

        /// <summary>Запустить метод по НОМЕРУ из списка (1-based). null = пропустить (0, вне диапазона или комбо).</summary>
        private static Mat? RunStage(int num, Image<Bgr, byte> img)
        {
            int idx = num - 1;
            if (idx < 0 || idx >= MethodRegistry.All.Count) return null;
            var m = MethodRegistry.All[idx];
            if (m is ChainMethod) return null;            // не вкладываем комбо в себя
            return m.Process(img, m.Parameters.ToDictionary(x => x.Key, x => x.Default));
        }

        private static Mat Normalize(Image<Bgr, byte> img)
        {
            var m = new Mat();
            img.Mat.ConvertTo(m, DepthType.Cv32F, 1.0 / 255.0);
            return m;
        }

        private static Image<Bgr, byte> ToImage(Mat float01)
        {
            using var m8 = new Mat();
            float01.ConvertTo(m8, DepthType.Cv8U, 255.0);
            return m8.ToImage<Bgr, byte>();
        }
    }
}
