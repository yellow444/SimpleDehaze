using Emgu.CV;
using Emgu.CV.Structure;

namespace SimpleDeHaze.Methods
{
    /// <summary>DCP с уточнением t фильтром на минимальном остовном дереве (MST). См. docs/methods/mst-graph-filter.md.</summary>
    public sealed class MstMethod : IDeHazeMethod
    {
        public string Name => "DCP - MST Tree Filter";

        public string Description =>
            "Классический DCP, но карта t уточняется по минимальному остовному дереву (MST) изображения.\n\n" +
            "Пиксели - вершины графа, веса рёбер - цветовая разница соседей. Строится MST; t агрегируется\n" +
            "вдоль дерева за два прохода (O(N)) с весом S(i,j) = exp(-D(i,j)/σ), где D - расстояние по дереву.\n" +
            "Дерево не пересекает резкие границы -> края точные, ореолов почти нет.\n\n" +
            "Полное разрешение (на больших кадрах может быть медленно).\n" +
            "Параметры: σ - ширина ядра близости; ω, patch - DCP.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("omega", "ω - доля удаляемой дымки", 0.3, 0.95, 0.5, search: true),
            new ParamDef("patch", "Патч тёмного канала",      1,   15,   5, 1, isInt: true),
            new ParamDef("min",   "t_min - нижний порог t",   0.01, 0.5, 0.1),
            new ParamDef("sigma", "σ - ширина ядра дерева",   0.02, 0.5, 0.1, log: true, search: true),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
            => DehazeCore.Run(input, p["omega"], (int)p["patch"], p["min"],
                              (i, t) => Refiners.Mst(i, t, p["sigma"]));
    }
}
