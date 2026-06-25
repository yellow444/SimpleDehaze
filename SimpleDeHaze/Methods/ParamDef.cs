namespace SimpleDeHaze.Methods
{
    /// <summary>Описание одного настраиваемого параметра метода (для авто-ползунка и авто-подбора в GUI).</summary>
    public sealed class ParamDef
    {
        public string Key { get; }
        public string Label { get; }
        public double Min { get; }
        public double Max { get; }
        public double Default { get; }
        public double Step { get; }
        public bool IsInt { get; }
        public bool Log { get; }
        /// <summary>Включать ли этот параметр в авто-подбор 'оптимальных параметров' (поиск по метрике).</summary>
        public bool Search { get; }

        public ParamDef(string key, string label, double min, double max, double @default,
                        double step = 0, bool isInt = false, bool log = false, bool search = false)
        {
            Key = key;
            Label = label;
            Min = min;
            Max = max;
            Default = @default;
            Step = step;
            IsInt = isInt;
            Log = log;
            Search = search;
        }
    }
}
