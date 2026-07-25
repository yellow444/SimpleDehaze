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
        /// <summary>
        /// Можно ли менять параметр при тщательном авто-подборе. false для режимных/структурных
        /// переключателей (напр. быстро/HQ, фактор апскейла), которые задаёт пользователь, а не метрика.
        /// </summary>
        public bool Tunable { get; }

        public ParamDef(string key, string label, double min, double max, double @default,
                        double step = 0, bool isInt = false, bool log = false, bool search = false, bool tunable = true)
        {
            if (string.IsNullOrWhiteSpace(key)) throw new ArgumentException("Ключ параметра не может быть пустым.", nameof(key));
            if (!double.IsFinite(min)) throw new ArgumentOutOfRangeException(nameof(min), "Минимум должен быть конечным числом.");
            if (!double.IsFinite(max) || max <= min) throw new ArgumentOutOfRangeException(nameof(max), "Максимум должен быть конечным и больше минимума.");
            if (!double.IsFinite(@default) || @default < min || @default > max)
                throw new ArgumentOutOfRangeException(nameof(@default), "Значение по умолчанию должно лежать в [min, max].");
            if (!double.IsFinite(step) || step < 0) throw new ArgumentOutOfRangeException(nameof(step), "Шаг должен быть конечным и неотрицательным.");
            if (log && min <= 0) throw new ArgumentOutOfRangeException(nameof(min), "Логарифмическая шкала требует min > 0.");

            Key = key;
            Label = label;
            Min = min;
            Max = max;
            Default = @default;
            Step = step;
            IsInt = isInt;
            Log = log;
            Search = search;
            Tunable = tunable;
        }

        /// <summary>
        /// Приводит произвольное значение к реально допустимому значению параметра. Дискретная сетка
        /// отсчитывается от <see cref="Min"/>: например, диапазон 3..51 с шагом 2 содержит только
        /// нечётные числа. Этим методом должны пользоваться и GUI, и автоматический подбор.
        /// </summary>
        public double Coerce(double value)
        {
            if (!double.IsFinite(value)) value = Default;
            value = Math.Clamp(value, Min, Max);

            double quantum = Step > 0 ? Step : (IsInt ? 1.0 : 0.0);
            if (quantum > 0)
            {
                double ticks = Math.Round((value - Min) / quantum, MidpointRounding.AwayFromZero);
                value = Min + ticks * quantum;
            }
            if (IsInt) value = Math.Round(value, MidpointRounding.AwayFromZero);
            return Math.Clamp(value, Min, Max);
        }

        /// <summary>Преобразует координату [0,1] в допустимое значение с учётом log/int/step.</summary>
        public double FromFraction(double fraction)
        {
            fraction = Math.Clamp(double.IsFinite(fraction) ? fraction : 0.0, 0.0, 1.0);
            double value = Log
                ? Min * Math.Pow(Max / Min, fraction)
                : Min + fraction * (Max - Min);
            return Coerce(value);
        }

        /// <summary>Преобразует значение параметра в каноническую координату [0,1].</summary>
        public double ToFraction(double value)
        {
            value = Coerce(value);
            double fraction = Log
                ? Math.Log(value / Min) / Math.Log(Max / Min)
                : (value - Min) / (Max - Min);
            return double.IsFinite(fraction) ? Math.Clamp(fraction, 0.0, 1.0) : 0.0;
        }
    }
}
