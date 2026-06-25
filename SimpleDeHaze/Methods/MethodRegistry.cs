namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// Реестр методов дехейзинга (порядок = порядок в выпадающем списке GUI). Чтобы добавить метод -
    /// реализуйте <see cref="IDeHazeMethod"/> и допишите его сюда: он сам появится в GUI и в --selftest.
    /// </summary>
    public static class MethodRegistry
    {
        public static IReadOnlyList<IDeHazeMethod> All { get; } = new IDeHazeMethod[]
        {
            // базовые приоры
            new DcpCpuMethod(),
            new DcpGpuMethod(),
            new HsvCapMethod(),
            // DCP с разными уточнителями карты t
            new FractionalMethod(),
            new BeltramiMethod(),
            new BeltramiGpuMethod(),
            new MstMethod(),
            new MattingMethod(),
            new MattingGpuMethod(),
            new MultiScaleDcpMethod(),
            new DualChannelMethod(),
            new AdaptiveSoftDcpMethod(),
            new WgifMethod(),
            new LocalAirlightMethod(),
            new TvMethod(),
            new DomainTransformMethod(),
            new FgsMethod(),
            new GradientDomainMethod(),
            new EnergyBasedDcpMethod(),
            // альтернативные пайплайны
            new ColorCubeMethod(),
            new PyramidFusionMethod(),
            new TarelMethod(),
            // гибрид физика + enhancement
            new HybridDcpClaheMethod(),
            // новые алгоритмы: атмосфера + спектр (поканально) + яркость зон
            new SpectralAdaptiveMethod(),
            new FastVeilMethod(),
            // enhancement-методы (не физическая модель дымки)
            new ClaheMethod(),
            new RetinexMethod(),
            new MsrcrMethod(),
        };
    }
}
