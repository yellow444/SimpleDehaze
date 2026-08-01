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
            // базовые приоры (первым - эталонный baseline без косметики)
            new CanonicalDcpMethod(),
            new A2crMethod(),
            new HcvA2crMethod(),
            new HcvRgbA2crFusionMethod(),
            new HcvA2crUtawMethod(),
            new HsvA2crMethod(),
            new HsvC3rMethod(),
            new DcpCpuMethod(),
            new DcpGpuMethod(),
            new HsvCapMethod(),
            new CapLocalMethod(),
            new ChromaticAnchorMethod(),
            new BraceDcpMethod(),
            new RfepDcpMethod(),
            // DCP с разными уточнителями карты t
            new FractionalMethod(),
            new BeltramiMethod(),
            new BeltramiGpuMethod(),
            new MstMethod(),
            new MattingMethod(),
            new MattingGpuMethod(),
            new MultiScaleDcpMethod(),
            new PfSfgfMethod(),
            new DualChannelMethod(),
            new AdaptiveSoftDcpMethod(),
            new WgifMethod(),
            new LocalAirlightMethod(),
            new LafTvMethod(),
            new TvMethod(),
            new DomainTransformMethod(),
            new FgsMethod(),
            new GradientDomainMethod(),
            new GdrSpMethod(),
            new EnergyBasedDcpMethod(),
            // альтернативные пайплайны
            new ColorCubeMethod(),
            new PyramidFusionMethod(),
            new TarelMethod(),
            // гибрид физика + enhancement
            new HybridDcpClaheMethod(),
            // новые алгоритмы: атмосфера + спектр (поканально) + яркость зон
            new ColorContrastRestoreMethod(),
            new SpectralAdaptiveMethod(),
            new LocalVisibilityFastMethod(),
            new LocalVisibilityQualityMethod(),
            new SilhouetteVisibilityMethod(),
            new VisibilityBoostMethod(),
            new FastVeilMethod(),
            // локально-адаптивная дымка (пространственно-неоднородная): баланс / точность / объекты / сочно
            new LocalHazeBalanceMethod(),
            new LocalHazeFidelityMethod(),
            new LocalHazeObjectsMethod(),
            new LocalHazeVividMethod(),
            new LocalHazeCoarseRevealMethod(),
            // комбо: многомасштабные контуры (Лапласиан-пирамида) и цепочка алгоритмов
            new LaplacianContourMethod(),
            new TransScaleLaplacianMethod(),
            new TransmissionAwareHsvEdgeMethod(),
            new TransmissionAwareHsvUtawMethod(),
            new TransmissionAwareHsvUtawGpuMethod(),
            new FractalHsvMethod(),
            new ChainMethod(),
            // enhancement-методы (не физическая модель дымки)
            new ClaheMethod(),
            new RetinexMethod(),
            new MsrcrMethod(),
        };

        /// <summary>
        /// Выделенные методы для быстрого знакомства с разными семействами алгоритмов. Звезда означает
        /// только curated-набор интерфейса, а не превосходство по качеству или универсальность.
        /// </summary>
        public static readonly HashSet<string> Recommended = new()
        {
            "A²CR-Dehaze (dual-gain recovery, эксперимент)",
            "HSV + многомасштабная шероховатость (эксперимент)",
            "Transmission-aware Laplacian (эксперимент)",
            "Transmission-aware HSV UTAW (эксперимент)",
            "Color Attenuation+ (адаптивная глубина, баланс белого)",
            "Color Attenuation Prior (HSV)",                        // лёгкий, хорошо держит контуры
            "Локальная дымка - баланс (цвет + объекты)",            // характер по умолчанию: цвет за дымкой + читаемость
            "Локальная дымка - точность (PSNR/MSE/SSIM)",           // лучший под метрики к эталону
            "Многомасштабные контуры (Лапласиан-пирамида)",         // контуры на всех масштабах
        };
    }
}
