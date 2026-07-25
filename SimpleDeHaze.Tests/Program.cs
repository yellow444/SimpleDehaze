namespace SimpleDeHaze.Tests;

internal static class Program
{
    public static int Main()
    {
        var physical = new PhysicalCoreTests();
        var benchmark = new BenchmarkContractTests();
        var a2cr = new A2crTests();
        var npy = new NpyReaderTests();
        var metrics = new BenchmarkMetricTests();
        var diode = new DiodeFormulaTests();
        var autoTune = new AutoTunerSearchTests();
        var tests = new (string Name, Action Run)[]
        {
            (nameof(physical.SrgbCurve_RoundTripsAllByteValues), physical.SrgbCurve_RoundTripsAllByteValues),
            (nameof(physical.BoundaryConstraint_KeepsStandardRecoveryInsideRgbCube), physical.BoundaryConstraint_KeepsStandardRecoveryInsideRgbCube),
            (nameof(physical.ChromaSafeBound_KeepsActualRecoveryInsideRgbCube), physical.ChromaSafeBound_KeepsActualRecoveryInsideRgbCube),
            (nameof(physical.IdentityMetrics_AreExactWithinNumericalTolerance), physical.IdentityMetrics_AreExactWithinNumericalTolerance),
            (nameof(physical.GamutProjection_PreservesRayAndKeepsEveryChannelInsideCube), physical.GamutProjection_PreservesRayAndKeepsEveryChannelInsideCube),
            (nameof(physical.ChromaticAnchor_ExactlyInvertsConstantHazeModelWithoutRegularization), physical.ChromaticAnchor_ExactlyInvertsConstantHazeModelWithoutRegularization),
            (nameof(physical.DenseHazeGate_IsSmoothMonotoneAndHasDeclaredEndpoints), physical.DenseHazeGate_IsSmoothMonotoneAndHasDeclaredEndpoints),
            (nameof(benchmark.CoreProfile_IsExplicitAndDoesNotPretendUnknownMethodIsSupported), benchmark.CoreProfile_IsExplicitAndDoesNotPretendUnknownMethodIsSupported),
            (nameof(benchmark.CoreProfile_DisablesDeclaredRfepPostprocessing), benchmark.CoreProfile_DisablesDeclaredRfepPostprocessing),
            (nameof(benchmark.CoreProfile_RecognizesHazeLinesWithoutCosmeticPostprocessing), benchmark.CoreProfile_RecognizesHazeLinesWithoutCosmeticPostprocessing),
            (nameof(benchmark.Manifest_UsesDeclaredSplitAndResolvesRelativePaths), benchmark.Manifest_UsesDeclaredSplitAndResolvesRelativePaths),
            (nameof(a2cr.RiskGain_RecoversClassicalInverseAndMinimizesQuadratic), a2cr.RiskGain_RecoversClassicalInverseAndMinimizesQuadratic),
            (nameof(a2cr.FeasibleProjector_KeepsEveryRandomPixelInsideCube), a2cr.FeasibleProjector_KeepsEveryRandomPixelInsideCube),
            (nameof(a2cr.EuclideanProjector_IsFeasibleAndNeverFartherThanRayProjection), a2cr.EuclideanProjector_IsFeasibleAndNeverFartherThanRayProjection),
            (nameof(a2cr.CachedFeasiblePolygon_MatchesExactProjection), a2cr.CachedFeasiblePolygon_MatchesExactProjection),
            (nameof(a2cr.Recovery_NoUncertainty_EqualsScalarAtmosphericInverse), a2cr.Recovery_NoUncertainty_EqualsScalarAtmosphericInverse),
            (nameof(a2cr.OpticalDepthFusion_UsesMedianAndReportsDisagreement), a2cr.OpticalDepthFusion_UsesMedianAndReportsDisagreement),
            (nameof(a2cr.TvRefiner_PreservesConstantGain), a2cr.TvRefiner_PreservesConstantGain),
            (nameof(a2cr.TvRefiner_ReducesTotalVariation), a2cr.TvRefiner_ReducesTotalVariation),
            (nameof(a2cr.JointTvSolver_DecreasesObjectiveAndKeepsEveryPixelFeasible), a2cr.JointTvSolver_DecreasesObjectiveAndKeepsEveryPixelFeasible),
            (nameof(a2cr.Method_DefaultPipeline_ProducesFiniteFeasibleOutput), a2cr.Method_DefaultPipeline_ProducesFiniteFeasibleOutput),
            (nameof(a2cr.HsvRecovery_InterpolatesHueAcrossCircularSeam), a2cr.HsvRecovery_InterpolatesHueAcrossCircularSeam),
            (nameof(a2cr.HsvRecovery_UncertaintyShrinksBothIndependentUpdates), a2cr.HsvRecovery_UncertaintyShrinksBothIndependentUpdates),
            (nameof(a2cr.HsvMethod_DefaultPipeline_ProducesFiniteFeasibleOutput), a2cr.HsvMethod_DefaultPipeline_ProducesFiniteFeasibleOutput),
            (nameof(npy.ReadsLittleEndianFloat32Matrix), npy.ReadsLittleEndianFloat32Matrix),
            (nameof(npy.ReadsBooleanMaskAsZeroOne), npy.ReadsBooleanMaskAsZeroOne),
            (nameof(metrics.HueError_IgnoresAchromaticPixelsAndIsZeroForIdentity), metrics.HueError_IgnoresAchromaticPixelsAndIsZeroForIdentity),
            (nameof(metrics.LpipsByteConversion_PreservesBytesAndScalesUnitFloats), metrics.LpipsByteConversion_PreservesBytesAndScalesUnitFloats),
            (nameof(metrics.Ciede2000_MatchesSharmaReferencePairs), metrics.Ciede2000_MatchesSharmaReferencePairs),
            (nameof(metrics.ChromaticFidelity_DetectsColorCollapseAndIsNeutralForGrayGt), metrics.ChromaticFidelity_DetectsColorCollapseAndIsNeutralForGrayGt),
            (nameof(metrics.LocalChromaExpansion_DetectsLocalizedColorExplosion), metrics.LocalChromaExpansion_DetectsLocalizedColorExplosion),
            (nameof(diode.Beta_ReachesTargetTransmissionAtP90Depth), diode.Beta_ReachesTargetTransmissionAtP90Depth),
            (nameof(diode.PoissonGaussianVariance_MatchesDeclaredHighCountApproximation), diode.PoissonGaussianVariance_MatchesDeclaredHighCountApproximation),
            (nameof(autoTune.Coverage_IndependentlyProbesEveryDimensionBeforeSoftBudget), autoTune.Coverage_IndependentlyProbesEveryDimensionBeforeSoftBudget),
            (nameof(autoTune.Search_FindsInteriorMixedScaleOptimumAndProgressNeverRegresses), autoTune.Search_FindsInteriorMixedScaleOptimumAndProgressNeverRegresses),
            (nameof(autoTune.QuantizedCandidates_AreCachedAndDoNotConsumeBudgetAgain), autoTune.QuantizedCandidates_AreCachedAndDoNotConsumeBudgetAgain),
            (nameof(autoTune.ParamDef_StepIsAnchoredAtMinimumAndRoundTrips), autoTune.ParamDef_StepIsAnchoredAtMinimumAndRoundTrips),
            (nameof(autoTune.Halton_UsesUniquePrimeBaseBeyondSixteenDimensions), autoTune.Halton_UsesUniquePrimeBaseBeyondSixteenDimensions),
            (nameof(autoTune.Registry_DefinitionsHaveUniqueKeysAndValidDefaults), autoTune.Registry_DefinitionsHaveUniqueKeysAndValidDefaults),
            (nameof(autoTune.Registry_StructuralModesStayFixedDuringThoroughTune), autoTune.Registry_StructuralModesStayFixedDuringThoroughTune),
            (nameof(autoTune.ChromaticAnchor_SearchCoordinatesAreIndependentAndActuallyProbed), autoTune.ChromaticAnchor_SearchCoordinatesAreIndependentAndActuallyProbed),
            (nameof(autoTune.ReferenceScore_IgnoresGtFittedDiagnosticMetrics), autoTune.ReferenceScore_IgnoresGtFittedDiagnosticMetrics),
            (nameof(autoTune.ReferenceScore_DoesNotDoubleCountMseAndRejectsStructuralCollapse), autoTune.ReferenceScore_DoesNotDoubleCountMseAndRejectsStructuralCollapse),
            (nameof(autoTune.NoReferenceHarshPenalty_UsesNaturalnessOnlyAsExtremeSafetyGuard), autoTune.NoReferenceHarshPenalty_UsesNaturalnessOnlyAsExtremeSafetyGuard),
            (nameof(autoTune.FullResolutionSafety_RejectsGoodhartExtremesButAllowsBoundedImprovement), autoTune.FullResolutionSafety_RejectsGoodhartExtremesButAllowsBoundedImprovement),
            (nameof(autoTune.Search_RejectsRunWhenEveryEvaluationFails), autoTune.Search_RejectsRunWhenEveryEvaluationFails),
        };

        int failed = 0;
        foreach (var test in tests)
        {
            try { test.Run(); Console.WriteLine($"PASS {test.Name}"); }
            catch (Exception ex) { failed++; Console.Error.WriteLine($"FAIL {test.Name}: {ex.Message}"); }
        }
        Console.WriteLine($"TESTS total={tests.Length} passed={tests.Length - failed} failed={failed}");
        return failed == 0 ? 0 : 1;
    }
}

internal static class TestAssert
{
    public static void True(bool value, string? message = null)
    {
        if (!value) throw new InvalidOperationException(message ?? "Expected true");
    }

    public static void False(bool value, string? message = null) => True(!value, message ?? "Expected false");

    public static void Equal<T>(T expected, T actual)
    {
        if (!EqualityComparer<T>.Default.Equals(expected, actual))
            throw new InvalidOperationException($"Expected {expected}, got {actual}");
    }

    public static void InRange(double value, double min, double max)
    {
        if (double.IsNaN(value) || value < min || value > max)
            throw new InvalidOperationException($"Expected [{min}, {max}], got {value}");
    }
}
