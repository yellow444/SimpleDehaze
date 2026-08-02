using System.Text.Json;

using SimpleDeHaze.Benchmarking;
using SimpleDeHaze.Methods;

namespace SimpleDeHaze.Tests;

public sealed class BenchmarkContractTests
{
    public void CoreProfile_IsExplicitAndDoesNotPretendUnknownMethodIsSupported()
    {
        var canonical = new CanonicalDcpMethod();
        var canonicalParams = canonical.Parameters.ToDictionary(x => x.Key, x => x.Default);
        TestAssert.True(BenchmarkProfiles.TryApplyCore(canonical, canonicalParams));
        var clahe = new ClaheMethod();
        var claheParams = clahe.Parameters.ToDictionary(x => x.Key, x => x.Default);
        TestAssert.False(BenchmarkProfiles.TryApplyCore(clahe, claheParams));
        var a2cr = new A2crMethod();
        var a2crParams = a2cr.Parameters.ToDictionary(x => x.Key, x => x.Default);
        TestAssert.True(BenchmarkProfiles.TryApplyCore(a2cr, a2crParams));
        TestAssert.Equal(0, a2crParams["tv"]);
    }

    public void CoreProfile_DisablesDeclaredRfepPostprocessing()
    {
        var method = new RfepDcpMethod();
        var parameters = method.Parameters.ToDictionary(x => x.Key, x => x.Default);
        TestAssert.True(BenchmarkProfiles.TryApplyCore(method, parameters));
        TestAssert.Equal(0, parameters["tone"]);
        TestAssert.Equal(method.Parameters.Single(x => x.Key == "color").Max, parameters["color"]);
    }

    public void CoreProfile_RecognizesHazeLinesWithoutCosmeticPostprocessing()
    {
        var method = new ColorCubeMethod();
        var parameters = method.Parameters.ToDictionary(x => x.Key, x => x.Default);
        TestAssert.True(BenchmarkProfiles.TryApplyCore(method, parameters));
    }

    public void Manifest_UsesDeclaredSplitAndResolvesRelativePaths()
    {
        string root = Path.Combine(AppContext.BaseDirectory, "manifest-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            File.WriteAllBytes(Path.Combine(root, "a.jpg"), new byte[] { 1 });
            File.WriteAllBytes(Path.Combine(root, "b.jpg"), new byte[] { 2 });
            string manifest = Path.Combine(root, "dataset.json");
            File.WriteAllText(manifest, JsonSerializer.Serialize(new
            {
                name = "synthetic", version = "1", root = ".",
                pairs = new[] { new { id = "a", hazy = "a.jpg", split = "val" }, new { id = "b", hazy = "b.jpg", split = "test" } }
            }));
            var loaded = DatasetCatalog.Load(manifest, "unused", "unused", "test", null, int.MaxValue);
            TestAssert.Equal("synthetic", loaded.Name);
            TestAssert.Equal("1", loaded.Version);
            TestAssert.Equal(1, loaded.Cases.Length);
            TestAssert.Equal("b", loaded.Cases[0].Id);
            TestAssert.Equal(Path.GetFullPath(Path.Combine(root, "b.jpg")), loaded.Cases[0].HazyPath);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
