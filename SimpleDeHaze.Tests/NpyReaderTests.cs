using System.Text;

using SimpleDeHaze.Benchmarking;

namespace SimpleDeHaze.Tests;

public sealed class NpyReaderTests
{
    public void ReadsLittleEndianFloat32Matrix()
    {
        string path = TemporaryPath();
        try
        {
            float[] expected = { 1.25f, -2.5f, 3.75f, 4, 5, 6 };
            WriteNpyV1(path, "<f4", "(2, 3, 1)", writer =>
            {
                foreach (float value in expected) writer.Write(value);
            });
            var actual = NpyReader.ReadFloat2D(path);
            TestAssert.Equal(2, actual.Rows);
            TestAssert.Equal(3, actual.Columns);
            TestAssert.Equal("<f4", actual.DType);
            TestAssert.InRange(actual.Values.Zip(expected).Max(x => Math.Abs(x.First - x.Second)), 0, 0);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    public void ReadsBooleanMaskAsZeroOne()
    {
        string path = TemporaryPath();
        try
        {
            byte[] expected = { 0, 1, 1, 0 };
            WriteNpyV1(path, "|b1", "(2, 2)", writer => writer.Write(expected));
            var actual = NpyReader.ReadFloat2D(path);
            TestAssert.True(actual.Values.SequenceEqual(new[] { 0f, 1f, 1f, 0f }));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    private static void WriteNpyV1(string path, string dtype, string shape, Action<BinaryWriter> writeValues)
    {
        string core = $"{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape}, }}";
        int padding = (16 - ((10 + core.Length + 1) % 16)) % 16;
        byte[] header = Encoding.ASCII.GetBytes(core + new string(' ', padding) + "\n");
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: false);
        writer.Write(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 1, 0 });
        writer.Write((ushort)header.Length);
        writer.Write(header);
        writeValues(writer);
    }

    private static string TemporaryPath()
    {
        string directory = Path.Combine(Environment.CurrentDirectory, ".test-tmp");
        Directory.CreateDirectory(directory);
        return Path.Combine(directory, $"simpledehaze-{Guid.NewGuid():N}.npy");
    }
}
