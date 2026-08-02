using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace SimpleDeHaze.Benchmarking;

/// <summary>Minimal, strict NumPy .npy reader for the two-dimensional DIODE depth and mask arrays.</summary>
internal static partial class NpyReader
{
    internal sealed record Array2D(float[] Values, int Rows, int Columns, string DType);

    public static Array2D ReadFloat2D(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20,
            FileOptions.SequentialScan);
        Span<byte> prefix = stackalloc byte[8];
        stream.ReadExactly(prefix);
        if (prefix[0] != 0x93 || Encoding.ASCII.GetString(prefix[1..6]) != "NUMPY")
            throw new InvalidDataException($"Not a NumPy .npy file: {path}");

        int major = prefix[6], minor = prefix[7];
        int headerLength;
        if (major == 1)
        {
            Span<byte> length = stackalloc byte[2]; stream.ReadExactly(length);
            headerLength = BinaryPrimitives.ReadUInt16LittleEndian(length);
        }
        else if (major is 2 or 3)
        {
            Span<byte> length = stackalloc byte[4]; stream.ReadExactly(length);
            headerLength = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(length));
        }
        else throw new InvalidDataException($"Unsupported .npy version {major}.{minor}: {path}");

        byte[] headerBytes = new byte[headerLength]; stream.ReadExactly(headerBytes);
        string header = (major == 3 ? Encoding.UTF8 : Encoding.ASCII).GetString(headerBytes);
        Match dtypeMatch = DTypeRegex().Match(header);
        Match fortranMatch = FortranRegex().Match(header);
        Match shapeMatch = ShapeRegex().Match(header);
        if (!dtypeMatch.Success || !fortranMatch.Success || !shapeMatch.Success)
            throw new InvalidDataException($"Malformed .npy header: {path}");
        if (bool.Parse(fortranMatch.Groups[1].Value))
            throw new InvalidDataException($"Fortran-ordered arrays are unsupported: {path}");

        string dtype = dtypeMatch.Groups[1].Value;
        int[] shape = shapeMatch.Groups[1].Value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(x => int.Parse(x, CultureInfo.InvariantCulture)).ToArray();
        bool supportedShape = shape.Length == 2 || shape.Length == 3 && shape[2] == 1;
        if (!supportedShape || shape[0] <= 0 || shape[1] <= 0)
            throw new InvalidDataException($"Expected a non-empty (H,W) or (H,W,1) array, got ({string.Join(',', shape)}): {path}");
        int count = checked(shape.Aggregate(1, (product, dimension) => checked(product * dimension)));
        float[] values = new float[count];

        switch (dtype)
        {
            case "<f4": case "=f4": case "|f4":
            {
                byte[] bytes = new byte[checked(count * 4)]; stream.ReadExactly(bytes);
                for (int i = 0; i < count; i++) values[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(i * 4, 4)));
                break;
            }
            case "<f8": case "=f8": case "|f8":
            {
                byte[] bytes = new byte[checked(count * 8)]; stream.ReadExactly(bytes);
                for (int i = 0; i < count; i++) values[i] = (float)BitConverter.Int64BitsToDouble(BinaryPrimitives.ReadInt64LittleEndian(bytes.AsSpan(i * 8, 8)));
                break;
            }
            case "|b1": case "|u1": case "<u1": case "=u1":
            {
                byte[] bytes = new byte[count]; stream.ReadExactly(bytes);
                for (int i = 0; i < count; i++) values[i] = bytes[i] == 0 ? 0f : 1f;
                break;
            }
            default:
                throw new InvalidDataException($"Unsupported .npy dtype '{dtype}': {path}");
        }
        if (stream.Position != stream.Length)
            throw new InvalidDataException($"Unexpected trailing bytes in .npy file: {path}");
        return new Array2D(values, shape[0], shape[1], dtype);
    }

    [GeneratedRegex("['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]", RegexOptions.CultureInvariant)]
    private static partial Regex DTypeRegex();

    [GeneratedRegex("['\"]fortran_order['\"]\\s*:\\s*(True|False)", RegexOptions.CultureInvariant)]
    private static partial Regex FortranRegex();

    [GeneratedRegex("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)", RegexOptions.CultureInvariant)]
    private static partial Regex ShapeRegex();
}
