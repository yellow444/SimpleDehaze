using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    internal sealed class ChromaticAnchorExecution : IDisposable
    {
        public Mat Result { get; }
        public Mat Transmission { get; }
        public Mat AdjustedDepth { get; }
        public Mat PhysicalSrgb { get; }
        public Mat AirlightSrgb { get; }
        public Mat ProjectionAlpha { get; }
        public MCvScalar GlobalAirlight { get; }
        public double RawOutOfGamutFraction { get; }
        public GamutProjectionSummary ProjectionSummary { get; }
        public bool LinearRadiance { get; }

        public ChromaticAnchorExecution(Mat result, Mat transmission, Mat adjustedDepth, Mat physicalSrgb,
            Mat airlightSrgb, Mat projectionAlpha, MCvScalar globalAirlight, double rawOutOfGamutFraction,
            GamutProjectionSummary projectionSummary, bool linearRadiance)
        {
            Result = result;
            Transmission = transmission;
            AdjustedDepth = adjustedDepth;
            PhysicalSrgb = physicalSrgb;
            AirlightSrgb = airlightSrgb;
            ProjectionAlpha = projectionAlpha;
            GlobalAirlight = globalAirlight;
            RawOutOfGamutFraction = rawOutOfGamutFraction;
            ProjectionSummary = projectionSummary;
            LinearRadiance = linearRadiance;
        }

        public void Dispose()
        {
            Result.Dispose();
            Transmission.Dispose();
            AdjustedDepth.Dispose();
            PhysicalSrgb.Dispose();
            AirlightSrgb.Dispose();
            ProjectionAlpha.Dispose();
        }
    }

    /// <summary>
    /// Экспериментальное восстановление крупномасштабного цвета за плотной дымкой. Transmission
    /// оценивается известным Color Attenuation Prior, но физическая инверсия выполняется в линейном RGB.
    /// Новая проверяемая часть — декомпозиция восстановления на локальную среднюю яркость airlight и
    /// глобально-якорную хроматичность плюс coarse-scale chroma denoising.
    /// </summary>
    public sealed class ChromaticAnchorMethod : IDeHazeMethod
    {
        private delegate void DiagnosticCapture(Mat transmission, Mat adjustedDepth, Mat recovered,
            Mat physicalSrgb, Mat[] airField, Mat projectionAlpha, MCvScalar globalAirlight,
            GamutProjectionSummary projectionSummary, bool linearRadiance);

        public string Name => "Chromatic Airlight Residual (CAR-Dehaze, эксперимент)";

        public string Description =>
            "Эксперимент для широких цветных объектов за плотной дымкой (например красноватая стена O-HAZE #08).\n\n" +
            "1. CAP в HSV оценивает t(x), но атмосферная инверсия выполняется в линейном RGB.\n" +
            "2. Средняя яркость восстанавливается с локальным полем A(x).\n" +
            "3. Для centered RGB используется C(J)=[C(I)-(1-t)C(A_g)]/max(t,t_c): локальное поле не может\n" +
            "   поглотить цвет широкой стены и затем нейтрализовать его как оттенок тумана.\n" +
            "4. В плотной дымке C(I) смешивается с крупномасштабной Gaussian-оценкой: цветовой регион\n" +
            "   сохраняется, поканальный шум усредняется.\n" +
            "5. Стабильная DCP-ветвь задаёт L*, а физическая ветвь — только низкочастотные a*/b*.\n" +
            "   Вес равен (1-t)^p·α^q, где α — точная gamut-проекция: недостоверные выбросы цвета\n" +
            "   автоматически подавляются, широкие цветные области с α≈1 сохраняются.\n\n" +
            "Статус: проверяемая гипотеза, а не доказанное превосходство; scene #08 — development case.";

        private const double Theta0 = 0.121779, Theta1 = 0.959710, Theta2 = -0.780245;

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("beta",    "β Color Attenuation",                    0.1,  3.0,  1.35, search: true),
            new ParamDef("depth",   "Адаптивная глубина плотных зон",         0.0,  2.5,  0.70, search: true),
            new ParamDef("rmin",    "Радиус min-фильтра CAP",                1,    25,   7,    1, isInt: true),
            new ParamDef("rguide",  "Радиус Guided Filter",                  5,    120,  40,   1, isInt: true),
            new ParamDef("eps",     "ε Guided Filter",                       1e-5, 1e-2, 1e-3, log: true),
            new ParamDef("patch",   "Патч оценки airlight",                  1,    15,   5,    1, isInt: true),
            new ParamDef("aRadius", "Радиус локальной яркости A(x)",         20,   240,  110,  1, isInt: true),
            new ParamDef("airMix",  "Глобальность яркости airlight",         0.0,  1.0,  0.75),
            new ParamDef("wb",      "Баланс по цвету airlight",              0.0,  1.0,  1.00, search: true),
            new ParamDef("min",     "t_min яркости",                         0.02, 0.4,  0.07),
            new ParamDef("chroma",  "t_floor хромы",                        0.08, 0.6,  0.12, search: true),
            new ParamDef("anchor",  "Вес глобальной хроматичности A_g",      0.0,  1.0,  1.00, search: true),
            new ParamDef("csigma",  "Масштаб устойчивой хромы σ",            0.0,  40.0, 10.0),
            new ParamDef("cmix",    "Coarse chroma в плотной дымке",         0.0,  1.0,  0.90, search: true),
            new ParamDef("baseOmega", "ω яркостной DCP-ветви",                0.30, 0.99, 0.95, search: true),
            new ParamDef("baseMin", "t_min яркостной DCP-ветви",             0.05, 0.50, 0.10, search: true),
            new ParamDef("backbone", "L*: 0=DCP, 1=RFEP, 2=A²CR, 3=adaptive", 0,    3,    3,    1, isInt: true, tunable: false),
            new ParamDef("fuse",    "Вес физической хромы",                   0.0,  1.0,  1.00, search: true),
            new ParamDef("cgain",   "Усиление крупномасштабной a*/b*",        0.5,  4.0,  1.20, search: true),
            new ParamDef("fsigma",  "Сглаживание chroma-donor σ",             0.0, 20.0,  4.00),
            new ParamDef("denseLo", "Начало dense-haze гейта",                0.0,  0.8,  0.60, search: true),
            new ParamDef("denseSpan", "Относительный диапазон dense-гейта",    0.25, 1.0,  0.50, search: true),
            new ParamDef("fpow",    "Степень маски плотности (1-t)",          0.25, 3.0,  1.00),
            new ParamDef("aconf",   "Степень доверия gamut α",                0.0,  4.0,  2.00),
            new ParamDef("tone",    "Мягкое восстановление тона",            0.0,  1.0,  0.00),
            new ParamDef("sat",     "Финишный вибранс",                      0.0,  0.6,  0.00),
            new ParamDef("detail",  "Микроконтраст",                         0.0,  0.5,  0.00),
            new ParamDef("smooth",  "Шумоподавление",                        0.0,  6.0,  0.00),
            new ParamDef("linear",  "Линейный радианс (0/1)",                0,    1,    1,    1, isInt: true, tunable: false),
            new ParamDef("project", "Gamut-проекция (0/1)",                  0,    1,    1,    1, isInt: true, tunable: false),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
            => ProcessCore(input, p, null);

        internal ChromaticAnchorExecution ProcessDetailed(Image<Bgr, byte> input,
            IReadOnlyDictionary<string, double> p)
        {
            Mat? transmission = null, adjustedDepth = null, physicalSrgb = null, airlightSrgb = null, projectionAlpha = null;
            MCvScalar globalAirlight = default;
            double outOfGamutFraction = double.NaN;
            GamutProjectionSummary projectionSummary = default;
            bool linearRadiance = true;

            Mat result = ProcessCore(input, p, (t, depth, recovered, physical, airField, alpha, airlight, projection, linear) =>
            {
                transmission = t.Clone();
                adjustedDepth = depth.Clone();
                physicalSrgb = physical.Clone();
                projectionAlpha = alpha.Clone();
                globalAirlight = airlight;
                projectionSummary = projection;
                linearRadiance = linear;
                outOfGamutFraction = OutOfGamutFraction(recovered);

                using var channels = new Emgu.CV.Util.VectorOfMat();
                foreach (Mat channel in airField) channels.Push(channel);
                using var mergedAirlight = new Mat();
                CvInvoke.Merge(channels, mergedAirlight);
                using var boundedAirlight = DeHazeCPU.Clip(mergedAirlight.Clone());
                airlightSrgb = ColorSpace.Encode(boundedAirlight, linear);
            });

            if (transmission == null || adjustedDepth == null || physicalSrgb == null || airlightSrgb == null || projectionAlpha == null)
            {
                result.Dispose();
                throw new InvalidOperationException("Не удалось получить диагностические карты CAR-Dehaze.");
            }
            return new ChromaticAnchorExecution(result, transmission, adjustedDepth, physicalSrgb,
                airlightSrgb, projectionAlpha, globalAirlight, outOfGamutFraction, projectionSummary, linearRadiance);
        }

        private Mat ProcessCore(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p,
            DiagnosticCapture? capture)
        {
            double beta = p["beta"], depthGain = p["depth"], eps = p["eps"];
            int rMin = (int)p["rmin"], rGuide = (int)p["rguide"], patch = (int)p["patch"], aRadius = (int)p["aRadius"];
            double airMix = p["airMix"], whiteBalance = p["wb"], tmin = p["min"], chroma = p["chroma"], anchor = p["anchor"];
            double coarseSigma = p["csigma"], coarseMix = p["cmix"];
            double baseOmega = p["baseOmega"], baseMin = p["baseMin"], fusion = p["fuse"], chromaGain = p["cgain"];
            int backbone = (int)p["backbone"];
            double fusionSigma = p["fsigma"], denseLo = p["denseLo"];
            double denseHi = denseLo + (1.0 - denseLo) * p["denseSpan"];
            double hazePower = p["fpow"], alphaPower = p["aconf"];
            double tone = p["tone"], sat = p["sat"], detail = p["detail"], smooth = p["smooth"];
            bool linear = !p.TryGetValue("linear", out double linearValue) || linearValue >= 0.5;
            bool project = !p.TryGetValue("project", out double projectValue) || projectValue >= 0.5;

            // CAP coefficients were trained on ordinary sRGB V/S, so t is estimated in feature space.
            using var feature = DehazeCore.Normalize(input);
            using var hsv = new Mat(); CvInvoke.CvtColor(feature, hsv, ColorConversion.Bgr2Hsv);
            var hsvChannels = hsv.Split();
            using var saturation = hsvChannels[1]; using var value = hsvChannels[2]; hsvChannels[0].Dispose();
            using var depth = new Mat(); CvInvoke.AddWeighted(value, Theta1, saturation, Theta2, Theta0, depth, DepthType.Cv32F);
            using (var element = CvInvoke.GetStructuringElement(ElementShape.Rectangle,
                new Size(2 * rMin + 1, 2 * rMin + 1), new Point(-1, -1)))
                CvInvoke.Erode(depth, depth, element, new Point(-1, -1), 1, BorderType.Reflect101, default);
            using var depthRefined = new Mat(); XImgprocInvoke.GuidedFilter(value, depth, depthRefined, rGuide, eps);
            using var adjustedDepth = new Mat();
            if (depthGain > 1e-6)
            {
                using var squared = new Mat(); CvInvoke.Multiply(depthRefined, depthRefined, squared);
                CvInvoke.AddWeighted(depthRefined, 1.0, squared, depthGain, 0.0, adjustedDepth);
            }
            else depthRefined.CopyTo(adjustedDepth);
            using var transmission = new Mat();
            using (var exponent = new Mat())
            {
                adjustedDepth.ConvertTo(exponent, DepthType.Cv32F, -beta);
                CvInvoke.Exp(exponent, transmission);
            }
            DehazeCore.Clamp01(transmission);

            // Инверсия только в линейном радиансе по умолчанию; sRGB-вариант оставлен для ablation.
            using var radiance = ColorSpace.Normalize(input, linear);
            MCvScalar globalAirlight = AtmosphericByDepth(radiance, depthRefined, 0.001);
            var airField = LocalHazeCore.AirlightField(radiance, patch, aRadius);
            if (airMix > 1e-6)
            {
                double[] global = { globalAirlight.V0, globalAirlight.V1, globalAirlight.V2 };
                for (int c = 0; c < 3; c++)
                    airField[c].ConvertTo(airField[c], DepthType.Cv32F, 1.0 - airMix, global[c] * airMix);
            }

            try
            {
                using var recovered = LocalHazeCore.RecoverChromaticAnchor(radiance, transmission, airField,
                    globalAirlight, tmin, chroma, anchor, coarseSigma, coarseMix);
                using var projection = project
                    ? GamutProjector.ProjectFromInput(radiance, recovered)
                    : null;
                using var bounded = projection != null
                    ? projection.Result.Clone()
                    : DeHazeCPU.Clip(recovered.Clone());
                LocalHazeCore.LocalAirlightWhiteBalance(bounded, airField, transmission, whiteBalance);
                using var srgb = ColorSpace.Encode(bounded, linear);

                using var identityAlpha = projection == null
                    ? new Mat(transmission.Size, DepthType.Cv32F, 1)
                    : null;
                if (identityAlpha != null) identityAlpha.SetTo(new MCvScalar(1.0));
                Mat projectionAlpha = projection?.Alpha ?? identityAlpha!;
                GamutProjectionSummary projectionSummary = projection?.Summary ??
                    new GamutProjectionSummary(0.0, 1.0, OutOfGamutFraction(recovered), 0.0);

                capture?.Invoke(transmission, adjustedDepth, recovered, srgb, airField, projectionAlpha,
                    globalAirlight, projectionSummary, linear);

                using var luminanceBase = BuildLuminanceBase(input, transmission, backbone,
                    baseOmega, baseMin, denseLo, denseHi);
                using var fused = FuseChroma(luminanceBase, srgb, transmission, projectionAlpha,
                    fusion, chromaGain, fusionSigma, denseLo, denseHi, hazePower, alphaPower);
                using var enhanced = DehazeCore.LabEnhance(fused, 0.0, 8, sat, detail);
                using var toned = DehazeCore.RestoreTone(enhanced, tone, 0.01);
                return smooth > 0.01 ? DehazeCore.BilateralDenoise(toned, smooth) : DeHazeCPU.Clip(toned.Clone());
            }
            finally
            {
                foreach (var channel in airField) channel.Dispose();
            }
        }

        private static double OutOfGamutFraction(Mat recovered)
        {
            int values = recovered.Rows * recovered.Cols * recovered.NumberOfChannels;
            var data = new float[values];
            recovered.CopyTo(data);
            int outside = data.Count(value => !float.IsFinite(value) || value < 0.0f || value > 1.0f);
            return values == 0 ? 0.0 : outside / (double)values;
        }

        private static Mat FuseChroma(Mat luminanceBase, Mat chromaDonor, Mat transmission, Mat projectionAlpha,
            double fusion, double chromaGain, double sigma, double denseLo, double denseHi,
            double hazePower, double alphaPower)
        {
            fusion = Math.Clamp(fusion, 0.0, 1.0);
            chromaGain = Math.Max(0.0, chromaGain);
            sigma = Math.Max(0.0, sigma);
            hazePower = Math.Max(0.05, hazePower);
            alphaPower = Math.Max(0.0, alphaPower);

            using var baseLab = new Mat(); CvInvoke.CvtColor(luminanceBase, baseLab, ColorConversion.Bgr2Lab);
            using var donorLab = new Mat(); CvInvoke.CvtColor(chromaDonor, donorLab, ColorConversion.Bgr2Lab);
            var baseChannels = baseLab.Split();
            var donorChannels = donorLab.Split();
            try
            {
                using var density = DenseHazeGate(transmission, denseLo, denseHi);
                if (Math.Abs(hazePower - 1.0) > 1e-6) CvInvoke.Pow(density, hazePower, density);
                if (alphaPower > 1e-6)
                {
                    using var confidence = new Mat(); CvInvoke.Pow(projectionAlpha, alphaPower, confidence);
                    CvInvoke.Multiply(density, confidence, density);
                }
                if (fusion < 1.0) CvInvoke.Multiply(density, new ScalarArray(fusion), density);
                if (sigma > 1e-3)
                    CvInvoke.GaussianBlur(density, density, new Size(0, 0), Math.Max(0.5, sigma * 0.5),
                        Math.Max(0.5, sigma * 0.5), BorderType.Reflect101);
                DehazeCore.Clamp01(density);

                using var output = new Emgu.CV.Util.VectorOfMat();
                output.Push(baseChannels[0]);
                for (int channel = 1; channel <= 2; channel++)
                {
                    using var donor = new Mat();
                    if (sigma > 1e-3)
                        CvInvoke.GaussianBlur(donorChannels[channel], donor, new Size(0, 0), sigma, sigma, BorderType.Reflect101);
                    else donorChannels[channel].CopyTo(donor);
                    if (Math.Abs(chromaGain - 1.0) > 1e-6)
                        CvInvoke.Multiply(donor, new ScalarArray(chromaGain), donor);
                    using var delta = new Mat(); CvInvoke.Subtract(donor, baseChannels[channel], delta);
                    CvInvoke.Multiply(delta, density, delta);
                    var fused = new Mat(); CvInvoke.Add(baseChannels[channel], delta, fused);
                    output.Push(fused); fused.Dispose();
                }
                using var fusedLab = new Mat(); CvInvoke.Merge(output, fusedLab);
                var result = new Mat(); CvInvoke.CvtColor(fusedLab, result, ColorConversion.Lab2Bgr);
                return DeHazeCPU.Clip(result);
            }
            finally
            {
                foreach (var channel in baseChannels) channel.Dispose();
                foreach (var channel in donorChannels) channel.Dispose();
            }
        }

        private static Mat BuildLuminanceBase(Image<Bgr, byte> input, Mat transmission, int backbone,
            double dcpOmega, double dcpMin, double denseLo, double denseHi)
        {
            if (backbone == 3)
            {
                using var rfep = RunBackbone(new RfepDcpMethod(), input);
                using var dcp = RunDcp(input, dcpOmega, dcpMin);
                return BlendLuminance(rfep, dcp, transmission, denseLo, denseHi);
            }
            IDeHazeMethod method = backbone switch
            {
                1 => new RfepDcpMethod(),
                2 => new A2crMethod(),
                _ => new CanonicalDcpMethod(),
            };
            if (method is CanonicalDcpMethod) return RunDcp(input, dcpOmega, dcpMin);
            return RunBackbone(method, input);
        }

        private static Mat RunDcp(Image<Bgr, byte> input, double omega, double minimumTransmission)
        {
            var method = new CanonicalDcpMethod();
            var parameters = method.Parameters.ToDictionary(definition => definition.Key, definition => definition.Default);
            parameters["omega"] = omega;
            parameters["min"] = minimumTransmission;
            parameters["linear"] = 0.0;
            return method.Process(input, parameters);
        }

        private static Mat RunBackbone(IDeHazeMethod method, Image<Bgr, byte> input)
        {
            var parameters = method.Parameters.ToDictionary(definition => definition.Key, definition => definition.Default);
            return method.Process(input, parameters);
        }

        private static Mat BlendLuminance(Mat broadBase, Mat denseBase, Mat transmission,
            double denseLo, double denseHi)
        {
            using var broadLab = new Mat(); CvInvoke.CvtColor(broadBase, broadLab, ColorConversion.Bgr2Lab);
            using var denseLab = new Mat(); CvInvoke.CvtColor(denseBase, denseLab, ColorConversion.Bgr2Lab);
            var broadChannels = broadLab.Split();
            var denseChannels = denseLab.Split();
            try
            {
                using var gate = DenseHazeGate(transmission, denseLo, denseHi);
                using var delta = new Mat(); CvInvoke.Subtract(denseChannels[0], broadChannels[0], delta);
                CvInvoke.Multiply(delta, gate, delta);
                CvInvoke.Add(broadChannels[0], delta, broadChannels[0]);
                using var merged = new Emgu.CV.Util.VectorOfMat();
                foreach (var channel in broadChannels) merged.Push(channel);
                using var lab = new Mat(); CvInvoke.Merge(merged, lab);
                var result = new Mat(); CvInvoke.CvtColor(lab, result, ColorConversion.Lab2Bgr);
                return DeHazeCPU.Clip(result);
            }
            finally
            {
                foreach (var channel in broadChannels) channel.Dispose();
                foreach (var channel in denseChannels) channel.Dispose();
            }
        }

        internal static Mat DenseHazeGate(Mat transmission, double denseLo, double denseHi)
        {
            denseLo = Math.Clamp(denseLo, 0.0, 0.99);
            denseHi = Math.Clamp(denseHi, denseLo + 0.01, 1.0);
            var gate = new Mat();
            transmission.ConvertTo(gate, DepthType.Cv32F, -1.0 / (denseHi - denseLo),
                (1.0 - denseLo) / (denseHi - denseLo));
            DehazeCore.Clamp01(gate);
            using var squared = new Mat(); CvInvoke.Multiply(gate, gate, squared);
            using var cubed = new Mat(); CvInvoke.Multiply(squared, gate, cubed);
            CvInvoke.AddWeighted(squared, 3.0, cubed, -2.0, 0.0, gate);
            return gate;
        }

        private static MCvScalar AtmosphericByDepth(Mat radiance, Mat depth, double topPercent)
        {
            int count = depth.Rows * depth.Cols;
            var depthValues = new float[count]; depth.CopyTo(depthValues);
            var channels = radiance.Split();
            var b = new float[count]; var g = new float[count]; var r = new float[count];
            channels[0].CopyTo(b); channels[1].CopyTo(g); channels[2].CopyTo(r);
            foreach (var channel in channels) channel.Dispose();

            int selected = Math.Max(1, (int)Math.Round(count * topPercent));
            int[] indices = Enumerable.Range(0, count).OrderByDescending(i => depthValues[i]).Take(selected).ToArray();
            double sumB = 0, sumG = 0, sumR = 0;
            foreach (int i in indices) { sumB += b[i]; sumG += g[i]; sumR += r[i]; }
            return new MCvScalar(sumB / selected, sumG / selected, sumR / selected);
        }
    }
}
