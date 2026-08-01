using System.Drawing;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV.XImgproc;

namespace SimpleDeHaze.Methods
{
    /// <summary>
    /// C³R-HSV (Cylindrical Confidence-Constrained Recovery) — HSV-аналог A²CR.
    ///
    /// Метод не рассматривает H, S и V как три независимых линейных канала. Вместо этого HSV
    /// записывается как z=(V, C*cos(H), C*sin(H)), где C=S*V. Такая запись непрерывна на границе
    /// H=0/360° и позволяет отдельно регуляризовать яркость, радиальную chroma и изменение hue.
    /// </summary>
    public sealed class HsvC3rMethod : IDeHazeMethod
    {
        private const double Theta0 = 0.121779;
        private const double Theta1 = 0.959710;
        private const double Theta2 = -0.780245;
        private const double Tiny = 1e-7;
        private const double UncertaintyScale = 0.01;
        private const int ProjectionIterations = 18;

        public string Name => "C³R-HSV: uncertainty-aware recovery (EXP)";

        public string Description =>
            "Экспериментальный HSV-аналог A²CR, который меняет оператор восстановления, " +
            "а не только prior карты t.\n\n" +
            "1. CAP-HSV и DCP независимо оценивают transmission. Оценки объединяются в " +
            "optical-depth, а их расхождение становится uncertainty map.\n" +
            "2. Recovery выполняется в linear-light HSV и непрерывных координатах " +
            "(V, C·cosH, C·sinH), C=S·V.\n" +
            "3. Остаток относительно atmospheric light раскладывается на value, radial-chroma " +
            "и tangential-chroma. Для каждой компоненты выводится свой gain из локальной энергии, " +
            "шума и uncertainty. При высокой неопределённости gain стремится к 1; при идеальной " +
            "модели — к 1/t.\n" +
            "4. Кандидат проецируется вдоль отрезка от входного пикселя в допустимый " +
            "saturation/value-конус. Исходный пиксель всегда допустим, поэтому post-hoc clipping " +
            "не нужен как часть алгоритма.\n" +
            "5. Hue trust region подавляет случайные повороты оттенка при малой насыщенности.\n\n" +
            "Метод оставлен без CLAHE/Retinex, чтобы benchmark измерял именно recovery.";

        public IReadOnlyList<ParamDef> Parameters { get; } = new[]
        {
            new ParamDef("beta",      "β CAP-HSV",                       0.1,   3.0,  1.0,  search: true),
            new ParamDef("omega",     "ω DCP",                           0.3,   1.0,  0.85, search: true),
            new ParamDef("capWeight", "Вес CAP в optical-depth fusion",  0.0,   1.0,  0.65, search: true),
            new ParamDef("patch",     "Радиус DCP-патча",                1,     25,   7,    1, isInt: true),
            new ParamDef("rmin",      "Радиус min-фильтра CAP",          1,     25,   7,    1, isInt: true),
            new ParamDef("rguide",    "Радиус Guided Filter",            5,     120,  50,   1, isInt: true),
            new ParamDef("eps",       "ε Guided Filter",                 1e-5,  1e-2, 1e-3, log: true),
            new ParamDef("min",       "t_min",                           0.02,  0.4,  0.08),
            new ParamDef("energy",    "Радиус локальной энергии",        1,     25,   7,    1, isInt: true),
            new ParamDef("noise",     "Штраф локального шума",           0.0,   4.0,  0.35, search: true),
            new ParamDef("unc",       "Штраф расхождения CAP/DCP",       0.0,   12.0, 2.0,  search: true),
            new ParamDef("hueGuard",  "Защита hue при слабой chroma",    0.0,   0.08, 0.008),
            new ParamDef("hueFloor",  "Порог надёжности hue",            0.005, 0.25, 0.05, log: true),
            new ParamDef("hueMax",    "Макс. коррекция hue, градусов",    0.0,   90.0, 24.0),
            new ParamDef("satRoom",   "Допустимое восстановление S",     0.0,   1.0,  0.35),
            new ParamDef("maxGain",   "Общий потолок gain",              1.0,   12.0, 6.0),
        };

        public Mat Process(Image<Bgr, byte> input, IReadOnlyDictionary<string, double> p)
        {
            float beta = (float)p["beta"];
            double omega = p["omega"];
            double capWeight = p["capWeight"];
            int patch = (int)p["patch"];
            int rMin = (int)p["rmin"];
            int rGuide = (int)p["rguide"];
            double guidedEps = p["eps"];
            double tMin = p["min"];
            int energyRadius = (int)p["energy"];
            double noiseWeight = p["noise"];
            double uncertaintyWeight = p["unc"];
            double hueGuard = p["hueGuard"];
            double hueFloor = p["hueFloor"];
            double hueMax = p["hueMax"];
            double saturationRoom = p["satRoom"];
            double maxGain = Math.Min(p["maxGain"], 1.0 / Math.Max(tMin, 1e-3));

            // CAP был выведен для обычного HSV/sRGB, поэтому prior оценивается до linearization.
            // Сам оператор восстановления ниже работает в linear-light HSV.
            using var inputSrgb = DehazeCore.Normalize(input);
            using var hsvSrgb = new Mat();
            CvInvoke.CvtColor(inputSrgb, hsvSrgb, ColorConversion.Bgr2Hsv);
            var hsvSrgbChannels = hsvSrgb.Split();
            hsvSrgbChannels[0].Dispose();
            using var saturationSrgb = hsvSrgbChannels[1];
            using var valueSrgb = hsvSrgbChannels[2];

            using var tCap = BuildCapTransmission(
                saturationSrgb, valueSrgb, beta, rMin, rGuide, guidedEps);

            // DCP и atmospheric inverse должны работать по радиансу, а не по gamma-coded sRGB.
            // CAP остаётся в исходном sRGB/HSV, поскольку его коэффициенты обучены именно там.
            using var inputLinear = ColorSpace.NormalizeLinear(input);
            using var darkLinear = DehazeCore.DarkChannel(inputLinear, patch);
            MCvScalar airlightLinear = DehazeCore.Atmospheric(inputLinear, darkLinear, 0.001);
            using var tDcpRaw = DehazeCore.RawTransmission(inputLinear, airlightLinear, omega, patch);
            using var linearGuide = ColorSpace.Luminance(inputLinear);
            using var tDcp = new Mat();
            XImgprocInvoke.GuidedFilter(linearGuide, tDcpRaw, tDcp, rGuide, guidedEps);

            using var transmission = new Mat();
            using var transmissionVariance = new Mat();
            FuseTransmissionInOpticalDepth(
                tCap, tDcp, capWeight, tMin, transmission, transmissionVariance);

            using var hsvLinear = new Mat();
            CvInvoke.CvtColor(inputLinear, hsvLinear, ColorConversion.Bgr2Hsv);
            var hsvLinearChannels = hsvLinear.Split();
            using var hue = hsvLinearChannels[0];
            using var saturation = hsvLinearChannels[1];
            using var value = hsvLinearChannels[2];

            BgrToHsv(
                airlightLinear.V0,
                airlightLinear.V1,
                airlightLinear.V2,
                out double airHue,
                out double airSaturation,
                out double airValue);

            double airChroma = airSaturation * airValue;
            double airAngle = airHue * Math.PI / 180.0;
            double airCx = airChroma * Math.Cos(airAngle);
            double airCy = airChroma * Math.Sin(airAngle);
            double airHueConfidence =
                airChroma * airChroma /
                (airChroma * airChroma + hueFloor * hueFloor + Tiny);

            // Для почти серого A направление hue(A) не определено. В этом случае ниже применяется
            // один изотропный chroma-gain, поэтому выбор осей не может поворачивать оттенок.
            bool chromaticAirlight = airHueConfidence >= 0.2;
            double radialX = chromaticAirlight ? airCx / Math.Max(airChroma, Tiny) : 1.0;
            double radialY = chromaticAirlight ? airCy / Math.Max(airChroma, Tiny) : 0.0;
            double tangentX = -radialY;
            double tangentY = radialX;

            int rows = inputSrgb.Rows;
            int cols = inputSrgb.Cols;
            int count = rows * cols;

            var hueData = CopySingleChannel(hue, count);
            var saturationData = CopySingleChannel(saturation, count);
            var valueData = CopySingleChannel(value, count);
            var transmissionData = CopySingleChannel(transmission, count);
            var transmissionVarianceData = CopySingleChannel(transmissionVariance, count);

            var inputCx = new float[count];
            var inputCy = new float[count];
            var deltaValue = new float[count];
            var radialResidual = new float[count];
            var tangentialResidual = new float[count];
            var hueConfidenceData = new float[count];

            for (int i = 0; i < count; i++)
            {
                double angle = hueData[i] * Math.PI / 180.0;
                double chroma = Math.Max(0.0, saturationData[i]) * Math.Max(0.0, valueData[i]);
                double cx = chroma * Math.Cos(angle);
                double cy = chroma * Math.Sin(angle);
                double dcx = cx - airCx;
                double dcy = cy - airCy;

                inputCx[i] = (float)cx;
                inputCy[i] = (float)cy;
                deltaValue[i] = (float)(valueData[i] - airValue);
                radialResidual[i] = (float)(dcx * radialX + dcy * radialY);
                tangentialResidual[i] = (float)(dcx * tangentX + dcy * tangentY);
                hueConfidenceData[i] = (float)(
                    chroma * chroma /
                    (chroma * chroma + hueFloor * hueFloor + Tiny));
            }

            using var deltaValueMat = DehazeCore.MatFromFloats(deltaValue, rows, cols);
            using var radialMat = DehazeCore.MatFromFloats(radialResidual, rows, cols);
            using var tangentialMat = DehazeCore.MatFromFloats(tangentialResidual, rows, cols);
            using var hueConfidenceMat =
                DehazeCore.MatFromFloats(hueConfidenceData, rows, cols);

            using var signalValue = LocalSecondMoment(deltaValueMat, energyRadius);
            using var signalRadial = LocalSecondMoment(radialMat, energyRadius);
            using var signalTangential = LocalSecondMoment(tangentialMat, energyRadius);
            using var noiseValue = LocalHighPassEnergy(
                deltaValueMat, energyRadius, noiseWeight);
            using var noiseRadial = LocalHighPassEnergy(
                radialMat, energyRadius, noiseWeight);
            using var noiseTangential = LocalHighPassEnergy(
                tangentialMat, energyRadius, noiseWeight);

            // Q в риске притягивает gain к identity (g=1), а Var(t) отдельно штрафует
            // нестабильность физической инверсии.
            using var penaltyValue = new Mat();
            transmissionVariance.ConvertTo(
                penaltyValue, DepthType.Cv32F, uncertaintyWeight, 1e-6);

            using var lowHueConfidence = new Mat();
            hueConfidenceMat.ConvertTo(lowHueConfidence, DepthType.Cv32F, -1.0, 1.0);
            DehazeCore.Clamp01(lowHueConfidence);

            using var penaltyRadial = penaltyValue.Clone();
            using (var weakColorPenalty = new Mat())
            {
                lowHueConfidence.ConvertTo(
                    weakColorPenalty, DepthType.Cv32F, hueGuard * 0.25);
                CvInvoke.Add(penaltyRadial, weakColorPenalty, penaltyRadial);
            }

            using var penaltyTangential = penaltyValue.Clone();
            using (var weakHuePenalty = new Mat())
            {
                double grayAirlightPenalty = hueGuard * (1.0 - airHueConfidence);
                lowHueConfidence.ConvertTo(
                    weakHuePenalty, DepthType.Cv32F, hueGuard, grayAirlightPenalty);
                CvInvoke.Add(penaltyTangential, weakHuePenalty, penaltyTangential);
            }

            using var gainValue = ComputeRiskGain(
                signalValue, noiseValue, penaltyValue,
                transmission, transmissionVariance, maxGain);
            using var gainRadial = ComputeRiskGain(
                signalRadial, noiseRadial, penaltyRadial,
                transmission, transmissionVariance, maxGain);
            using var gainTangential = ComputeRiskGain(
                signalTangential, noiseTangential, penaltyTangential,
                transmission, transmissionVariance, maxGain);

            var gainValueData = CopySingleChannel(gainValue, count);
            var gainRadialData = CopySingleChannel(gainRadial, count);
            var gainTangentialData = CopySingleChannel(gainTangential, count);

            var outputHue = new float[count];
            var outputSaturation = new float[count];
            var outputValue = new float[count];

            for (int i = 0; i < count; i++)
            {
                double gv = gainValueData[i];
                double gr = gainRadialData[i];
                double gt = gainTangentialData[i];

                if (!chromaticAirlight)
                {
                    double isotropicChromaGain = 0.5 * (gr + gt);
                    gr = isotropicChromaGain;
                    gt = isotropicChromaGain;
                }

                double candidateValue = airValue + gv * deltaValue[i];
                double candidateCx =
                    airCx +
                    gr * radialResidual[i] * radialX +
                    gt * tangentialResidual[i] * tangentX;
                double candidateCy =
                    airCy +
                    gr * radialResidual[i] * radialY +
                    gt * tangentialResidual[i] * tangentY;

                double t = Math.Clamp((double)transmissionData[i], 0.0, 1.0);
                double tVariance = Math.Max(0.0, transmissionVarianceData[i]);
                double uncertaintyConfidence =
                    tVariance / (tVariance + UncertaintyScale);
                double hueConfidence = Math.Clamp((double)hueConfidenceData[i], 0.0, 1.0);

                // Не разрешаем восстановлению автоматически занимать весь HSV-конус.
                // Коридор расширяется относительно входной S только при выраженной дымке,
                // надёжном hue и согласии независимых prior.
                double room =
                    saturationRoom *
                    (1.0 - t) *
                    hueConfidence *
                    (1.0 - uncertaintyConfidence);
                double saturationCap =
                    saturationData[i] +
                    (1.0 - saturationData[i]) * room;
                saturationCap = Math.Clamp(saturationCap, saturationData[i], 1.0);

                ProjectAlongInputRay(
                    valueData[i],
                    inputCx[i],
                    inputCy[i],
                    candidateValue,
                    candidateCx,
                    candidateCy,
                    saturationCap,
                    out double recoveredValue,
                    out double recoveredCx,
                    out double recoveredCy);

                double recoveredChroma =
                    Math.Sqrt(recoveredCx * recoveredCx + recoveredCy * recoveredCy);
                double recoveredHue = recoveredChroma > Tiny
                    ? NormalizeHue(Math.Atan2(recoveredCy, recoveredCx) * 180.0 / Math.PI)
                    : NormalizeHue(hueData[i]);

                double allowedHueShift =
                    hueMax *
                    (1.0 - t) *
                    hueConfidence *
                    airHueConfidence *
                    (1.0 - uncertaintyConfidence);
                double hueShift = Math.Clamp(
                    WrapHueDelta(recoveredHue - hueData[i]),
                    -allowedHueShift,
                    allowedHueShift);
                recoveredHue = NormalizeHue(hueData[i] + hueShift);

                // Поворот при неизменном радиусе сохраняет cone-feasibility.
                outputHue[i] = (float)recoveredHue;
                outputValue[i] = (float)Math.Clamp(recoveredValue, 0.0, 1.0);
                outputSaturation[i] = recoveredValue > Tiny
                    ? (float)Math.Clamp(recoveredChroma / recoveredValue, 0.0, 1.0)
                    : 0.0f;
            }

            using var outputHueMat = DehazeCore.MatFromFloats(outputHue, rows, cols);
            using var outputSaturationMat =
                DehazeCore.MatFromFloats(outputSaturation, rows, cols);
            using var outputValueMat = DehazeCore.MatFromFloats(outputValue, rows, cols);
            using var outputHsv = new Mat();
            using (var channels = new VectorOfMat())
            {
                channels.Push(outputHueMat);
                channels.Push(outputSaturationMat);
                channels.Push(outputValueMat);
                CvInvoke.Merge(channels, outputHsv);
            }

            using var outputLinear = new Mat();
            CvInvoke.CvtColor(outputHsv, outputLinear, ColorConversion.Hsv2Bgr);
            using var outputSrgb = ColorSpace.ToSrgb(outputLinear);
            return outputSrgb.Clone();
        }

        private static Mat BuildCapTransmission(
            Mat saturation,
            Mat value,
            float beta,
            int minRadius,
            int guideRadius,
            double guidedEps)
        {
            using var depth = new Mat();
            CvInvoke.AddWeighted(
                value, Theta1, saturation, Theta2, Theta0, depth, DepthType.Cv32F);

            using (var element = CvInvoke.GetStructuringElement(
                       ElementShape.Rectangle,
                       new Size(2 * minRadius + 1, 2 * minRadius + 1),
                       new Point(-1, -1)))
            {
                CvInvoke.Erode(
                    depth, depth, element, new Point(-1, -1),
                    1, BorderType.Reflect101, default);
            }

            using var refinedDepth = new Mat();
            XImgprocInvoke.GuidedFilter(
                value, depth, refinedDepth, guideRadius, guidedEps);

            var transmission = new Mat();
            using var negativeDepth = new Mat();
            refinedDepth.ConvertTo(negativeDepth, DepthType.Cv32F, -beta);
            CvInvoke.Exp(negativeDepth, transmission);
            return transmission;
        }

        /// <summary>
        /// Fusion выполняется в optical depth D=-ln(t). Для двух оценок:
        /// mean(D)=w*D_CAP+(1-w)*D_DCP,
        /// Var(D)=w(1-w)(D_CAP-D_DCP)^2,
        /// Var(t)≈t^2 Var(D).
        /// </summary>
        internal static void FuseTransmissionInOpticalDepth(
            Mat tCap,
            Mat tDcp,
            double capWeight,
            double tMin,
            Mat tMean,
            Mat tVariance)
        {
            int count = tCap.Rows * tCap.Cols;
            var capData = CopySingleChannel(tCap, count);
            var dcpData = CopySingleChannel(tDcp, count);
            var meanData = new float[count];
            var varianceData = new float[count];

            double w = Math.Clamp(capWeight, 0.0, 1.0);
            double minTransmission = Math.Max(1e-4, tMin);

            for (int i = 0; i < count; i++)
            {
                double cap = Math.Clamp((double)capData[i], minTransmission, 1.0);
                double dcp = Math.Clamp((double)dcpData[i], minTransmission, 1.0);
                double capDepth = -Math.Log(cap);
                double dcpDepth = -Math.Log(dcp);
                double meanDepth = w * capDepth + (1.0 - w) * dcpDepth;
                double meanTransmission =
                    Math.Clamp(Math.Exp(-meanDepth), minTransmission, 1.0);
                // Disagreement is evidence independent of the fusion preference. Using
                // w(1-w) here allowed capWeight=0/1 to erase uncertainty, giving AutoTuner a
                // shortcut instead of a real quality trade-off. The equal-prior variance is
                // retained even when the mean deliberately favors one estimator.
                double depthDifference = capDepth - dcpDepth;
                double depthVariance = 0.25 * depthDifference * depthDifference;

                meanData[i] = (float)meanTransmission;
                varianceData[i] = (float)Math.Clamp(
                    meanTransmission * meanTransmission * depthVariance,
                    0.0,
                    1.0);
            }

            using var meanMat =
                DehazeCore.MatFromFloats(meanData, tCap.Rows, tCap.Cols);
            using var varianceMat =
                DehazeCore.MatFromFloats(varianceData, tCap.Rows, tCap.Cols);
            meanMat.CopyTo(tMean);
            varianceMat.CopyTo(tVariance);
        }

        private static Mat LocalSecondMoment(Mat component, int radius)
        {
            using var squared = new Mat();
            CvInvoke.Multiply(component, component, squared);

            var result = new Mat();
            int r = Math.Max(1, radius);
            CvInvoke.Blur(
                squared,
                result,
                new Size(2 * r + 1, 2 * r + 1),
                new Point(-1, -1));
            CvInvoke.Add(result, new ScalarArray(Tiny), result);
            return result;
        }

        private static Mat LocalHighPassEnergy(
            Mat component,
            int radius,
            double weight)
        {
            var result = new Mat(component.Size, DepthType.Cv32F, 1);
            result.SetTo(new MCvScalar(0.0));
            if (weight <= Tiny)
                return result;

            using var smooth = new Mat();
            CvInvoke.GaussianBlur(component, smooth, new Size(0, 0), 1.0);

            using var high = new Mat();
            CvInvoke.Subtract(component, smooth, high);
            CvInvoke.Multiply(high, high, high);

            int r = Math.Max(1, radius);
            CvInvoke.Blur(
                high,
                result,
                new Size(2 * r + 1, 2 * r + 1),
                new Point(-1, -1));
            CvInvoke.Multiply(result, new ScalarArray(weight), result);
            return result;
        }

        /// <summary>
        /// Минимизатор локального риска
        /// R(g)=S[(g*t-1)^2+g^2 Var(t)] + N*g^2 + Q*(g-1)^2:
        /// g*=(S*t+Q)/(S*(t^2+Var(t))+N+Q).
        /// </summary>
        internal static Mat ComputeRiskGain(
            Mat signal,
            Mat noise,
            Mat modelPenalty,
            Mat transmission,
            Mat transmissionVariance,
            double maxGain)
        {
            using var transmissionSquared = new Mat();
            CvInvoke.Multiply(transmission, transmission, transmissionSquared);
            CvInvoke.Add(
                transmissionSquared,
                transmissionVariance,
                transmissionSquared);

            // LocalSecondMoment observes the hazy residual. Convert it to an estimate of the
            // latent component energy before inserting it into the risk, matching the A²CR
            // derivation: S=max(E[d_obs²]-N,0)/(t²+Var(t)).
            using var latentSignal = new Mat();
            CvInvoke.Subtract(signal, noise, latentSignal);
            using (var zero = new Mat(latentSignal.Size, DepthType.Cv32F, 1))
            {
                zero.SetTo(new MCvScalar(0));
                CvInvoke.Max(latentSignal, zero, latentSignal);
            }
            using (var stableSecondMoment = transmissionSquared.Clone())
            {
                CvInvoke.Add(stableSecondMoment, new ScalarArray(1e-8), stableSecondMoment);
                CvInvoke.Divide(latentSignal, stableSecondMoment, latentSignal);
            }

            using var numerator = new Mat();
            CvInvoke.Multiply(latentSignal, transmission, numerator);
            CvInvoke.Add(numerator, modelPenalty, numerator);

            using var denominator = new Mat();
            CvInvoke.Multiply(latentSignal, transmissionSquared, denominator);
            CvInvoke.Add(denominator, noise, denominator);
            CvInvoke.Add(denominator, modelPenalty, denominator);
            CvInvoke.Add(denominator, new ScalarArray(1e-8), denominator);

            var gain = new Mat();
            CvInvoke.Divide(numerator, denominator, gain);

            using (var identity = new Mat(gain.Size, DepthType.Cv32F, 1))
            {
                identity.SetTo(new MCvScalar(1.0));
                CvInvoke.Max(gain, identity, gain);
            }

            using (var globalCap = new Mat(gain.Size, DepthType.Cv32F, 1))
            {
                globalCap.SetTo(new MCvScalar(maxGain));
                CvInvoke.Min(gain, globalCap, gain);
            }

            using (var inverseTransmission = new Mat())
            using (var ones = new Mat(transmission.Size, DepthType.Cv32F, 1))
            {
                ones.SetTo(new MCvScalar(1.0));
                CvInvoke.Divide(ones, transmission, inverseTransmission);
                CvInvoke.Min(gain, inverseTransmission, gain);
            }

            return gain;
        }

        /// <summary>
        /// Ищет максимальный alpha на отрезке z(alpha)=z0+alpha(z1-z0), alpha in [0,1],
        /// для которого 0&lt;=V&lt;=1 и ||c||&lt;=saturationCap*V. z0 — входной HSV-пиксель,
        /// поэтому допустимое множество никогда не пусто.
        /// </summary>
        internal static void ProjectAlongInputRay(
            double inputValue,
            double inputCx,
            double inputCy,
            double candidateValue,
            double candidateCx,
            double candidateCy,
            double saturationCap,
            out double outputValue,
            out double outputCx,
            out double outputCy)
        {
            double deltaValue = candidateValue - inputValue;
            double deltaCx = candidateCx - inputCx;
            double deltaCy = candidateCy - inputCy;
            double alpha = 1.0;

            if (deltaValue > Tiny)
                alpha = Math.Min(alpha, (1.0 - inputValue) / deltaValue);
            else if (deltaValue < -Tiny)
                alpha = Math.Min(alpha, inputValue / -deltaValue);

            alpha = Math.Clamp(alpha, 0.0, 1.0);

            if (!IsInsideSaturationCone(
                    inputValue,
                    inputCx,
                    inputCy,
                    deltaValue,
                    deltaCx,
                    deltaCy,
                    alpha,
                    saturationCap))
            {
                double low = 0.0;
                double high = alpha;
                for (int iteration = 0; iteration < ProjectionIterations; iteration++)
                {
                    double middle = 0.5 * (low + high);
                    if (IsInsideSaturationCone(
                            inputValue,
                            inputCx,
                            inputCy,
                            deltaValue,
                            deltaCx,
                            deltaCy,
                            middle,
                            saturationCap))
                    {
                        low = middle;
                    }
                    else
                    {
                        high = middle;
                    }
                }
                alpha = low;
            }

            outputValue = inputValue + alpha * deltaValue;
            outputCx = inputCx + alpha * deltaCx;
            outputCy = inputCy + alpha * deltaCy;
        }

        internal static bool IsInsideSaturationCone(
            double inputValue,
            double inputCx,
            double inputCy,
            double deltaValue,
            double deltaCx,
            double deltaCy,
            double alpha,
            double saturationCap)
        {
            double value = inputValue + alpha * deltaValue;
            if (value < 0.0 || value > 1.0)
                return false;

            double cx = inputCx + alpha * deltaCx;
            double cy = inputCy + alpha * deltaCy;
            double radiusSquared = cx * cx + cy * cy;
            double maximumRadius = saturationCap * value;
            return radiusSquared <= maximumRadius * maximumRadius + 1e-10;
        }

        private static float[] CopySingleChannel(Mat source, int count)
        {
            var data = new float[count];
            source.CopyTo(data);
            return data;
        }

        private static void BgrToHsv(
            double blue,
            double green,
            double red,
            out double hue,
            out double saturation,
            out double value)
        {
            double maximum = Math.Max(red, Math.Max(green, blue));
            double minimum = Math.Min(red, Math.Min(green, blue));
            double chroma = maximum - minimum;
            value = maximum;
            saturation = maximum > Tiny ? chroma / maximum : 0.0;

            if (chroma <= Tiny)
            {
                hue = 0.0;
                return;
            }

            if (maximum == red)
                hue = 60.0 * ((green - blue) / chroma % 6.0);
            else if (maximum == green)
                hue = 60.0 * ((blue - red) / chroma + 2.0);
            else
                hue = 60.0 * ((red - green) / chroma + 4.0);

            hue = NormalizeHue(hue);
        }

        private static double NormalizeHue(double hue)
        {
            double result = hue % 360.0;
            return result < 0.0 ? result + 360.0 : result;
        }

        private static double WrapHueDelta(double delta)
        {
            double wrapped = (delta + 180.0) % 360.0;
            if (wrapped < 0.0)
                wrapped += 360.0;
            return wrapped - 180.0;
        }
    }
}
