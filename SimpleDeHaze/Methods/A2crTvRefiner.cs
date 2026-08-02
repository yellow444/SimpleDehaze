namespace SimpleDeHaze.Methods
{
    internal readonly record struct A2crJointTvResult(
        float[] GainParallel,
        float[] GainPerpendicular,
        int Iterations,
        double RelativeChange,
        double ObjectiveBefore,
        double ObjectiveAfter,
        double ConstraintProjectionFraction);

    /// <summary>TV solvers for A²CR gain maps.</summary>
    internal static class A2crTvRefiner
    {
        /// <summary>
        /// Condat/Vũ primal-dual solver for the joint convex A2CR recovery objective:
        /// quadratic uncertainty risks + gain coupling + edge-aware TV, subject to the exact
        /// per-pixel polygon formed by RGB feasibility and the gain box. The smooth risk is
        /// handled by an explicit gradient and the non-smooth terms by exact proximal projections.
        /// </summary>
        public static A2crJointTvResult SolveJoint(
            float[] initialParallel,
            float[] initialPerpendicular,
            float[] riskAParallel,
            float[] riskBParallel,
            float[] riskAPerpendicular,
            float[] riskBPerpendicular,
            float[] inputBgr,
            double[] airlightBgr,
            int rows,
            int cols,
            double tvWeight,
            double edgeScale,
            double coupling,
            int iterations,
            float gainMaximum,
            bool constrainRgb)
        {
            int n = rows * cols;
            if (initialParallel.Length != n || initialPerpendicular.Length != n ||
                riskAParallel.Length != n || riskBParallel.Length != n ||
                riskAPerpendicular.Length != n || riskBPerpendicular.Length != n ||
                inputBgr.Length != n * 3 || airlightBgr.Length != 3)
                throw new ArgumentException("A²CR joint-TV arrays have inconsistent shape");
            if (tvWeight <= 0 || iterations <= 0)
                return new A2crJointTvResult((float[])initialParallel.Clone(),
                    (float[])initialPerpendicular.Clone(), 0, 0, double.NaN, double.NaN, 0);

            coupling = Math.Max(0, coupling);
            edgeScale = Math.Max(1e-5, edgeScale);
            gainMaximum = Math.Max(1, gainMaximum);

            var polygonP = constrainRgb ? new float[n * A2crFeasibleProjector.MaximumPolygonVertices] : Array.Empty<float>();
            var polygonQ = constrainRgb ? new float[n * A2crFeasibleProjector.MaximumPolygonVertices] : Array.Empty<float>();
            var polygonCount = constrainRgb ? new byte[n] : Array.Empty<byte>();
            if (constrainRgb)
            {
                for (int i = 0; i < n; i++)
                {
                    int j = i * 3, offset = i * A2crFeasibleProjector.MaximumPolygonVertices;
                    polygonCount[i] = (byte)A2crFeasibleProjector.BuildPolygon(
                        inputBgr[j], inputBgr[j + 1], inputBgr[j + 2],
                        airlightBgr[0], airlightBgr[1], airlightBgr[2], gainMaximum,
                        polygonP.AsSpan(offset, A2crFeasibleProjector.MaximumPolygonVertices),
                        polygonQ.AsSpan(offset, A2crFeasibleProjector.MaximumPolygonVertices));
                }
            }

            var edgeWeight = EdgeWeights(inputBgr, rows, cols, edgeScale);
            var gainP = new float[n]; var gainQ = new float[n];
            long initialProjected = 0;
            for (int i = 0; i < n; i++)
            {
                var projected = Project(i, initialParallel[i], initialPerpendicular[i]);
                gainP[i] = (float)projected.GainParallel;
                gainQ[i] = (float)projected.GainPerpendicular;
                if (projected.WasProjected) initialProjected++;
            }

            double objectiveBefore = Objective(gainP, gainQ, riskAParallel, riskBParallel,
                riskAPerpendicular, riskBPerpendicular, edgeWeight, rows, cols, tvWeight, coupling);
            double bestObjective = objectiveBefore;
            var bestP = (float[])gainP.Clone(); var bestQ = (float[])gainQ.Clone();
            long bestProjected = initialProjected;
            var dualPx = new float[n]; var dualPy = new float[n];
            var dualQx = new float[n]; var dualQy = new float[n];
            var nextP = new float[n]; var nextQ = new float[n];
            var extrapolatedP = new float[n]; var extrapolatedQ = new float[n];

            double lipschitz = 0;
            for (int i = 0; i < n; i++)
            {
                double h11 = 2.0 * (Math.Max(0, riskAParallel[i]) + coupling);
                double h22 = 2.0 * (Math.Max(0, riskAPerpendicular[i]) + coupling);
                double h12 = -2.0 * coupling;
                double eigenMax = 0.5 * (h11 + h22 +
                    Math.Sqrt((h11 - h22) * (h11 - h22) + 4 * h12 * h12));
                lipschitz = Math.Max(lipschitz, eigenMax);
            }
            double tau = lipschitz > 1e-12 ? Math.Min(10.0, 0.95 / lipschitz) : 0.25;
            // ||forward-gradient||^2 <= 8. This choice satisfies
            // 1/tau - sigma*||K||^2 >= L/2 for tau <= 0.95/L.
            double sigma = 0.49 / (8.0 * tau);
            double relativeChange = double.PositiveInfinity;
            long projectedLast = initialProjected;
            int iterationsUsed = 0;

            for (int iteration = 0; iteration < iterations; iteration++)
            {
                double delta2 = 0, scale2 = 0;
                projectedLast = 0;
                for (int y = 0; y < rows; y++)
                for (int x = 0; x < cols; x++)
                {
                    int i = y * cols + x;
                    double divergenceP = dualPx[i] - (x > 0 ? dualPx[i - 1] : 0) +
                                         dualPy[i] - (y > 0 ? dualPy[i - cols] : 0);
                    double divergenceQ = dualQx[i] - (x > 0 ? dualQx[i - 1] : 0) +
                                         dualQy[i] - (y > 0 ? dualQy[i - cols] : 0);
                    double difference = gainP[i] - gainQ[i];
                    double gradientP = 2 * riskAParallel[i] * gainP[i] - 2 * riskBParallel[i] +
                                       2 * coupling * difference;
                    double gradientQ = 2 * riskAPerpendicular[i] * gainQ[i] - 2 * riskBPerpendicular[i] -
                                       2 * coupling * difference;
                    var projected = Project(i,
                        gainP[i] - tau * (gradientP - divergenceP),
                        gainQ[i] - tau * (gradientQ - divergenceQ));
                    nextP[i] = (float)projected.GainParallel;
                    nextQ[i] = (float)projected.GainPerpendicular;
                    if (projected.WasProjected) projectedLast++;
                    extrapolatedP[i] = 2 * nextP[i] - gainP[i];
                    extrapolatedQ[i] = 2 * nextQ[i] - gainQ[i];
                    double dp = nextP[i] - gainP[i], dq = nextQ[i] - gainQ[i];
                    delta2 += dp * dp + dq * dq;
                    scale2 += gainP[i] * gainP[i] + gainQ[i] * gainQ[i];
                }

                for (int y = 0; y < rows; y++)
                for (int x = 0; x < cols; x++)
                {
                    int i = y * cols + x;
                    double px = dualPx[i] + sigma * (x + 1 < cols ? extrapolatedP[i + 1] - extrapolatedP[i] : 0);
                    double py = dualPy[i] + sigma * (y + 1 < rows ? extrapolatedP[i + cols] - extrapolatedP[i] : 0);
                    double qx = dualQx[i] + sigma * (x + 1 < cols ? extrapolatedQ[i + 1] - extrapolatedQ[i] : 0);
                    double qy = dualQy[i] + sigma * (y + 1 < rows ? extrapolatedQ[i + cols] - extrapolatedQ[i] : 0);
                    double radius = tvWeight * edgeWeight[i];
                    double normP = Math.Sqrt(px * px + py * py);
                    double normQ = Math.Sqrt(qx * qx + qy * qy);
                    double scaleP = normP > radius && normP > 0 ? radius / normP : 1;
                    double scaleQ = normQ > radius && normQ > 0 ? radius / normQ : 1;
                    dualPx[i] = (float)(px * scaleP); dualPy[i] = (float)(py * scaleP);
                    dualQx[i] = (float)(qx * scaleQ); dualQy[i] = (float)(qy * scaleQ);
                }

                (gainP, nextP) = (nextP, gainP);
                (gainQ, nextQ) = (nextQ, gainQ);
                iterationsUsed = iteration + 1;
                relativeChange = Math.Sqrt(delta2 / Math.Max(1e-20, scale2));
                double objective = Objective(gainP, gainQ, riskAParallel, riskBParallel,
                    riskAPerpendicular, riskBPerpendicular, edgeWeight, rows, cols, tvWeight, coupling);
                if (objective < bestObjective)
                {
                    bestObjective = objective;
                    bestProjected = projectedLast;
                    Array.Copy(gainP, bestP, n);
                    Array.Copy(gainQ, bestQ, n);
                }
                if (iteration >= 9 && relativeChange < 1e-6) break;
            }

            return new A2crJointTvResult(bestP, bestQ, iterationsUsed, relativeChange,
                objectiveBefore, bestObjective, bestProjected / (double)n);

            A2crGainPairProjection Project(int index, double proposedP, double proposedQ)
            {
                if (!constrainRgb)
                {
                    double boundedP = Math.Clamp(proposedP, 1, gainMaximum);
                    double boundedQ = Math.Clamp(proposedQ, 1, gainMaximum);
                    double distance = Math.Sqrt((boundedP - proposedP) * (boundedP - proposedP) +
                                                (boundedQ - proposedQ) * (boundedQ - proposedQ));
                    return new A2crGainPairProjection(boundedP, boundedQ, distance, distance > 1e-10);
                }
                int offset = index * A2crFeasibleProjector.MaximumPolygonVertices;
                int count = polygonCount[index];
                return A2crFeasibleProjector.ProjectToPolygon(proposedP, proposedQ,
                    polygonP.AsSpan(offset, count), polygonQ.AsSpan(offset, count));
            }
        }

        public static float[] Denoise(float[] source, int rows, int cols, double weight, int iterations,
            float minimum, float maximum)
        {
            if (weight <= 0 || iterations <= 0) return (float[])source.Clone();
            int n = source.Length;
            var px = new float[n]; var py = new float[n]; var div = new float[n]; var u = new float[n];
            float w = (float)weight;
            const float tau = 0.249f;
            for (int iteration = 0; iteration < iterations; iteration++)
            {
                Divergence(px, py, div, rows, cols);
                for (int i = 0; i < n; i++) u[i] = source[i] + w * div[i];
                for (int y = 0; y < rows; y++)
                for (int x = 0; x < cols; x++)
                {
                    int i = y * cols + x;
                    float gx = x + 1 < cols ? u[i + 1] - u[i] : 0;
                    float gy = y + 1 < rows ? u[i + cols] - u[i] : 0;
                    float scale = 1f + tau * MathF.Sqrt(gx * gx + gy * gy) / Math.Max(1e-8f, w);
                    px[i] = (px[i] + tau * gx / Math.Max(1e-8f, w)) / scale;
                    py[i] = (py[i] + tau * gy / Math.Max(1e-8f, w)) / scale;
                }
            }
            Divergence(px, py, div, rows, cols);
            for (int i = 0; i < n; i++) u[i] = Math.Clamp(source[i] + w * div[i], minimum, maximum);
            return u;
        }

        private static float[] EdgeWeights(float[] inputBgr, int rows, int cols, double scale)
        {
            int n = rows * cols;
            var luminance = new float[n]; var weights = new float[n];
            for (int i = 0; i < n; i++)
            {
                int j = i * 3;
                luminance[i] = 0.0722f * inputBgr[j] + 0.7152f * inputBgr[j + 1] + 0.2126f * inputBgr[j + 2];
            }
            double scale2 = scale * scale;
            for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
            {
                int i = y * cols + x;
                double dx = x + 1 < cols ? luminance[i + 1] - luminance[i] : 0;
                double dy = y + 1 < rows ? luminance[i + cols] - luminance[i] : 0;
                weights[i] = (float)(1.0 / Math.Sqrt(1.0 + (dx * dx + dy * dy) / scale2));
            }
            return weights;
        }

        internal static double Objective(float[] gainP, float[] gainQ,
            float[] riskAP, float[] riskBP, float[] riskAQ, float[] riskBQ,
            float[] edgeWeight, int rows, int cols, double tvWeight, double coupling)
        {
            double objective = 0;
            for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
            {
                int i = y * cols + x;
                double p = gainP[i], q = gainQ[i], difference = p - q;
                objective += riskAP[i] * p * p - 2 * riskBP[i] * p +
                             riskAQ[i] * q * q - 2 * riskBQ[i] * q +
                             coupling * difference * difference;
                double dxP = x + 1 < cols ? gainP[i + 1] - p : 0;
                double dyP = y + 1 < rows ? gainP[i + cols] - p : 0;
                double dxQ = x + 1 < cols ? gainQ[i + 1] - q : 0;
                double dyQ = y + 1 < rows ? gainQ[i + cols] - q : 0;
                objective += tvWeight * edgeWeight[i] *
                    (Math.Sqrt(dxP * dxP + dyP * dyP) + Math.Sqrt(dxQ * dxQ + dyQ * dyQ));
            }
            return objective;
        }

        private static void Divergence(float[] px, float[] py, float[] output, int rows, int cols)
        {
            for (int y = 0; y < rows; y++)
            for (int x = 0; x < cols; x++)
            {
                int i = y * cols + x;
                float dx = px[i] - (x > 0 ? px[i - 1] : 0);
                float dy = py[i] - (y > 0 ? py[i - cols] : 0);
                output[i] = dx + dy;
            }
        }
    }
}
