namespace SimpleDeHaze.Methods
{
    internal readonly record struct A2crProjection(
        double GainParallel, double GainPerpendicular, double Alpha,
        double B, double G, double R);

    internal readonly record struct A2crEuclideanProjection(
        double GainParallel, double GainPerpendicular,
        double B, double G, double R,
        double Distance, bool WasProjected);

    internal readonly record struct A2crGainPairProjection(
        double GainParallel, double GainPerpendicular,
        double Distance, bool WasProjected);

    /// <summary>
    /// Exact maximum feasible step from the identity gains (1,1) towards a proposed dual gain.
    /// Since (1,1) reconstructs the input pixel, the ray always starts inside the RGB cube.
    /// </summary>
    internal static class A2crFeasibleProjector
    {
        internal const int MaximumPolygonVertices = 10;

        public static A2crProjection Project(double b, double g, double r,
            double airB, double airG, double airR, double gainParallel, double gainPerpendicular)
        {
            Components(b, g, r, airB, airG, airR,
                out double pb, out double pg, out double pr,
                out double qb, out double qg, out double qr);

            double dB = (gainParallel - 1.0) * pb + (gainPerpendicular - 1.0) * qb;
            double dG = (gainParallel - 1.0) * pg + (gainPerpendicular - 1.0) * qg;
            double dR = (gainParallel - 1.0) * pr + (gainPerpendicular - 1.0) * qr;
            double alpha = Math.Min(1.0, Math.Min(MaxStep(b, dB), Math.Min(MaxStep(g, dG), MaxStep(r, dR))));
            alpha = Math.Clamp(alpha, 0.0, 1.0);

            return new A2crProjection(
                1.0 + alpha * (gainParallel - 1.0),
                1.0 + alpha * (gainPerpendicular - 1.0),
                alpha,
                b + alpha * dB,
                g + alpha * dG,
                r + alpha * dR);
        }

        /// <summary>
        /// Exact Euclidean projection of a proposed gain pair onto the intersection of
        /// [1,gainMax]^2 and the six linear RGB-cube half-spaces. In two dimensions the
        /// closest point is either the proposal itself, a projection onto one active edge,
        /// or an intersection of two active edges, so exhaustive boundary enumeration is
        /// deterministic and exact up to floating-point tolerance.
        /// </summary>
        public static A2crEuclideanProjection ProjectEuclidean(double b, double g, double r,
            double airB, double airG, double airR, double gainParallel, double gainPerpendicular,
            double gainMax)
        {
            gainMax = Math.Max(1.0, gainMax);
            Components(b, g, r, airB, airG, airR,
                out double pb, out double pg, out double pr,
                out double qb, out double qg, out double qr);

            // Half-spaces are represented as a*x+b*y <= c.
            double[] ca =
            {
                -1, 1, 0, 0,
                -pb, pb, -pg, pg, -pr, pr,
            };
            double[] cb =
            {
                0, 0, -1, 1,
                -qb, qb, -qg, qg, -qr, qr,
            };
            double[] cc =
            {
                -1, gainMax, -1, gainMax,
                airB, 1-airB, airG, 1-airG, airR, 1-airR,
            };

            double bestP = 1.0, bestQ = 1.0;
            double bestDistance2 = SquaredDistance(gainParallel, gainPerpendicular, bestP, bestQ);
            if (Feasible(gainParallel, gainPerpendicular, ca, cb, cc))
            {
                bestP = gainParallel;
                bestQ = gainPerpendicular;
                bestDistance2 = 0.0;
            }
            else
            {
                for (int i = 0; i < ca.Length; i++)
                {
                    double norm2 = ca[i] * ca[i] + cb[i] * cb[i];
                    if (norm2 <= 1e-24) continue;
                    double excess = ca[i] * gainParallel + cb[i] * gainPerpendicular - cc[i];
                    double candidateP = gainParallel - excess * ca[i] / norm2;
                    double candidateQ = gainPerpendicular - excess * cb[i] / norm2;
                    Consider(candidateP, candidateQ);
                }

                for (int i = 0; i < ca.Length; i++)
                for (int j = i + 1; j < ca.Length; j++)
                {
                    double determinant = ca[i] * cb[j] - ca[j] * cb[i];
                    if (Math.Abs(determinant) <= 1e-18) continue;
                    double candidateP = (cc[i] * cb[j] - cc[j] * cb[i]) / determinant;
                    double candidateQ = (ca[i] * cc[j] - ca[j] * cc[i]) / determinant;
                    Consider(candidateP, candidateQ);
                }
            }

            double outB = airB + bestP * pb + bestQ * qb;
            double outG = airG + bestP * pg + bestQ * qg;
            double outR = airR + bestP * pr + bestQ * qr;
            double distance = Math.Sqrt(Math.Max(0, bestDistance2));
            return new A2crEuclideanProjection(bestP, bestQ, outB, outG, outR,
                distance, distance > 1e-10);

            void Consider(double candidateP, double candidateQ)
            {
                if (!double.IsFinite(candidateP) || !double.IsFinite(candidateQ) ||
                    !Feasible(candidateP, candidateQ, ca, cb, cc)) return;
                double distance2 = SquaredDistance(gainParallel, gainPerpendicular, candidateP, candidateQ);
                if (distance2 < bestDistance2)
                {
                    bestDistance2 = distance2;
                    bestP = candidateP;
                    bestQ = candidateQ;
                }
            }
        }

        /// <summary>Builds the per-pixel feasible polygon once for repeated solver projections.</summary>
        internal static int BuildPolygon(double b, double g, double r,
            double airB, double airG, double airR, double gainMax,
            Span<float> gainParallelVertices, Span<float> gainPerpendicularVertices)
        {
            if (gainParallelVertices.Length < MaximumPolygonVertices ||
                gainPerpendicularVertices.Length < MaximumPolygonVertices)
                throw new ArgumentException("Polygon buffers are too small");

            gainMax = Math.Max(1.0, gainMax);
            Components(b, g, r, airB, airG, airR,
                out double pb, out double pg, out double pr,
                out double qb, out double qg, out double qr);
            Span<double> ca = stackalloc double[10];
            Span<double> cb = stackalloc double[10];
            Span<double> cc = stackalloc double[10];
            FillConstraints(pb, pg, pr, qb, qg, qr, airB, airG, airR, gainMax, ca, cb, cc);

            Span<double> vp = stackalloc double[MaximumPolygonVertices];
            Span<double> vq = stackalloc double[MaximumPolygonVertices];
            int count = 0;
            for (int i = 0; i < ca.Length; i++)
            for (int j = i + 1; j < ca.Length; j++)
            {
                double determinant = ca[i] * cb[j] - ca[j] * cb[i];
                if (Math.Abs(determinant) <= 1e-18) continue;
                double candidateP = (cc[i] * cb[j] - cc[j] * cb[i]) / determinant;
                double candidateQ = (ca[i] * cc[j] - ca[j] * cc[i]) / determinant;
                if (!Feasible(candidateP, candidateQ, ca, cb, cc)) continue;
                bool duplicate = false;
                for (int k = 0; k < count; k++)
                    if (SquaredDistance(candidateP, candidateQ, vp[k], vq[k]) <= 1e-14)
                    { duplicate = true; break; }
                if (duplicate) continue;
                if (count < MaximumPolygonVertices)
                {
                    vp[count] = candidateP;
                    vq[count] = candidateQ;
                    count++;
                }
            }

            if (count == 0)
            {
                vp[0] = vq[0] = 1.0;
                count = 1;
            }
            else if (count > 2)
            {
                double centerP = 0, centerQ = 0;
                for (int i = 0; i < count; i++) { centerP += vp[i]; centerQ += vq[i]; }
                centerP /= count; centerQ /= count;
                for (int i = 1; i < count; i++)
                {
                    double keyP = vp[i], keyQ = vq[i];
                    double keyAngle = Math.Atan2(keyQ - centerQ, keyP - centerP);
                    int k = i - 1;
                    while (k >= 0 && Math.Atan2(vq[k] - centerQ, vp[k] - centerP) > keyAngle)
                    {
                        vp[k + 1] = vp[k]; vq[k + 1] = vq[k]; k--;
                    }
                    vp[k + 1] = keyP; vq[k + 1] = keyQ;
                }
            }

            for (int i = 0; i < count; i++)
            {
                gainParallelVertices[i] = (float)vp[i];
                gainPerpendicularVertices[i] = (float)vq[i];
            }
            return count;
        }

        /// <summary>Fast exact projection onto a polygon previously produced by <see cref="BuildPolygon"/>.</summary>
        internal static A2crGainPairProjection ProjectToPolygon(double gainParallel, double gainPerpendicular,
            ReadOnlySpan<float> polygonParallel, ReadOnlySpan<float> polygonPerpendicular)
        {
            if (polygonParallel.Length != polygonPerpendicular.Length || polygonParallel.Length == 0)
                throw new ArgumentException("Invalid feasible polygon");
            if (InsideConvexPolygon(gainParallel, gainPerpendicular, polygonParallel, polygonPerpendicular))
                return new A2crGainPairProjection(gainParallel, gainPerpendicular, 0, false);

            double bestP = polygonParallel[0], bestQ = polygonPerpendicular[0];
            double bestDistance2 = SquaredDistance(gainParallel, gainPerpendicular, bestP, bestQ);
            int edges = polygonParallel.Length == 1 ? 0 : polygonParallel.Length;
            for (int i = 0; i < edges; i++)
            {
                int j = (i + 1) % polygonParallel.Length;
                double ax = polygonParallel[i], ay = polygonPerpendicular[i];
                double dx = polygonParallel[j] - ax, dy = polygonPerpendicular[j] - ay;
                double length2 = dx * dx + dy * dy;
                double fraction = length2 <= 1e-24 ? 0 :
                    Math.Clamp(((gainParallel - ax) * dx + (gainPerpendicular - ay) * dy) / length2, 0, 1);
                double candidateP = ax + fraction * dx, candidateQ = ay + fraction * dy;
                double distance2 = SquaredDistance(gainParallel, gainPerpendicular, candidateP, candidateQ);
                if (distance2 < bestDistance2)
                {
                    bestDistance2 = distance2;
                    bestP = candidateP;
                    bestQ = candidateQ;
                }
            }
            double distance = Math.Sqrt(Math.Max(0, bestDistance2));
            return new A2crGainPairProjection(bestP, bestQ, distance, distance > 1e-10);
        }

        private static bool InsideConvexPolygon(double x, double y,
            ReadOnlySpan<float> polygonX, ReadOnlySpan<float> polygonY)
        {
            if (polygonX.Length == 1)
                return SquaredDistance(x, y, polygonX[0], polygonY[0]) <= 1e-14;
            if (polygonX.Length == 2)
            {
                double dx = polygonX[1] - polygonX[0], dy = polygonY[1] - polygonY[0];
                double length2 = dx * dx + dy * dy;
                double fraction = length2 <= 1e-24 ? 0 :
                    Math.Clamp(((x - polygonX[0]) * dx + (y - polygonY[0]) * dy) / length2, 0, 1);
                return SquaredDistance(x, y, polygonX[0] + fraction * dx,
                    polygonY[0] + fraction * dy) <= 1e-14;
            }
            double sign = 0;
            for (int i = 0; i < polygonX.Length; i++)
            {
                int j = (i + 1) % polygonX.Length;
                double cross = (polygonX[j] - polygonX[i]) * (y - polygonY[i]) -
                               (polygonY[j] - polygonY[i]) * (x - polygonX[i]);
                if (Math.Abs(cross) <= 2e-8) continue;
                double current = Math.Sign(cross);
                if (sign == 0) sign = current;
                else if (current != sign) return false;
            }
            return true;
        }

        private static bool Feasible(double gainParallel, double gainPerpendicular,
            ReadOnlySpan<double> a, ReadOnlySpan<double> b, ReadOnlySpan<double> c)
        {
            const double tolerance = 2e-10;
            for (int i = 0; i < a.Length; i++)
                if (a[i] * gainParallel + b[i] * gainPerpendicular > c[i] + tolerance)
                    return false;
            return true;
        }

        private static void FillConstraints(double pb, double pg, double pr,
            double qb, double qg, double qr, double airB, double airG, double airR,
            double gainMax, Span<double> ca, Span<double> cb, Span<double> cc)
        {
            ca[0] = -1; ca[1] = 1; ca[2] = 0; ca[3] = 0;
            ca[4] = -pb; ca[5] = pb; ca[6] = -pg; ca[7] = pg; ca[8] = -pr; ca[9] = pr;
            cb[0] = 0; cb[1] = 0; cb[2] = -1; cb[3] = 1;
            cb[4] = -qb; cb[5] = qb; cb[6] = -qg; cb[7] = qg; cb[8] = -qr; cb[9] = qr;
            cc[0] = -1; cc[1] = gainMax; cc[2] = -1; cc[3] = gainMax;
            cc[4] = airB; cc[5] = 1-airB; cc[6] = airG; cc[7] = 1-airG; cc[8] = airR; cc[9] = 1-airR;
        }

        private static double SquaredDistance(double x, double y, double px, double py)
        {
            double dx = x - px, dy = y - py;
            return dx * dx + dy * dy;
        }

        private static void Components(double b, double g, double r,
            double airB, double airG, double airR,
            out double pb, out double pg, out double pr,
            out double qb, out double qg, out double qr)
        {
            double norm = Math.Sqrt(airB * airB + airG * airG + airR * airR);
            double ub, ug, ur;
            if (norm > 1e-12)
            {
                ub = airB / norm; ug = airG / norm; ur = airR / norm;
            }
            else
            {
                ub = ug = ur = 1.0 / Math.Sqrt(3.0);
            }

            double db = b - airB, dg = g - airG, dr = r - airR;
            double dot = ub * db + ug * dg + ur * dr;
            pb = ub * dot; pg = ug * dot; pr = ur * dot;
            qb = db - pb; qg = dg - pg; qr = dr - pr;
        }

        private static double MaxStep(double input, double delta)
        {
            if (delta > 1e-15) return Math.Max(0, (1.0 - input) / delta);
            if (delta < -1e-15) return Math.Max(0, input / -delta);
            return 1.0;
        }
    }
}
