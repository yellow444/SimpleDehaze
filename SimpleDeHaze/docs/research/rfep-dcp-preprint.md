# Boundary-Constrained Prior Fusion for Non-Learning Single-Image Dehazing

> arXiv-style draft. This is a Markdown manuscript skeleton; convert to LaTeX after experiments
> on external datasets and baselines.
>
> **CORRECTION (2026-07-27).** An earlier version of this draft claimed the transmission lower
> bound derived from `0 <= J <= 1` as its first contribution. That bound is **not new**: it is the
> boundary constraint of Meng et al., *Efficient Image Dehazing with Boundary Constraint and
> Contextual Regularization*, ICCV 2013 (with radiance limits `C0=0, C1=1`). The claim has been
> removed. What remains is (i) prior fusion, (ii) a feasibility set derived for the *chroma-safe*
> recovery operator actually used by the code, and (iii) the engineering composition. The method
> is renamed accordingly. See `NOVELTY.md` at the repository root for the authoritative claim list.

## Abstract

Single-image dehazing remains ill-posed, and learning-based methods may suffer from domain
shift, opaque failure modes, and high deployment costs. We study a non-learning prior-based
pipeline that inserts a known boundary-constraint projection layer between prior fusion and
image recovery. The standard lower envelope follows Meng et al. (ICCV 2013); for the actual
chroma-safe recovery operator we derive and numerically test a separate admissible interval.
The pipeline combines a dark-channel
transmission estimate, a robust HSV bright-region prior, sky-aware confidence fusion, fast
guided refinement, and chroma-safe recovery. The method is fully deterministic, requires no
training data, and exposes interpretable parameters. The planned evaluation uses raw PSNR,
SSIM and CIEDE2000 as primary metrics; GT-aligned values are diagnostic only.

## 1. Introduction

Atmospheric scattering is commonly modeled as

$$
I(x)=J(x)t(x)+A(1-t(x)).
$$

Classical priors such as DCP provide transparent and efficient dehazing, but fail on bright,
low-texture regions such as sky, white walls, snow, and glare. In such regions, a slightly
underestimated transmission causes the inversion

$$
J_c(x)=\frac{I_c(x)-A_c}{t(x)}+A_c
$$

to leave the valid RGB cube. Subsequent clipping hides the violation but leaves visual
artifacts: color halos, saturation spikes, and burned highlights.

This paper asks a simple question: before recovering `J`, can we cheaply test whether the
chosen transmission is physically feasible?

## 2. Contributions

*Not claimed:* the transmission lower bound derived from radiance limits. That is the boundary
constraint of Meng et al. (ICCV 2013) and is used here as a known result.

1. **Feasibility set for chroma-safe recovery.** The implementation does not use the standard
   inversion but `J_c = A_c + d̄/max(t,m) + δ_c/max(t,q)`, for which the classical boundary
   constraint is not valid. We derive the corresponding admissible set, which turns out to be an
   *interval* rather than a ray in the regime `m <= t < q`, and show it is generally weaker than
   the classical bound — i.e. projecting chroma-safe recovery onto `t_box` is over-conservative.
   Correctness is verified by randomized testing (`--mathtest`) against an independently written
   scalar reference.
2. **Engineering bright-region prior.** Quantile-normalized `V-S` in HSV space avoids trained
   CAP coefficients; novelty is not claimed for this component.
3. **Projection before and after edge-aware refinement**, with an explicit `strict` mode in which
   the guarantee actually holds (unscaled bound, hard projection) and a relaxed default in which
   it does not — the violation rate is measured rather than assumed.
4. **Reproducible C# / Emgu.CV implementation** with a canonical DCP baseline, explicit
   `--profile=core` configurations (unsupported methods are rejected), and headless CSV benchmarking
   with dataset manifests, exact parameters, repeated timing and hardware/memory metadata.

## 3. Related Work

**Dark Channel Prior.** DCP estimates haze density from the minimum color channel over local
patches and remains a strong transparent baseline.

**Boundary Constraint and Contextual Regularization.** Boundary-constraint methods derive
transmission constraints from the imaging model and solve a regularized optimization problem.
RFEP is related but uses the constraint as a fast insertion layer after prior fusion rather
than as a standalone global solver.

**Bright-region Priors.** RSVT and related HSV-space methods address color distortion in
bright regions. RFEP uses robust HSV statistics for confidence and alternative transmission
estimation, then guards the final inversion through RGB-cube feasibility.

**Fast Guided Filter.** We use subsampled guided refinement for practical runtime.

## 4. Method

### 4.1 Prior Fusion

DCP estimate:

$$
t_D(x)=1-\omega\,dark(I/A).
$$

Robust HSV estimate:

$$
q(x)=V(x)-S(x),
\quad
z(x)=\frac{q(x)-median(q)}{IQR(q)/1.349+\varepsilon},
$$

$$
t_H(x)=\exp(-\alpha\,clip(z(x),0,z_{max})).
$$

DCP confidence:

$$
w_D(x)=(1-Sky(x))\exp(-\lambda_v(V(x)-\tau_v)_+)
\exp(-\lambda_s(\tau_s-S(x))_+).
$$

Implementation uses the product of the three factors:

$$
w_D=(1-Sky)\cdot e^{-\lambda_v(\cdot)}\cdot e^{-\lambda_s(\cdot)}.
$$

Fused map:

$$
t_{mix}=t_H+w_D(t_D-t_H).
$$

### 4.2 Boundary Constraint (Meng et al., ICCV 2013 — prior work, not a contribution)

From

$$
0\le \frac{I_c-A_c}{t}+A_c\le 1
$$

we obtain

$$
t_{box,c}(x)=
\begin{cases}
\frac{I_c(x)-A_c}{1-A_c+\varepsilon}, & I_c(x)>A_c,\\
\frac{A_c-I_c(x)}{A_c+\varepsilon}, & I_c(x)<A_c,\\
0, & otherwise.
\end{cases}
$$

The channelwise envelope is

$$
t_{box}(x)=\max_c t_{box,c}(x).
$$

### 4.3 Projection

Soft projection:

$$
t_{proj}=t_{mix}+\rho\max(t_{box}-t_{mix},0).
$$

`rho=1` is a hard projection; `rho<1` trades strict feasibility for stronger haze removal.

### 4.4 Refinement and Recovery

RFEP-DCP refines `t_proj` with fast guided filtering, applies a second projection, and recovers
the image using chroma-safe decomposition:

$$
J_c=A_c+\frac{\bar d}{\max(t,t_{min})}
+\frac{\delta_c}{\max(t,chromaFloor)}.
$$

## 5. Experiments

### 5.1 Datasets

Planned:

- RESIDE SOTS indoor/outdoor;
- I-HAZE;
- O-HAZE;
- NH-HAZE / HD-NH-HAZE for non-uniform haze.

### 5.2 Baselines

Classical baselines:

- DCP;
- CAP-HSV;
- BCCR;
- PF-DCP;
- non-local haze-lines;
- Tarel;
- RSVT;
- current SimpleDeHaze variants.

### 5.3 Metrics

- PSNR and SSIM;
- exposure/white-balance aligned PSNR and SSIM (**diagnostic only, never used for ranking**);
- CIEDE2000;
- no-reference score;
- clipping percentage;
- colorfulness ratio;
- runtime and milliseconds per megapixel.

### 5.4 Ablations

- DCP only;
- DCP + robust HSV;
- DCP + robust HSV + confidence fusion;
- fusion + RFEP before refinement;
- fusion + RFEP before and after refinement;
- RFEP + chroma-safe recovery.

## 6. Current Repository Evidence

Implemented files:

- `Methods/RfepDcpMethod.cs`;
- `Methods/BraceDcpMethod.cs`;
- `Methods/PfSfgfMethod.cs`;
- `Methods/LafTvMethod.cs`;
- `Methods/GdrSpMethod.cs`;
- `Program.cs --benchmark`;
- `Metrics.cs` with CIEDE2000 and runtime diagnostics.

Smoke commands:

```powershell
dotnet build SimpleDeHaze.sln -c Debug
dotnet run --project SimpleDeHaze\SimpleDeHaze.csproj -- --selftest
dotnet run --project SimpleDeHaze\SimpleDeHaze.csproj -- --benchmark --limit=1 --out=benchmark_one.csv
```

Historical smoke evidence on bundled `01_outdoor_hazy.jpg` (aligned columns are diagnostic and
must not be used for ranking):

| Method | PSNR aligned | SSIM aligned | CIEDE2000 aligned | score | ms/MP |
|---|---:|---:|---:|---:|---:|
| DCP CPU | 16.97 | 0.752 | 14.19 | 72.3 | 796 |
| BRACE-DCP | 16.60 | 0.602 | 14.66 | 60.4 | 617 |
| RFEP-DCP | 16.62 | 0.611 | 14.65 | 58.2 | 489 |
| PF-SFGF | 16.96 | 0.751 | 14.18 | 61.4 | 476 |
| LAF-TV/WLS | 17.59 | 0.838 | 13.34 | 59.2 | 253 |
| GDR-SP | 16.93 | 0.740 | 14.27 | 54.3 | 1346 |

These are sanity-check numbers only. Publication claims require RESIDE/I-HAZE/O-HAZE-style
evaluation and external non-learning baselines.

### 4.5 Admissible Set for the Chroma-Safe Operator (candidate contribution)

With `u = max(t,m)`, `v = max(t,q)`, `q >= m`, and `e_c = A_c + δ_c/q`, the recovery
`J_c = e_c + d̄/u` is monotone in `t` on `m <= t < q`, giving per channel

- `d̄ > 0`: `t >= d̄/(1-e_c)` if `e_c < 1`, and additionally `t <= d̄/(-e_c)` when `e_c < 0`;
- `d̄ < 0`: `t >= |d̄|/e_c` if `e_c > 0`, and additionally `t <= |d̄|/(e_c-1)` when `e_c > 1`;
- `d̄ = 0`: admissible iff `0 <= e_c <= 1`.

The admissible region is therefore the interval `[t_lo, t_hi]` intersected over channels. Because
the effective luminance denominator is `u = max(t,m)`, admissibility requires `max(t_lo, m) <= t_hi`.
If the interval is empty or `t_lo >= q`, feasibility is only reachable in the regime `t >= q`, where
the operator degenerates to the standard inversion and the classical bound applies:
`t >= max(t_box, q)`.

## 7. Limitations

- The method depends on atmospheric light estimation.
- The derived set is *sufficient*: in the interval case we return its left end, which is the minimal
  admissible transmission, but we do not attempt to exploit slack in the `t >= q` regime.
- With the default relaxed projection (`strict=0`) no feasibility guarantee holds; only `strict=1`
  is guaranteed, and at a cost in dehazing strength.
- The RGB-box bound can make the method conservative in scenes where aggressive enhancement is
  visually preferred.
- Official NIQE/BRISQUE require external learned/statistical models; the repository does not
  report them. Its own diagnostics are named `natur_dev_own` and `artifact_dev_own`.
- Strong arXiv claims require external baselines and datasets, not only the bundled O-HAZE-like
  sample pairs.

## 8. Conclusion

RFEP-DCP demonstrates that prior-based dehazing can be made more reliable by checking physical
feasibility before radiance recovery. The projection layer is deterministic, local, fast, and
compatible with existing DCP-like transmission priors.
