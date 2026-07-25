# AutoTuner audit source notes

These are supporting notes for the rendered technical report, not a second report surface.

## Scope and evidence

- Development/failure case: O-HAZE scene 08. It is not a blind test because it was inspected while developing CAR and the tuner safeguards.
- Processing grain: 714×800 output; coarse search at 480 px, fine search at 340 px; final candidates reprocessed at 800 px.
- Reference and no-reference audits both use 320 requested unique evaluations. GT is withheld from the `ObjectVisibility` search and used only for post-selection diagnostics.
- Canonical consolidated evidence: `benchmark_results/autotune-audit-final-summary.json`.
- Regression suite at handoff: 40/40 passed in Release.

## Chart map

- Section: reference-result comparison.
- Question: did reference tuning improve every declared 0–100 component rather than only the scalar objective?
- Form: grouped native bar chart.
- Fields: `metric`, `score`, grouped by `stage`; raw PSNR, SSIM, CIEDE2000, clipping, and chroma score remain in each source row.
- Takeaway: the tuned result improves objective, raw reference score, SSIM, normalized PSNR, normalized ΔE fidelity, and salient chroma on this development case.
- Palette: two-root comparison (neutral/start versus blue/tuned), with stage labels and grouped position as non-color distinction.

No second quantitative chart is used: the no-GT safety decision and wall ROI are exact audit cases with only 2–4 rows, so tables are more honest and readable than sparse charts.

## Reproduction

```text
dotnet build SimpleDeHaze.sln -c Release
dotnet run --project SimpleDeHaze.Tests/SimpleDeHaze.Tests.csproj -c Release --no-build
dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release --no-build -- --autotune-audit --thorough --maxeval=320 --evalmaxdim=340 --maxdim=800 --goal=ref "--methods=^Chromatic Airlight" --out=benchmark_results/autotune-audit-car-scene8-reference-guarded-final-320
dotnet run --project SimpleDeHaze/SimpleDeHaze.csproj -c Release --no-build -- --autotune-audit --thorough --maxeval=320 --evalmaxdim=340 --maxdim=800 --goal=obj "--methods=^Chromatic Airlight" --out=benchmark_results/autotune-audit-car-scene8-thorough-object-guarded-320
```

## Interpretation constraints

- A finite deterministic budget cannot prove the global optimum of a discontinuous 27-dimensional objective. “Best” means best verified safe finalist among the searched canonical candidates.
- Full-image reference tuning is not ROI tuning. On the fixed wall ROI, CAR defaults retain the target `a*` direction better than the globally tuned configuration.
- The no-GT safety result demonstrates protection on this development case, not universal perceptual validity. It should be repeated on held-out images and other registered methods.
