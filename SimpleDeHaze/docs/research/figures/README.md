# Publication figure provenance

Figures are regenerated with:

```powershell
python tools/build_publication_figures.py
```

| Figure | Analytical question | Source | Rendering |
|---|---|---|---|
| `diode-bootstrap-effects` | Does B9 improve paired frame-level metrics over B0/B8, and at what runtime cost? | `benchmark_results/publication-statistics.json`; 500 frames, 45 recipes averaged within frame, 10 000 stratified bootstrap iterations | faceted dot and 95% interval, blue B9-B0, orange B9-B8, dashed zero |
| `scene8-car-wall` | Does global tuning preserve the same local wall color as CAR defaults? | two `scene8-wall-study.csv` files and corresponding full/ROI PNGs | four-column full image plus identical ROI crops |

Both PNG and vector PDF exports are produced. Scene 08 is explicitly labelled as a development
case. No chart uses a truncated absolute-magnitude bar or an unreported denominator.

