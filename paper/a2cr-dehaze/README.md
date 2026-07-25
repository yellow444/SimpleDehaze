# A2CR arXiv source package

This directory is self-contained for compilation and upload. It is a research preprint draft,
not an arXiv submission record and not a claim of state-of-the-art performance.

Build on Windows with MiKTeX:

```powershell
.\build.ps1
```

The script uses `latexmk` when Perl is available and otherwise runs the explicit
`pdflatex -> bibtex -> pdflatex -> pdflatex` sequence.

The checked PDF is `output/pdf/a2cr-dehaze-preprint.pdf` at repository root. The two vector figures
are generated from reviewed raw artifacts with `python tools/build_publication_figures.py`, then
copied into `figures/` to keep the upload independent of the repository layout.

Before an actual submission, the author should still:

1. choose the arXiv category and replace the draft date if needed;
2. register a DOI/archive release for the exact code and result artifacts;
3. freeze a clean commit and disclose that the current real-paired evaluation is not blind;
4. ideally add a pre-registered external blind test and full external BCCR implementation;
5. check dataset and figure redistribution terms for the selected submission bundle.

Reproducibility commands, hashes, split definitions, metric conventions, and raw-result paths are
in `../../REPRODUCIBILITY.md` and `../../SimpleDeHaze/docs/research/a2cr-data-protocol.md`.
