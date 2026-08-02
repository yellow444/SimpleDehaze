# Public branch contents

This branch is a clean public snapshot with an independent Git history. It contains:

- application, benchmark, CUDA and test source code;
- build, dataset-preparation and analysis scripts;
- public documentation and article sources;
- the A2CR LaTeX source package and its publication figures;
- dataset source URLs, manifests and reproducibility instructions.

It intentionally excludes private planning notes, local benchmark outputs, generated archives and
PDF builds, downloaded datasets, bundled dataset images, package caches, binaries and profiling
artifacts. Downloaded research datasets remain governed by their original providers' terms and can
be prepared locally with the scripts under `tools/`.

The allowlist is deliberate. Before merging new files into this branch, review them for credentials,
personal contact data, machine-specific paths, third-party redistribution rights and generated
artifacts.
