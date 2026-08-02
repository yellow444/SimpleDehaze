# A²CR publication checklist

## Recommended order

1. Freeze the exact source/result commit and create a repository release.
2. Submit the LaTeX source package to arXiv under `cs.CV`.
3. Wait until the submission is announced and has an arXiv identifier.
4. Add that identifier and link to the Habr draft, then publish the Habr adaptation.

The arXiv paper does not need to mention Habr. Publishing the Habr article first would not turn it
into an arXiv affiliation or scholarly reference, but arXiv-first gives the work a stable canonical
identifier and timestamp before the popular explanation appears. A future journal or conference
may have its own preprint policy and must be checked separately.

## Author and affiliation

Use the following arXiv Authors field:

    Maksim Sitnikov (Independent Researcher)

Do not enter `MSc`, `Master of Computer Science`, `Grad Student`, or the fact that this is a second
higher-education degree in the arXiv Authors field. arXiv metadata explicitly excludes degree
suffixes, and an academic degree is not a current organizational affiliation. The PDF likewise uses
the neutral and accurate `Independent Researcher` line.

For Habr, put a short credential in the profile rather than in the evidence section of the article:

    Независимый исследователь и разработчик, магистр компьютерных наук.

The fact that the master's degree is a second higher education is personal background, not evidence
for the method, and is better omitted unless a separate autobiographical article needs that context.

## Paste-ready arXiv metadata

Category:

    cs.CV - Computer Vision and Pattern Recognition

Title:

    Uncertainty-Aware Airlight-Aligned Dual-Gain Recovery with Exact RGB Feasibility for Training-Free Single-Image Dehazing

Authors:

    Maksim Sitnikov (Independent Researcher)

Abstract (ASCII-safe):

    The inverse atmospheric-scattering model amplifies transmission, airlight, and sensor-noise errors when a scalar gain 1/t is applied in dense haze. We introduce A^2CR, a training-free recovery operator that decomposes the airlight-centered RGB residual into components parallel and perpendicular to the airlight vector. Two gains are obtained from analytic quadratic risks containing transmission, airlight, and noise uncertainty. Each gain pair is constrained by the exact convex polygon induced by the six RGB-cube inequalities and gain bounds. An edge-aware joint total-variation problem is solved with a primal-dual algorithm and exact polygon projection. On a controlled DIODE RGB-D evaluation comprising 500 frames and 22,500 reproducible haze recipes, the full variant improves over scalar inversion by 2.951 dB PSNR, 0.191 SSIM, and -0.978 CIEDE2000, while eliminating post-recovery RGB violations. Results on four real paired datasets are mixed and were not blind: the method is strongest on Dense-Haze but does not dominate the tested classical baselines elsewhere. We therefore claim an interpretable constrained-recovery study, not state-of-the-art dehazing. Code, executable tests, manifests, and statistical artifacts accompany the paper.

Comments:

    5 pages, 1 figure; code and reproducibility artifacts available in the accompanying repository.

Journal reference and DOI:

    Leave blank until an actual publication or DOI exists.

## Habr publication edits after arXiv announcement

Add near the beginning or in the reproducibility section:

    Препринт: https://arxiv.org/abs/YYMM.NNNNN

Do not describe the Habr article as a translation: it is an original Russian technical adaptation by
the same author. Keep the controlled/real-data limitations and do not replace them with a marketing
claim.
