# Toyon

Binary topological transform, live in your browser. Nothing is uploaded.

Binarize an image, then repeatedly apply a custom 3×3 on/off element until nothing changes. Each step a pixel stays on only if every on-cell sees foreground. Turn the centre off and shapes drift instead of shrinking. Works on the default photo, your own image, or your webcam.

## Breeze

A still photo set swaying by physics, fully in the browser ([open it](https://asanayei.github.io/toyon/#breeze)).

- The photo becomes an edge-aware grid graph of colour patches.
- Its smoothest graph-Laplacian eigenvectors are the vibration modes; frequency ~ sqrt(eigenvalue).
- Each mode is a damped spring driven by turbulent wind and travelling gusts. Drag to pull, release to ring.
- Blurry regions are treated as far away and move less (depth from defocus).
- The mode sum is a motion field that backward-warps the photo, on CPU, WebGL2 or WebGPU.
- Save a 6 s video straight from the canvas.

Inspired by [Generative Image Dynamics](https://generative-dynamics.github.io/) (Li et al., CVPR 2024), with the spectral modes built analytically instead of by a diffusion model.

Plain HTML + JavaScript, no libraries.

Toyon photo by [John Rusk](https://commons.wikimedia.org/wiki/File:J20151125-0008%E2%80%94Heteromeles_arbutifolia_(23303059776).jpg), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).
