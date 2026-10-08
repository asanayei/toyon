# Toyon

Binary topological transform, live in your browser. Nothing is uploaded.

Binarize an image, then repeatedly apply a custom 3×3 on/off element until nothing changes. Each step a pixel stays on only if every on-cell sees foreground. Turn the centre off and shapes drift instead of shrinking. Works on the default photo, your own image, or your webcam.

## Depth 3D

The berries as a 3D point cloud ([open it](https://asanayei.github.io/toyon/#depth)).

- Depth precomputed once with Apple's [Depth Pro](https://github.com/apple/ml-depth-pro) (`tools/make_depth.py`), which also estimates the camera's field of view (about 12°).
- Stored as `depth.png` (16-bit inverse depth in two channels) plus `depth.json`.
- The browser unprojects all 614k pixels on the GPU with plain WebGL2. Drag to orbit, scroll to zoom.
- Scan sweeps a glowing sheet through the depth; Peel removes everything in front of it.
- Wind is depth-aware: a 48×32×10 grid of springs in inverse depth. Gusts reach deeper layers later and weaker, and front twigs are springier. Points move as whole berries over a background built from the farthest nearby content, so nothing stretches or ghosts.
- Save an 8 s video from the original camera.

Depth Pro is used under Apple's research licence (non-commercial).

Plain HTML + JavaScript, no libraries.

Toyon photo by [John Rusk](https://commons.wikimedia.org/wiki/File:J20151125-0008%E2%80%94Heteromeles_arbutifolia_(23303059776).jpg), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).
