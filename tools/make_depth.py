"""Precompute the depth asset for the Depth tab (run once, on a machine with a GPU).

Runs Apple's Depth Pro on toyon.jpg, which gives metric depth plus the camera's field of view,
and writes:
  depth.png   960x640, R/G = 16-bit inverse depth (high/low byte), B = 255 (reserved)
  depth.json  { w, h, f (focal length in px at this size), near, far }

Usage: python tools/make_depth.py   (needs torch + transformers)
"""
import json
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

W, H = 960, 640
img = Image.open('toyon.jpg').convert('RGB')
proc = AutoImageProcessor.from_pretrained('apple/DepthPro-hf')
model = AutoModelForDepthEstimation.from_pretrained('apple/DepthPro-hf', dtype=torch.float16).cuda().eval()
inp = {k: (v.cuda().half() if v.dtype == torch.float32 else v.cuda()) for k, v in proc(images=img, return_tensors='pt').items()}
with torch.no_grad():
    out = model(**inp)
post = proc.post_process_depth_estimation(out, target_sizes=[(img.height, img.width)])[0]
fov = float(post['field_of_view'])                       # horizontal, degrees
z = np.asarray(Image.fromarray(post['predicted_depth'].float().cpu().numpy()).resize((W, H), Image.BILINEAR))

near, far = float(np.percentile(z, 0.1)), float(np.percentile(z, 99.9))
z = np.clip(z, near, far)
inv = (1 / z - 1 / far) / (1 / near - 1 / far)           # 0 = far, 1 = near
q = np.round(inv * 65535).astype(np.uint32)

# Depth edges: a pixel straddling a big depth jump (berry rim against the background) would
# float in mid-air when unprojected ("flying pixels"). Snap it to whichever side it is closer to.
pad = np.pad(z, 1, mode='edge')
win = np.stack([pad[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)])
lo, hi = win.min(0), win.max(0)
edge = (hi - lo) / z > 0.15
snapped = np.where(z - lo < hi - z, lo, hi)
z = np.where(edge, snapped, z)
inv = (1 / z - 1 / far) / (1 / near - 1 / far)
q = np.round(inv * 65535).astype(np.uint32)

rgb = np.stack([q >> 8, q & 255, np.full_like(q, 255)], -1).astype(np.uint8)
Image.fromarray(rgb).save('depth.png', optimize=True)
f = 0.5 * W / np.tan(np.radians(fov) / 2)
json.dump({'w': W, 'h': H, 'f': round(f, 2), 'near': round(near, 4), 'far': round(far, 4)}, open('depth.json', 'w'))
print(f'fov {fov:.2f} deg, f {f:.0f} px, depth {near:.2f}-{far:.2f} m, edges {edge.mean():.1%}')
