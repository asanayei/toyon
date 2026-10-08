"""Generative fill for the Depth tab: make the point cloud hold up from wide angles.

A single photo only shows the front of every berry. Orbit the camera and gaps open up: behind
berries, around the edges of the frame. This script fills them with generated content and lifts
it into 3D ("inpaint and lift", as in Text2Room / LucidDreamer):

  for each camera on a widening ring around the original view:
    1. render the current cloud (photo points + everything generated so far) from that camera
    2. find the holes (big uncovered regions; tiny cracks are left to the browser's point size)
    3. SDXL inpainting paints the holes, conditioned on what is visible around them
    4. Depth Pro estimates depth for the painted view; an affine fit plus a smooth residual
       ties it to the rendered depth around the hole, so new geometry meets old without seams
    5. the painted hole pixels become new 3D points

Everything happens in the browser's default "Depth 20%" space (zscale 0.2), the geometry
people actually orbit. Points are written back in the browser's own terms (original pixel
position + normalised inverse depth), so the Depth slider and the wind still apply to them.

Writes gen.bin: N points x 12 bytes, all little-endian:
  int16 px*8, int16 py*8   position in original-photo pixels (may lie outside the frame)
  uint16 q                 inverse depth, (q16 / 65535) * 2 - 0.5  ->  -0.5 .. 1.5
  uint8 r, g, b, size      size = footprint in original pixels * 32
  uint16 pad
Usage: python tools/gen_fill.py  [--views N] [--debug DIR]
"""
import argparse, json, math, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--views', type=int, default=99)
ap.add_argument('--debug', default='')
ap.add_argument('--steps', type=int, default=30)
args = ap.parse_args()
dev = 'cuda'
torch.manual_seed(7)

meta = json.load(open('depth.json'))
W0, H0, f0, near, far = meta['w'], meta['h'], meta['f'], meta['near'], meta['far']
ZSCALE = 0.2
pivot = math.sqrt(near * far)
RW, RH = 1152, 768                                  # render / inpaint size (3:2, multiple of 64)
fn = (RH / 2) / ((H0 / 2) / f0)                     # same vertical fov as the browser at zoom 1

# ---------- the photo as points, in scaled space ----------
dep = np.asarray(Image.open('depth.png').convert('RGB')).astype(np.int64)
q0 = (dep[..., 0] * 256 + dep[..., 1]) / 65535.0
col0 = np.asarray(Image.open('toyon.jpg').convert('RGB').resize((W0, H0), Image.LANCZOS)).astype(np.float32) / 255
py0, px0 = np.mgrid[0:H0, 0:W0].astype(np.float64)


def q_to_z(q): return 1 / ((1 / far) + q * (1 / near - 1 / far))
def z_to_q(z): return (1 / z - 1 / far) / (1 / near - 1 / far)
def zs_of(z): return pivot + (z - pivot) * ZSCALE
def z_of(zs): return pivot + (zs - pivot) / ZSCALE


def to_world(px, py, q):
    zs = zs_of(q_to_z(q))
    rx, ry = (px + 0.5 - W0 / 2) / f0, (py + 0.5 - H0 / 2) / f0
    return np.stack([rx * zs, -ry * zs, -zs], -1)


def from_world(P):
    zs = -P[:, 2]
    px = P[:, 0] / zs * f0 + W0 / 2 - 0.5
    py = -P[:, 1] / zs * f0 + H0 / 2 - 0.5
    return px, py, z_to_q(np.maximum(z_of(zs), 0.3))


P = to_world(px0, py0, q0).reshape(-1, 3)
C = col0.reshape(-1, 3)
S = (zs_of(q_to_z(q0)) / f0).reshape(-1)            # lateral footprint (scaled metres)
pts = [torch.tensor(P, dtype=torch.float32, device=dev)]
cols = [torch.tensor(C, dtype=torch.float32, device=dev)]
sizes = [torch.tensor(S, dtype=torch.float32, device=dev)]
gen = []                                            # (P, C, S) of generated points, numpy


def camera(yaw, pitch):
    target = np.array([0, 0, -pivot])
    eye = target + pivot * np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)])
    z = eye - target; z /= np.linalg.norm(z)
    x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])                         # world -> camera rotation
    return R, eye


def render(R, eye):
    """Z-buffered disc splats, like the browser. Returns rgb, view depth (0 = empty), coverage."""
    Pw, Cw, Sw = torch.cat(pts), torch.cat(cols), torch.cat(sizes)
    Rt = torch.tensor(R, dtype=torch.float32, device=dev)
    Pc = (Pw - torch.tensor(eye, dtype=torch.float32, device=dev)) @ Rt.T
    D = -Pc[:, 2]
    ok = D > 0.05
    u = fn * Pc[:, 0] / D + RW / 2 - 0.5
    v = -fn * Pc[:, 1] / D + RH / 2 - 0.5
    rad = (0.75 * fn * Sw / D).clamp(0.5, 3.0)      # shader: diameter 1.5 * footprint
    key = torch.full((RH * RW,), 2 ** 62, dtype=torch.int64, device=dev)
    idx = torch.arange(len(D), device=dev)
    dq = (D.clamp(0, 50) * 1e6).long()
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            uu, vv = torch.round(u).long() + dx, torch.round(v).long() + dy
            # disc test against the true centre
            m = ok & ((uu - u) ** 2 + (vv - v) ** 2 <= rad ** 2 + 0.25) & (uu >= 0) & (uu < RW) & (vv >= 0) & (vv < RH)
            k = (dq[m] << 22) | idx[m]
            key.scatter_reduce_(0, vv[m] * RW + uu[m], k, 'amin')
    hit = key < 2 ** 62
    win = (key & (2 ** 22 - 1)).clamp(max=len(D) - 1)
    rgb = torch.where(hit[:, None], Cw[win], torch.zeros(3, device=dev)).reshape(RH, RW, 3)
    depth = torch.where(hit, D[win], torch.zeros((), device=dev)).reshape(RH, RW)
    return rgb, depth, hit.reshape(RH, RW)


def holes(hit):
    """Uncovered regions worth painting: close tiny cracks first, then grow a margin."""
    miss = (~hit).float()[None, None]
    # cracks: a missing pixel mostly surrounded by covered ones
    frac = F.avg_pool2d(miss, 5, 1, 2)
    big = (miss > 0) & (frac > 0.45)
    big = F.max_pool2d(big.float(), 5, 1, 2) * miss   # back inside the true gap
    lift = big[0, 0] > 0
    # SDXL works on 8x8-pixel latents and greys out every masked pixel of its input, so a thin
    # mask comes back as grey specks. Paint whole latent blocks, plus one block of margin.
    blk = F.max_pool2d(big, 8, 8)
    blk = F.max_pool2d(blk, 3, 1, 1)
    paint = F.interpolate(blk, scale_factor=8, mode='nearest')[0, 0] > 0
    return lift, paint


# ---------- models ----------
from diffusers import AutoPipelineForInpainting
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

pipe = AutoPipelineForInpainting.from_pretrained(
    'diffusers/stable-diffusion-xl-1.0-inpainting-0.1', torch_dtype=torch.float16, variant='fp16').to(dev)
pipe.set_progress_bar_config(disable=True)
dproc = AutoImageProcessor.from_pretrained('apple/DepthPro-hf')
dmodel = AutoModelForDepthEstimation.from_pretrained('apple/DepthPro-hf', dtype=torch.float16).to(dev).eval()
PROMPT = ('close-up photo of a toyon shrub, clusters of glossy red berries, dark green serrated leaves, '
          'thin branches, natural daylight, shallow depth of field, sharp detail')
NEG = 'text, watermark, people, smear, cartoon, painting, frame, border, spiky, hairy, thorns, fuzzy'


def mono_depth(img):
    inp = {k: (v.to(dev).half() if v.dtype == torch.float32 else v.to(dev)) for k, v in dproc(images=img, return_tensors='pt').items()}
    with torch.no_grad():
        out = dmodel(**inp)
    return dproc.post_process_depth_estimation(out, target_sizes=[(RH, RW)])[0]['predicted_depth'].float()


def smooth_fill(val, w, sigma):
    """Normalised Gaussian convolution: spread known values (weight w) into the unknown."""
    k = int(3 * sigma) | 1
    g = torch.exp(-torch.arange(-k, k + 1, device=dev).float() ** 2 / (2 * sigma ** 2))
    def blur(x):
        x = F.conv2d(x[None, None], g.view(1, 1, 1, -1), padding=(0, k))
        return F.conv2d(x, g.view(1, 1, -1, 1), padding=(k, 0))[0, 0]
    return blur(val * w) / blur(w).clamp(min=1e-6)


def sdxl(img, need, prompt, seed):
    """Inpaint `need` (bool HxW) in img (HxWx3 float). Returns the painted picture as a tensor."""
    # SDXL works on 8x8-pixel latents and greys out every masked pixel of its input, so a thin
    # mask comes back as grey specks. Paint whole latent blocks, plus one block of margin.
    blk = F.max_pool2d(need.float()[None, None], 8, 8)
    blk = F.max_pool2d(blk, 3, 1, 1)
    pm = F.interpolate(blk, scale_factor=8, mode='nearest')[0, 0] > 0
    src = Image.fromarray((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    mask = Image.fromarray((pm.cpu().numpy() * 255).astype(np.uint8))
    out = pipe(prompt=prompt, negative_prompt=NEG, image=src, mask_image=mask, width=RW, height=RH,
               num_inference_steps=args.steps, strength=0.99, guidance_scale=6.0,
               generator=torch.Generator(dev).manual_seed(seed)).images[0]
    return torch.tensor(np.asarray(out), device=dev).float() / 255, out, pm


def fit_q(img_pil, q_known, good):
    """Mono depth -> normalised inverse depth q, tied to known q (affine + smooth residual)."""
    inv = 1 / mono_depth(img_pil).clamp(min=0.05)
    A = torch.stack([inv[good], torch.ones_like(inv[good])], 1)
    a_, b_ = torch.linalg.lstsq(A, q_known[good][:, None]).solution[:, 0]
    fit = a_ * inv + b_
    resid = torch.where(good, q_known - fit, torch.zeros_like(fit))
    return fit + smooth_fill(resid, good.float(), 20)


# ---------- stage 1: layers behind the photo, from the original camera ----------
# Like 3D Photo Inpainting (Shih et al. 2020): peel the near berries off, paint what is behind
# them, repeat a little deeper. Done from the original viewpoint, where SDXL sees the real photo
# best. The canvas is a little wider than the photo, so the first pass also extends the edges.
SC = 0.75                                           # photo covers 75% of the canvas
OX, OY = RW * (1 - SC) / 2, RH * (1 - SC) / 2
KPX = W0 / (RW * SC)                                # original pixels per canvas pixel
iw, ih = round(RW * SC), round(RH * SC)
I = torch.zeros(RH, RW, 3, device=dev)
Q = torch.zeros(RH, RW, device=dev)
inner = torch.zeros(RH, RW, dtype=torch.bool, device=dev)
oy, ox = round(OY), round(OX)
I[oy:oy + ih, ox:ox + iw] = torch.tensor(np.asarray(Image.open('toyon.jpg').convert('RGB').resize((iw, ih), Image.LANCZOS)), device=dev).float() / 255
Q[oy:oy + ih, ox:ox + iw] = torch.tensor(np.asarray(Image.fromarray(q0.astype(np.float32)).resize((iw, ih), Image.NEAREST)), device=dev)
inner[oy:oy + ih, ox:ox + iw] = True


def lift_canvas(region, Iq, Qq, stride):
    ys, xs = torch.nonzero(region, as_tuple=True)
    keep = ((ys % stride) == 0) & ((xs % stride) == 0)
    ys, xs = ys[keep], xs[keep]
    px = ((xs.float() + 0.5 - OX) * KPX - 0.5).cpu().numpy().astype(np.float64)
    py = ((ys.float() + 0.5 - OY) * KPX - 0.5).cpu().numpy().astype(np.float64)
    q = Qq[ys, xs].clamp(-0.4, 1.4).cpu().numpy().astype(np.float64)
    Pw = torch.tensor(to_world(px, py, q), dtype=torch.float32, device=dev)
    Sn = torch.tensor(zs_of(q_to_z(q)) / f0 * KPX * stride, dtype=torch.float32, device=dev)
    pts.append(Pw); cols.append(Iq[ys, xs]); sizes.append(Sn)
    gen.append((Pw.cpu().numpy(), Iq[ys, xs].cpu().numpy(), Sn.cpu().numpy()))
    return len(ys)


if args.debug: os.makedirs(args.debug, exist_ok=True)
# 1a. extend past the frame
edge = ~inner
# start the margins from a blurred mirror of the photo, or SDXL just keeps them black
inner_img = I[oy:oy + ih, ox:ox + iw].permute(2, 0, 1)[None]
mirror = F.pad(inner_img, (ox, RW - iw - ox, oy, RH - ih - oy), mode='reflect')[0].permute(1, 2, 0)
soft = torch.stack([smooth_fill(mirror[..., c], torch.ones(RH, RW, device=dev), 12) for c in range(3)], -1)
I = torch.where(inner[..., None], I, soft)
out_t, out_pil, _ = sdxl(I, edge, PROMPT, 11)
Q = torch.where(inner, Q, fit_q(out_pil, Q, inner))
I = torch.where(inner[..., None], I, out_t)
print(f'outpaint: +{lift_canvas(edge, I, Q, 2)} points', flush=True)
DEEP = Q.clone()                                    # farthest surface so far along each ray
if args.debug: out_pil.save(f'{args.debug}/L0.jpg')
# 1b. peel layers
LAYERS = [(0.35, 'red toyon berries and thin branches deeper inside the shrub, partly shaded, photo'),
          (0.60, 'branches and leaves deep inside a toyon shrub, a few red berries, in shade, photo'),
          (0.85, 'dark green foliage in soft shade, out of focus background, photo')]
for li, (tau, prompt) in enumerate(LAYERS):
    front = Q > 1 - tau
    region = F.max_pool2d(front.float()[None, None], 5, 1, 2)[0, 0] > 0    # take the rims too
    out_t, out_pil, pm = sdxl(I, region, prompt, 20 + li)
    good = ~pm
    qn = fit_q(out_pil, Q, good)
    qn = torch.minimum(qn, torch.full_like(qn, 1 - tau - 0.04))       # must lie behind the peel
    # Only keep paint that lies behind everything already on that ray. Anything else would
    # duplicate a surface and the two would show through each other as speckle.
    new = region & (qn < DEEP - 0.03)
    I = torch.where(new[..., None], out_t, I)
    Q = torch.where(new, qn, Q)
    DEEP = torch.where(new, qn, DEEP)
    print(f'layer {li} (t<{tau}): {front.float().mean():.0%} peeled, +{lift_canvas(new, I, Q, 2)} points', flush=True)
    if args.debug: out_pil.save(f'{args.debug}/L{li + 1}.jpg')

# ---------- stage 2: widening ring of new cameras for whatever is still missing ----------
# widening rings: each later view sees what the earlier ones already filled
VIEWS = []
for yaw, pitch in [(0.25, 0), (-0.25, 0), (0, 0.18), (0, -0.18),
                   (0.5, 0), (-0.5, 0), (0.35, 0.22), (-0.35, 0.22), (0.35, -0.22), (-0.35, -0.22),
                   (0.8, 0), (-0.8, 0), (0, 0.4), (0, -0.4), (0.6, 0.3), (-0.6, 0.3), (0.6, -0.3), (-0.6, -0.3),
                   (1.1, 0.1), (-1.1, 0.1)]:
    VIEWS.append((yaw, pitch))
VIEWS = VIEWS[:args.views]

for vi, (yaw, pitch) in enumerate(VIEWS):
    R, eye = camera(yaw, pitch)
    rgb, depth, hit = render(R, eye)
    lift, paint = holes(hit)
    print(f'view {vi} yaw {yaw:+.2f} pitch {pitch:+.2f}: holes {lift.float().mean():.1%}', flush=True)
    if lift.sum() < 500:
        continue
    # cracks inside covered areas: a cheap fill so SDXL sees a clean picture
    known = hit.float()
    base = torch.stack([smooth_fill(rgb[..., c], known, 1.5) for c in range(3)], -1)
    base = torch.where((hit | paint)[..., None], rgb, base)
    src = Image.fromarray((base.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    mask = Image.fromarray((paint.cpu().numpy() * 255).astype(np.uint8))
    out = pipe(prompt=PROMPT, negative_prompt=NEG, image=src, mask_image=mask, width=RW, height=RH,
               num_inference_steps=args.steps, strength=0.99, guidance_scale=7.0,
               generator=torch.Generator(dev).manual_seed(100 + vi)).images[0]
    painted = torch.tensor(np.asarray(out), device=dev).float() / 255
    # depth: affine fit of mono depth to rendered depth where both exist, then a smooth residual
    md = mono_depth(out)
    good = hit & ~paint & (depth > 0)
    a_, b_ = torch.linalg.lstsq(torch.stack([md[good], torch.ones_like(md[good])], 1), depth[good][:, None]).solution[:, 0]
    fit = a_ * md + b_
    resid = torch.where(good, depth - fit, torch.zeros_like(depth))
    D = fit + smooth_fill(resid, good.float(), 25)
    D = D.clamp(pivot * 0.6, pivot * 2.0)
    # lift painted hole pixels (every 2nd pixel: plenty dense, half the file)
    ys, xs = torch.nonzero(lift, as_tuple=True)
    keep = ((ys % 2) == 0) & ((xs % 2) == 0)
    ys, xs = ys[keep], xs[keep]
    d = D[ys, xs]
    cx = (xs.float() + 0.5 - RW / 2) / fn * d
    cy = -(ys.float() + 0.5 - RH / 2) / fn * d
    Pc = torch.stack([cx, cy, -d], 1)
    Pw = Pc @ torch.tensor(R, dtype=torch.float32, device=dev) + torch.tensor(eye, dtype=torch.float32, device=dev)
    Cn = painted[ys, xs]
    Sn = 2 * d / fn
    # never in front of what the photo (and the layers) already show along the original rays
    gpx, gpy, gq = from_world(Pw.cpu().numpy().astype(np.float64))
    cxs = np.round((gpx + 0.5) / KPX + OX - 0.5).astype(int); cys = np.round((gpy + 0.5) / KPX + OY - 0.5).astype(int)
    inside = (cxs >= 0) & (cxs < RW) & (cys >= 0) & (cys < RH)
    deep = np.full(len(gq), np.inf)
    deep[inside] = DEEP.cpu().numpy()[cys[inside], cxs[inside]]
    ok_t = torch.tensor(gq < deep - 0.02, device=dev)
    Pw, Cn, Sn, d = Pw[ok_t], Cn[ok_t], Sn[ok_t], d[ok_t]
    pts.append(Pw); cols.append(Cn); sizes.append(Sn)
    gen.append((Pw.cpu().numpy(), Cn.cpu().numpy(), Sn.cpu().numpy()))
    print(f'   +{len(d)} points (fit a={a_:.3f} b={b_:.3f})', flush=True)
    if args.debug:
        tag = f'{args.debug}/v{vi:02d}'
        src.save(tag + '_in.jpg'); mask.save(tag + '_mask.png'); out.save(tag + '_out.jpg')

# ---------- write gen.bin ----------
Pg = np.concatenate([g[0] for g in gen]); Cg = np.concatenate([g[1] for g in gen]); Sg = np.concatenate([g[2] for g in gen])
px, py, q = from_world(Pg.astype(np.float64))
zs = -Pg[:, 2]
size = Sg / (zs / f0)                               # footprint in original-pixel units at that depth
ok = (np.abs(px * 8) < 32000) & (np.abs(py * 8) < 32000) & (q > -0.5) & (q < 1.5)
rec = np.zeros(ok.sum(), dtype=[('x', '<i2'), ('y', '<i2'), ('q', '<u2'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('s', 'u1'), ('pad', '<u2')])
rec['x'] = np.round(px[ok] * 8); rec['y'] = np.round(py[ok] * 8)
rec['q'] = np.round((q[ok] + 0.5) / 2 * 65535)
c8 = np.round(np.clip(Cg[ok], 0, 1) * 255).astype(np.uint8)
rec['r'], rec['g'], rec['b'] = c8[:, 0], c8[:, 1], c8[:, 2]
rec['s'] = np.clip(np.round(size[ok] * 32), 1, 255)
rec.tofile('gen.bin')
print(f'gen.bin: {ok.sum()} points, {os.path.getsize("gen.bin") / 1e6:.1f} MB')
