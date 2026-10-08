// Breeze: make a still photo sway, fully in the browser.
//
// Idea (after Generative Image Dynamics, Li et al. CVPR 2024, and Davis et al.'s image-space
// modal bases): motion in a scene like this is a sum of a few vibration modes, each with its
// own shape and frequency. Instead of predicting those with a diffusion model, we build them:
//   1. The photo becomes a grid graph of colour patches; neighbours with similar colour are
//      strongly linked, so a berry cluster holds together and edges come apart.
//   2. The smoothest eigenvectors of that graph's Laplacian are its vibration modes, like the
//      modes of a drum skin cut to the shape of the plant. Frequency ~ sqrt(eigenvalue).
//   3. Each mode is a damped spring (one for x, one for y) driven by turbulent wind and by gusts
//      that travel across the frame. Dragging the image pulls the springs; letting go rings them.
//   4. Blurry regions are taken as far away (depth from defocus) and move less.
//   5. Each frame, the mode sum gives a displacement field and the photo is backward-warped.

// ---------- small helpers ----------
function mulberry32(a) {
  return () => {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function gaussRand(rng) {
  return Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng());
}

// Cyclic Jacobi eigen-decomposition of a small symmetric matrix (row-major, m x m).
function jacobiEigen(A, m) {
  const a = Float64Array.from(A), v = new Float64Array(m * m);
  for (let i = 0; i < m; i++) v[i * m + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < m; p++) for (let q = p + 1; q < m; q++) off += a[p * m + q] ** 2;
    if (off < 1e-20) break;
    for (let p = 0; p < m; p++) {
      for (let q = p + 1; q < m; q++) {
        const apq = a[p * m + q];
        if (Math.abs(apq) < 1e-15) continue;
        const th = (a[q * m + q] - a[p * m + p]) / (2 * apq);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < m; k++) {
          const akp = a[k * m + p], akq = a[k * m + q];
          a[k * m + p] = c * akp - s * akq; a[k * m + q] = s * akp + c * akq;
        }
        for (let k = 0; k < m; k++) {
          const apk = a[p * m + k], aqk = a[q * m + k];
          a[p * m + k] = c * apk - s * aqk; a[q * m + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < m; k++) {
          const vkp = v[k * m + p], vkq = v[k * m + q];
          v[k * m + p] = c * vkp - s * vkq; v[k * m + q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { values: Array.from({ length: m }, (_, i) => a[i * m + i]), vectors: v };
}

// ---------- 1-2. modes from the image ----------
// rgba: analysis-size pixels (a few hundred px wide). Returns K mode shapes on a gw x gh grid.
function analyzeModes(rgba, w, h, K = 12, gw = 80, iters = 200) {
  const gh = Math.max(8, Math.round(gw * h / w)), n = gw * gh;
  const col = new Float32Array(n * 3), cnt = new Float32Array(n), sharp = new Float32Array(n);
  const lum = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) lum[i] = 0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2];
  for (let y = 0; y < h; y++) {
    const gy = Math.min(gh - 1, (y * gh / h) | 0);
    for (let x = 0; x < w; x++) {
      const gi = gy * gw + Math.min(gw - 1, (x * gw / w) | 0), p = y * w + x;
      col[3 * gi] += rgba[4 * p]; col[3 * gi + 1] += rgba[4 * p + 1]; col[3 * gi + 2] += rgba[4 * p + 2];
      cnt[gi]++;
      if (x > 0 && y > 0 && x < w - 1 && y < h - 1)
        sharp[gi] += Math.abs(4 * lum[p] - lum[p - 1] - lum[p + 1] - lum[p - w] - lum[p + w]);
    }
  }
  for (let i = 0; i < n; i++) {
    const c = Math.max(1, cnt[i]);
    col[3 * i] /= c; col[3 * i + 1] /= c; col[3 * i + 2] /= c; sharp[i] /= c;
  }

  // Depth from defocus: sharp patches are near (move freely), blurry ones far (move little).
  const sorted = Float32Array.from(sharp).sort();
  const lo = sorted[(n * 0.1) | 0], hi = sorted[(n * 0.9) | 0] + 1e-6;
  let flex = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const s = Math.min(1, Math.max(0, (sharp[i] - lo) / (hi - lo)));
    flex[i] = 0.15 + 0.85 * s * s * (3 - 2 * s);
  }
  for (let pass = 0; pass < 3; pass++) {           // soften so the depth map has no seams
    const f2 = new Float32Array(n);
    for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
      let s = 0, c = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy;
        if (xx >= 0 && yy >= 0 && xx < gw && yy < gh) { s += flex[yy * gw + xx]; c++; }
      }
      f2[y * gw + x] = s / c;
    }
    flex = f2;
  }

  // Edge-aware graph: right and down links, weight exp(-colour distance^2 / sigma^2).
  const d2 = (i, j) => (col[3 * i] - col[3 * j]) ** 2 + (col[3 * i + 1] - col[3 * j + 1]) ** 2 + (col[3 * i + 2] - col[3 * j + 2]) ** 2;
  const all = [];
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
    const i = y * gw + x;
    if (x < gw - 1) all.push(d2(i, i + 1));
    if (y < gh - 1) all.push(d2(i, i + gw));
  }
  all.sort((a, b) => a - b);
  const sigma2 = all[(all.length / 2) | 0] * 1.5 + 1;
  const wr = new Float32Array(n), wd = new Float32Array(n), deg = new Float32Array(n);
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
    const i = y * gw + x;
    if (x < gw - 1) { const v = Math.exp(-d2(i, i + 1) / sigma2) + 1e-3; wr[i] = v; deg[i] += v; deg[i + 1] += v; }
    if (y < gh - 1) { const v = Math.exp(-d2(i, i + gw) / sigma2) + 1e-3; wd[i] = v; deg[i] += v; deg[i + gw] += v; }
  }
  const isq = deg.map(d => 1 / Math.sqrt(d));

  // M = (I + D^-1/2 W D^-1/2) / 2: its top eigenvectors are the smoothest Laplacian modes.
  const tmp = new Float64Array(n);
  function applyM(x, out) {
    for (let i = 0; i < n; i++) tmp[i] = x[i] * isq[i];
    out.fill(0);
    for (let y = 0; y < gh; y++) for (let xx = 0; xx < gw; xx++) {
      const i = y * gw + xx;
      if (xx < gw - 1) { out[i] += wr[i] * tmp[i + 1]; out[i + 1] += wr[i] * tmp[i]; }
      if (y < gh - 1) { out[i] += wd[i] * tmp[i + gw]; out[i + gw] += wd[i] * tmp[i]; }
    }
    for (let i = 0; i < n; i++) out[i] = 0.5 * (x[i] + out[i] * isq[i]);
  }

  // The constant mode (sqrt(deg)) is a rigid shift; project it out and find the next K.
  const v0 = new Float64Array(n);
  let nv = 0;
  for (let i = 0; i < n; i++) { v0[i] = Math.sqrt(deg[i]); nv += deg[i]; }
  for (let i = 0; i < n; i++) v0[i] /= Math.sqrt(nv);

  // Start from low-frequency cosines (already close to the answer), then subspace iteration.
  const pairs = [];
  for (let a = 0; a < 8; a++) for (let b = 0; b < 8; b++) if (a || b) pairs.push([a, b, (a / gw) ** 2 + (b / gh) ** 2]);
  pairs.sort((p, q) => p[2] - q[2]);
  let V = Array.from({ length: K }, (_, k) => {
    const [a, b] = pairs[k], v = new Float64Array(n);
    for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++)
      v[y * gw + x] = Math.cos(Math.PI * a * (x + 0.5) / gw) * Math.cos(Math.PI * b * (y + 0.5) / gh) * v0[y * gw + x];
    return v;
  });
  const dot = (a, b) => { let s = 0; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; };
  function orthonormalize() {
    for (let k = 0; k < K; k++) {
      const v = V[k];
      for (const u of [v0, ...V.slice(0, k)]) { const c = dot(v, u); for (let i = 0; i < n; i++) v[i] -= c * u[i]; }
      const s = Math.sqrt(dot(v, v)) || 1;
      for (let i = 0; i < n; i++) v[i] /= s;
    }
  }
  orthonormalize();
  let W = V.map(() => new Float64Array(n));
  for (let it = 0; it < iters; it++) {
    for (let k = 0; k < K; k++) applyM(V[k], W[k]);
    [V, W] = [W, V];
    if (it % 8 === 7) orthonormalize();
  }
  orthonormalize();

  // Rayleigh-Ritz: rotate the subspace into individual eigenvectors, smoothest first.
  for (let k = 0; k < K; k++) applyM(V[k], W[k]);
  const H = new Float64Array(K * K);
  for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) H[a * K + b] = dot(V[a], W[b]);
  const { values, vectors } = jacobiEigen(H, K);
  const order = values.map((v, i) => [v, i]).sort((p, q) => q[0] - p[0]);

  const phi = [], mu = [];
  for (const [lam, j] of order) {
    const u = new Float32Array(n);
    for (let a = 0; a < K; a++) { const c = vectors[a * K + j]; for (let i = 0; i < n; i++) u[i] += c * V[a][i]; }
    let mx = 0;
    for (let i = 0; i < n; i++) { u[i] *= isq[i]; mx = Math.max(mx, Math.abs(u[i] * flex[i])); }
    for (let i = 0; i < n; i++) u[i] /= mx || 1;
    phi.push(u);
    mu.push(Math.max(1e-6, 2 - 2 * lam));          // normalised Laplacian eigenvalue
  }
  // Membrane physics: frequency grows with sqrt(eigenvalue). Lowest mode sways at ~0.55 Hz.
  const freq = mu.map(m => Math.min(3.2, 0.55 * Math.sqrt(m / mu[0])));
  return { gw, gh, n, K, phi, mu, freq, flex, aspect: w / h };
}

// ---------- 3. driven springs ----------
class Dynamics {
  constructor(modes, seed = 7) {
    this.m = modes;
    const K = modes.K;
    this.q = new Float64Array(2 * K); this.v = new Float64Array(2 * K);   // [x0..xK-1, y0..yK-1]
    this.turb = new Float64Array(2 * K);
    this.omega = new Float64Array(2 * K);
    for (let k = 0; k < K; k++) {
      this.omega[k] = 2 * Math.PI * modes.freq[k];
      this.omega[K + k] = 2 * Math.PI * modes.freq[k] * 1.3;     // hanging fruit is stiffer up-down
    }
    this.zeta = 0.07;
    this.rng = mulberry32(seed);
    this.env = 0;                                   // slow swell of the wind
    this.gust = { c: -0.5, speed: 0.3, amp: 0, wait: 1.5 };
    this.t = 0;
    // Generalised-force weights: how much a unit push at each cell excites each mode.
    const { n, phi, flex } = modes;
    this.norm = phi.map(p => { let s = 0; for (let i = 0; i < n; i++) s += flex[i] * p[i] * p[i]; return 1 / s; });
    this.gx = new Float32Array(n);
    for (let i = 0; i < n; i++) this.gx[i] = ((i % modes.gw) + 0.5) / modes.gw;
    this.gproj = new Float64Array(K);
    this.field = new Float32Array(2 * n);
  }

  // Generalised force of a gust band centred at c (in image widths) on each mode.
  projectGust(c, width) {
    const { n, K, phi, flex } = this.m;
    this.gproj.fill(0);
    if (this.gust.amp <= 0) return;
    for (let i = 0; i < n; i++) {
      const g = Math.exp(-(((this.gx[i] - c) / width) ** 2)) * flex[i];
      if (g < 1e-3) continue;
      for (let k = 0; k < K; k++) this.gproj[k] += phi[k][i] * g;
    }
    for (let k = 0; k < K; k++) this.gproj[k] *= this.norm[k];
  }

  // wind in [0, 1]. grab: { i: grid cell, tx, ty: target displacement in image widths } or null.
  step(dt, wind, grab) {
    const { K, phi, flex } = this.m, rng = this.rng;
    const A = 0.0035 * wind;                        // typical sway, in image widths
    // Wind statistics: slow swell, per-mode turbulence, gusts that sweep across.
    const a = Math.exp(-dt / 4);
    this.env = this.env * a + Math.sqrt(1 - a * a) * gaussRand(rng);
    const b = Math.exp(-dt / 0.35);
    for (let j = 0; j < 2 * K; j++) this.turb[j] = this.turb[j] * b + Math.sqrt(1 - b * b) * gaussRand(rng);
    const G = this.gust;
    if (G.amp > 0) { G.c += G.speed * dt; if (G.c > 1.6) G.amp = 0; }
    else if ((G.wait -= dt) <= 0) {
      G.c = -0.6; G.speed = 0.25 + 0.2 * rng(); G.amp = (1.2 + 1.5 * rng()) * wind; G.wait = 3 + 6 * rng();
    }
    this.projectGust(G.c, 0.22);

    // Grabbing: the point force that would hold the grabbed cell at the target, found from
    // the cell's compliance (how far it moves per unit force, summed over modes).
    let pull = null;
    if (grab) {
      const fp = flex[grab.i];
      let cx = 0, cy = 0;
      for (let k = 0; k < K; k++) {
        const r = (phi[k][grab.i] * fp) ** 2 * this.norm[k];
        cx += r / this.omega[k] ** 2; cy += r / this.omega[K + k] ** 2;
      }
      pull = [grab.tx / cx, grab.ty / cy];
    }
    const sub = Math.max(1, Math.ceil(dt * 480)), h = dt / sub;
    for (let s = 0; s < sub; s++) {
      for (let j = 0; j < 2 * K; j++) {
        const k = j % K, isY = j >= K, w = this.omega[j], fall = Math.sqrt(this.omega[0] / w);
        // Forces are scaled by w^2 so each spring's static stretch equals the push; 'fall' gives
        // higher modes less energy (a 1/f-like spectrum, as in real wind-blown plants).
        const swell = 1 + 0.4 * this.env;
        let f = isY
          ? A * fall * (0.45 * swell * this.turb[j] + 0.15 * this.gproj[k] * G.amp)
          : A * fall * (0.9 * swell * this.turb[j] + this.gproj[k] * G.amp);
        f *= w * w;
        let damp = 2 * this.zeta * w;
        if (pull) {
          f += (isY ? pull[1] : pull[0]) * phi[k][grab.i] * flex[grab.i] * this.norm[k];
          damp += 5;
        }
        this.v[j] += h * (f - w * w * this.q[j] - damp * this.v[j]);
        this.q[j] += h * this.v[j];
      }
    }
    this.t += dt;
  }

  // Displacement field in texture units (u, v), one vec2 per grid cell.
  computeField() {
    const { n, K, phi, flex, aspect } = this.m, F = this.field, q = this.q;
    for (let i = 0; i < n; i++) {
      let dx = 0, dy = 0;
      for (let k = 0; k < K; k++) { const p = phi[k][i]; dx += q[k] * p; dy += q[K + k] * p; }
      F[2 * i] = dx * flex[i]; F[2 * i + 1] = dy * flex[i] * aspect;
    }
    return F;
  }
}

// ---------- 5a. CPU renderer: backward warp with bilinear sampling ----------
function warpCPU(src, out, W, H, F, gw, gh, show) {
  for (let y = 0; y < H; y++) {
    let gy = (y + 0.5) / H * gh - 0.5;
    gy = gy < 0 ? 0 : gy > gh - 1 ? gh - 1 : gy;
    const y0 = gy | 0, y1 = Math.min(gh - 1, y0 + 1), fy = gy - y0;
    for (let x = 0; x < W; x++) {
      let gx = (x + 0.5) / W * gw - 0.5;
      gx = gx < 0 ? 0 : gx > gw - 1 ? gw - 1 : gx;
      const x0 = gx | 0, x1 = Math.min(gw - 1, x0 + 1), fx = gx - x0;
      const a = 2 * (y0 * gw + x0), b = 2 * (y0 * gw + x1), c = 2 * (y1 * gw + x0), d = 2 * (y1 * gw + x1);
      const du = (F[a] * (1 - fx) + F[b] * fx) * (1 - fy) + (F[c] * (1 - fx) + F[d] * fx) * fy;
      const dv = (F[a + 1] * (1 - fx) + F[b + 1] * fx) * (1 - fy) + (F[c + 1] * (1 - fx) + F[d + 1] * fx) * fy;
      let sx = x - du * W, sy = y - dv * H;
      sx = sx < 0 ? 0 : sx > W - 1.001 ? W - 1.001 : sx;
      sy = sy < 0 ? 0 : sy > H - 1.001 ? H - 1.001 : sy;
      const ix = sx | 0, iy = sy | 0, ax = sx - ix, ay = sy - iy;
      const p = 4 * (iy * W + ix), q = p + 4 * W, o = 4 * (y * W + x);
      for (let ch = 0; ch < 3; ch++) {
        let v = (src[p + ch] * (1 - ax) + src[p + 4 + ch] * ax) * (1 - ay) + (src[q + ch] * (1 - ax) + src[q + 4 + ch] * ax) * ay;
        if (show) {
          const m = Math.min(1, Math.hypot(du, dv / (W / H)) / 0.008), ang = Math.atan2(dv, du);
          v = v * 0.3 * (1 - m) + 255 * m * (0.5 + 0.5 * Math.cos(ang + ch * 2.094));
        }
        out[o + ch] = v;
      }
      out[o + 3] = 255;
    }
  }
}

// ---------- 5b. GPU renderers ----------
const GL_VS = `#version 300 es
out vec2 uv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  uv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const GL_FS = `#version 300 es
precision highp float;
in vec2 uv;
uniform sampler2D img, disp;
uniform float show, aspect;
out vec4 color;
void main() {
  vec2 d = texture(disp, uv).xy;
  vec3 c = texture(img, uv - d).rgb;
  float m = clamp(length(vec2(d.x, d.y / aspect)) / 0.008, 0.0, 1.0);
  vec3 hue = 0.5 + 0.5 * cos(atan(d.y, d.x) + vec3(0.0, 2.094, 4.188));
  color = vec4(mix(c, mix(c * 0.3, hue, m), show), 1.0);
}`;

function makeWebGL(canvas, image, gw, gh) {
  const gl = canvas.getContext('webgl2', { antialias: false });
  if (!gl) throw new Error('WebGL2 not available');
  const sh = (type, src) => {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, GL_VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, GL_FS));
  gl.linkProgram(prog); gl.useProgram(prog);
  const tex = (unit, filter) => {
    const t = gl.createTexture(); gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  };
  tex(0, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
  tex(1, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG16F, gw, gh, 0, gl.RG, gl.FLOAT, null);
  gl.uniform1i(gl.getUniformLocation(prog, 'img'), 0);
  gl.uniform1i(gl.getUniformLocation(prog, 'disp'), 1);
  const uShow = gl.getUniformLocation(prog, 'show');
  gl.uniform1f(gl.getUniformLocation(prog, 'aspect'), canvas.width / canvas.height);
  return {
    name: 'WebGL2',
    draw(F, show) {
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gw, gh, gl.RG, gl.FLOAT, F);
      gl.uniform1f(uShow, show ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    },
    destroy() { gl.getExtension('WEBGL_lose_context')?.loseContext(); },
  };
}

const WGSL = `
struct U { gw: f32, gh: f32, show: f32, aspect: f32 };
@group(0) @binding(0) var img: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<storage, read> disp: array<vec2f>;
@group(0) @binding(3) var<uniform> u: U;
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VO; o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0); o.uv = vec2f(p.x, 1.0 - p.y); return o;
}
fn at(x: i32, y: i32) -> vec2f { return disp[u32(y) * u32(u.gw) + u32(x)]; }
fn field(uv: vec2f) -> vec2f {
  let g = clamp(uv * vec2f(u.gw, u.gh) - 0.5, vec2f(0.0), vec2f(u.gw - 1.0, u.gh - 1.0));
  let i0 = vec2i(floor(g)); let i1 = min(i0 + 1, vec2i(i32(u.gw) - 1, i32(u.gh) - 1)); let f = g - floor(g);
  return mix(mix(at(i0.x, i0.y), at(i1.x, i0.y), f.x), mix(at(i0.x, i1.y), at(i1.x, i1.y), f.x), f.y);
}
@fragment fn fs(o: VO) -> @location(0) vec4f {
  let d = field(o.uv);
  let c = textureSample(img, smp, o.uv - d).rgb;
  let m = clamp(length(vec2f(d.x, d.y / u.aspect)) / 0.008, 0.0, 1.0);
  let hue = 0.5 + 0.5 * cos(atan2(d.y, d.x) + vec3f(0.0, 2.094, 4.188));
  return vec4f(mix(c, mix(c * 0.3, hue, m), u.show), 1.0);
}`;

async function makeWebGPU(canvas, image, gw, gh) {
  if (!navigator.gpu) throw new Error('WebGPU not available in this browser');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('no WebGPU adapter');
  const device = await adapter.requestDevice();
  const ctx = canvas.getContext('webgpu'), format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: 'opaque' });
  const bmp = await createImageBitmap(image);
  const tex = device.createTexture({
    size: [bmp.width, bmp.height], format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture({ source: bmp }, { texture: tex }, [bmp.width, bmp.height]);
  const disp = device.createBuffer({ size: gw * gh * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const mod = device.createShaderModule({ code: WGSL });
  const pipe = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: mod, entryPoint: 'vs' },
    fragment: { module: mod, entryPoint: 'fs', targets: [{ format }] },
  });
  const bind = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: tex.createView() },
      { binding: 1, resource: device.createSampler({ magFilter: 'linear', minFilter: 'linear' }) },
      { binding: 2, resource: { buffer: disp } },
      { binding: 3, resource: { buffer: uni } },
    ],
  });
  const u = new Float32Array([gw, gh, 0, canvas.width / canvas.height]);
  return {
    name: 'WebGPU',
    draw(F, show) {
      u[2] = show ? 1 : 0;
      device.queue.writeBuffer(uni, 0, u);
      device.queue.writeBuffer(disp, 0, F);
      const enc = device.createCommandEncoder();
      const pass = enc.beginRenderPass({ colorAttachments: [{
        view: ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1],
      }] });
      pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.draw(3); pass.end();
      device.queue.submit([enc.finish()]);
    },
    destroy() { device.destroy(); },
  };
}

if (typeof module !== 'undefined') module.exports = { analyzeModes, Dynamics, warpCPU, jacobiEigen };
