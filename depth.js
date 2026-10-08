// Depth 3D: the photo as a point cloud, from a precomputed Depth Pro depth map (tools/make_depth.py).
//
// Every pixel of the 960x640 depth map is one point. The vertex shader reads its depth straight
// from the depth texture (no vertex buffers), unprojects it with the camera's real focal length,
// and colours it from the photo. Scaling depth slides points along their own camera ray, so from
// the original viewpoint the cloud always looks exactly like the photo.

const DEPTH_VS = `#version 300 es
uniform sampler2D depthTex, colorTex;
uniform highp sampler3D windTex;
uniform mat4 viewProj;
uniform vec2 size;
uniform float f, near, far, zscale, zpivot, pxScale, scan, scanMode, colorMode, windOn, genPass;
// Generated points (gen.bin) come as vertex attributes instead of from the depth texture.
in vec2 aPos;     // original-photo pixel position * 8
in float aQ;      // inverse depth, normalised: q = aQ * 2 - 0.5
in vec4 aCol;     // rgb + footprint (a * 255 / 32 original pixels)
out vec3 vColor;

vec3 turbo(float x) {
  vec4 v4 = vec4(1.0, x, x * x, x * x * x);
  vec2 v2 = v4.zw * v4.z;
  return vec3(
    dot(v4, vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234)) + dot(v2, vec2(-152.94239396, 59.28637943)),
    dot(v4, vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333)) + dot(v2, vec2(4.27729857, 2.82956604)),
    dot(v4, vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771)) + dot(v2, vec2(-89.90310912, 27.34824973)));
}

void main() {
  vec2 p; float q, valid = 1.0, foot = 1.0;
  if (genPass > 0.5) {
    p = aPos / 8.0; q = aQ * 2.0 - 0.5; foot = aCol.a * 255.0 / 32.0;
  } else {
    int W = int(size.x);
    ivec2 ip = ivec2(gl_VertexID % W, gl_VertexID / W);
    vec4 d = texelFetch(depthTex, ip, 0);
    p = vec2(ip);
    q = (floor(d.r * 255.0 + 0.5) * 256.0 + floor(d.g * 255.0 + 0.5)) / 65535.0;
    valid = d.b;
  }
  float z = 1.0 / mix(1.0 / far, 1.0 / near, q);
  float t = 1.0 - q;          // 0 = nearest, 1 = farthest, even in inverse depth (where the detail is)
  // In peel mode drop everything in front of the scan plane.
  if (valid < 0.5 || (scanMode > 1.5 && t < scan - 0.004)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

  float zs = zpivot + (z - zpivot) * zscale;
  vec2 ray = (p + 0.5 - 0.5 * size) / f;
  vec3 P = vec3(ray.x * zs, -ray.y * zs, -zs);
  vec2 uv = (p + 0.5) / size;
  // Wind: a sway in metres from the 3D spring grid, at this point's place and depth. Scaled by
  // zs / z so it projects to the same pixels whatever the Depth slider says.
  if (windOn > 0.5) P.xy += texture(windTex, vec3(uv, t)).xy * zs / z;
  gl_Position = viewProj * vec4(P, 1.0);
  gl_PointSize = clamp(1.5 * foot * pxScale * zs / f / gl_Position.w, 1.0, 24.0);

  vec3 c = colorMode > 0.5 ? turbo(0.1 + 0.85 * clamp(q, 0.0, 1.0)) : genPass > 0.5 ? aCol.rgb : texture(colorTex, uv).rgb;
  t = clamp(t, 0.0, 1.0);
  c *= 1.0 - 0.35 * t;                                  // a little atmosphere: far is dimmer
  if (scanMode > 0.5) {
    // A glowing sheet sweeps from front to back; in Scan mode what is behind it waits in shadow.
    float glow = exp(-pow((t - scan) / 0.02, 2.0));
    float behind = scanMode < 1.5 && t > scan ? 0.4 : 1.0;
    c = c * behind + glow * vec3(0.55, 1.25, 1.45);
  }
  vColor = c;
}`;

const DEPTH_FS = `#version 300 es
precision mediump float;
in vec3 vColor;
out vec4 color;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  if (dot(c, c) > 1.0) discard;
  color = vec4(vColor, 1.0);
}`;

const BACK_VS = `#version 300 es
out vec2 uv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  uv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.999, 1.0);
}`;
const BACK_FS = `#version 300 es
precision mediump float;
in vec2 uv;
uniform sampler2D back;
uniform float alpha;
out vec4 color;
void main() { color = vec4(mix(vec3(0.03, 0.035, 0.045), texture(back, uv).rgb * 0.85, alpha), 1.0); }`;

// ---------- tiny matrix helpers (column-major, like GL) ----------
function perspective(fovy, aspect, n, f) {
  const t = 1 / Math.tan(fovy / 2);
  return [t / aspect, 0, 0, 0, 0, t, 0, 0, 0, 0, (f + n) / (n - f), -1, 0, 0, 2 * f * n / (n - f), 0];
}
function lookAt(e, c, up) {
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const norm = a => { const l = Math.hypot(...a); return a.map(v => v / l); };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const z = norm(sub(e, c)), x = norm(cross(up, z)), y = cross(z, x);
  return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, e), -dot(y, e), -dot(z, e), 1];
}
function mul(a, b) {
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return o;
}

function makeDepthView(canvas, colorImg, depthImg, meta, backImg) {
  let genCount = 0, genVao = null;
  const gl = canvas.getContext('webgl2', { antialias: true, preserveDrawingBuffer: true });
  if (!gl) throw new Error('WebGL2 not available');
  const sh = (type, src) => {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const link = (vs, fs) => {
    const pr = gl.createProgram();
    gl.attachShader(pr, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(pr);
    if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(pr));
    return pr;
  };
  const back = link(BACK_VS, BACK_FS), prog = link(DEPTH_VS, DEPTH_FS);
  gl.useProgram(prog);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);   // depth bytes must arrive untouched
  const tex = (unit, img, filter) => {
    const t = gl.createTexture(); gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  };
  tex(0, depthImg, gl.NEAREST);
  tex(1, colorImg, gl.LINEAR);
  if (backImg) tex(3, backImg, gl.LINEAR);
  let windDims = null;
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_3D, gl.createTexture());
  for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
    [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE]])
    gl.texParameteri(gl.TEXTURE_3D, k, v);
  const U = {};
  for (const name of ['depthTex', 'colorTex', 'viewProj', 'size', 'f', 'near', 'far', 'zscale', 'zpivot', 'pxScale', 'scan', 'scanMode', 'colorMode', 'windOn', 'windTex', 'genPass'])
    U[name] = gl.getUniformLocation(prog, name);
  gl.uniform1i(U.depthTex, 0); gl.uniform1i(U.colorTex, 1); gl.uniform1i(U.windTex, 2);
  gl.uniform2f(U.size, meta.w, meta.h);
  gl.uniform1f(U.f, meta.f); gl.uniform1f(U.near, meta.near); gl.uniform1f(U.far, meta.far);
  gl.useProgram(back);
  gl.uniform1i(gl.getUniformLocation(back, 'back'), 3);
  const uBackAlpha = gl.getUniformLocation(back, 'alpha');
  gl.useProgram(prog);
  const vfov = 2 * Math.atan(meta.h / 2 / meta.f);

  // Orbit camera around a point at the scene's middle depth; yaw = pitch = 0, dist = pivot is
  // exactly the original camera.
  const emptyVao = gl.createVertexArray();
  return {
    vfov,
    // gen.bin from tools/gen_fill.py: 12-byte records (int16 x8, int16 y8, uint16 q, rgba8, pad)
    setGenerated(buf) {
      genVao = gl.createVertexArray(); gl.bindVertexArray(genVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ARRAY_BUFFER, buf, gl.STATIC_DRAW);
      const at = (name, n, type, norm, off) => {
        const loc = gl.getAttribLocation(prog, name);
        if (loc < 0) return;
        gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, n, type, norm, 12, off);
      };
      at('aPos', 2, gl.SHORT, false, 0); at('aQ', 1, gl.UNSIGNED_SHORT, true, 4); at('aCol', 4, gl.UNSIGNED_BYTE, true, 6);
      gl.bindVertexArray(null);
      genCount = buf.byteLength / 12;
    },
    get genCount() { return genCount; },
    draw({ yaw, pitch, dist, zoom, zscale, zpivot, scan, scanMode, colorMode, wind, generated }) {
      const W = canvas.width, H = canvas.height;
      gl.viewport(0, 0, W, H);
      gl.clear(gl.DEPTH_BUFFER_BIT);
      // Backdrop: a blurred photo, only near the original viewpoint, so gaps a moving berry
      // uncovers show soft background instead of black.
      gl.disable(gl.DEPTH_TEST);
      gl.useProgram(back);
      const off = Math.abs(yaw) + Math.abs(pitch) + Math.abs(zoom - 1) * 0.3 + Math.abs(dist - zpivot);
      gl.uniform1f(uBackAlpha, backImg && wind ? Math.max(0, 1 - off / 0.04) : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.enable(gl.DEPTH_TEST);
      gl.useProgram(prog);
      if (wind) {
        gl.activeTexture(gl.TEXTURE2);
        const { gx, gy, gz, field } = wind;
        if (!windDims) { gl.texImage3D(gl.TEXTURE_3D, 0, gl.RG16F, gx, gy, gz, 0, gl.RG, gl.FLOAT, field); windDims = 1; }
        else gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, 0, gx, gy, gz, gl.RG, gl.FLOAT, field);
      }
      gl.uniform1f(U.windOn, wind ? 1 : 0);
      const fovy = vfov / zoom, target = [0, 0, -zpivot];
      const eye = [
        target[0] + dist * Math.sin(yaw) * Math.cos(pitch),
        target[1] + dist * Math.sin(pitch),
        target[2] + dist * Math.cos(yaw) * Math.cos(pitch),
      ];
      const proj = perspective(fovy, W / H, 0.05, 100);
      gl.uniformMatrix4fv(U.viewProj, false, mul(proj, lookAt(eye, target, [0, 1, 0])));
      gl.uniform1f(U.pxScale, H / 2 / Math.tan(fovy / 2));
      gl.uniform1f(U.zscale, zscale); gl.uniform1f(U.zpivot, zpivot);
      gl.uniform1f(U.scan, scan); gl.uniform1f(U.scanMode, scanMode); gl.uniform1f(U.colorMode, colorMode);
      gl.bindVertexArray(emptyVao);
      gl.uniform1f(U.genPass, 0);
      gl.drawArrays(gl.POINTS, 0, meta.w * meta.h);
      if (generated && genCount) {
        gl.bindVertexArray(genVao);
        gl.uniform1f(U.genPass, 1);
        gl.drawArrays(gl.POINTS, 0, genCount);
      }
      gl.bindVertexArray(null);
    },
  };
}

function mulberry32(a) {
  return () => {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function gaussRand(rng) { return Math.sqrt(-2 * Math.log(rng() + 1e-12)) * Math.cos(2 * Math.PI * rng()); }

// ---------- depth-aware wind ----------
// A 3D grid of damped springs: 48 x 32 across the picture, 10 layers deep (in inverse depth,
// where the berries are). Each cell sways in metres. Depth matters in three ways:
//   - gust fronts sweep across AND into the bush, so back layers are hit later;
//   - the inside and back of the bush are sheltered, so they get less wind;
//   - front twigs are light (springy, ~1 Hz), deeper branches heavy (slow, ~0.5 Hz).
// Far layers also move fewer pixels for the same sway, simply by perspective.
class DepthWind {
  constructor(meta, seed = 5) {
    this.gx = 48; this.gy = 32; this.gz = 10;
    const n = this.gx * this.gy * this.gz;
    this.n = n;
    this.u = new Float32Array(2 * n); this.v = new Float32Array(2 * n);
    this.field = this.u;                             // the texture is the spring state itself
    this.omega = new Float32Array(n); this.shelter = new Float32Array(n);
    this.xs = new Float32Array(n); this.ys = new Float32Array(n); this.zs = new Float32Array(n);
    for (let k = 0; k < this.gz; k++) for (let j = 0; j < this.gy; j++) for (let i = 0; i < this.gx; i++) {
      const c = (k * this.gy + j) * this.gx + i, s = (k + 0.5) / this.gz;
      this.xs[c] = (i + 0.5) / this.gx; this.ys[c] = (j + 0.5) / this.gy; this.zs[c] = s;
      this.omega[c] = 2 * Math.PI * (1.0 - 0.5 * s);
      this.shelter[c] = 1 - 0.55 * s;
    }
    this.rng = mulberry32(seed);
    this.t = 0; this.env = 0; this.fronts = []; this.wait = 0.3;
    this.eddies = Array.from({ length: 5 }, () => ({
      kx: 1.5 + 3 * this.rng(), ky: 1 + 2.5 * this.rng(), kz: 1 + 3 * this.rng(),
      ph: 6.28 * this.rng(), a: 0.12 + 0.12 * this.rng(),
    }));
  }
  step(dt, wind) {
    const rng = this.rng, U = 0.22 + 0.3 * wind, A = 0.0018 * wind;   // A: sway in metres
    const a = Math.exp(-dt / 5);
    this.env = this.env * a + Math.sqrt(1 - a * a) * gaussRand(rng);
    if ((this.wait -= dt) <= 0) {
      this.fronts.push({ c: -0.5, amp: (0.7 + 0.9 * rng()) * (0.4 + wind), w: 0.2 + 0.15 * rng() });
      this.wait = 2 + 4 * rng();
    }
    for (const fr of this.fronts) fr.c += U * dt;
    this.fronts = this.fronts.filter(fr => fr.c < 2.2);
    this.t += dt;
    const sub = Math.max(1, Math.ceil(dt * 120)), h = dt / sub, z = 0.18;
    for (let c = 0; c < this.n; c++) {
      const x = this.xs[c], y = this.ys[c], s = this.zs[c];
      let p = 0.5 + 0.2 * this.env;
      for (const fr of this.fronts) p += fr.amp * Math.exp(-(((x + 0.6 * s - fr.c) / fr.w) ** 2));
      for (const e of this.eddies) p += e.a * Math.sin(6.28 * (e.kx * (x - U * this.t) + e.ky * y + e.kz * s) + e.ph);
      p = Math.max(0, p) * A * this.shelter[c];
      const fx = p, fy = -0.25 * p;                  // downwind, and a little droop
      const w = this.omega[c], w2 = w * w, d = 2 * z * w;
      let ux = this.u[2 * c], uy = this.u[2 * c + 1], vx = this.v[2 * c], vy = this.v[2 * c + 1];
      for (let st = 0; st < sub; st++) {
        vx += h * (w2 * (fx - ux) - d * vx); vy += h * (w2 * (fy - uy) - d * vy);
        ux += h * vx; uy += h * vy;
      }
      this.u[2 * c] = ux; this.u[2 * c + 1] = uy; this.v[2 * c] = vx; this.v[2 * c + 1] = vy;
    }
  }
}

if (typeof module !== 'undefined') module.exports = { perspective, lookAt, mul, DepthWind };
