// Depth 3D: the photo as a point cloud, from a precomputed Depth Pro depth map (tools/make_depth.py).
//
// Every pixel of the 960x640 depth map is one point. The vertex shader reads its depth straight
// from the depth texture (no vertex buffers), unprojects it with the camera's real focal length,
// and colours it from the photo. Scaling depth slides points along their own camera ray, so from
// the original viewpoint the cloud always looks exactly like the photo.

const DEPTH_VS = `#version 300 es
uniform sampler2D depthTex, colorTex;
uniform mat4 viewProj;
uniform vec2 size;
uniform float f, near, far, zscale, zpivot, pxScale, scan, scanMode, colorMode;
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
  int W = int(size.x);
  ivec2 p = ivec2(gl_VertexID % W, gl_VertexID / W);
  vec4 d = texelFetch(depthTex, p, 0);
  float q = (floor(d.r * 255.0 + 0.5) * 256.0 + floor(d.g * 255.0 + 0.5)) / 65535.0;
  float z = 1.0 / mix(1.0 / far, 1.0 / near, q);
  float t = 1.0 - q;          // 0 = nearest, 1 = farthest, even in inverse depth (where the detail is)
  // Drop points on depth edges (they would float between berry and background), and in peel
  // mode everything in front of the scan plane.
  if (d.b < 0.5 || (scanMode > 1.5 && t < scan - 0.004)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

  float zs = zpivot + (z - zpivot) * zscale;
  vec2 ray = (vec2(p) + 0.5 - 0.5 * size) / f;
  vec3 P = vec3(ray.x * zs, -ray.y * zs, -zs);
  gl_Position = viewProj * vec4(P, 1.0);
  gl_PointSize = clamp(1.5 * pxScale * zs / f / gl_Position.w, 1.0, 24.0);

  vec2 uv = (vec2(p) + 0.5) / size;
  vec3 c = colorMode > 0.5 ? turbo(0.1 + 0.85 * q) : texture(colorTex, uv).rgb;
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

function makeDepthView(canvas, colorImg, depthImg, meta) {
  const gl = canvas.getContext('webgl2', { antialias: true, preserveDrawingBuffer: true });
  if (!gl) throw new Error('WebGL2 not available');
  const sh = (type, src) => {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, DEPTH_VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, DEPTH_FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
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
  const U = {};
  for (const name of ['depthTex', 'colorTex', 'viewProj', 'size', 'f', 'near', 'far', 'zscale', 'zpivot', 'pxScale', 'scan', 'scanMode', 'colorMode'])
    U[name] = gl.getUniformLocation(prog, name);
  gl.uniform1i(U.depthTex, 0); gl.uniform1i(U.colorTex, 1);
  gl.uniform2f(U.size, meta.w, meta.h);
  gl.uniform1f(U.f, meta.f); gl.uniform1f(U.near, meta.near); gl.uniform1f(U.far, meta.far);
  gl.enable(gl.DEPTH_TEST);
  const vfov = 2 * Math.atan(meta.h / 2 / meta.f);

  // Orbit camera around a point at the scene's middle depth; yaw = pitch = 0, dist = pivot is
  // exactly the original camera.
  return {
    vfov,
    draw({ yaw, pitch, dist, zoom, zscale, zpivot, scan, scanMode, colorMode }) {
      const W = canvas.width, H = canvas.height;
      gl.viewport(0, 0, W, H);
      gl.clearColor(0.03, 0.035, 0.045, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
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
      gl.drawArrays(gl.POINTS, 0, meta.w * meta.h);
    },
  };
}

if (typeof module !== 'undefined') module.exports = { perspective, lookAt, mul };
