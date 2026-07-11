// 2D image resize (single-channel f32), nearest or bilinear.
//
// Source-pixel mapping uses the standard half-pixel-center convention
// (align_corners = false): dst pixel (dx, dy) samples source coordinate
// ((dx + 0.5) * src/dst - 0.5, ...). The CPU reference implementation in
// lib.rs MUST use the identical formula so GPU/CPU parity tests can hold
// a tight tolerance.

struct Params {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    mode: u32, // 0 = nearest, 1 = bilinear
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

fn sample_nearest(x: f32, y: f32) -> f32 {
    let sx = u32(clamp(round(x), 0.0, f32(params.src_w) - 1.0));
    let sy = u32(clamp(round(y), 0.0, f32(params.src_h) - 1.0));
    return src[sy * params.src_w + sx];
}

fn sample_bilinear(x: f32, y: f32) -> f32 {
    let x0f = clamp(floor(x), 0.0, f32(params.src_w) - 1.0);
    let y0f = clamp(floor(y), 0.0, f32(params.src_h) - 1.0);
    let x1f = min(x0f + 1.0, f32(params.src_w) - 1.0);
    let y1f = min(y0f + 1.0, f32(params.src_h) - 1.0);
    let fx = clamp(x - x0f, 0.0, 1.0);
    let fy = clamp(y - y0f, 0.0, 1.0);
    let ix0 = u32(x0f);
    let iy0 = u32(y0f);
    let ix1 = u32(x1f);
    let iy1 = u32(y1f);
    let v00 = src[iy0 * params.src_w + ix0];
    let v10 = src[iy0 * params.src_w + ix1];
    let v01 = src[iy1 * params.src_w + ix0];
    let v11 = src[iy1 * params.src_w + ix1];
    let top = mix(v00, v10, fx);
    let bot = mix(v01, v11, fx);
    return mix(top, bot, fy);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dx = gid.x;
    let dy = gid.y;
    if (dx >= params.dst_w || dy >= params.dst_h) {
        return;
    }
    let scale_x = f32(params.src_w) / f32(params.dst_w);
    let scale_y = f32(params.src_h) / f32(params.dst_h);
    let sx = max((f32(dx) + 0.5) * scale_x - 0.5, 0.0);
    let sy = max((f32(dy) + 0.5) * scale_y - 0.5, 0.0);

    var v: f32;
    if (params.mode == 0u) {
        v = sample_nearest(sx, sy);
    } else {
        v = sample_bilinear(sx, sy);
    }
    dst[dy * params.dst_w + dx] = v;
}
