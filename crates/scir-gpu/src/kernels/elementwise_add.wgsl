// Elementwise vector addition: out[i] = a[i] + b[i].
// Mirrors the CUDA `add_vec_f32` PTX kernel in lib.rs (same op, parity target).

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&out)) {
        return;
    }
    out[i] = a[i] + b[i];
}
