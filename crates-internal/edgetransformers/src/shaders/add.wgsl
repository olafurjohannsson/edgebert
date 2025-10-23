/// A uniform struct to pass metadata about the tensor size.
/// This prevents the shader from accessing memory out of bounds.
struct AddUniforms {
    size: u32,
};

/// @group(0) @binding(0): Uniforms containing metadata.
@group(0) @binding(0) var<uniform> uniforms: AddUniforms;

/// @group(0) @binding(1): Read-only input tensor 'a'.
@group(0) @binding(1) var<storage, read> a: array<f32>;

/// @group(0) @binding(2): Read-only input tensor 'b'.
@group(0) @binding(2) var<storage, read> b: array<f32>;

/// @group(0) @binding(3): Writable output tensor.
@group(0) @binding(3) var<storage, read_write> output: array<f32>;


@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Boundary check to ensure memory safety.
    if (idx >= uniforms.size) {
        return;
    }

    output[idx] = a[idx] + b[idx];
}