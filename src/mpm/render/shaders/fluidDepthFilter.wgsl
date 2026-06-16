struct FluidUniforms {
  projection_matrix: mat4x4f,
  inv_projection_matrix: mat4x4f,
  view_matrix: mat4x4f,
  inv_view_matrix: mat4x4f,
  model_matrix: mat4x4f,
  screen_params: vec4f,
  fluid_params: vec4f,
  contour_params: vec4f,
}

struct FilterUniforms {
  direction: vec4f,
}

struct FragmentInput {
  @location(0) uv: vec2f,
}

@group(0) @binding(0) var depth_texture: texture_2d<f32>;
@group(0) @binding(1) var<uniform> filter_uniforms: FilterUniforms;
@group(0) @binding(2) var<uniform> fluid_uniforms: FluidUniforms;

fn load_depth(pixel: vec2i) -> f32 {
  let size = vec2i(textureDimensions(depth_texture));
  let clamped = clamp(pixel, vec2i(0), size - vec2i(1));
  return textureLoad(depth_texture, vec2u(clamped), 0).r;
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
  let size = vec2i(textureDimensions(depth_texture));
  let pixel = clamp(vec2i(input.uv * vec2f(size)), vec2i(0), size - vec2i(1));
  let center_depth = load_depth(pixel);
  if (center_depth >= -0.0001) {
    return vec4f(0.0);
  }

  let radius = 8;
  let sigma = f32(radius) / 3.0;
  let two_sigma = 2.0 * sigma * sigma;
  let sigma_depth = max(1.0, fluid_uniforms.fluid_params.x * 3.5);
  let two_sigma_depth = 2.0 * sigma_depth * sigma_depth;
  let dir = vec2i(filter_uniforms.direction.xy);

  var sum = 0.0;
  var weight_sum = 0.0;
  for (var i = -radius; i <= radius; i++) {
    let sample_depth = load_depth(pixel + dir * i);
    if (sample_depth >= -0.0001) {
      continue;
    }

    let spatial_weight = exp(-f32(i * i) / two_sigma);
    let depth_delta = sample_depth - center_depth;
    let depth_weight = exp(-(depth_delta * depth_delta) / two_sigma_depth);
    let weight = spatial_weight * depth_weight;
    sum += sample_depth * weight;
    weight_sum += weight;
  }

  if (weight_sum <= 0.0) {
    return vec4f(center_depth, 0.0, 0.0, 1.0);
  }
  return vec4f(sum / weight_sum, 0.0, 0.0, 1.0);
}
