struct FilterUniforms {
  direction: vec4f,
}

struct FragmentInput {
  @location(0) uv: vec2f,
}

@group(0) @binding(0) var thickness_texture: texture_2d<f32>;
@group(0) @binding(1) var<uniform> filter_uniforms: FilterUniforms;

fn load_thickness(pixel: vec2i) -> f32 {
  let size = vec2i(textureDimensions(thickness_texture));
  let clamped = clamp(pixel, vec2i(0), size - vec2i(1));
  return textureLoad(thickness_texture, vec2u(clamped), 0).r;
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
  let size = vec2i(textureDimensions(thickness_texture));
  let pixel = clamp(vec2i(input.uv * vec2f(size)), vec2i(0), size - vec2i(1));
  let radius = 12;
  let sigma = f32(radius) / 3.0;
  let two_sigma = 2.0 * sigma * sigma;
  let dir = vec2i(filter_uniforms.direction.xy);

  var sum = 0.0;
  var weight_sum = 0.0;
  for (var i = -radius; i <= radius; i++) {
    let weight = exp(-f32(i * i) / two_sigma);
    sum += load_thickness(pixel + dir * i) * weight;
    weight_sum += weight;
  }

  return vec4f(sum / max(weight_sum, 1e-6), 0.0, 0.0, 1.0);
}
