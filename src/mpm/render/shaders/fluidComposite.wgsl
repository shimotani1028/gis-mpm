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

struct FragmentInput {
  @location(0) uv: vec2f,
}

@group(0) @binding(0) var depth_texture: texture_2d<f32>;
@group(0) @binding(1) var thickness_texture: texture_2d<f32>;
@group(0) @binding(2) var attribute_texture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> uniforms: FluidUniforms;

fn texture_pixel(uv: vec2f) -> vec2i {
  let size = vec2i(textureDimensions(depth_texture));
  return clamp(vec2i(uv * vec2f(size)), vec2i(0), size - vec2i(1));
}

fn load_depth(pixel: vec2i) -> f32 {
  let size = vec2i(textureDimensions(depth_texture));
  let clamped = clamp(pixel, vec2i(0), size - vec2i(1));
  return textureLoad(depth_texture, vec2u(clamped), 0).r;
}

fn load_thickness(pixel: vec2i) -> f32 {
  let size = vec2i(textureDimensions(thickness_texture));
  let clamped = clamp(pixel, vec2i(0), size - vec2i(1));
  return textureLoad(thickness_texture, vec2u(clamped), 0).r;
}

fn load_attribute(pixel: vec2i) -> f32 {
  let size = vec2i(textureDimensions(attribute_texture));
  let clamped = clamp(pixel, vec2i(0), size - vec2i(1));
  return textureLoad(attribute_texture, vec2u(clamped), 0).r;
}

fn view_position_from_depth(uv: vec2f, view_z: f32) -> vec3f {
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let p00 = uniforms.projection_matrix[0].x;
  let p11 = uniforms.projection_matrix[1].y;
  return vec3f(
    ndc.x * -view_z / p00,
    ndc.y * -view_z / p11,
    view_z,
  );
}

fn safe_view_position(uv: vec2f, pixel: vec2i, fallback_depth: f32) -> vec3f {
  var depth = load_depth(pixel);
  if (depth >= -0.0001) {
    depth = fallback_depth;
  }
  return view_position_from_depth(uv, depth);
}

fn contour_color(t: f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0);
  let blue = vec3f(0.08, 0.28, 1.00);
  let cyan = vec3f(0.00, 0.72, 1.00);
  let yellow = vec3f(1.00, 0.88, 0.35);
  let red = vec3f(1.00, 0.15, 0.10);
  if (x < 0.34) {
    return mix(blue, cyan, x / 0.34);
  }
  if (x < 0.68) {
    return mix(cyan, yellow, (x - 0.34) / 0.34);
  }
  return mix(yellow, red, (x - 0.68) / 0.32);
}

fn fluid_body_color(pixel: vec2i, thickness: f32) -> vec3f {
  if (uniforms.contour_params.x < 0.5) {
    let shallow_color = vec3f(0.04, 0.74, 1.00);
    let deep_color = vec3f(0.00, 0.26, 0.72);
    let depth_mix = clamp(1.0 - exp(-thickness * 1.2), 0.0, 1.0);
    return mix(shallow_color, deep_color, depth_mix);
  }

  let weighted_attribute = load_attribute(pixel);
  let attribute_value = weighted_attribute / max(thickness, 0.0001);
  let range_min = uniforms.contour_params.y;
  let range_max = max(range_min + 0.000001, uniforms.contour_params.z);
  let normalized = (attribute_value - range_min) / (range_max - range_min);
  return contour_color(normalized);
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
  let pixel = texture_pixel(input.uv);
  let depth = load_depth(pixel);
  if (depth >= -0.0001) {
    discard;
  }

  let thickness = load_thickness(pixel);
  if (thickness <= 0.0001) {
    discard;
  }

  let texel = uniforms.screen_params.zw;
  let view_position = view_position_from_depth(input.uv, depth);
  var dx = safe_view_position(input.uv + vec2f(texel.x, 0.0), pixel + vec2i(1, 0), depth) - view_position;
  var dy = safe_view_position(input.uv + vec2f(0.0, texel.y), pixel + vec2i(0, 1), depth) - view_position;
  let dx2 = view_position - safe_view_position(input.uv - vec2f(texel.x, 0.0), pixel - vec2i(1, 0), depth);
  let dy2 = view_position - safe_view_position(input.uv - vec2f(0.0, texel.y), pixel - vec2i(0, 1), depth);

  if (abs(dx.z) > abs(dx2.z)) {
    dx = dx2;
  }
  if (abs(dy.z) > abs(dy2.z)) {
    dy = dy2;
  }

  let normal = -normalize(cross(dx, dy));
  let ray_dir = normalize(view_position);
  let light_dir = normalize((uniforms.view_matrix * vec4f(-0.35, 0.82, 0.45, 0.0)).xyz);
  let half_dir = normalize(light_dir - ray_dir);

  let diffuse = max(0.0, dot(light_dir, normal));
  let specular = pow(max(0.0, dot(half_dir, normal)), 180.0);
  let fresnel = clamp(0.02 + 0.98 * pow(1.0 - max(0.0, dot(normal, -ray_dir)), 5.0), 0.0, 1.0);

  let reflection_dir = reflect(ray_dir, normal);
  let reflection_world = normalize((uniforms.inv_view_matrix * vec4f(reflection_dir, 0.0)).xyz);
  let sky_mix = clamp(reflection_world.y * 0.5 + 0.5, 0.0, 1.0);
  let sky_color = mix(vec3f(0.48, 0.70, 0.96), vec3f(0.90, 0.98, 1.0), sky_mix);

  let body_color = fluid_body_color(pixel, thickness);
  let lit_color = body_color * (0.48 + 0.34 * diffuse);
  let reflected_color = sky_color * (0.35 + 0.65 * fresnel);
  let color = lit_color + reflected_color * fresnel * 0.45 + vec3f(specular * 1.45);

  let max_alpha = uniforms.fluid_params.y;
  let coverage_alpha = 1.0 - exp(-thickness * 3.0);
  let visibility_floor = smoothstep(0.0001, 0.028, thickness) * 0.22;
  let alpha = clamp(max_alpha * max(visibility_floor, coverage_alpha), 0.0, max_alpha);

  // Overlay canvas は premultiplied alpha で CSS 合成されるため、RGB も alpha 乗算して返す。
  return vec4f(color * alpha, alpha);
}
