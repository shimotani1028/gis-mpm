struct Particle {
  position: vec4f,
  velocity: vec4f,
  c0: vec4f,
  c1: vec4f,
  c2: vec4f,
};

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

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) attribute_value: f32,
  @location(2) valid: f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: FluidUniforms;

fn particle_attribute(particle: Particle) -> f32 {
  let mode = uniforms.contour_params.x;
  if (mode < 1.5) {
    return length(particle.velocity.xyz);
  }
  // velocity.w には P2G で計算した density / rest_density の密度比を入れている。
  return max(0.0, particle.velocity.w);
}

@vertex
fn vs(
  @builtin(vertex_index) vertex_index: u32,
  @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(0.5, 0.5),
    vec2f(0.5, -0.5),
    vec2f(-0.5, -0.5),
    vec2f(0.5, 0.5),
    vec2f(-0.5, -0.5),
    vec2f(-0.5, 0.5),
  );

  let particle = particles[instance_index];
  var out: VertexOutput;
  if (particle.position.w <= 0.5 || uniforms.contour_params.x < 0.5) {
    out.position = vec4f(2.0, 2.0, 0.0, 1.0);
    out.uv = vec2f(0.0);
    out.attribute_value = 0.0;
    out.valid = 0.0;
    return out;
  }

  let radius = uniforms.fluid_params.x;
  let corner = corners[vertex_index] * radius * 2.0;
  let world_position = uniforms.model_matrix * vec4f(particle.position.xyz, 1.0);
  let center_view_position = (uniforms.view_matrix * world_position).xyz;

  out.position = uniforms.projection_matrix * vec4f(
    center_view_position + vec3f(corner, 0.0),
    1.0,
  );
  out.uv = corners[vertex_index] + vec2f(0.5);
  out.attribute_value = particle_attribute(particle);
  out.valid = 1.0;
  return out;
}

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4f {
  if (input.valid < 0.5) {
    discard;
  }

  let normal_xy = input.uv * 2.0 - vec2f(1.0);
  let r2 = dot(normal_xy, normal_xy);
  if (r2 > 1.0) {
    discard;
  }

  // thickness と同じ重みを掛けておく。composite では blur 済み thickness で割り、
  // 画面 pixel 内に重なった粒子の加重平均値へ戻す。
  let sphere_thickness = sqrt(1.0 - r2);
  let particle_thickness = uniforms.fluid_params.z * sphere_thickness;
  return vec4f(input.attribute_value * particle_thickness, 0.0, 0.0, 1.0);
}
