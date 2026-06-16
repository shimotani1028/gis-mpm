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
  @location(1) center_view_position: vec3f,
  @location(2) valid: f32,
}

struct FragmentOutput {
  @location(0) color: vec4f,
  @builtin(frag_depth) depth: f32,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: FluidUniforms;

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
  if (particle.position.w <= 0.5) {
    out.position = vec4f(2.0, 2.0, 0.0, 1.0);
    out.uv = vec2f(0.0);
    out.center_view_position = vec3f(0.0);
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
  out.center_view_position = center_view_position;
  out.valid = 1.0;
  return out;
}

@fragment
fn fs(input: VertexOutput) -> FragmentOutput {
  if (input.valid < 0.5) {
    discard;
  }

  let normal_xy = input.uv * 2.0 - vec2f(1.0);
  let r2 = dot(normal_xy, normal_xy);
  if (r2 > 1.0) {
    discard;
  }

  let radius = uniforms.fluid_params.x;
  let normal_z = sqrt(1.0 - r2);
  let view_position = vec4f(
    input.center_view_position + vec3f(normal_xy * radius, normal_z * radius),
    1.0,
  );
  let clip_position = uniforms.projection_matrix * view_position;

  var out: FragmentOutput;
  out.depth = clip_position.z / clip_position.w;
  out.color = vec4f(view_position.z, 0.0, 0.0, 1.0);
  return out;
}
