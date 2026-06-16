struct Particle {
  position: vec4f,
  velocity: vec4f,
  c0: vec4f,
  c1: vec4f,
  c2: vec4f,
};

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> positions: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> contour_attributes: array<vec4f>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&particles)) {
    return;
  }

  // Three.js 側では vec4 stride の instanced attribute として読み、xyz だけを使う。
  // 将来 kill 境界で inactive になった粒子は十分遠くへ逃がして描画対象から外す。
  if (particles[id.x].position.w <= 0.5) {
    positions[id.x] = vec4f(0.0, -1000000.0, 0.0, 0.0);
    contour_attributes[id.x] = vec4f(0.0);
    return;
  }
  positions[id.x] = vec4f(particles[id.x].position.xyz, 1.0);
  // y 成分は内部密度そのものではなく、基準密度に対する密度比。
  contour_attributes[id.x] = vec4f(
    length(particles[id.x].velocity.xyz),
    max(0.0, particles[id.x].velocity.w),
    1.0,
    0.0
  );
}
