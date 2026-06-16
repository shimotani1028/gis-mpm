struct Particle {
  position: vec4f,
  velocity: vec4f,
  c0: vec4f,
  c1: vec4f,
  c2: vec4f,
};

struct Cell {
  vx: atomic<i32>,
  vy: atomic<i32>,
  vz: atomic<i32>,
  mass: i32,
};

struct SimUniforms {
  grid_size: vec4f,
  real_box_size: vec4f,
  material_params: vec4f,
  boundary_params: vec4f,
};

override fixed_point_multiplier: f32;
override stiffness: f32;
override rest_density: f32;
// dt は runtime 可変なので uniforms 経由 (sim.boundary_params.z)

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(2) var<uniform> sim: SimUniforms;

fn encodeFixedPoint(value: f32) -> i32 {
  let limit = 2147483000.0;
  return i32(clamp(value * fixed_point_multiplier, -limit, limit));
}

fn decodeFixedPoint(value: i32) -> f32 {
  return f32(value) / fixed_point_multiplier;
}

fn gridSizeI() -> vec3i {
  return vec3i(i32(sim.grid_size.x), i32(sim.grid_size.y), i32(sim.grid_size.z));
}

fn cellIndex(cell: vec3i) -> u32 {
  let grid = gridSizeI();
  return u32(cell.x * grid.y * grid.z + cell.y * grid.z + cell.z);
}

fn isCellInside(cell: vec3i) -> bool {
  let grid = gridSizeI();
  return cell.x >= 0
    && cell.y >= 0
    && cell.z >= 0
    && cell.x < grid.x
    && cell.y < grid.y
    && cell.z < grid.z;
}

fn quadraticWeights(cell_diff: vec3f) -> array<vec3f, 3> {
  var weights: array<vec3f, 3>;
  let d0 = vec3f(0.5) - cell_diff;
  let d1 = cell_diff;
  let d2 = vec3f(0.5) + cell_diff;
  weights[0] = 0.5 * d0 * d0;
  weights[1] = vec3f(0.75) - d1 * d1;
  weights[2] = 0.5 * d2 * d2;
  return weights;
}

fn affineMatrix(particle: Particle) -> mat3x3f {
  return mat3x3f(particle.c0.xyz, particle.c1.xyz, particle.c2.xyz);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&particles)) {
    return;
  }

  var particle = particles[id.x];
  if (particle.position.w <= 0.5) {
    return;
  }

  let cell_idx = vec3i(floor(particle.position.xyz));
  let cell_diff = particle.position.xyz - (vec3f(cell_idx) + vec3f(0.5));
  let weights = quadraticWeights(cell_diff);

  // 1 回目の P2G で作った grid mass を粒子側へ gather して、毎 substep の体積を推定する。
  var density = 0.0;
  for (var gx = 0u; gx < 3u; gx++) {
    for (var gy = 0u; gy < 3u; gy++) {
      for (var gz = 0u; gz < 3u; gz++) {
        let cell = cell_idx + vec3i(i32(gx) - 1, i32(gy) - 1, i32(gz) - 1);
        if (!isCellInside(cell)) {
          continue;
        }
        let weight = weights[gx].x * weights[gy].y * weights[gz].z;
        density += decodeFixedPoint(cells[cellIndex(cell)].mass) * weight;
      }
    }
  }

  let safe_density = max(density, 0.0001);
  // velocity.w は物理計算には使わず、描画コンター用の最新密度比として保持する。
  particle.velocity.w = safe_density / max(rest_density, 0.0001);
  particles[id.x] = particle;
  let volume = sim.boundary_params.w / safe_density;

  // 粘性流体モデル: stress = -pressure * I + viscosity * (grad_v + grad_v^T)
  let pressure = max(0.0, stiffness * (pow(safe_density / rest_density, 5.0) - 1.0));
  let c = affineMatrix(particle);
  var stress = mat3x3f(
    vec3f(-pressure, 0.0, 0.0),
    vec3f(0.0, -pressure, 0.0),
    vec3f(0.0, 0.0, -pressure)
  );
  stress = stress + sim.material_params.x * (c + transpose(c));

  // MLS-MPM の fused momentum/force update。quadratic kernel なので Mp^-1 = 4。
  let eq_16_term0 = -volume * 4.0 * stress * sim.boundary_params.z;

  for (var gx = 0u; gx < 3u; gx++) {
    for (var gy = 0u; gy < 3u; gy++) {
      for (var gz = 0u; gz < 3u; gz++) {
        let cell = cell_idx + vec3i(i32(gx) - 1, i32(gy) - 1, i32(gz) - 1);
        if (!isCellInside(cell)) {
          continue;
        }
        let weight = weights[gx].x * weights[gy].y * weights[gz].z;
        let cell_dist = (vec3f(cell) + vec3f(0.5)) - particle.position.xyz;
        let momentum = (eq_16_term0 * cell_dist) * weight;
        let index = cellIndex(cell);

        atomicAdd(&cells[index].vx, encodeFixedPoint(momentum.x));
        atomicAdd(&cells[index].vy, encodeFixedPoint(momentum.y));
        atomicAdd(&cells[index].vz, encodeFixedPoint(momentum.z));
      }
    }
  }
}
