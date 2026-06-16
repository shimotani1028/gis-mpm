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
  mass: atomic<i32>,
};

struct SimUniforms {
  grid_size: vec4f,
  real_box_size: vec4f,
  material_params: vec4f,
  boundary_params: vec4f,
};

override fixed_point_multiplier: f32;

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(2) var<uniform> sim: SimUniforms;

fn encodeFixedPoint(value: f32) -> i32 {
  // fixed_point_multiplier=1e7 を使うため、GPU 側の異常値で i32 範囲を越えないよう丸める。
  let limit = 2147483000.0;
  return i32(clamp(value * fixed_point_multiplier, -limit, limit));
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

fn affineVelocity(particle: Particle, cell_dist: vec3f) -> vec3f {
  // c0/c1/c2 は APIC/MLS-MPM の affine momentum matrix の列として保存する。
  return particle.c0.xyz * cell_dist.x
    + particle.c1.xyz * cell_dist.y
    + particle.c2.xyz * cell_dist.z;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&particles)) {
    return;
  }

  let particle = particles[id.x];
  if (particle.position.w <= 0.5) {
    return;
  }

  let cell_idx = vec3i(floor(particle.position.xyz));
  let cell_diff = particle.position.xyz - (vec3f(cell_idx) + vec3f(0.5));
  let weights = quadraticWeights(cell_diff);

  for (var gx = 0u; gx < 3u; gx++) {
    for (var gy = 0u; gy < 3u; gy++) {
      for (var gz = 0u; gz < 3u; gz++) {
        let cell = cell_idx + vec3i(i32(gx) - 1, i32(gy) - 1, i32(gz) - 1);
        if (!isCellInside(cell)) {
          continue;
        }
        let weight = weights[gx].x * weights[gy].y * weights[gz].z;
        let cell_dist = (vec3f(cell) + vec3f(0.5)) - particle.position.xyz;

        let mass_contrib = weight * sim.boundary_params.w;
        let momentum_contrib = mass_contrib * (particle.velocity.xyz + affineVelocity(particle, cell_dist));
        let index = cellIndex(cell);

        atomicAdd(&cells[index].mass, encodeFixedPoint(mass_contrib));
        atomicAdd(&cells[index].vx, encodeFixedPoint(momentum_contrib.x));
        atomicAdd(&cells[index].vy, encodeFixedPoint(momentum_contrib.y));
        atomicAdd(&cells[index].vz, encodeFixedPoint(momentum_contrib.z));
      }
    }
  }
}
