struct Particle {
  position: vec4f,
  velocity: vec4f,
  c0: vec4f,
  c1: vec4f,
  c2: vec4f,
};

struct Cell {
  vx: i32,
  vy: i32,
  vz: i32,
  mass: i32,
};

struct SimUniforms {
  grid_size: vec4f,
  real_box_size: vec4f,
  material_params: vec4f,
  boundary_params: vec4f,
  collision_params: vec4f,
};

struct TerrainBoundaryCell {
  surface: vec4f,
  material: vec4f,
};

struct BuildingSideBoundaryCell {
  wall: vec4f,
  material: vec4f,
};

struct TerrainSample {
  height: f32,
  normal: vec3f,
  mode: f32,
};

struct BuildingSideSample {
  signed_distance: f32,
  base: f32,
  top: f32,
  mode: f32,
  normal: vec3f,
};

struct PositionVelocityCorrection {
  position: vec3f,
  velocity: vec3f,
};

override fixed_point_multiplier: f32;
// dt は runtime 可変なので uniforms 経由 (sim.boundary_params.z)

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
@group(0) @binding(2) var<uniform> sim: SimUniforms;
@group(0) @binding(3) var<storage, read> terrain_boundaries: array<TerrainBoundaryCell>;
@group(0) @binding(4) var<storage, read> building_side_boundaries: array<BuildingSideBoundaryCell>;

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

fn cellVelocity(cell: vec3i) -> vec3f {
  if (!isCellInside(cell)) {
    return vec3f(0.0);
  }
  let c = cells[cellIndex(cell)];
  return vec3f(
    decodeFixedPoint(c.vx),
    decodeFixedPoint(c.vy),
    decodeFixedPoint(c.vz)
  );
}

fn terrainIndex(x: i32, z: i32) -> u32 {
  let depth = i32(sim.material_params.w);
  return u32(x * depth + z);
}

fn readTerrainCell(x: i32, z: i32) -> TerrainBoundaryCell {
  let width = i32(sim.material_params.z);
  let depth = i32(sim.material_params.w);
  let safe_x = clamp(x, 0, width - 1);
  let safe_z = clamp(z, 0, depth - 1);
  return terrain_boundaries[terrainIndex(safe_x, safe_z)];
}

fn readBuildingSideCell(x: i32, z: i32) -> BuildingSideBoundaryCell {
  let width = i32(sim.material_params.z);
  let depth = i32(sim.material_params.w);
  let safe_x = clamp(x, 0, width - 1);
  let safe_z = clamp(z, 0, depth - 1);
  return building_side_boundaries[terrainIndex(safe_x, safe_z)];
}

fn dominantSurfaceMaterial(
  c00: TerrainBoundaryCell,
  c10: TerrainBoundaryCell,
  c01: TerrainBoundaryCell,
  c11: TerrainBoundaryCell
) -> vec4f {
  // height は滑らかに補間し、接触面の有効/無効 mode は最も高い surface のセルから取る。
  // G2P 側では material response を使わず、接触面の幾何だけを読む。
  var height = c00.surface.x;
  var material = c00.material;
  if (c10.surface.x > height) {
    height = c10.surface.x;
    material = c10.material;
  }
  if (c01.surface.x > height) {
    height = c01.surface.x;
    material = c01.material;
  }
  if (c11.surface.x > height) {
    material = c11.material;
  }
  return material;
}

fn sampleTerrain(pos: vec3f) -> TerrainSample {
  let width = i32(sim.material_params.z);
  let depth = i32(sim.material_params.w);
  if (sim.boundary_params.x < 0.5 || width <= 1 || depth <= 1) {
    return TerrainSample(-1000000.0, vec3f(0.0, 1.0, 0.0), 0.0);
  }

  // Particle rendering maps grid coordinate to ENU with x / gridSize.x.
  // Terrain contact must use the same denominator; using gridSize.x - 1
  // shifts the sampled DEM by hundreds of meters for wide domains.
  let gx = clamp(
    pos.x / max(1.0, sim.real_box_size.x) * f32(width - 1),
    0.0,
    f32(width - 1)
  );
  let gz = clamp(
    pos.z / max(1.0, sim.real_box_size.z) * f32(depth - 1),
    0.0,
    f32(depth - 1)
  );
  let x0 = i32(floor(gx));
  let z0 = i32(floor(gz));
  let x1 = min(x0 + 1, width - 1);
  let z1 = min(z0 + 1, depth - 1);
  let tx = gx - f32(x0);
  let tz = gz - f32(z0);

  let c00 = readTerrainCell(x0, z0);
  let c10 = readTerrainCell(x1, z0);
  let c01 = readTerrainCell(x0, z1);
  let c11 = readTerrainCell(x1, z1);
  let h0 = c00.surface.x + (c10.surface.x - c00.surface.x) * tx;
  let h1 = c01.surface.x + (c11.surface.x - c01.surface.x) * tx;
  let n0 = c00.surface.yzw + (c10.surface.yzw - c00.surface.yzw) * tx;
  let n1 = c01.surface.yzw + (c11.surface.yzw - c01.surface.yzw) * tx;
  let material = dominantSurfaceMaterial(c00, c10, c01, c11);
  return TerrainSample(
    h0 + (h1 - h0) * tz,
    normalize(n0 + (n1 - n0) * tz),
    material.w
  );
}

fn sampleBuildingSide(pos: vec3f) -> BuildingSideSample {
  let width = i32(sim.material_params.z);
  let depth = i32(sim.material_params.w);
  if (width <= 1 || depth <= 1) {
    return BuildingSideSample(
      1000000.0,
      0.0,
      0.0,
      0.0,
      vec3f(0.0, 0.0, 0.0)
    );
  }

  // side-wall は signed distance と水平法線を持つ nearest sample として読む。
  // heightfield と違い、空セルを bilinear で混ぜると法線が消えるため補間しない。
  let gx = clamp(
    pos.x / max(1.0, sim.real_box_size.x) * f32(width - 1),
    0.0,
    f32(width - 1)
  );
  let gz = clamp(
    pos.z / max(1.0, sim.real_box_size.z) * f32(depth - 1),
    0.0,
    f32(depth - 1)
  );
  let sample_x = clamp(i32(floor(gx + 0.5)), 0, width - 1);
  let sample_z = clamp(i32(floor(gz + 0.5)), 0, depth - 1);
  let cell = readBuildingSideCell(sample_x, sample_z);
  let normal_xz = cell.wall.yz;
  let normal_len = length(normal_xz);
  if (cell.material.w < 0.5 || normal_len <= 0.000001) {
    return BuildingSideSample(
      1000000.0,
      0.0,
      0.0,
      0.0,
      vec3f(0.0, 0.0, 0.0)
    );
  }

  let normal = vec3f(normal_xz.x / normal_len, 0.0, normal_xz.y / normal_len);
  let sample_pos = vec3f(
    f32(sample_x) / max(1.0, f32(width - 1)) * sim.real_box_size.x,
    pos.y,
    f32(sample_z) / max(1.0, f32(depth - 1)) * sim.real_box_size.z
  );

  return BuildingSideSample(
    cell.wall.x + dot(pos - sample_pos, normal),
    cell.material.z,
    cell.wall.w,
    cell.material.w,
    normal
  );
}

fn removePenetratingVelocity(velocity: vec3f, normal: vec3f) -> vec3f {
  let vn = dot(velocity, normal);
  if (vn >= 0.0) {
    return velocity;
  }

  // G2P 側は material response を掛けない。
  // grid 側で拘束しきれず境界内へ向いた法線速度だけを消して、二重反発を避ける。
  return velocity - vn * normal;
}

fn applyDomainClampCorrection(
  previous_position: vec3f,
  position: vec3f,
  velocity: vec3f,
  lower_bound: vec3f,
  upper_bound: vec3f
) -> PositionVelocityCorrection {
  var correction = PositionVelocityCorrection(position, velocity);

  // domain は軸に揃った固定平面 6 枚なので、建物側面のような field 再サンプルは不要。
  // 前位置と次位置の signed distance だけで crossing 時刻を近似できる。
  correction = applyBoundaryCrossingCorrection(
    previous_position,
    correction.position,
    correction.velocity,
    vec3f(1.0, 0.0, 0.0),
    previous_position.x - lower_bound.x,
    correction.position.x - lower_bound.x,
    0.0,
    1000000.0
  );
  correction = applyBoundaryCrossingCorrection(
    previous_position,
    correction.position,
    correction.velocity,
    vec3f(-1.0, 0.0, 0.0),
    upper_bound.x - previous_position.x,
    upper_bound.x - correction.position.x,
    0.0,
    1000000.0
  );
  correction = applyBoundaryCrossingCorrection(
    previous_position,
    correction.position,
    correction.velocity,
    vec3f(0.0, 1.0, 0.0),
    previous_position.y - lower_bound.y,
    correction.position.y - lower_bound.y,
    0.0,
    1000000.0
  );
  correction = applyBoundaryCrossingCorrection(
    previous_position,
    correction.position,
    correction.velocity,
    vec3f(0.0, -1.0, 0.0),
    upper_bound.y - previous_position.y,
    upper_bound.y - correction.position.y,
    0.0,
    1000000.0
  );
  correction = applyBoundaryCrossingCorrection(
    previous_position,
    correction.position,
    correction.velocity,
    vec3f(0.0, 0.0, 1.0),
    previous_position.z - lower_bound.z,
    correction.position.z - lower_bound.z,
    0.0,
    1000000.0
  );
  correction = applyBoundaryCrossingCorrection(
    previous_position,
    correction.position,
    correction.velocity,
    vec3f(0.0, 0.0, -1.0),
    upper_bound.z - previous_position.z,
    upper_bound.z - correction.position.z,
    0.0,
    1000000.0
  );

  return PositionVelocityCorrection(clamp(correction.position, lower_bound, upper_bound), correction.velocity);
}

fn applyBoundaryCrossingCorrection(
  previous_position: vec3f,
  position: vec3f,
  velocity: vec3f,
  normal: vec3f,
  previous_distance: f32,
  current_distance: f32,
  clearance: f32,
  recovery_limit: f32
) -> PositionVelocityCorrection {
  if (current_distance >= clearance) {
    return PositionVelocityCorrection(position, velocity);
  }

  let corrected_velocity = removePenetratingVelocity(velocity, normal);

  if (previous_distance > clearance) {
    // signed distance が clearance を跨いだなら、その substep 内で境界面を横切ったと見なす。
    // G2P では反発・摩擦を掛けず、衝突時刻から非侵入速度で残り時間だけ進める。
    let denom = max(0.000001, previous_distance - current_distance);
    let hit_t = clamp((previous_distance - clearance) / denom, 0.0, 1.0);
    let hit_position = previous_position + (position - previous_position) * hit_t;
    let remaining_dt = sim.boundary_params.z * (1.0 - hit_t);
    return PositionVelocityCorrection(hit_position + corrected_velocity * remaining_dt, corrected_velocity);
  }

  // 既に内側から始まった粒子には明確な衝突時刻がない。
  // 大きな瞬間移動を避けるため、境界ごとの上限内で外側へ戻す。
  let recovery = min(clearance - current_distance, recovery_limit);
  return PositionVelocityCorrection(position + normal * recovery, corrected_velocity);
}

fn sideSampleAppliesAtY(side: BuildingSideSample, y: f32) -> bool {
  let vertical_slack = 0.25;
  return side.mode >= 0.5 && y >= side.base - vertical_slack && y <= side.top + vertical_slack;
}

fn terrainSignedDistance(position: vec3f, terrain: TerrainSample) -> f32 {
  let terrain_floor = terrain.height + sim.real_box_size.w;
  let surface_position = vec3f(position.x, terrain_floor, position.z);
  return dot(position - surface_position, terrain.normal);
}

fn applyTerrainCorrection(
  previous_position: vec3f,
  position: vec3f,
  velocity: vec3f
) -> PositionVelocityCorrection {
  let terrain = sampleTerrain(position);
  if (terrain.mode <= 0.5) {
    return PositionVelocityCorrection(position, velocity);
  }

  let current_distance = terrainSignedDistance(position, terrain);
  if (current_distance >= 0.0) {
    return PositionVelocityCorrection(position, velocity);
  }

  let previous_distance = current_distance + dot(previous_position - position, terrain.normal);
  let crossed_surface = previous_distance > 0.0;
  var correction = applyBoundaryCrossingCorrection(
    previous_position,
    position,
    velocity,
    terrain.normal,
    previous_distance,
    current_distance,
    0.0,
    0.25
  );

  if (crossed_surface) {
    // 地形は曲面を tangent plane で近似しているため、残り dt で進めた候補を再サンプルする。
    // まだ地形内なら、side-wall の最終押し戻しと同じ安全弁で外側へ出す。
    let final_terrain = sampleTerrain(correction.position);
    if (final_terrain.mode > 0.5) {
      let final_distance = terrainSignedDistance(correction.position, final_terrain);
      if (final_distance < 0.0) {
        correction = PositionVelocityCorrection(
          correction.position + final_terrain.normal * -final_distance,
          correction.velocity
        );
      }
    }
  }

  return correction;
}

fn applyBuildingSideCorrection(
  previous_position: vec3f,
  position: vec3f,
  velocity: vec3f
) -> PositionVelocityCorrection {
  let side = sampleBuildingSide(position);
  if (!sideSampleAppliesAtY(side, position.y)) {
    return PositionVelocityCorrection(position, velocity);
  }

  let clearance = max(0.05, sim.real_box_size.w);
  if (side.signed_distance >= clearance) {
    return PositionVelocityCorrection(position, velocity);
  }

  let previous_distance = side.signed_distance + dot(previous_position - position, side.normal);
  var correction = applyBoundaryCrossingCorrection(
    previous_position,
    position,
    velocity,
    side.normal,
    previous_distance,
    side.signed_distance,
    clearance,
    0.25
  );

  if (previous_distance > clearance) {
    let final_side = sampleBuildingSide(correction.position);
    if (sideSampleAppliesAtY(final_side, correction.position.y) && final_side.signed_distance < clearance) {
      correction = PositionVelocityCorrection(
        correction.position + final_side.normal * (clearance - final_side.signed_distance),
        correction.velocity
      );
    }
  }

  return correction;
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

  var new_velocity = vec3f(0.0);
  var b0 = vec3f(0.0);
  var b1 = vec3f(0.0);
  var b2 = vec3f(0.0);

  for (var gx = 0u; gx < 3u; gx++) {
    for (var gy = 0u; gy < 3u; gy++) {
      for (var gz = 0u; gz < 3u; gz++) {
        let cell = cell_idx + vec3i(i32(gx) - 1, i32(gy) - 1, i32(gz) - 1);
        let weight = weights[gx].x * weights[gy].y * weights[gz].z;
        let dist = (vec3f(cell) + vec3f(0.5)) - particle.position.xyz;
        let weighted_velocity = cellVelocity(cell) * weight;

        new_velocity += weighted_velocity;
        b0 += weighted_velocity * dist.x;
        b1 += weighted_velocity * dist.y;
        b2 += weighted_velocity * dist.z;
      }
    }
  }

  var corrected_velocity = new_velocity;
  particle.c0 = vec4f(b0 * 4.0, 0.0);
  particle.c1 = vec4f(b1 * 4.0, 0.0);
  particle.c2 = vec4f(b2 * 4.0, 0.0);

  // 粒子側は数値的な安全弁として箱内 clamp する。
  // x/z は壁セルへ kernel が届く 1 cell 内側を維持し、y は originY=minTerrainY の
  // 表示領域に合わせて地形 clearance まで許す。底面近くでは P2G/G2P 側で
  // `isCellInside()` を見るため、近傍セルが範囲外になっても安全に skip できる。
  let lower_bound = vec3f(1.001, sim.real_box_size.w, 1.001);
  let upper_bound = vec3f(
    sim.real_box_size.x - 1.001,
    sim.real_box_size.y - sim.real_box_size.w,
    sim.real_box_size.z - 1.001
  );
  let unclamped_next_position =
    particle.position.xyz + corrected_velocity * sim.boundary_params.z;
  var domain_correction = applyDomainClampCorrection(
    particle.position.xyz,
    unclamped_next_position,
    corrected_velocity,
    lower_bound,
    upper_bound
  );
  var next_position = domain_correction.position;
  corrected_velocity = domain_correction.velocity;

  // 地形も domain / side-wall と同じ crossing ベースの位置復帰で扱う。
  // DEM heightfield は現在位置の tangent plane として signed distance を作り、
  // 復帰後の候補がまだ地形内なら再サンプルして押し戻す。
  let terrain_correction = applyTerrainCorrection(
    particle.position.xyz,
    next_position,
    corrected_velocity
  );
  next_position = terrain_correction.position;
  corrected_velocity = terrain_correction.velocity;

  // 建物側面は DEM 床とは別の horizontal side-wall として処理する。
  // G2P では摩擦・反発を掛けず、前位置からの crossing を見て位置復帰だけ行う。
  let before_side_position = next_position;
  let side_correction = applyBuildingSideCorrection(
    particle.position.xyz,
    next_position,
    corrected_velocity
  );
  corrected_velocity = side_correction.velocity;
  domain_correction = applyDomainClampCorrection(
    before_side_position,
    side_correction.position,
    corrected_velocity,
    lower_bound,
    upper_bound
  );
  next_position = domain_correction.position;
  corrected_velocity = domain_correction.velocity;

  // velocity.w は描画コンター用の密度比なので、速度 xyz の更新でも維持する。
  particle.velocity = vec4f(corrected_velocity, particle.velocity.w);
  particle.position = vec4f(
    next_position,
    1.0
  );

  particles[id.x] = particle;
}
