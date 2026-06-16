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
};

struct BoundaryPlane {
  normal_offset: vec4f,
  material: vec4f,
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
  friction: f32,
  restitution: f32,
  mode: f32,
};

struct BuildingSideSample {
  signed_distance: f32,
  base: f32,
  top: f32,
  mode: f32,
  normal: vec3f,
  friction: f32,
  restitution: f32,
};

override fixed_point_multiplier: f32;
override gravity_y: f32;
// dt は runtime 可変なので uniforms 経由 (sim.boundary_params.z)

@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(1) var<uniform> sim: SimUniforms;
@group(0) @binding(2) var<storage, read> domain_boundaries: array<BoundaryPlane>;
@group(0) @binding(3) var<storage, read> terrain_boundaries: array<TerrainBoundaryCell>;
@group(0) @binding(4) var<storage, read> building_side_boundaries: array<BuildingSideBoundaryCell>;

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

fn cellCoord(index: u32) -> vec3i {
  let grid = gridSizeI();
  let i = i32(index);
  let x = i / (grid.y * grid.z);
  let y = (i / grid.z) % grid.y;
  let z = i % grid.z;
  return vec3i(x, y, z);
}

fn applyBoundaryVelocity(
  velocity: vec3f,
  normal: vec3f,
  friction: f32,
  restitution: f32
) -> vec3f {
  let vn = dot(velocity, normal);
  if (vn >= 0.0) {
    return velocity;
  }

  let safe_friction = max(0.0, friction);
  let safe_restitution = max(0.0, restitution);
  let closing_speed = -vn;
  let v_normal = -safe_restitution * vn * normal;
  let v_tangent = velocity - vn * normal;
  let tangent_speed = length(v_tangent);
  if (tangent_speed <= 0.000001) {
    return v_normal;
  }

  // 1 回だけの近似 Coulomb 投影。
  // 法線方向の衝突 impulse を unit mass の速度変化として見なし、
  // 接線方向の減速量を mu * normalImpulse 以下に制限する。
  let normal_impulse = (1.0 + safe_restitution) * closing_speed;
  let friction_delta = min(tangent_speed, safe_friction * normal_impulse);
  let v_tangent_after = v_tangent * ((tangent_speed - friction_delta) / tangent_speed);
  return v_normal + v_tangent_after;
}

fn applyDomainBoundaries(pos: vec3f, velocity: vec3f) -> vec3f {
  var v = velocity;
  let boundary_thickness = 1.0;
  let boundary_count = u32(sim.material_params.y);
  for (var i = 0u; i < boundary_count; i++) {
    let boundary = domain_boundaries[i];
    let signed_distance = dot(pos, boundary.normal_offset.xyz) + boundary.normal_offset.w;
    if (signed_distance < boundary_thickness) {
      v = applyBoundaryVelocity(
        v,
        boundary.normal_offset.xyz,
        boundary.material.x,
        boundary.material.y
      );
    }
  }
  return v;
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
  // heightfield の高さは補間するが、摩擦/反発は「接触面を作っているセル」の値を使う。
  // 建物セルと地形セルの境界で friction が中間値になると、object friction を分けた意味が薄れる。
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
  if (width <= 1 || depth <= 1) {
    return TerrainSample(-1000000.0, vec3f(0.0, 1.0, 0.0), 0.0, 0.0, 1.0);
  }

  // Match the render transform: worldX = originX + gridX * (xLength / gridSize.x).
  // The previous gridSize - 1 denominator sampled DEM points too far toward +x/+z.
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
    material.x,
    material.y,
    material.w
  );
}

fn applyTerrainBoundary(pos: vec3f, velocity: vec3f) -> vec3f {
  if (sim.boundary_params.x < 0.5) {
    return velocity;
  }

  let width = i32(sim.material_params.z);
  let depth = i32(sim.material_params.w);
  if (width <= 1 || depth <= 1) {
    return velocity;
  }

  // ---- 早期 return: nearest 1 セルだけ読んで、明らかに地形上空なら即抜ける ----
  //
  // 本来の sampleTerrain は 4 セル bilinear + vec3 normalize() を必ず実行する。
  // updateGrid は active cell × substep 回走るため、地形より十分上空のセル
  // (大半のケース) でも毎回 4 read + normalize のコストを払っていた。
  //
  // ここでは bilinear と同じ floor 由来のコーナーセル (h00 相当) を 1 個だけ読む。
  // 高解像度 heightfield 化後は隣接セルの高さ差が小さくなったため、薄い slack だけ足し、
  // 地形から明らかに離れている cell では bilinear + normalize を省く。
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
  let approx_h = readTerrainCell(x0, z0).surface.x;
  let slack = 0.5;
  if (pos.y >= approx_h + sim.boundary_params.y + slack) {
    return velocity;
  }

  // ---- 地形近傍のみ、本来の bilinear + normalize を実行する ----
  let terrain = sampleTerrain(pos);
  if (pos.y >= terrain.height + sim.boundary_params.y) {
    return velocity;
  }

  return applyBoundaryVelocity(velocity, terrain.normal, terrain.friction, terrain.restitution);
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
      vec3f(0.0, 0.0, 0.0),
      0.0,
      0.0
    );
  }

  // side-wall は CPU 側で distance field 風の 2D 境界グリッドへ焼いてある。
  // bilinear で空セルと混ぜると signed distance と法線が壊れるため、nearest sample を使う。
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
      vec3f(0.0, 0.0, 0.0),
      0.0,
      0.0
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
    normal,
    cell.material.x,
    cell.material.y
  );
}

fn applyBuildingSideBoundary(pos: vec3f, velocity: vec3f) -> vec3f {
  let side = sampleBuildingSide(pos);
  if (side.mode < 0.5) {
    return velocity;
  }

  let vertical_slack = 0.25;
  if (pos.y < side.base - vertical_slack || pos.y > side.top + vertical_slack) {
    return velocity;
  }

  // Domain wall と同じく、有限厚みの境界帯で速度を拘束する。
  // ここでは上向き成分を持たない水平法線だけを使い、建物側面を急斜面として登らせない。
  let side_thickness = max(0.5, sim.boundary_params.y);
  if (side.signed_distance >= side_thickness) {
    return velocity;
  }

  return applyBoundaryVelocity(velocity, side.normal, side.friction, side.restitution);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&cells)) {
    return;
  }

  if (cells[id.x].mass <= 0) {
    return;
  }

  let mass = decodeFixedPoint(cells[id.x].mass);
  var velocity = vec3f(
    decodeFixedPoint(cells[id.x].vx),
    decodeFixedPoint(cells[id.x].vy),
    decodeFixedPoint(cells[id.x].vz)
  ) / mass;

  velocity.y = velocity.y + gravity_y * sim.boundary_params.z;

  // すべての境界は法線方向速度 + 接線摩擦の共通関数で扱う。
  let cell = cellCoord(id.x);
  let pos = vec3f(cell) + vec3f(0.5);
  velocity = applyDomainBoundaries(pos, velocity);
  velocity = applyBuildingSideBoundary(pos, velocity);
  velocity = applyTerrainBoundary(pos, velocity);

  cells[id.x].vx = encodeFixedPoint(velocity.x);
  cells[id.x].vy = encodeFixedPoint(velocity.y);
  cells[id.x].vz = encodeFixedPoint(velocity.z);
}
