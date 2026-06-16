import type { Terrain } from '../../terrain/Terrain';

export type MpmBoundaryType = 'domain' | 'terrain' | 'object';
export type MpmBoundaryMode = 'wall' | 'kill';

export interface MpmBoundaryMaterial {
  friction: number;
  restitution: number;
}

export interface MpmBoundary {
  type: MpmBoundaryType;
  mode: MpmBoundaryMode;
  material: MpmBoundaryMaterial;
}

export interface MpmGridSize {
  x: number;
  y: number;
  z: number;
}

export interface MpmBoxMeters {
  originX: number;
  originY: number;
  originZ: number;
  xLength: number;
  yLength: number;
  zLength: number;
}

export interface MpmBoxInputMeters {
  originX: number;
  originZ: number;
  xLength: number;
  yLength: number;
  zLength: number;
}

export interface TerrainBoundaryResolution {
  width: number;
  depth: number;
}

export interface MpmHeightfieldObstacle {
  readonly mpmBoundaryType: Exclude<MpmBoundaryType, 'domain'>;
  sampleHeight(
    worldX: number,
    worldZ: number,
    influenceRadiusM: number,
  ): number | undefined;
}

export interface MpmSideBoundarySample {
  signedDistanceM: number;
  normal: { x: number; z: number };
  baseHeight: number;
  topHeight: number;
}

export interface MpmSideBoundaryObstacle {
  readonly mpmBoundaryType: Exclude<MpmBoundaryType, 'domain'>;
  sampleSideBoundary(
    worldX: number,
    worldZ: number,
    influenceRadiusM: number,
  ): MpmSideBoundarySample | undefined;
}

export const BOUNDARY_TYPE_CODE: Record<MpmBoundaryType, number> = {
  domain: 1,
  terrain: 2,
  object: 3,
};

export const BOUNDARY_MODE_CODE: Record<MpmBoundaryMode, number> = {
  wall: 1,
  kill: 2,
};

const DOMAIN_PLANE_FLOATS = 8;
const TERRAIN_CELL_FLOATS = 8;
const BUILDING_SIDE_CELL_FLOATS = 8;
const EMPTY_BUILDING_SIDE_DISTANCE = 1_000_000;

export interface DomainBoundaryPlane extends MpmBoundary {
  normal: { x: number; y: number; z: number };
  offset: number;
}

export class DomainBoundary implements MpmBoundary {
  readonly type = 'domain';
  readonly planes: DomainBoundaryPlane[];

  constructor(
    readonly gridSize: MpmGridSize,
    readonly material: MpmBoundaryMaterial,
    readonly mode: MpmBoundaryMode = 'wall',
  ) {
    this.planes = [
      this.createPlane({ x: 1, y: 0, z: 0 }, 0),
      this.createPlane({ x: -1, y: 0, z: 0 }, gridSize.x),
      this.createPlane({ x: 0, y: 1, z: 0 }, 0),
      this.createPlane({ x: 0, y: -1, z: 0 }, gridSize.y),
      this.createPlane({ x: 0, y: 0, z: 1 }, 0),
      this.createPlane({ x: 0, y: 0, z: -1 }, gridSize.z),
    ];
  }

  toGpuData(): Float32Array {
    const data = new Float32Array(this.planes.length * DOMAIN_PLANE_FLOATS);
    for (let i = 0; i < this.planes.length; i++) {
      const plane = this.planes[i]!;
      const base = i * DOMAIN_PLANE_FLOATS;
      data[base] = plane.normal.x;
      data[base + 1] = plane.normal.y;
      data[base + 2] = plane.normal.z;
      data[base + 3] = plane.offset;
      data[base + 4] = plane.material.friction;
      data[base + 5] = plane.material.restitution;
      data[base + 6] = BOUNDARY_TYPE_CODE[plane.type];
      data[base + 7] = BOUNDARY_MODE_CODE[plane.mode];
    }
    return data;
  }

  private createPlane(
    normal: { x: number; y: number; z: number },
    offset: number,
  ): DomainBoundaryPlane {
    return {
      type: this.type,
      mode: this.mode,
      material: this.material,
      normal,
      offset,
    };
  }
}

export class TerrainBoundary implements MpmBoundary {
  readonly type = 'terrain';
  readonly width: number;
  readonly depth: number;
  readonly originY: number;
  readonly maxHeightGrid: number;
  readonly missingSamples: number;
  readonly clampedSamples: number;
  readonly objectSamples: number;
  readonly objectMaterial: MpmBoundaryMaterial | undefined;
  private readonly gridSize: MpmGridSize;
  private readonly cells: Float32Array;
  private readonly heights: Float32Array;

  private constructor(params: {
    material: MpmBoundaryMaterial;
    mode: MpmBoundaryMode;
    gridSize: MpmGridSize;
    width: number;
    depth: number;
    originY: number;
    maxHeightGrid: number;
    missingSamples: number;
    clampedSamples: number;
    objectSamples: number;
    objectMaterial?: MpmBoundaryMaterial | undefined;
    cells: Float32Array;
    heights: Float32Array;
  }) {
    this.material = params.material;
    this.mode = params.mode;
    this.gridSize = params.gridSize;
    this.width = params.width;
    this.depth = params.depth;
    this.originY = params.originY;
    this.maxHeightGrid = params.maxHeightGrid;
    this.missingSamples = params.missingSamples;
    this.clampedSamples = params.clampedSamples;
    this.objectSamples = params.objectSamples;
    this.objectMaterial = params.objectMaterial;
    this.cells = params.cells;
    this.heights = params.heights;
  }

  readonly material: MpmBoundaryMaterial;
  readonly mode: MpmBoundaryMode;

  static flat(
    gridSize: MpmGridSize,
    material: MpmBoundaryMaterial,
    originY: number,
    mode: MpmBoundaryMode = 'wall',
    resolution: number | TerrainBoundaryResolution = gridSize.x,
  ): TerrainBoundary {
    const { width, depth } = normalizeTerrainResolution(resolution, gridSize);
    const heights = new Float32Array(width * depth);
    return TerrainBoundary.fromGridHeights({
      gridSize,
      width,
      depth,
      material,
      mode,
      originY,
      missingSamples: width * depth,
      worldHeightsGridY: heights,
    });
  }

  static fromTerrain(params: {
    terrain: Terrain;
    gridSize: MpmGridSize;
    box: MpmBoxInputMeters;
    material: MpmBoundaryMaterial;
    objectMaterial?: MpmBoundaryMaterial | undefined;
    obstacles?: MpmHeightfieldObstacle[] | undefined;
    fallbackOriginY: number;
    mode?: MpmBoundaryMode;
    resolution?: number | TerrainBoundaryResolution;
  }): TerrainBoundary {
    const { width, depth } = normalizeTerrainResolution(
      params.resolution ?? params.gridSize.x,
      params.gridSize,
    );
    const worldHeights = new Float32Array(width * depth);
    const cellTypeCodes = new Float32Array(width * depth);
    const cellFrictions = new Float32Array(width * depth);
    const cellRestitutions = new Float32Array(width * depth);
    let originY = Number.POSITIVE_INFINITY;
    let missingSamples = 0;
    let objectSamples = 0;
    const objectMaterial = params.objectMaterial ?? params.material;
    const obstacles = params.obstacles ?? [];
    const influenceRadiusM = terrainBoundarySampleRadiusM(params.box, width, depth);

    for (let x = 0; x < width; x++) {
      const tx = width <= 1 ? 0 : x / (width - 1);
      const worldX = params.box.originX + params.box.xLength * tx;
      for (let z = 0; z < depth; z++) {
        const tz = depth <= 1 ? 0 : z / (depth - 1);
        const worldZ = params.box.originZ + params.box.zLength * tz;
        const sample = params.terrain.sampleSurface(worldX, worldZ);
        const index = x * depth + z;
        let boundaryType: MpmBoundaryType = 'terrain';
        let material = params.material;
        if (sample) {
          worldHeights[index] = sample.height;
          originY = Math.min(originY, sample.height);
        } else {
          worldHeights[index] = params.fallbackOriginY;
          missingSamples++;
        }

        // 建物などの heightfield 障害物は、地形と同じ衝突式で使うため
        // ここで terrain heightfield へ合成する。type/material はセルごとに保持する。
        for (const obstacle of obstacles) {
          const obstacleHeight = obstacle.sampleHeight(
            worldX,
            worldZ,
            influenceRadiusM,
          );
          if (
            obstacleHeight === undefined ||
            !Number.isFinite(obstacleHeight) ||
            obstacleHeight <= worldHeights[index]!
          ) {
            continue;
          }
          worldHeights[index] = obstacleHeight;
          originY = Math.min(originY, obstacleHeight);
          boundaryType = obstacle.mpmBoundaryType;
          material = boundaryType === 'object' ? objectMaterial : params.material;
        }
        if (boundaryType === 'object') objectSamples++;
        cellTypeCodes[index] = BOUNDARY_TYPE_CODE[boundaryType];
        cellFrictions[index] = material.friction;
        cellRestitutions[index] = material.restitution;
      }
    }

    const yScale = params.box.yLength / params.gridSize.y;
    if (!Number.isFinite(originY)) {
      originY = params.fallbackOriginY;
    }
    const maxTerrainHeightGrid = Math.max(0, params.gridSize.y - 1e-3);
    const worldHeightsGridY = new Float32Array(width * depth);
    let clampedSamples = 0;

    for (let i = 0; i < worldHeights.length; i++) {
      const heightGridY = (worldHeights[i]! - originY) / yScale;
      const clamped = Math.min(maxTerrainHeightGrid, Math.max(0, heightGridY));
      if (clamped !== heightGridY) clampedSamples++;
      worldHeightsGridY[i] = clamped;
    }

    return TerrainBoundary.fromGridHeights({
      gridSize: params.gridSize,
      width,
      depth,
      material: params.material,
      mode: params.mode ?? 'wall',
      originY,
      missingSamples,
      clampedSamples,
      worldHeightsGridY,
      cellTypeCodes,
      cellFrictions,
      cellRestitutions,
      objectSamples,
      objectMaterial: objectSamples > 0 ? objectMaterial : undefined,
    });
  }

  toGpuData(): Float32Array {
    return this.cells;
  }

  sampleHeightGrid(x: number, z: number): number {
    const gx = this.gridXToBoundaryX(x);
    const gz = this.gridZToBoundaryZ(z);
    const x0 = Math.floor(gx);
    const z0 = Math.floor(gz);
    const x1 = Math.min(x0 + 1, this.width - 1);
    const z1 = Math.min(z0 + 1, this.depth - 1);
    const tx = gx - x0;
    const tz = gz - z0;
    const h00 = this.heightAt(x0, z0);
    const h10 = this.heightAt(x1, z0);
    const h01 = this.heightAt(x0, z1);
    const h11 = this.heightAt(x1, z1);
    const h0 = h00 + (h10 - h00) * tx;
    const h1 = h01 + (h11 - h01) * tx;
    return h0 + (h1 - h0) * tz;
  }

  sampleNormalGrid(x: number, z: number): { x: number; y: number; z: number } {
    const gx = this.gridXToBoundaryX(x);
    const gz = this.gridZToBoundaryZ(z);
    const x0 = Math.floor(gx);
    const z0 = Math.floor(gz);
    const x1 = Math.min(x0 + 1, this.width - 1);
    const z1 = Math.min(z0 + 1, this.depth - 1);
    const tx = gx - x0;
    const tz = gz - z0;
    const n00 = this.normalAt(x0, z0);
    const n10 = this.normalAt(x1, z0);
    const n01 = this.normalAt(x0, z1);
    const n11 = this.normalAt(x1, z1);
    const nx0 = mix(n00.x, n10.x, tx);
    const ny0 = mix(n00.y, n10.y, tx);
    const nz0 = mix(n00.z, n10.z, tx);
    const nx1 = mix(n01.x, n11.x, tx);
    const ny1 = mix(n01.y, n11.y, tx);
    const nz1 = mix(n01.z, n11.z, tx);
    const nx = mix(nx0, nx1, tz);
    const ny = mix(ny0, ny1, tz);
    const nz = mix(nz0, nz1, tz);
    const len = Math.hypot(nx, ny, nz) || 1;
    return { x: nx / len, y: ny / len, z: nz / len };
  }

  private static fromGridHeights(params: {
    gridSize: MpmGridSize;
    width: number;
    depth: number;
    material: MpmBoundaryMaterial;
    mode: MpmBoundaryMode;
    originY: number;
    missingSamples: number;
    clampedSamples?: number;
    cellTypeCodes?: Float32Array;
    cellFrictions?: Float32Array;
    cellRestitutions?: Float32Array;
    objectSamples?: number;
    objectMaterial?: MpmBoundaryMaterial | undefined;
    worldHeightsGridY: Float32Array;
  }): TerrainBoundary {
    const width = params.width;
    const depth = params.depth;
    const cells = new Float32Array(width * depth * TERRAIN_CELL_FLOATS);
    const heights = params.worldHeightsGridY;
    let maxHeightGrid = 0;
    // heightfield は box 全幅 [0, gridSize.x] / [0, gridSize.z] に張る。
    // 描画側も grid coordinate * (boxLength / gridSize) で ENU へ写像するため、
    // 法線の水平距離も gridSize 基準にし、(gridSize - 1) 基準の横ズレを作らない。
    const dxGrid = Math.max(1, params.gridSize.x) / Math.max(1, width - 1);
    const dzGrid = Math.max(1, params.gridSize.z) / Math.max(1, depth - 1);

    for (let x = 0; x < width; x++) {
      for (let z = 0; z < depth; z++) {
        const h = heights[x * depth + z] ?? 0;
        maxHeightGrid = Math.max(maxHeightGrid, h);
        const hW = heights[Math.max(0, x - 1) * depth + z] ?? h;
        const hE = heights[Math.min(width - 1, x + 1) * depth + z] ?? h;
        const hS = heights[x * depth + Math.max(0, z - 1)] ?? h;
        const hN = heights[x * depth + Math.min(depth - 1, z + 1)] ?? h;
        const dx = Math.max(1, Math.min(width - 1, x + 1) - Math.max(0, x - 1)) * dxGrid;
        const dz = Math.max(1, Math.min(depth - 1, z + 1) - Math.max(0, z - 1)) * dzGrid;
        const dhdx = (hE - hW) / dx;
        const dhdz = (hN - hS) / dz;
        const nx = -dhdx;
        const ny = 1;
        const nz = -dhdz;
        const len = Math.hypot(nx, ny, nz) || 1;
        const index = x * depth + z;
        const base = (x * depth + z) * TERRAIN_CELL_FLOATS;
        cells[base] = h;
        cells[base + 1] = nx / len;
        cells[base + 2] = ny / len;
        cells[base + 3] = nz / len;
        cells[base + 4] = params.cellFrictions?.[index] ?? params.material.friction;
        cells[base + 5] =
          params.cellRestitutions?.[index] ?? params.material.restitution;
        cells[base + 6] =
          params.cellTypeCodes?.[index] ?? BOUNDARY_TYPE_CODE.terrain;
        cells[base + 7] = BOUNDARY_MODE_CODE[params.mode];
      }
    }

    return new TerrainBoundary({
      material: params.material,
      mode: params.mode,
      gridSize: params.gridSize,
      width,
      depth,
      originY: params.originY,
      maxHeightGrid,
      missingSamples: params.missingSamples,
      clampedSamples: params.clampedSamples ?? 0,
      objectSamples: params.objectSamples ?? 0,
      objectMaterial: params.objectMaterial,
      cells,
      heights,
    });
  }

  private heightAt(x: number, z: number): number {
    return this.heights[x * this.depth + z] ?? 0;
  }

  private normalAt(x: number, z: number): { x: number; y: number; z: number } {
    const base = (x * this.depth + z) * TERRAIN_CELL_FLOATS;
    return {
      x: this.cells[base + 1] ?? 0,
      y: this.cells[base + 2] ?? 1,
      z: this.cells[base + 3] ?? 0,
    };
  }

  private gridXToBoundaryX(x: number): number {
    return clamp(
      x / Math.max(1, this.gridSize.x) * (this.width - 1),
      0,
      this.width - 1,
    );
  }

  private gridZToBoundaryZ(z: number): number {
    return clamp(
      z / Math.max(1, this.gridSize.z) * (this.depth - 1),
      0,
      this.depth - 1,
    );
  }
}

export class BuildingSideBoundary implements MpmBoundary {
  readonly type = 'object';
  readonly material: MpmBoundaryMaterial;
  readonly mode: MpmBoundaryMode;
  readonly width: number;
  readonly depth: number;
  readonly sideSamples: number;

  private constructor(params: {
    material: MpmBoundaryMaterial;
    mode: MpmBoundaryMode;
    width: number;
    depth: number;
    sideSamples: number;
    cells: Float32Array;
  }) {
    this.material = params.material;
    this.mode = params.mode;
    this.width = params.width;
    this.depth = params.depth;
    this.sideSamples = params.sideSamples;
    this.cells = params.cells;
  }

  private readonly cells: Float32Array;

  static empty(
    gridSize: MpmGridSize,
    material: MpmBoundaryMaterial,
    mode: MpmBoundaryMode = 'wall',
    resolution: number | TerrainBoundaryResolution = gridSize.x,
  ): BuildingSideBoundary {
    const { width, depth } = normalizeTerrainResolution(resolution, gridSize);
    return new BuildingSideBoundary({
      material,
      mode,
      width,
      depth,
      sideSamples: 0,
      cells: emptyBuildingSideCells(width, depth),
    });
  }

  static fromObstacles(params: {
    gridSize: MpmGridSize;
    box: MpmBoxInputMeters;
    originY: number;
    material: MpmBoundaryMaterial;
    mode?: MpmBoundaryMode;
    resolution?: number | TerrainBoundaryResolution;
    obstacles?: MpmSideBoundaryObstacle[] | undefined;
  }): BuildingSideBoundary {
    const { width, depth } = normalizeTerrainResolution(
      params.resolution ?? params.gridSize.x,
      params.gridSize,
    );
    const cells = emptyBuildingSideCells(width, depth);
    const obstacles = params.obstacles ?? [];
    if (obstacles.length === 0) {
      return new BuildingSideBoundary({
        material: params.material,
        mode: params.mode ?? 'wall',
        width,
        depth,
        sideSamples: 0,
        cells,
      });
    }

    const cellSizeX = params.box.xLength / Math.max(1, params.gridSize.x);
    const cellSizeY = params.box.yLength / Math.max(1, params.gridSize.y);
    const cellSizeZ = params.box.zLength / Math.max(1, params.gridSize.z);
    const modeCode = BOUNDARY_MODE_CODE[params.mode ?? 'wall'];
    const sampleRadiusM = terrainBoundarySampleRadiusM(params.box, width, depth);
    const searchRadiusM =
      sampleRadiusM + 0.75 * Math.max(cellSizeX, cellSizeZ);
    const maxY = Math.max(0, params.gridSize.y - 1e-3);
    let sideSamples = 0;

    for (let x = 0; x < width; x++) {
      const tx = width <= 1 ? 0 : x / (width - 1);
      const worldX = params.box.originX + params.box.xLength * tx;
      for (let z = 0; z < depth; z++) {
        const tz = depth <= 1 ? 0 : z / (depth - 1);
        const worldZ = params.box.originZ + params.box.zLength * tz;
        let bestSample: MpmSideBoundarySample | undefined;
        let bestScore = Number.POSITIVE_INFINITY;

        for (const obstacle of obstacles) {
          const sample = obstacle.sampleSideBoundary(worldX, worldZ, searchRadiusM);
          if (sample === undefined) continue;
          const score = sideBoundarySampleScore(sample);
          if (score < bestScore) {
            bestScore = score;
            bestSample = sample;
          }
        }
        if (bestSample === undefined) continue;

        const normalLen = Math.hypot(bestSample.normal.x, bestSample.normal.z);
        if (normalLen <= 1e-6) continue;
        const nxWorld = bestSample.normal.x / normalLen;
        const nzWorld = bestSample.normal.z / normalLen;
        const gridNormalMetric = Math.hypot(
          nxWorld * cellSizeX,
          nzWorld * cellSizeZ,
        );
        if (gridNormalMetric <= 1e-6) continue;

        const baseGridY = clamp(
          (bestSample.baseHeight - params.originY) / cellSizeY,
          0,
          maxY,
        );
        const topGridY = clamp(
          (bestSample.topHeight - params.originY) / cellSizeY,
          0,
          maxY,
        );
        if (topGridY <= baseGridY + 1e-3) continue;

        const base = (x * depth + z) * BUILDING_SIDE_CELL_FLOATS;
        // signedDistanceM は世界メートルでの footprint からの距離。
        // shader 側の粒子・grid 速度は MPM grid 単位なので、水平スケールを含めた
        // grid 座標系の符号付き距離と法線へ変換して保存する。
        cells[base] = bestSample.signedDistanceM / gridNormalMetric;
        cells[base + 1] = (nxWorld * cellSizeX) / gridNormalMetric;
        cells[base + 2] = (nzWorld * cellSizeZ) / gridNormalMetric;
        cells[base + 3] = topGridY;
        cells[base + 4] = params.material.friction;
        cells[base + 5] = params.material.restitution;
        cells[base + 6] = baseGridY;
        cells[base + 7] = modeCode;
        sideSamples++;
      }
    }

    return new BuildingSideBoundary({
      material: params.material,
      mode: params.mode ?? 'wall',
      width,
      depth,
      sideSamples,
      cells,
    });
  }

  toGpuData(): Float32Array {
    return this.cells;
  }
}

function terrainBoundarySampleRadiusM(
  box: MpmBoxInputMeters,
  width: number,
  depth: number,
): number {
  const dx = box.xLength / Math.max(1, width - 1);
  const dz = box.zLength / Math.max(1, depth - 1);
  // Heightfield は点サンプルなので、footprint が格子点の間に落ちる小建物を
  // 完全に見逃さないよう、1 セル半径だけ障害物判定を広げる。
  return Math.hypot(dx, dz) * 0.5;
}

function emptyBuildingSideCells(width: number, depth: number): Float32Array {
  const cells = new Float32Array(width * depth * BUILDING_SIDE_CELL_FLOATS);
  for (let i = 0; i < width * depth; i++) {
    cells[i * BUILDING_SIDE_CELL_FLOATS] = EMPTY_BUILDING_SIDE_DISTANCE;
  }
  return cells;
}

function sideBoundarySampleScore(sample: MpmSideBoundarySample): number {
  const distance = Math.abs(sample.signedDistanceM);
  // footprint 内部は既に建物 solid に入っている状態なので、外側の近傍 wall より優先する。
  return sample.signedDistanceM < 0 ? -1_000_000 + distance : distance;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function normalizeTerrainResolution(
  resolution: number | TerrainBoundaryResolution,
  gridSize: MpmGridSize,
): TerrainBoundaryResolution {
  if (typeof resolution === 'number') {
    const size = Math.max(2, Math.floor(resolution));
    return { width: size, depth: size };
  }
  return {
    width: Math.max(2, Math.floor(resolution.width || gridSize.x)),
    depth: Math.max(2, Math.floor(resolution.depth || gridSize.z)),
  };
}

function mix(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
