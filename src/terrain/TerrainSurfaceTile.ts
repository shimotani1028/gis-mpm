// =============================================================================
// TerrainSurfaceTile: DEM 標高だけを持つ地表サンプル用タイル
// -----------------------------------------------------------------------------
// 描画用 TileMesh と MPM 接触用 coverage の両方で同じ x/z -> 標高変換を使う。
// MPM coverage では Three.js の Geometry/Material を作らず、この軽量クラスだけを
// 保持することで、広い計算領域でも GPU メッシュを大量に増やさない。
// =============================================================================

import { tileSizeAndCenterENU, type TileKey } from '../geo/tileMath';
import type { TileData } from './TileLoader';

/** 地形衝突用に返す、ワールド ENU 座標での高さと法線。 */
export interface TerrainSurfaceSample {
  height: number;
  normal: { x: number; y: number; z: number };
}

/**
 * DEM 1 タイル分の CPU サンプラ。
 *
 * 座標系:
 * - worldX = East [m]
 * - worldZ = North [m]
 * - DEM row 0 は北端、col 0 は西端
 */
export class TerrainSurfaceTile {
  /** タイル識別子（デバッグと tile key 逆引き用） */
  readonly key: TileKey;

  protected readonly elevations: Float32Array;
  protected readonly invalidMask: Uint8Array | undefined;
  protected readonly cols: number;
  protected readonly rows: number;
  protected readonly widthM: number;
  protected readonly heightM: number;
  protected readonly centerX: number;
  protected readonly centerY: number;
  protected readonly dx: number;
  protected readonly dz: number;

  constructor(key: TileKey, data: TileData) {
    this.key = key;
    this.elevations = data.elevations;
    this.invalidMask = data.invalidMask;
    this.cols = data.width;
    this.rows = data.height;

    const { widthM, heightM, centerX, centerY } = tileSizeAndCenterENU(
      key.z, key.x, key.y,
    );
    this.widthM = widthM;
    this.heightM = heightM;
    this.centerX = centerX;
    this.centerY = centerY;
    // GSI DEM PNG の各画素はタイル外周そのものではなく画素中心の標高として扱う。
    // タイル境界の連続サンプルは Terrain 側で隣接タイルをまたいで行うため、
    // 単体タイル内の座標も「pixel center が index 整数」になるように合わせる。
    this.dx = widthM / this.cols;
    this.dz = heightM / this.rows;
  }

  /** ワールド ENU の x/z から、このタイル上の高さと法線を返す。範囲外なら undefined。 */
  sampleSurface(worldX: number, worldZ: number): TerrainSurfaceSample | undefined {
    const localX = worldX - this.centerX;
    const localZ = worldZ - this.centerY;
    const col = (localX + this.widthM * 0.5) / this.dx - 0.5;
    const row = (this.heightM * 0.5 - localZ) / this.dz - 0.5;
    if (col < -0.5 || row < -0.5 || col > this.cols - 0.5 || row > this.rows - 0.5) {
      return undefined;
    }

    const height = this.sampleElevationAtGrid(col, row);
    const colW = Math.max(0, col - 1);
    const colE = Math.min(this.cols - 1, col + 1);
    const rowN = Math.max(0, row - 1);
    const rowS = Math.min(this.rows - 1, row + 1);
    const hW = this.sampleElevationAtGrid(colW, row);
    const hE = this.sampleElevationAtGrid(colE, row);
    const hN = this.sampleElevationAtGrid(col, rowN);
    const hS = this.sampleElevationAtGrid(col, rowS);
    const dhdx = (hE - hW) / Math.max((colE - colW) * this.dx, 1e-6);
    const dhdz = (hN - hS) / Math.max((rowS - rowN) * this.dz, 1e-6);

    // y = height(x,z) の上向き法線。z は北向きを正にするため dhdz は北-南の勾配。
    const nx = -dhdx;
    const ny = 1;
    const nz = -dhdz;
    const len = Math.hypot(nx, ny, nz) || 1;
    return {
      height,
      normal: { x: nx / len, y: ny / len, z: nz / len },
    };
  }

  /** global tile sampler から使う、1 画素中心の標高値。 */
  heightAtPixel(col: number, row: number): number {
    return this.elevationAt(col, row);
  }

  /**
   * raw DEM で有効だった 1 画素中心の標高値。
   * worker 側の有限値 fallback で埋まった no-data は undefined として扱う。
   */
  heightAtValidPixel(col: number, row: number): number | undefined {
    const safeCol = Math.min(this.cols - 1, Math.max(0, col));
    const safeRow = Math.min(this.rows - 1, Math.max(0, row));
    const index = safeRow * this.cols + safeCol;
    if (this.invalidMask?.[index] === 1) return undefined;
    const value = this.elevations[index];
    return Number.isFinite(value) ? value : undefined;
  }

  private sampleElevationAtGrid(col: number, row: number): number {
    const c0 = Math.floor(col);
    const r0 = Math.floor(row);
    const c1 = Math.min(c0 + 1, this.cols - 1);
    const r1 = Math.min(r0 + 1, this.rows - 1);
    const tx = col - c0;
    const tz = row - r0;
    const h00 = this.elevationAt(c0, r0);
    const h10 = this.elevationAt(c1, r0);
    const h01 = this.elevationAt(c0, r1);
    const h11 = this.elevationAt(c1, r1);
    const h0 = h00 + (h10 - h00) * tx;
    const h1 = h01 + (h11 - h01) * tx;
    return h0 + (h1 - h0) * tz;
  }

  private elevationAt(col: number, row: number): number {
    const safeCol = Math.min(this.cols - 1, Math.max(0, col));
    const safeRow = Math.min(this.rows - 1, Math.max(0, row));
    return this.elevations[safeRow * this.cols + safeCol] ?? 0;
  }
}
