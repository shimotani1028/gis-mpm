// =============================================================================
// 1 タイル分の Three.js メッシュ
// -----------------------------------------------------------------------------
// 構成:
//   - 既定では DEM 256x256 の画素中心値から、257x257 のセル境界メッシュを作る
//   - bridge mode では DEM 画素中心 256x256 だけを描き、隣接タイル間のセルは
//     TileBridgeMesh が A col255 / B col0 のような edge 画素中心同士を直接つなぐ
//   - タイル外周には必要に応じて下向きの skirt を追加し、LOD 境界の隙間を隠す
//   - 標高に応じた色は color attribute に焼き込む
//
// メッシュの位置と回転:
//   - 頂点ローカル座標を ENU の (x_East, y_Up, z_North) として作る
//   - row 0 = 北端、col 0 = 西端
//   - メッシュ自体はタイル中心 (centerX, 0, centerY) に置く
//
// 寿命管理:
//   - dispose() で Geometry / Material を解放
//   - LRU キャッシュから外れる際に確実に呼ぶこと（GPU メモリリーク防止）
// =============================================================================

import {
  BufferGeometry,
  Float32BufferAttribute,
  Mesh,
  Uint32BufferAttribute,
} from 'three/webgpu';

import { TILE_PIXELS, TILE_SKIRT_DEPTH_M } from '../config';
import type { TileKey } from '../geo/tileMath';
import { buildTerrainMaterial } from './material';
import type { TileData } from './TileLoader';
import { TerrainSurfaceTile } from './TerrainSurfaceTile';

export type { TerrainSurfaceSample } from './TerrainSurfaceTile';

export interface TileMeshBoundaryNeighbors {
  north?: TileMesh;
  south?: TileMesh;
  west?: TileMesh;
  east?: TileMesh;
  northwest?: TileMesh;
  northeast?: TileMesh;
  southwest?: TileMesh;
  southeast?: TileMesh;
}

interface DemGeometryData {
  geometry: BufferGeometry;
  positions: Float32Array;
  normals: Float32Array;
  colors: Float32Array;
  sampleCols: number;
  sampleRows: number;
  gridCols: number;
  gridRows: number;
  dx: number;
  dz: number;
  northSkirt: number[];
  southSkirt: number[];
  westSkirt: number[];
  eastSkirt: number[];
  skirtDepthM: number;
  mode: TileMeshGeometryMode;
}

interface TileMeshOptions {
  geometryMode?: TileMeshGeometryMode;
  skirtDepthM?: number;
}

export type TileMeshGeometryMode = 'cell-boundary' | 'pixel-center';
export type TileBridgeDirection = 'east' | 'south';
export type TileEdge = 'north' | 'south' | 'west' | 'east';
export type TileLodBridgeAxis = 'x' | 'y';

export interface TileLodBridgeSide {
  tile: TileMesh;
  edge: TileEdge;
}

export interface TileLodBridgeSpan {
  axis: TileLodBridgeAxis;
  commonZoom: number;
  start: number;
  end: number;
}

const NO_DATA_SEARCH_RADIUS_PIXELS = 2;

/**
 * 1 タイル分のメッシュを表すラッパ。Three.js の Mesh と寿命管理 API を持つ。
 */
export class TileMesh extends TerrainSurfaceTile {
  /** Three.js シーンに add するためのメッシュ本体 */
  readonly mesh: Mesh;

  // 内部リソース（dispose で解放）
  private readonly geometry: BufferGeometry;
  private readonly material: ReturnType<typeof buildTerrainMaterial>;
  private readonly geometryData: DemGeometryData;
  private boundarySignature: string | undefined = undefined;

  constructor(key: TileKey, data: TileData, options: TileMeshOptions = {}) {
    super(key, data);

    // ---- ジオメトリのサイズと位置 ----
    // DEM の画素値は画素中心の高さとして扱う。描画メッシュも CPU sampler と
    // 同じ「タイル幅 / 画素数」の座標系で作り、端画素を外周へ引き伸ばさない。
    this.geometryData = buildDemGeometry(
      data,
      this.widthM,
      this.heightM,
      this.centerX,
      this.centerY,
      options.geometryMode ?? 'cell-boundary',
      options.skirtDepthM ?? TILE_SKIRT_DEPTH_M,
    );
    this.geometry = this.geometryData.geometry;

    // ---- マテリアル ----
    this.material = buildTerrainMaterial();

    // ---- メッシュ ----
    this.mesh = new Mesh(this.geometry, this.material);

    // 頂点はローカル ENU、メッシュ位置はタイル中心 ENU。
    // y は頂点に標高として入っているため、メッシュ自体の y は 0 に置く。
    this.mesh.position.set(this.centerX, 0, this.centerY);

    // BufferGeometry の bounding volume は実標高を含んでいるので、標準の
    // frustum culling を使える。旧実装のような shader displacement 由来の
    // 誤カリングは起きない。
    this.mesh.frustumCulled = true;

    // 名前にタイルキーを入れておくと DevTools のシーン検査で便利
    this.mesh.name = `tile_${key.z}_${key.x}_${key.y}`;
  }

  /**
   * タイル外周とその 1 セル内側の高さを、CPU sampler と同じ画素中心 bilinear で再計算する。
   * 隣接 DEM が表示中なら境界の反対側の画素も読み、未ロードなら自タイル内で
   * clamp する。端高さを平均して「見た目だけ合わせる」処理ではなく、同じ座標で
   * 同じ DEM サンプル式を使うための境界帯更新に限定する。
   */
  updateBoundarySampling(neighbors: TileMeshBoundaryNeighbors): void {
    // pixel-center mode ではタイル境界の人工頂点を持たない。
    // 隣接タイル間のセルは TileBridgeMesh が実際の edge 画素中心同士で張る。
    if (this.geometryData.mode !== 'cell-boundary') return;

    const nextSignature = makeBoundarySignature(neighbors);
    if (nextSignature === this.boundarySignature) return;
    this.boundarySignature = nextSignature;

    const data = this.geometryData;
    const lastRow = data.gridRows - 1;
    const lastCol = data.gridCols - 1;
    const rows = boundaryBandIndices(lastRow);
    const cols = boundaryBandIndices(lastCol);

    for (const row of rows) {
      for (let col = 0; col < data.gridCols; col++) {
        this.setGridHeight(row, col, this.sampleGridHeight(col, row, neighbors));
      }
    }
    for (let row = 0; row < data.gridRows; row++) {
      for (const col of cols) {
        this.setGridHeight(row, col, this.sampleGridHeight(col, row, neighbors));
      }
    }

    // 法線と色は高さを更新した境界帯だけ再計算する。
    this.updatePerimeterNormals();
    this.updatePerimeterColors();
    this.updateSkirts({
      north: neighbors.north !== undefined,
      south: neighbors.south !== undefined,
      west: neighbors.west !== undefined,
      east: neighbors.east !== undefined,
    });

    const positionAttr = this.geometry.getAttribute('position');
    const normalAttr = this.geometry.getAttribute('normal');
    const colorAttr = this.geometry.getAttribute('color');
    positionAttr.needsUpdate = true;
    normalAttr.needsUpdate = true;
    colorAttr.needsUpdate = true;
    this.geometry.computeBoundingBox();
    this.geometry.computeBoundingSphere();
  }

  /** GPU リソースを解放する。LRU evict 時に必ず呼ぶこと */
  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }

  /** bridge cell 生成用に、指定した DEM 画素中心の world 座標と高さを返す。 */
  pixelCenterVertex(col: number, row: number): { x: number; y: number; z: number } {
    const safeCol = Math.min(this.cols - 1, Math.max(0, col));
    const safeRow = Math.min(this.rows - 1, Math.max(0, row));
    return {
      x: this.centerX - this.widthM * 0.5 + (safeCol + 0.5) * this.dx,
      y: this.heightAtPixel(safeCol, safeRow),
      z: this.centerY + this.heightM * 0.5 - (safeRow + 0.5) * this.dz,
    };
  }

  /** bridge cell のライティングを通常 tile edge と揃えるための頂点法線。 */
  pixelCenterNormal(col: number, row: number): { x: number; y: number; z: number } {
    const safeCol = Math.min(this.geometryData.gridCols - 1, Math.max(0, col));
    const safeRow = Math.min(this.geometryData.gridRows - 1, Math.max(0, row));
    const p = (safeRow * this.geometryData.gridCols + safeCol) * 3;
    return {
      x: this.geometryData.normals[p] ?? 0,
      y: this.geometryData.normals[p + 1] ?? 1,
      z: this.geometryData.normals[p + 2] ?? 0,
    };
  }

  /**
   * mixed LOD bridge 用に、tile edge 上の連続座標 t=[0,1] の world 頂点を返す。
   * t は west/east edge では north→south、north/south edge では west→east。
   */
  edgeVertexAt(edge: TileEdge, t: number): { x: number; y: number; z: number } {
    const pixelCol =
      edge === 'west'
        ? 0
        : edge === 'east'
          ? this.cols - 1
          : t * this.cols - 0.5;
    const pixelRow =
      edge === 'north'
        ? 0
        : edge === 'south'
          ? this.rows - 1
          : t * this.rows - 0.5;
    return {
      x: this.centerX - this.widthM * 0.5 + (pixelCol + 0.5) * this.dx,
      y: this.samplePixelCoordinate(pixelCol, pixelRow, {}),
      z: this.centerY + this.heightM * 0.5 - (pixelRow + 0.5) * this.dz,
    };
  }

  /**
   * edgeVertexAt() と同じ連続座標で、表示 tile mesh の edge 法線を線形補間する。
   * LOD bridge 側で独自に normal を作ると、bridge だけ別ライティングの帯になる。
   */
  edgeNormalAt(edge: TileEdge, t: number): { x: number; y: number; z: number } {
    const pixelCol =
      edge === 'west'
        ? 0
        : edge === 'east'
          ? this.geometryData.gridCols - 1
          : t * this.geometryData.gridCols - 0.5;
    const pixelRow =
      edge === 'north'
        ? 0
        : edge === 'south'
          ? this.geometryData.gridRows - 1
          : t * this.geometryData.gridRows - 0.5;
    return this.normalAtPixelCoordinate(pixelCol, pixelRow);
  }

  get pixelCols(): number {
    return this.cols;
  }

  get pixelRows(): number {
    return this.rows;
  }

  private sampleGridHeight(
    gridCol: number,
    gridRow: number,
    neighbors: TileMeshBoundaryNeighbors,
  ): number {
    // grid vertex は画素中心 index の中間にあるため、境界 0 は -0.5、
    // 内部境界 1 は 0.5、東端 256 は 255.5 をサンプルする。
    return this.samplePixelCoordinate(gridCol - 0.5, gridRow - 0.5, neighbors);
  }

  private samplePixelCoordinate(
    pixelCol: number,
    pixelRow: number,
    neighbors: TileMeshBoundaryNeighbors,
  ): number {
    const validHeight = this.sampleValidPixelCoordinate(pixelCol, pixelRow, neighbors);
    if (validHeight !== undefined) return validHeight;
    const nearestHeight = this.nearestValidPixelCoordinate(
      pixelCol,
      pixelRow,
      neighbors,
      NO_DATA_SEARCH_RADIUS_PIXELS,
    );
    if (nearestHeight !== undefined) return nearestHeight;
    return this.sampleFallbackPixelCoordinate(pixelCol, pixelRow, neighbors);
  }

  private sampleValidPixelCoordinate(
    pixelCol: number,
    pixelRow: number,
    neighbors: TileMeshBoundaryNeighbors,
  ): number | undefined {
    const c0 = Math.floor(pixelCol);
    const r0 = Math.floor(pixelRow);
    const c1 = c0 + 1;
    const r1 = r0 + 1;
    const tx = pixelCol - c0;
    const ty = pixelRow - r0;

    const h00 = this.validHeightAtPixelWithNeighbors(c0, r0, neighbors);
    const h10 = this.validHeightAtPixelWithNeighbors(c1, r0, neighbors);
    const h01 = this.validHeightAtPixelWithNeighbors(c0, r1, neighbors);
    const h11 = this.validHeightAtPixelWithNeighbors(c1, r1, neighbors);

    let total = 0;
    let weightSum = 0;
    const add = (height: number | undefined, weight: number): void => {
      if (height === undefined || weight <= 0) return;
      total += height * weight;
      weightSum += weight;
    };
    add(h00, (1 - tx) * (1 - ty));
    add(h10, tx * (1 - ty));
    add(h01, (1 - tx) * ty);
    add(h11, tx * ty);
    return weightSum > 0 ? total / weightSum : undefined;
  }

  private nearestValidPixelCoordinate(
    pixelCol: number,
    pixelRow: number,
    neighbors: TileMeshBoundaryNeighbors,
    radius: number,
  ): number | undefined {
    const centerCol = Math.round(pixelCol);
    const centerRow = Math.round(pixelRow);
    let bestDistSq = Number.POSITIVE_INFINITY;
    let total = 0;
    let count = 0;

    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const sampleCol = centerCol + dx;
        const sampleRow = centerRow + dy;
        const height = this.validHeightAtPixelWithNeighbors(sampleCol, sampleRow, neighbors);
        if (height === undefined) continue;

        const distSq = (sampleCol - pixelCol) ** 2 + (sampleRow - pixelRow) ** 2;
        if (distSq + 1e-9 < bestDistSq) {
          bestDistSq = distSq;
          total = height;
          count = 1;
        } else if (Math.abs(distSq - bestDistSq) <= 1e-9) {
          total += height;
          count++;
        }
      }
    }

    return count > 0 ? total / count : undefined;
  }

  private sampleFallbackPixelCoordinate(
    pixelCol: number,
    pixelRow: number,
    neighbors: TileMeshBoundaryNeighbors,
  ): number {
    const c0 = Math.floor(pixelCol);
    const r0 = Math.floor(pixelRow);
    const c1 = c0 + 1;
    const r1 = r0 + 1;
    const tx = pixelCol - c0;
    const ty = pixelRow - r0;

    const h00 = this.fallbackHeightAtPixelWithNeighbors(c0, r0, neighbors);
    const h10 = this.fallbackHeightAtPixelWithNeighbors(c1, r0, neighbors);
    const h01 = this.fallbackHeightAtPixelWithNeighbors(c0, r1, neighbors);
    const h11 = this.fallbackHeightAtPixelWithNeighbors(c1, r1, neighbors);
    const h0 = h00 + (h10 - h00) * tx;
    const h1 = h01 + (h11 - h01) * tx;
    return h0 + (h1 - h0) * ty;
  }

  private validHeightAtPixelWithNeighbors(
    col: number,
    row: number,
    neighbors: TileMeshBoundaryNeighbors,
  ): number | undefined {
    if (col >= 0 && col < this.cols && row >= 0 && row < this.rows) {
      return this.heightAtValidPixel(col, row);
    }

    const offsetX = col < 0 ? -1 : col >= this.cols ? 1 : 0;
    const offsetY = row < 0 ? -1 : row >= this.rows ? 1 : 0;
    const neighbor = neighborForOffset(neighbors, offsetX, offsetY);
    if (!neighbor) return undefined;

    const neighborCol =
      offsetX < 0 ? neighbor.cols + col : offsetX > 0 ? col - this.cols : col;
    const neighborRow =
      offsetY < 0 ? neighbor.rows + row : offsetY > 0 ? row - this.rows : row;
    return neighbor.heightAtValidPixel(neighborCol, neighborRow);
  }

  private fallbackHeightAtPixelWithNeighbors(
    col: number,
    row: number,
    neighbors: TileMeshBoundaryNeighbors,
  ): number {
    if (col >= 0 && col < this.cols && row >= 0 && row < this.rows) {
      return this.heightAtSamplePixel(col, row);
    }

    const offsetX = col < 0 ? -1 : col >= this.cols ? 1 : 0;
    const offsetY = row < 0 ? -1 : row >= this.rows ? 1 : 0;
    const neighbor = neighborForOffset(neighbors, offsetX, offsetY);
    if (!neighbor) return this.heightAtSamplePixel(col, row);

    const neighborCol =
      offsetX < 0 ? neighbor.cols + col : offsetX > 0 ? col - this.cols : col;
    const neighborRow =
      offsetY < 0 ? neighbor.rows + row : offsetY > 0 ? row - this.rows : row;
    return neighbor.heightAtSamplePixel(neighborCol, neighborRow);
  }

  private heightAtSamplePixel(col: number, row: number): number {
    const safeCol = Math.min(this.cols - 1, Math.max(0, col));
    const safeRow = Math.min(this.rows - 1, Math.max(0, row));
    return this.elevations[safeRow * this.cols + safeCol] ?? 0;
  }

  private gridHeight(row: number, col: number): number {
    return this.geometryData.positions[(row * this.geometryData.gridCols + col) * 3 + 1] ?? 0;
  }

  private setGridHeight(row: number, col: number, height: number): void {
    this.geometryData.positions[(row * this.geometryData.gridCols + col) * 3 + 1] = height;
  }

  private setVertexNormal(index: number, x: number, y: number, z: number): void {
    const p = index * 3;
    const len = Math.hypot(x, y, z) || 1;
    this.geometryData.normals[p] = x / len;
    this.geometryData.normals[p + 1] = y / len;
    this.geometryData.normals[p + 2] = z / len;
  }

  private normalAtPixelCoordinate(pixelCol: number, pixelRow: number): { x: number; y: number; z: number } {
    const data = this.geometryData;
    const col = Math.min(data.gridCols - 1, Math.max(0, pixelCol));
    const row = Math.min(data.gridRows - 1, Math.max(0, pixelRow));
    const c0 = Math.floor(col);
    const r0 = Math.floor(row);
    const c1 = Math.min(c0 + 1, data.gridCols - 1);
    const r1 = Math.min(r0 + 1, data.gridRows - 1);
    const tx = col - c0;
    const ty = row - r0;

    const n00 = this.normalAtGridIndex(r0, c0);
    const n10 = this.normalAtGridIndex(r0, c1);
    const n01 = this.normalAtGridIndex(r1, c0);
    const n11 = this.normalAtGridIndex(r1, c1);
    const x0 = n00.x + (n10.x - n00.x) * tx;
    const y0 = n00.y + (n10.y - n00.y) * tx;
    const z0 = n00.z + (n10.z - n00.z) * tx;
    const x1 = n01.x + (n11.x - n01.x) * tx;
    const y1 = n01.y + (n11.y - n01.y) * tx;
    const z1 = n01.z + (n11.z - n01.z) * tx;
    const x = x0 + (x1 - x0) * ty;
    const y = y0 + (y1 - y0) * ty;
    const z = z0 + (z1 - z0) * ty;
    const len = Math.hypot(x, y, z) || 1;
    return { x: x / len, y: y / len, z: z / len };
  }

  private normalAtGridIndex(row: number, col: number): { x: number; y: number; z: number } {
    const p = (row * this.geometryData.gridCols + col) * 3;
    return {
      x: this.geometryData.normals[p] ?? 0,
      y: this.geometryData.normals[p + 1] ?? 1,
      z: this.geometryData.normals[p + 2] ?? 0,
    };
  }

  private updateNormalAt(row: number, col: number): void {
    const data = this.geometryData;
    const rowN = Math.max(row - 1, 0);
    const rowS = Math.min(row + 1, data.gridRows - 1);
    const colW = Math.max(col - 1, 0);
    const colE = Math.min(col + 1, data.gridCols - 1);
    const northZ = (rowS - rowN) * data.dz;
    const eastX = (colE - colW) * data.dx;
    const hN = this.gridHeight(rowN, col);
    const hS = this.gridHeight(rowS, col);
    const hW = this.gridHeight(row, colW);
    const hE = this.gridHeight(row, colE);
    const northY = hN - hS;
    const eastY = hE - hW;
    this.setVertexNormal(
      row * data.gridCols + col,
      -northZ * eastY,
      northZ * eastX,
      -northY * eastX,
    );
  }

  private updatePerimeterNormals(): void {
    const data = this.geometryData;
    const rows = boundaryBandIndices(data.gridRows - 1);
    const cols = boundaryBandIndices(data.gridCols - 1);

    for (const row of rows) {
      for (let col = 0; col < data.gridCols; col++) {
        this.updateNormalAt(row, col);
      }
    }
    for (let row = 0; row < data.gridRows; row++) {
      for (const col of cols) {
        this.updateNormalAt(row, col);
      }
    }
  }

  private updateColorAt(row: number, col: number): void {
    const data = this.geometryData;
    const index = row * data.gridCols + col;
    const p = index * 3;
    const color = elevationColor(this.gridHeight(row, col));
    data.colors[p] = color.r;
    data.colors[p + 1] = color.g;
    data.colors[p + 2] = color.b;
  }

  private updatePerimeterColors(): void {
    const data = this.geometryData;
    const rows = boundaryBandIndices(data.gridRows - 1);
    const cols = boundaryBandIndices(data.gridCols - 1);

    for (const row of rows) {
      for (let col = 0; col < data.gridCols; col++) {
        this.updateColorAt(row, col);
      }
    }
    for (let row = 0; row < data.gridRows; row++) {
      for (const col of cols) {
        this.updateColorAt(row, col);
      }
    }
  }

  private updateSkirts(stitched: {
    north: boolean;
    south: boolean;
    west: boolean;
    east: boolean;
  }): void {
    const data = this.geometryData;
    if (data.northSkirt.length === 0) return;
    const lastRow = data.gridRows - 1;
    const lastCol = data.gridCols - 1;
    for (let col = 0; col < data.gridCols; col++) {
      this.syncSkirtVertex(data.northSkirt[col]!, col, stitched.north);
      this.syncSkirtVertex(
        data.southSkirt[col]!,
        lastRow * data.gridCols + col,
        stitched.south,
      );
    }
    for (let row = 0; row < data.gridRows; row++) {
      this.syncSkirtVertex(data.westSkirt[row]!, row * data.gridCols, stitched.west);
      this.syncSkirtVertex(
        data.eastSkirt[row]!,
        row * data.gridCols + lastCol,
        stitched.east,
      );
    }
  }

  private syncSkirtVertex(skirtIndex: number, sourceIndex: number, stitched: boolean): void {
    const data = this.geometryData;
    const source = sourceIndex * 3;
    const target = skirtIndex * 3;
    data.positions[target] = data.positions[source]!;
    // Same-zoom internal tile edges already share the resampled boundary surface.
    // Keeping a 30m vertical skirt there renders as a dark/light band on GSI terrain.
    data.positions[target + 1] =
      data.positions[source + 1]! - (stitched ? 0 : data.skirtDepthM);
    data.positions[target + 2] = data.positions[source + 2]!;
    data.normals[target] = data.normals[source]!;
    data.normals[target + 1] = data.normals[source + 1]!;
    data.normals[target + 2] = data.normals[source + 2]!;
    data.colors[target] = data.colors[source]!;
    data.colors[target + 1] = data.colors[source + 1]!;
    data.colors[target + 2] = data.colors[source + 2]!;
  }
}

/**
 * 隣接する 2 枚の pixel-center TileMesh の間に、実際の DEM サンプル間セルを張る。
 *
 * 例: west tile の col255 と east tile の col0 は、共有境界の左右 0.5px にある
 * 通常の隣接サンプルなので、その 2 列を直接 quad strip で結ぶ。
 */
export class TileBridgeMesh {
  readonly mesh: Mesh;

  private readonly geometry: BufferGeometry;
  private readonly material: ReturnType<typeof buildTerrainMaterial>;

  constructor(a: TileMesh, b: TileMesh, direction: TileBridgeDirection) {
    const data = buildBridgeGeometry(a, b, direction);
    this.geometry = data.geometry;
    this.material = buildTerrainMaterial();
    this.mesh = new Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = true;
    this.mesh.name = `tile_bridge_${direction}_${a.key.z}_${a.key.x}_${a.key.y}`;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}

/** 4 枚のタイルが接する角にできる 1 セルを張る bridge。 */
export class TileCornerBridgeMesh {
  readonly mesh: Mesh;

  private readonly geometry: BufferGeometry;
  private readonly material: ReturnType<typeof buildTerrainMaterial>;

  constructor(northwest: TileMesh, northeast: TileMesh, southwest: TileMesh, southeast: TileMesh) {
    const data = buildCornerBridgeGeometry(northwest, northeast, southwest, southeast);
    this.geometry = data.geometry;
    this.material = buildTerrainMaterial();
    this.mesh = new Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = true;
    this.mesh.name = `tile_bridge_corner_${northwest.key.z}_${northwest.key.x}_${northwest.key.y}`;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}

/**
 * 異なる zoom の pixel-center TileMesh が接する辺を張る bridge。
 *
 * coarse 側と fine 側では edge 頂点数が違うため、共通 zoom 上の境界重なり区間で
 * 両 edge を同じ t 列に再サンプルして ruled surface として三角形化する。
 * DEM 高さを平均して合わせるのではなく、表示中の各 edge surface 上の点だけを使う。
 */
export class TileLodBridgeMesh {
  readonly mesh: Mesh;

  private readonly geometry: BufferGeometry;
  private readonly material: ReturnType<typeof buildTerrainMaterial>;

  constructor(a: TileLodBridgeSide, b: TileLodBridgeSide, span: TileLodBridgeSpan) {
    const data = buildLodBridgeGeometry(a, b, span);
    this.geometry = data.geometry;
    this.material = buildTerrainMaterial();
    this.mesh = new Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = true;
    this.mesh.name = `tile_lod_bridge_${a.edge}_${a.tile.key.z}_${a.tile.key.x}_${a.tile.key.y}_${b.edge}_${b.tile.key.z}_${b.tile.key.x}_${b.tile.key.y}`;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}

// -----------------------------------------------------------------------------
// DEM → BufferGeometry
// -----------------------------------------------------------------------------

interface Rgb {
  r: number;
  g: number;
  b: number;
}

/**
 * 標高 [m] から地表色を返す。旧 TSL マテリアルと同じ色帯を CPU 側で再現する。
 */
function elevationColor(h: number): Rgb {
  const green = { r: 0.28, g: 0.42, b: 0.20 };
  const brown = { r: 0.45, g: 0.34, b: 0.22 };
  const white = { r: 0.95, g: 0.95, b: 0.96 };

  const tBrown = smoothstep(400, 1400, h);
  const tWhite = smoothstep(2200, 3500, h);
  const base = mixRgb(green, brown, tBrown);
  return mixRgb(base, white, tWhite);
}

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

function mixRgb(a: Rgb, b: Rgb, t: number): Rgb {
  return {
    r: a.r + (b.r - a.r) * t,
    g: a.g + (b.g - a.g) * t,
    b: a.b + (b.b - a.b) * t,
  };
}

function boundaryBandIndices(lastIndex: number): number[] {
  return Array.from(
    new Set([
      0,
      Math.min(1, lastIndex),
      Math.max(lastIndex - 1, 0),
      lastIndex,
    ]),
  );
}

function gpuStableLocalCoordinate(worldCoordinate: number, meshWorldCoordinate: number): number {
  // WebGPU では matrix と vertex attribute が float32 になる。Hokota のように
  // ENU 原点から遠い tile では、隣接 tile が別々に `center + local` を計算すると
  // cm 単位の隙間が出るため、共有 world 境界を float32 化してから local へ戻す。
  const worldF32 = Math.fround(worldCoordinate);
  const meshF32 = Math.fround(meshWorldCoordinate);
  return Math.fround(worldF32 - meshF32);
}

function buildDemGeometry(
  data: TileData,
  widthM: number,
  heightM: number,
  centerX: number,
  centerZ: number,
  mode: TileMeshGeometryMode,
  skirtDepthM: number,
): DemGeometryData {
  const sampleCols = data.width;
  const sampleRows = data.height;
  const gridCols = mode === 'pixel-center' ? sampleCols : sampleCols + 1;
  const gridRows = mode === 'pixel-center' ? sampleRows : sampleRows + 1;
  const gridVertexCount = gridCols * gridRows;
  const useSkirt = mode === 'cell-boundary' && skirtDepthM > 0;
  const skirtVertexCount = useSkirt ? gridCols * 2 + gridRows * 2 : 0;
  const vertexCount = gridVertexCount + skirtVertexCount;

  const terrainQuadCount =
    mode === 'pixel-center'
      ? Math.max(0, sampleCols - 1) * Math.max(0, sampleRows - 1)
      : sampleCols * sampleRows;
  const skirtQuadCount = useSkirt ? sampleCols * 2 + sampleRows * 2 : 0;
  const indexCount = (terrainQuadCount + skirtQuadCount) * 6;

  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const indices = new Uint32Array(indexCount);

  // DEM の画素値はセル中心の値なので、セル境界 grid は 256 画素に対して 257 本。
  // ここを width / 255 にすると描画だけが 1 画素分引き伸ばされる。
  const dx = widthM / sampleCols;
  const dz = heightM / sampleRows;

  let vertexCursor = 0;
  let indexCursor = 0;

  const setVertex = (x: number, y: number, z: number, color: Rgb): number => {
    const i = vertexCursor++;
    const p = i * 3;
    positions[p] = x;
    positions[p + 1] = y;
    positions[p + 2] = z;
    colors[p] = color.r;
    colors[p + 1] = color.g;
    colors[p + 2] = color.b;
    return i;
  };

  const setNormal = (i: number, x: number, y: number, z: number): void => {
    const p = i * 3;
    const len = Math.hypot(x, y, z) || 1;
    normals[p] = x / len;
    normals[p + 1] = y / len;
    normals[p + 2] = z / len;
  };

  const addTri = (a: number, b: number, c: number): void => {
    indices[indexCursor++] = a;
    indices[indexCursor++] = b;
    indices[indexCursor++] = c;
  };

  // row 0 は PNG の上端 = 北端。ローカル z は北を正にする。
  for (let row = 0; row < gridRows; row++) {
    const pixelRow = mode === 'pixel-center' ? row : row - 0.5;
    const worldZ = centerZ + heightM * 0.5 - (pixelRow + 0.5) * dz;
    const z = gpuStableLocalCoordinate(worldZ, centerZ);
    for (let col = 0; col < gridCols; col++) {
      const pixelCol = mode === 'pixel-center' ? col : col - 0.5;
      const elevation = sampleTilePixelCoordinate(data, pixelCol, pixelRow);
      const worldX = centerX - widthM * 0.5 + (pixelCol + 0.5) * dx;
      const x = gpuStableLocalCoordinate(worldX, centerX);
      setVertex(x, elevation, z, elevationColor(elevation));
    }
  }

  // 地表法線は DEM 勾配から計算する。skirt 面を含めて computeVertexNormals() すると
  // タイル外周の法線が縦壁方向へ倒れ、境界が暗線として目立つため使わない。
  const gridHeight = (row: number, col: number): number => {
    return positions[(row * gridCols + col) * 3 + 1] ?? 0;
  };

  for (let row = 0; row < gridRows; row++) {
    const rowN = Math.max(row - 1, 0);
    const rowS = Math.min(row + 1, gridRows - 1);
    const northZ = (rowS - rowN) * dz;

    for (let col = 0; col < gridCols; col++) {
      const colW = Math.max(col - 1, 0);
      const colE = Math.min(col + 1, gridCols - 1);
      const eastX = (colE - colW) * dx;

      const hN = gridHeight(rowN, col);
      const hS = gridHeight(rowS, col);
      const hW = gridHeight(row, colW);
      const hE = gridHeight(row, colE);

      // north x east で、平坦面なら +Y を向く。
      const northY = hN - hS;
      const eastY = hE - hW;
      setNormal(
        row * gridCols + col,
        -northZ * eastY,
        northZ * eastX,
        -northY * eastX,
      );
    }
  }

  // 地表面。Three.js の FrontSide で見えるよう、平面時に法線が +Y になる順序で張る。
  for (let row = 0; row < gridRows - 1; row++) {
    for (let col = 0; col < gridCols - 1; col++) {
      const nw = row * gridCols + col;
      const ne = nw + 1;
      const sw = (row + 1) * gridCols + col;
      const se = sw + 1;
      addTri(nw, ne, sw);
      addTri(ne, se, sw);
    }
  }

  // Skirt は cell-boundary mode だけで使う。pixel-center mode では隣接タイル間を
  // TileBridgeMesh が実セルとして張るため、人工的な縦壁を作らない。
  const addSkirtVertex = (sourceIndex: number): number => {
    const p = sourceIndex * 3;
    const i = setVertex(
      positions[p]!,
      positions[p + 1]! - skirtDepthM,
      positions[p + 2]!,
      {
        r: colors[p]!,
        g: colors[p + 1]!,
        b: colors[p + 2]!,
      },
    );
    setNormal(i, normals[p]!, normals[p + 1]!, normals[p + 2]!);
    return i;
  };

  const northSkirt = new Array<number>();
  const southSkirt = new Array<number>();
  const westSkirt = new Array<number>();
  const eastSkirt = new Array<number>();
  if (useSkirt) {
    for (let col = 0; col < gridCols; col++) {
      northSkirt[col] = addSkirtVertex(col);
      southSkirt[col] = addSkirtVertex((gridRows - 1) * gridCols + col);
    }
    for (let col = 0; col < gridCols - 1; col++) {
      const n0 = col;
      const n1 = col + 1;
      const ns0 = northSkirt[col]!;
      const ns1 = northSkirt[col + 1]!;
      addTri(n0, ns0, n1);
      addTri(n1, ns0, ns1);

      const s0 = (gridRows - 1) * gridCols + col;
      const s1 = s0 + 1;
      const ss0 = southSkirt[col]!;
      const ss1 = southSkirt[col + 1]!;
      addTri(s0, s1, ss0);
      addTri(s1, ss1, ss0);
    }

    for (let row = 0; row < gridRows; row++) {
      westSkirt[row] = addSkirtVertex(row * gridCols);
      eastSkirt[row] = addSkirtVertex(row * gridCols + (gridCols - 1));
    }
    for (let row = 0; row < gridRows - 1; row++) {
      const w0 = row * gridCols;
      const w1 = (row + 1) * gridCols;
      const ws0 = westSkirt[row]!;
      const ws1 = westSkirt[row + 1]!;
      addTri(w0, w1, ws0);
      addTri(w1, ws1, ws0);

      const e0 = row * gridCols + (gridCols - 1);
      const e1 = (row + 1) * gridCols + (gridCols - 1);
      const es0 = eastSkirt[row]!;
      const es1 = eastSkirt[row + 1]!;
      addTri(e0, es0, e1);
      addTri(e1, es0, es1);
    }
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new Float32BufferAttribute(normals, 3));
  geometry.setAttribute('color', new Float32BufferAttribute(colors, 3));
  geometry.setIndex(new Uint32BufferAttribute(indices, 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return {
    geometry,
    positions,
    normals,
    colors,
    sampleCols,
    sampleRows,
    gridCols,
    gridRows,
    dx,
    dz,
    northSkirt,
    southSkirt,
    westSkirt,
    eastSkirt,
    skirtDepthM,
    mode,
  };
}

function buildBridgeGeometry(
  a: TileMesh,
  b: TileMesh,
  direction: TileBridgeDirection,
): { geometry: BufferGeometry } {
  const sampleCount =
    direction === 'east'
      ? Math.min(a.pixelRows, b.pixelRows)
      : Math.min(a.pixelCols, b.pixelCols);
  const vertexCount = sampleCount * 2;
  const indexCount = Math.max(0, sampleCount - 1) * 6;

  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const indices = new Uint32Array(indexCount);

  const setVertex = (
    index: number,
    vertex: { x: number; y: number; z: number },
    normal: { x: number; y: number; z: number },
  ): void => {
    const p = index * 3;
    positions[p] = Math.fround(vertex.x);
    positions[p + 1] = Math.fround(vertex.y);
    positions[p + 2] = Math.fround(vertex.z);
    normals[p] = normal.x;
    normals[p + 1] = normal.y;
    normals[p + 2] = normal.z;
    const color = elevationColor(vertex.y);
    colors[p] = color.r;
    colors[p + 1] = color.g;
    colors[p + 2] = color.b;
  };

  for (let i = 0; i < sampleCount; i++) {
    if (direction === 'east') {
      setVertex(
        i * 2,
        a.pixelCenterVertex(a.pixelCols - 1, i),
        a.pixelCenterNormal(a.pixelCols - 1, i),
      );
      setVertex(
        i * 2 + 1,
        b.pixelCenterVertex(0, i),
        b.pixelCenterNormal(0, i),
      );
    } else {
      setVertex(
        i * 2,
        a.pixelCenterVertex(i, a.pixelRows - 1),
        a.pixelCenterNormal(i, a.pixelRows - 1),
      );
      setVertex(
        i * 2 + 1,
        b.pixelCenterVertex(i, 0),
        b.pixelCenterNormal(i, 0),
      );
    }
  }

  let indexCursor = 0;
  const addTri = (ia: number, ib: number, ic: number): void => {
    indices[indexCursor++] = ia;
    indices[indexCursor++] = ib;
    indices[indexCursor++] = ic;
  };

  for (let i = 0; i < sampleCount - 1; i++) {
    const nw = i * 2;
    const sw = nw + 1;
    const ne = (i + 1) * 2;
    const se = ne + 1;

    if (direction === 'east') {
      // east bridge のペアは [westEdge, eastEdge]。i は north->south に進む。
      addTri(nw, sw, ne);
      addTri(ne, sw, se);
    } else {
      // south bridge のペアは [northEdge, southEdge]。i は west->east に進む。
      addTri(nw, ne, sw);
      addTri(ne, se, sw);
    }
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new Float32BufferAttribute(normals, 3));
  geometry.setAttribute('color', new Float32BufferAttribute(colors, 3));
  geometry.setIndex(new Uint32BufferAttribute(indices, 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return { geometry };
}

function buildCornerBridgeGeometry(
  northwest: TileMesh,
  northeast: TileMesh,
  southwest: TileMesh,
  southeast: TileMesh,
): { geometry: BufferGeometry } {
  const vertices = [
    northwest.pixelCenterVertex(northwest.pixelCols - 1, northwest.pixelRows - 1),
    northeast.pixelCenterVertex(0, northeast.pixelRows - 1),
    southwest.pixelCenterVertex(southwest.pixelCols - 1, 0),
    southeast.pixelCenterVertex(0, 0),
  ];
  const positions = new Float32Array(12);
  const normals = new Float32Array(12);
  const colors = new Float32Array(12);
  const vertexNormals = [
    northwest.pixelCenterNormal(northwest.pixelCols - 1, northwest.pixelRows - 1),
    northeast.pixelCenterNormal(0, northeast.pixelRows - 1),
    southwest.pixelCenterNormal(southwest.pixelCols - 1, 0),
    southeast.pixelCenterNormal(0, 0),
  ];
  for (let i = 0; i < vertices.length; i++) {
    const vertex = vertices[i]!;
    const normal = vertexNormals[i]!;
    const p = i * 3;
    positions[p] = Math.fround(vertex.x);
    positions[p + 1] = Math.fround(vertex.y);
    positions[p + 2] = Math.fround(vertex.z);
    normals[p] = normal.x;
    normals[p + 1] = normal.y;
    normals[p + 2] = normal.z;
    const color = elevationColor(vertex.y);
    colors[p] = color.r;
    colors[p + 1] = color.g;
    colors[p + 2] = color.b;
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new Float32BufferAttribute(normals, 3));
  geometry.setAttribute('color', new Float32BufferAttribute(colors, 3));
  geometry.setIndex(new Uint32BufferAttribute(new Uint32Array([0, 1, 2, 1, 3, 2]), 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return { geometry };
}

function buildLodBridgeGeometry(
  a: TileLodBridgeSide,
  b: TileLodBridgeSide,
  span: TileLodBridgeSpan,
): { geometry: BufferGeometry } {
  const ordered = orderLodBridgeSides(a, b, span.axis);
  const events = collectLodBridgeEvents(ordered.primary.tile, ordered.secondary.tile, span);
  const vertexCount = events.length * 2;
  const indexCount = Math.max(0, events.length - 1) * 6;
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const indices = new Uint32Array(indexCount);

  const setVertex = (
    index: number,
    side: TileLodBridgeSide,
    t: number,
  ): void => {
    const vertex = side.tile.edgeVertexAt(side.edge, t);
    const normal = side.tile.edgeNormalAt(side.edge, t);
    const p = index * 3;
    positions[p] = Math.fround(vertex.x);
    positions[p + 1] = Math.fround(vertex.y);
    positions[p + 2] = Math.fround(vertex.z);
    normals[p] = normal.x;
    normals[p + 1] = normal.y;
    normals[p + 2] = normal.z;
    const color = elevationColor(vertex.y);
    colors[p] = color.r;
    colors[p + 1] = color.g;
    colors[p + 2] = color.b;
  };

  for (let i = 0; i < events.length; i++) {
    const event = events[i]!;
    setVertex(i * 2, ordered.primary, axisTForTile(ordered.primary.tile.key, span.axis, span.commonZoom, event));
    setVertex(
      i * 2 + 1,
      ordered.secondary,
      axisTForTile(ordered.secondary.tile.key, span.axis, span.commonZoom, event),
    );
  }

  let indexCursor = 0;
  const addTri = (ia: number, ib: number, ic: number): void => {
    indices[indexCursor++] = ia;
    indices[indexCursor++] = ib;
    indices[indexCursor++] = ic;
  };

  for (let i = 0; i < events.length - 1; i++) {
    const p0 = i * 2;
    const s0 = p0 + 1;
    const p1 = (i + 1) * 2;
    const s1 = p1 + 1;

    if (span.axis === 'y') {
      // primary = west side, secondary = east side, event order = north -> south.
      addTri(p0, s0, p1);
      addTri(p1, s0, s1);
    } else {
      // primary = north side, secondary = south side, event order = west -> east.
      addTri(p0, p1, s0);
      addTri(p1, s1, s0);
    }
  }

  const geometry = new BufferGeometry();
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new Float32BufferAttribute(normals, 3));
  geometry.setAttribute('color', new Float32BufferAttribute(colors, 3));
  geometry.setIndex(new Uint32BufferAttribute(indices, 1));
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return { geometry };
}

function orderLodBridgeSides(
  a: TileLodBridgeSide,
  b: TileLodBridgeSide,
  axis: TileLodBridgeAxis,
): { primary: TileLodBridgeSide; secondary: TileLodBridgeSide } {
  if (axis === 'y') {
    if (a.edge === 'east' && b.edge === 'west') return { primary: a, secondary: b };
    if (a.edge === 'west' && b.edge === 'east') return { primary: b, secondary: a };
  } else {
    if (a.edge === 'south' && b.edge === 'north') return { primary: a, secondary: b };
    if (a.edge === 'north' && b.edge === 'south') return { primary: b, secondary: a };
  }
  return { primary: a, secondary: b };
}

function collectLodBridgeEvents(
  a: TileMesh,
  b: TileMesh,
  span: TileLodBridgeSpan,
): number[] {
  const events = [span.start, span.end];
  addTileAxisPixelCenters(events, a.key, span.axis, span.commonZoom, span.start, span.end);
  addTileAxisPixelCenters(events, b.key, span.axis, span.commonZoom, span.start, span.end);
  events.sort((v0, v1) => v0 - v1);

  const unique: number[] = [];
  const eps = 1e-8;
  for (const event of events) {
    const clamped = Math.min(span.end, Math.max(span.start, event));
    if (unique.length > 0 && Math.abs(unique[unique.length - 1]! - clamped) <= eps) continue;
    unique.push(clamped);
  }
  return unique.length >= 2 ? unique : [span.start, span.end];
}

function addTileAxisPixelCenters(
  events: number[],
  key: TileKey,
  axis: TileLodBridgeAxis,
  commonZoom: number,
  spanStart: number,
  spanEnd: number,
): void {
  const range = tileAxisRangeAtZoom(key, axis, commonZoom);
  const pixelCount = TILE_PIXELS;
  const eps = 1e-8;
  for (let i = 0; i < pixelCount; i++) {
    const t = (i + 0.5) / pixelCount;
    const event = range.start + (range.end - range.start) * t;
    if (event <= spanStart + eps || event >= spanEnd - eps) continue;
    events.push(event);
  }
}

function axisTForTile(
  key: TileKey,
  axis: TileLodBridgeAxis,
  commonZoom: number,
  event: number,
): number {
  const range = tileAxisRangeAtZoom(key, axis, commonZoom);
  return (event - range.start) / (range.end - range.start);
}

function tileAxisRangeAtZoom(
  key: TileKey,
  axis: TileLodBridgeAxis,
  commonZoom: number,
): { start: number; end: number } {
  const scale = 2 ** (commonZoom - key.z);
  const tileIndex = axis === 'x' ? key.x : key.y;
  return {
    start: tileIndex * scale,
    end: (tileIndex + 1) * scale,
  };
}

function sampleTilePixelCoordinate(data: TileData, pixelCol: number, pixelRow: number): number {
  const c0 = Math.floor(pixelCol);
  const r0 = Math.floor(pixelRow);
  const c1 = c0 + 1;
  const r1 = r0 + 1;
  const tx = pixelCol - c0;
  const ty = pixelRow - r0;

  const h00 = tileElevationAt(data, c0, r0);
  const h10 = tileElevationAt(data, c1, r0);
  const h01 = tileElevationAt(data, c0, r1);
  const h11 = tileElevationAt(data, c1, r1);
  const h0 = h00 + (h10 - h00) * tx;
  const h1 = h01 + (h11 - h01) * tx;
  return h0 + (h1 - h0) * ty;
}

function tileElevationAt(data: TileData, col: number, row: number): number {
  const safeCol = Math.min(data.width - 1, Math.max(0, col));
  const safeRow = Math.min(data.height - 1, Math.max(0, row));
  return data.elevations[safeRow * data.width + safeCol] ?? 0;
}

function neighborForOffset(
  neighbors: TileMeshBoundaryNeighbors,
  offsetX: number,
  offsetY: number,
): TileMesh | undefined {
  if (offsetX === 0 && offsetY < 0) return neighbors.north;
  if (offsetX === 0 && offsetY > 0) return neighbors.south;
  if (offsetX < 0 && offsetY === 0) return neighbors.west;
  if (offsetX > 0 && offsetY === 0) return neighbors.east;
  if (offsetX < 0 && offsetY < 0) return neighbors.northwest;
  if (offsetX > 0 && offsetY < 0) return neighbors.northeast;
  if (offsetX < 0 && offsetY > 0) return neighbors.southwest;
  if (offsetX > 0 && offsetY > 0) return neighbors.southeast;
  return undefined;
}

function makeBoundarySignature(neighbors: TileMeshBoundaryNeighbors): string {
  return [
    tileMeshKey(neighbors.north),
    tileMeshKey(neighbors.south),
    tileMeshKey(neighbors.west),
    tileMeshKey(neighbors.east),
    tileMeshKey(neighbors.northwest),
    tileMeshKey(neighbors.northeast),
    tileMeshKey(neighbors.southwest),
    tileMeshKey(neighbors.southeast),
  ].join('|');
}

function tileMeshKey(tile: TileMesh | undefined): string {
  if (!tile) return '';
  return `${tile.key.z}/${tile.key.x}/${tile.key.y}`;
}
