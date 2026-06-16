// =============================================================================
// BuildingObstacleLayer
// -----------------------------------------------------------------------------
// MVP import layer for simple building obstacles.
// Reads GeoJSON Polygon/MultiPolygon footprints and renders them as simple
// extruded building meshes in the app's ENU world coordinates.
// =============================================================================

import {
  BufferGeometry,
  DoubleSide,
  Float32BufferAttribute,
  Group,
  Mesh,
  MeshStandardMaterial,
  ShapeUtils,
  Uint32BufferAttribute,
  Vector2,
} from 'three/webgpu';

import { lonLatToENU, withAoiCenter, type AoiCenter } from '../geo/tileMath';
import type {
  MpmHeightfieldObstacle,
  MpmSideBoundaryObstacle,
  MpmSideBoundarySample,
} from '../mpm/boundary/MpmBoundary';

type GeoJsonPosition = [number, number, number?];
type LinearRing = GeoJsonPosition[];
type PolygonCoordinates = LinearRing[];
type MultiPolygonCoordinates = PolygonCoordinates[];

export interface BuildingGeoJsonFeature {
  type: 'Feature';
  id?: string;
  properties?: {
    baseHeight?: number;
    topHeight?: number;
    height?: number;
    measuredHeight?: number;
    [key: string]: unknown;
  };
  geometry:
    | { type: 'Polygon'; coordinates: PolygonCoordinates }
    | { type: 'MultiPolygon'; coordinates: MultiPolygonCoordinates };
}

export interface BuildingGeoJsonMetadataBounds {
  minLon: number;
  minLat: number;
  maxLon: number;
  maxLat: number;
  minHeight: number;
  maxHeight: number;
}

export interface BuildingGeoJson {
  type: 'FeatureCollection';
  name?: string;
  crs?: {
    type: 'name';
    properties: { name: string };
  };
  metadata?: {
    sourceZip?: string;
    sourceEntries?: string[];
    sourceEntry?: string;
    verticalDatum?: string;
    coordinateOrder?: string;
    featureCount?: number;
    bounds: BuildingGeoJsonMetadataBounds;
  };
  features: BuildingGeoJsonFeature[];
}

export interface BuildingLayerBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  minZ: number;
  maxZ: number;
}

export interface BuildingLayerHorizontalBounds {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
}

interface BuildingObstacleShape {
  points: Array<{ x: number; z: number }>;
  bounds: { minX: number; maxX: number; minZ: number; maxZ: number };
  baseHeight: number;
  topHeight: number;
  winding: 1 | -1;
}

interface BuildingObstacleLayerOptions {
  aoiCenter?: AoiCenter;
  placement?: 'absolute' | 'terrain-draped';
  terrainHeightSampler?: (worldX: number, worldZ: number) => number | undefined;
  // 表示専用 chunk では MPM 用の重い footprint index を作らない。
  // 計算用 layer は Reset/Apply 時に計算領域に掛かる建物だけで別途作る。
  buildMpmObstacles?: boolean;
  includeBounds?: BuildingLayerHorizontalBounds;
}

const BUILDING_OBSTACLE_INDEX_CELL_SIZE_M = 64;
const MAX_BUILDING_LAYER_FEATURES = 200_000;
const MAX_BUILDING_LAYER_RINGS = 300_000;
const MAX_BUILDING_LAYER_RING_VERTICES = 20_000;
const MAX_BUILDING_LAYER_VERTICES = 2_000_000;
const MAX_BUILDING_LAYER_INDICES = 6_000_000;

export class BuildingObstacleLayer extends Group implements MpmHeightfieldObstacle, MpmSideBoundaryObstacle {
  readonly mpmBoundaryType = 'object';
  readonly bounds: BuildingLayerBounds;
  readonly buildingCount: number;

  private readonly mesh: Mesh;
  private readonly material: MeshStandardMaterial;
  private readonly obstacleIndex: Map<string, BuildingObstacleShape[]>;
  private readonly obstacleIndexCellSizeM: number;

  private constructor(params: {
    geometry: BufferGeometry;
    bounds: BuildingLayerBounds;
    buildingCount: number;
    obstacleIndex: Map<string, BuildingObstacleShape[]>;
    obstacleIndexCellSizeM: number;
  }) {
    super();
    this.name = 'BuildingObstacleLayer';
    this.bounds = params.bounds;
    this.buildingCount = params.buildingCount;
    this.obstacleIndex = params.obstacleIndex;
    this.obstacleIndexCellSizeM = params.obstacleIndexCellSizeM;
    this.material = new MeshStandardMaterial({
      color: 0xb7c1c8,
      roughness: 0.82,
      metalness: 0.0,
      // PLATEAU の footprint はリング向きがデータにより変わる。
      // 片面描画だと裏向き面が消え、建物が破片のように見えるため両面表示にする。
      side: DoubleSide,
    });
    this.mesh = new Mesh(params.geometry, this.material);
    this.mesh.name = 'Imported building obstacles';
    this.add(this.mesh);
  }

  static async load(
    url: string,
    options: BuildingObstacleLayerOptions = {},
  ): Promise<BuildingObstacleLayer> {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`building geojson fetch failed: ${res.status} ${res.statusText}`);
    }
    const geojson = (await res.json()) as BuildingGeoJson;
    return BuildingObstacleLayer.fromGeoJson(geojson, options);
  }

  static fromGeoJson(
    geojson: BuildingGeoJson,
    options: BuildingObstacleLayerOptions = {},
  ): BuildingObstacleLayer {
    if (options.aoiCenter) {
      const { aoiCenter, ...nextOptions } = options;
      return withAoiCenter(aoiCenter, () =>
        BuildingObstacleLayer.fromGeoJson(geojson, nextOptions),
      );
    }

    const positions: number[] = [];
    const indices: number[] = [];
    const obstacleShapes: BuildingObstacleShape[] = [];
    const buildMpmObstacles = options.buildMpmObstacles !== false;
    const includeBounds = options.includeBounds;
    const bounds = emptyBounds();
    let buildingCount = 0;
    let ringCount = 0;

    for (const feature of geojson.features) {
      if (buildingCount >= MAX_BUILDING_LAYER_FEATURES) {
        throw new Error(`building feature 数が上限 ${MAX_BUILDING_LAYER_FEATURES} を超えています`);
      }
      const sourceBaseHeight = finiteNumber(feature.properties?.baseHeight);
      const sourceTopHeight =
        finiteNumber(feature.properties?.topHeight) ??
        (sourceBaseHeight !== undefined
          ? sourceBaseHeight + (finiteNumber(feature.properties?.height) ?? 8)
          : undefined);
      if (
        sourceBaseHeight === undefined ||
        sourceTopHeight === undefined ||
        sourceTopHeight <= sourceBaseHeight
      ) {
        continue;
      }
      const buildingHeight = Math.max(
        1,
        finiteNumber(feature.properties?.measuredHeight) ??
          sourceTopHeight - sourceBaseHeight,
      );

      const polygons =
        feature.geometry.type === 'Polygon'
          ? [feature.geometry.coordinates]
          : feature.geometry.coordinates;
      let addedForFeature = false;
      for (const polygon of polygons) {
        const outerRing = polygon[0];
        if (!outerRing || outerRing.length < 4) continue;
        validateRingCoordinates(outerRing);
        const points = cleanRing(outerRing).map(([lon, lat, sourceHeight]) => {
          const enu = lonLatToENU(lon, lat);
          const sourceBaseHeightAtPoint = finiteNumber(sourceHeight);
          return sourceBaseHeightAtPoint === undefined
            ? { x: enu.x, z: enu.y }
            : { x: enu.x, z: enu.y, sourceHeight: sourceBaseHeightAtPoint };
        });
        if (points.length < 3) continue;
        if (points.length > MAX_BUILDING_LAYER_RING_VERTICES) {
          throw new Error(`building ring vertex 数が上限 ${MAX_BUILDING_LAYER_RING_VERTICES} を超えています`);
        }
        const footprint = footprintBounds(points);
        if (includeBounds && !horizontalBoundsIntersect(footprint, includeBounds)) {
          continue;
        }
        ringCount++;
        if (ringCount > MAX_BUILDING_LAYER_RINGS) {
          throw new Error(`building ring 数が上限 ${MAX_BUILDING_LAYER_RINGS} を超えています`);
        }

        const baseHeights = placedBaseHeights(
          points,
          sourceBaseHeight,
          options,
        );
        const baseHeight = Math.min(...baseHeights);
        const topHeight = Math.max(...baseHeights) + buildingHeight;
        appendExtrudedPolygon(
          positions,
          indices,
          points,
          baseHeights,
          topHeight,
          bounds,
        );
        if (positions.length / 3 > MAX_BUILDING_LAYER_VERTICES) {
          throw new Error(`building geometry vertex 数が上限 ${MAX_BUILDING_LAYER_VERTICES} を超えています`);
        }
        if (indices.length > MAX_BUILDING_LAYER_INDICES) {
          throw new Error(`building geometry index 数が上限 ${MAX_BUILDING_LAYER_INDICES} を超えています`);
        }
        if (buildMpmObstacles) {
          obstacleShapes.push({
            points: points.map((point) => ({ x: point.x, z: point.z })),
            bounds: footprint,
            baseHeight,
            topHeight,
            winding: polygonWinding(points),
          });
        }
        addedForFeature = true;
      }
      if (addedForFeature) buildingCount++;
    }

    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new Float32BufferAttribute(positions, 3));
    geometry.setIndex(new Uint32BufferAttribute(indices, 1));
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();

    const obstacleIndex = buildMpmObstacles
      ? buildObstacleIndex(
          obstacleShapes,
          BUILDING_OBSTACLE_INDEX_CELL_SIZE_M,
        )
      : new Map<string, BuildingObstacleShape[]>();

    return new BuildingObstacleLayer({
      geometry,
      bounds,
      buildingCount,
      obstacleIndex,
      obstacleIndexCellSizeM: BUILDING_OBSTACLE_INDEX_CELL_SIZE_M,
    });
  }

  sampleHeight(
    worldX: number,
    worldZ: number,
    influenceRadiusM = 0,
  ): number | undefined {
    let height: number | undefined = undefined;
    const shapes = candidateObstacleShapes(
      this.obstacleIndex,
      this.obstacleIndexCellSizeM,
      worldX,
      worldZ,
      influenceRadiusM,
    );
    for (const shape of shapes) {
      if (!footprintMayContainSample(shape, worldX, worldZ, influenceRadiusM)) {
        continue;
      }
      // MPM heightfield は点サンプルなので、格子点が footprint のすぐ外側でも
      // 建物がその cell に掛かっていれば障害物として拾う。
      if (
        pointInFootprint(worldX, worldZ, shape.points) ||
        (influenceRadiusM > 0 &&
          distanceToFootprint(worldX, worldZ, shape.points) <= influenceRadiusM)
      ) {
        height = Math.max(height ?? Number.NEGATIVE_INFINITY, shape.topHeight);
      }
    }
    return height;
  }

  sampleSideBoundary(
    worldX: number,
    worldZ: number,
    influenceRadiusM = 0,
  ): MpmSideBoundarySample | undefined {
    let bestSample: MpmSideBoundarySample | undefined;
    let bestScore = Number.POSITIVE_INFINITY;
    const shapes = candidateObstacleShapes(
      this.obstacleIndex,
      this.obstacleIndexCellSizeM,
      worldX,
      worldZ,
      influenceRadiusM,
    );
    for (const shape of shapes) {
      if (!footprintMayContainSample(shape, worldX, worldZ, influenceRadiusM)) {
        continue;
      }

      const inside = pointInFootprint(worldX, worldZ, shape.points);
      const edge = nearestFootprintEdge(worldX, worldZ, shape);
      if (edge === undefined) continue;

      const distance = Math.sqrt(edge.distanceSq);
      if (!inside && distance > influenceRadiusM) {
        continue;
      }

      const normal = sideBoundaryNormal(worldX, worldZ, inside, edge);
      const signedDistanceM = inside ? -distance : distance;
      const score = signedDistanceM < 0 ? -1_000_000 + distance : distance;
      if (score >= bestScore) continue;

      bestScore = score;
      bestSample = {
        signedDistanceM,
        normal,
        baseHeight: shape.baseHeight,
        topHeight: shape.topHeight,
      };
    }

    return bestSample;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    this.material.dispose();
  }
}

function buildObstacleIndex(
  shapes: BuildingObstacleShape[],
  cellSizeM: number,
): Map<string, BuildingObstacleShape[]> {
  const index = new Map<string, BuildingObstacleShape[]>();
  for (const shape of shapes) {
    const minX = obstacleIndexCell(shape.bounds.minX, cellSizeM);
    const maxX = obstacleIndexCell(shape.bounds.maxX, cellSizeM);
    const minZ = obstacleIndexCell(shape.bounds.minZ, cellSizeM);
    const maxZ = obstacleIndexCell(shape.bounds.maxZ, cellSizeM);
    for (let ix = minX; ix <= maxX; ix++) {
      for (let iz = minZ; iz <= maxZ; iz++) {
        const key = obstacleIndexKey(ix, iz);
        const bucket = index.get(key);
        if (bucket) {
          bucket.push(shape);
        } else {
          index.set(key, [shape]);
        }
      }
    }
  }
  return index;
}

function candidateObstacleShapes(
  index: Map<string, BuildingObstacleShape[]>,
  cellSizeM: number,
  x: number,
  z: number,
  radiusM: number,
): BuildingObstacleShape[] {
  if (index.size === 0) return [];

  const minX = obstacleIndexCell(x - radiusM, cellSizeM);
  const maxX = obstacleIndexCell(x + radiusM, cellSizeM);
  const minZ = obstacleIndexCell(z - radiusM, cellSizeM);
  const maxZ = obstacleIndexCell(z + radiusM, cellSizeM);
  const seen = new Set<BuildingObstacleShape>();
  const candidates: BuildingObstacleShape[] = [];
  for (let ix = minX; ix <= maxX; ix++) {
    for (let iz = minZ; iz <= maxZ; iz++) {
      const bucket = index.get(obstacleIndexKey(ix, iz));
      if (!bucket) continue;
      for (const shape of bucket) {
        if (seen.has(shape)) continue;
        seen.add(shape);
        candidates.push(shape);
      }
    }
  }
  return candidates;
}

function obstacleIndexCell(value: number, cellSizeM: number): number {
  return Math.floor(value / cellSizeM);
}

function obstacleIndexKey(x: number, z: number): string {
  return `${x}/${z}`;
}

function horizontalBoundsIntersect(
  a: BuildingLayerHorizontalBounds,
  b: BuildingLayerHorizontalBounds,
): boolean {
  return (
    a.minX <= b.maxX &&
    a.maxX >= b.minX &&
    a.minZ <= b.maxZ &&
    a.maxZ >= b.minZ
  );
}

function footprintBounds(
  points: Array<{ x: number; z: number }>,
): BuildingObstacleShape['bounds'] {
  const bounds = {
    minX: Number.POSITIVE_INFINITY,
    maxX: Number.NEGATIVE_INFINITY,
    minZ: Number.POSITIVE_INFINITY,
    maxZ: Number.NEGATIVE_INFINITY,
  };
  for (const point of points) {
    bounds.minX = Math.min(bounds.minX, point.x);
    bounds.maxX = Math.max(bounds.maxX, point.x);
    bounds.minZ = Math.min(bounds.minZ, point.z);
    bounds.maxZ = Math.max(bounds.maxZ, point.z);
  }
  return bounds;
}

function footprintMayContainSample(
  shape: BuildingObstacleShape,
  x: number,
  z: number,
  radius: number,
): boolean {
  return (
    x >= shape.bounds.minX - radius &&
    x <= shape.bounds.maxX + radius &&
    z >= shape.bounds.minZ - radius &&
    z <= shape.bounds.maxZ + radius
  );
}

function pointInFootprint(
  x: number,
  z: number,
  points: Array<{ x: number; z: number }>,
): boolean {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const a = points[i]!;
    const b = points[j]!;
    const crossesZ = a.z > z !== b.z > z;
    if (!crossesZ) continue;
    const edgeX = ((b.x - a.x) * (z - a.z)) / (b.z - a.z) + a.x;
    if (x < edgeX) inside = !inside;
  }
  return inside;
}

function distanceToFootprint(
  x: number,
  z: number,
  points: Array<{ x: number; z: number }>,
): number {
  let minDistanceSq = Number.POSITIVE_INFINITY;
  for (let i = 0; i < points.length; i++) {
    const a = points[i]!;
    const b = points[(i + 1) % points.length]!;
    minDistanceSq = Math.min(minDistanceSq, pointSegmentDistanceSq(x, z, a, b));
  }
  return Math.sqrt(minDistanceSq);
}

interface NearestFootprintEdge {
  distanceSq: number;
  closestX: number;
  closestZ: number;
  fallbackNormal: { x: number; z: number };
}

function nearestFootprintEdge(
  x: number,
  z: number,
  shape: BuildingObstacleShape,
): NearestFootprintEdge | undefined {
  let best: NearestFootprintEdge | undefined;
  for (let i = 0; i < shape.points.length; i++) {
    const a = shape.points[i]!;
    const b = shape.points[(i + 1) % shape.points.length]!;
    const closest = closestPointOnSegment(x, z, a, b);
    const distanceSq = (x - closest.x) ** 2 + (z - closest.z) ** 2;
    if (best && distanceSq >= best.distanceSq) continue;

    const edgeDx = b.x - a.x;
    const edgeDz = b.z - a.z;
    const edgeLen = Math.hypot(edgeDx, edgeDz);
    if (edgeLen <= 1e-12) continue;

    // x/z 平面の polygon winding から、ring の向きに依存しない外向き法線を作る。
    const fallbackNormal =
      shape.winding > 0
        ? { x: edgeDz / edgeLen, z: -edgeDx / edgeLen }
        : { x: -edgeDz / edgeLen, z: edgeDx / edgeLen };
    best = {
      distanceSq,
      closestX: closest.x,
      closestZ: closest.z,
      fallbackNormal,
    };
  }
  return best;
}

function sideBoundaryNormal(
  x: number,
  z: number,
  inside: boolean,
  edge: NearestFootprintEdge,
): { x: number; z: number } {
  const dx = inside ? edge.closestX - x : x - edge.closestX;
  const dz = inside ? edge.closestZ - z : z - edge.closestZ;
  const len = Math.hypot(dx, dz);
  if (len > 1e-6) {
    return { x: dx / len, z: dz / len };
  }
  return edge.fallbackNormal;
}

function pointSegmentDistanceSq(
  x: number,
  z: number,
  a: { x: number; z: number },
  b: { x: number; z: number },
): number {
  const closest = closestPointOnSegment(x, z, a, b);
  return (x - closest.x) ** 2 + (z - closest.z) ** 2;
}

function closestPointOnSegment(
  x: number,
  z: number,
  a: { x: number; z: number },
  b: { x: number; z: number },
): { x: number; z: number } {
  const dx = b.x - a.x;
  const dz = b.z - a.z;
  const lengthSq = dx * dx + dz * dz;
  if (lengthSq <= 1e-12) {
    return { x: a.x, z: a.z };
  }
  const t = Math.min(1, Math.max(0, ((x - a.x) * dx + (z - a.z) * dz) / lengthSq));
  return {
    x: a.x + dx * t,
    z: a.z + dz * t,
  };
}

function polygonWinding(points: Array<{ x: number; z: number }>): 1 | -1 {
  let area2 = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i]!;
    const b = points[(i + 1) % points.length]!;
    area2 += a.x * b.z - b.x * a.z;
  }
  return area2 >= 0 ? 1 : -1;
}

function placedBaseHeights(
  points: Array<{ x: number; z: number; sourceHeight?: number }>,
  sourceBaseHeight: number,
  options: BuildingObstacleLayerOptions,
): number[] {
  if (options.placement === 'terrain-draped' && options.terrainHeightSampler) {
    return points.map((point) => {
      const terrainHeight = options.terrainHeightSampler?.(point.x, point.z);
      if (terrainHeight !== undefined && Number.isFinite(terrainHeight)) {
        return terrainHeight;
      }
      return point.sourceHeight ?? sourceBaseHeight;
    });
  }

  return points.map((point) => point.sourceHeight ?? sourceBaseHeight);
}

function appendExtrudedPolygon(
  positions: number[],
  indices: number[],
  points: Array<{ x: number; z: number }>,
  baseHeights: number[],
  topHeight: number,
  bounds: BuildingLayerBounds,
): void {
  const baseIndex = positions.length / 3;

  // bottom ring: absolute GeoJSON altitude if present, otherwise scalar base;
  // in PLATEAU sample mode this ring is draped to DEM height per footprint vertex.
  for (let i = 0; i < points.length; i++) {
    const p = points[i]!;
    const baseHeight = baseHeights[i]!;
    positions.push(p.x, baseHeight, p.z);
    includePoint(bounds, p.x, baseHeight, p.z);
  }

  // top ring: keep a flat roof for the MVP extrusion. The height comes from
  // CityGML measuredHeight when available, so the roof is not an arbitrary lift.
  for (const p of points) {
    positions.push(p.x, topHeight, p.z);
    includePoint(bounds, p.x, topHeight, p.z);
  }

  const n = points.length;

  appendTopFace(indices, baseIndex + n, points);

  // Side faces.
  for (let i = 0; i < n; i++) {
    const next = (i + 1) % n;
    const b0 = baseIndex + i;
    const b1 = baseIndex + next;
    const t0 = baseIndex + n + i;
    const t1 = baseIndex + n + next;
    indices.push(b0, b1, t1, b0, t1, t0);
  }
}

function cleanRing(ring: LinearRing): LinearRing {
  if (ring.length < 2) return ring;
  const withoutClosingPoint = (() => {
    const first = ring[0]!;
    const last = ring[ring.length - 1]!;
    if (first[0] === last[0] && first[1] === last[1]) return ring.slice(0, -1);
    return ring;
  })();

  const cleaned: LinearRing = [];
  for (const point of withoutClosingPoint) {
    const previous = cleaned[cleaned.length - 1];
    if (previous && previous[0] === point[0] && previous[1] === point[1]) continue;
    cleaned.push(point);
  }
  return cleaned;
}

function validateRingCoordinates(ring: LinearRing): void {
  for (const [lon, lat, height] of ring) {
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
      throw new Error('building GeoJSON に非有限の緯度経度が含まれています');
    }
    if (lon < -180 || lon > 180 || lat < -90 || lat > 90) {
      throw new Error(`building GeoJSON 座標が範囲外です: lon=${lon}, lat=${lat}`);
    }
    if (height !== undefined && !Number.isFinite(height)) {
      throw new Error('building GeoJSON に非有限の高さが含まれています');
    }
  }
}

function appendTopFace(
  indices: number[],
  topBaseIndex: number,
  points: Array<{ x: number; z: number }>,
): void {
  const triangles = triangulateFootprint(points);
  if (triangles.length > 0) {
    for (const triangle of triangles) {
      const [a, b, c] = triangle;
      if (a === undefined || b === undefined || c === undefined) continue;
      indices.push(topBaseIndex + a, topBaseIndex + b, topBaseIndex + c);
    }
    return;
  }

  // 入力が自己交差などで triangulation できない場合の最後の fallback。
  // この経路は凸またはほぼ凸の footprint だけを想定する。
  for (let i = 1; i < points.length - 1; i++) {
    indices.push(topBaseIndex, topBaseIndex + i, topBaseIndex + i + 1);
  }
}

function triangulateFootprint(points: Array<{ x: number; z: number }>): number[][] {
  if (points.length < 3) return [];
  const origin = points[0]!;
  const contour = points.map((point) => new Vector2(point.x - origin.x, point.z - origin.z));
  return ShapeUtils.triangulateShape(contour, []);
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function emptyBounds(): BuildingLayerBounds {
  return {
    minX: Number.POSITIVE_INFINITY,
    maxX: Number.NEGATIVE_INFINITY,
    minY: Number.POSITIVE_INFINITY,
    maxY: Number.NEGATIVE_INFINITY,
    minZ: Number.POSITIVE_INFINITY,
    maxZ: Number.NEGATIVE_INFINITY,
  };
}

function includePoint(
  bounds: BuildingLayerBounds,
  x: number,
  y: number,
  z: number,
): void {
  bounds.minX = Math.min(bounds.minX, x);
  bounds.maxX = Math.max(bounds.maxX, x);
  bounds.minY = Math.min(bounds.minY, y);
  bounds.maxY = Math.max(bounds.maxY, y);
  bounds.minZ = Math.min(bounds.minZ, z);
  bounds.maxZ = Math.max(bounds.maxZ, z);
}
