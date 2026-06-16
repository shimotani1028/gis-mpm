import type {
  BuildingGeoJson,
  BuildingGeoJsonMetadataBounds,
} from './BuildingObstacleLayer';

const MIB = 1024 * 1024;

// CityGML の XML 解析は worker でも main thread fallback でも同じ制限を使う。
// 巨大な自治体 ZIP では「正しい XML だがブラウザで扱うには大きすぎる」ケースがある。
const MAX_CITYGML_BUILDINGS = 200_000;
const MAX_CITYGML_RINGS = 300_000;
const MAX_CITYGML_RING_VERTICES = 20_000;
const MAX_CITYGML_TOTAL_VERTICES = 2_000_000;
const MAX_CITYGML_POSITION_VALUES = 10_000_000;
const MAX_POSLIST_TEXT_BYTES = 4 * MIB;

interface CityGmlPoint {
  lat: number;
  lon: number;
  height?: number;
}

interface BuildingRing {
  coords: Array<[number, number, number?]>;
  heights: number[];
}

export interface CityGmlImportBudget {
  rings: number;
  vertices: number;
  positionValues: number;
}

export interface ParsedCityGmlBuildingFeatures {
  features: BuildingGeoJson['features'];
  budget: CityGmlImportBudget;
}

export function emptyCityGmlImportBudget(): CityGmlImportBudget {
  return {
    rings: 0,
    vertices: 0,
    positionValues: 0,
  };
}

export function addCityGmlImportBudget(
  total: CityGmlImportBudget,
  addition: CityGmlImportBudget,
): void {
  total.rings += addition.rings;
  total.vertices += addition.vertices;
  total.positionValues += addition.positionValues;

  // worker は 1 entry 内の上限を守る。main thread 側では chunk 全体の累積上限を守る。
  if (total.rings > MAX_CITYGML_RINGS) {
    throw new Error(`GML ring 数が上限 ${MAX_CITYGML_RINGS} を超えています`);
  }
  if (total.vertices > MAX_CITYGML_TOTAL_VERTICES) {
    throw new Error(`GML vertex 総数が上限 ${MAX_CITYGML_TOTAL_VERTICES} を超えています`);
  }
  if (total.positionValues > MAX_CITYGML_POSITION_VALUES) {
    throw new Error(`GML posList 数値総数が上限 ${MAX_CITYGML_POSITION_VALUES} を超えています`);
  }
}

export function assertCityGmlFeatureCount(featureCount: number): void {
  if (featureCount > MAX_CITYGML_BUILDINGS) {
    throw new Error(`建物 feature 数が上限 ${MAX_CITYGML_BUILDINGS} を超えています`);
  }
}

export function parseCityGmlBuildingFeatures(
  gml: string,
  sourceEntry: string,
  featureIndexOffset = 0,
): ParsedCityGmlBuildingFeatures {
  const features: BuildingGeoJson['features'] = [];
  const budget = emptyCityGmlImportBudget();
  const buildingRe = /<bldg:Building\b[\s\S]*?<\/bldg:Building>/g;

  for (const match of gml.matchAll(buildingRe)) {
    assertCityGmlFeatureCount(featureIndexOffset + features.length + 1);
    const feature = buildFeature(
      match[0],
      sourceEntry,
      featureIndexOffset + features.length,
      budget,
    );
    if (feature) features.push(feature);
  }

  return { features, budget };
}

export function boundsForBuildingFeatures(
  features: BuildingGeoJson['features'],
): BuildingGeoJsonMetadataBounds {
  const bounds = {
    minLon: Number.POSITIVE_INFINITY,
    minLat: Number.POSITIVE_INFINITY,
    maxLon: Number.NEGATIVE_INFINITY,
    maxLat: Number.NEGATIVE_INFINITY,
    minHeight: Number.POSITIVE_INFINITY,
    maxHeight: Number.NEGATIVE_INFINITY,
  };

  for (const feature of features) {
    const polygons =
      feature.geometry.type === 'Polygon'
        ? [feature.geometry.coordinates]
        : feature.geometry.coordinates;
    for (const polygon of polygons) {
      for (const ring of polygon) {
        for (const [lon, lat] of ring) {
          bounds.minLon = Math.min(bounds.minLon, lon);
          bounds.maxLon = Math.max(bounds.maxLon, lon);
          bounds.minLat = Math.min(bounds.minLat, lat);
          bounds.maxLat = Math.max(bounds.maxLat, lat);
        }
      }
    }
    bounds.minHeight = Math.min(bounds.minHeight, feature.properties?.baseHeight ?? bounds.minHeight);
    bounds.maxHeight = Math.max(bounds.maxHeight, feature.properties?.topHeight ?? bounds.maxHeight);
  }

  return bounds;
}

function buildFeature(
  buildingXml: string,
  sourceEntry: string,
  index: number,
  budget: CityGmlImportBudget,
): BuildingGeoJson['features'][number] | undefined {
  const rings = ringsFromFootprint(buildingXml, budget);
  if (rings.length === 0) return undefined;

  const id = attr(buildingXml, 'gml:id') ?? `building-${index}`;
  const buildingId = textContentByLocalName(buildingXml, 'buildingID');
  const measured = measuredHeight(buildingXml);
  const solidRange = heightRangeFromSolid(buildingXml, budget);
  const footprintHeights = rings.flatMap((ring) => ring.heights);
  const baseHeight = solidRange?.min ?? finiteMin(footprintHeights);
  const topHeight =
    solidRange?.max ??
    (baseHeight !== undefined ? baseHeight + (measured ?? 8) : undefined);

  if (
    baseHeight === undefined ||
    topHeight === undefined ||
    !Number.isFinite(baseHeight) ||
    !Number.isFinite(topHeight) ||
    topHeight <= baseHeight
  ) {
    return undefined;
  }

  const properties: NonNullable<BuildingGeoJson['features'][number]['properties']> = {
    gmlId: id,
    source: sourceEntry,
    baseHeight,
    topHeight,
    height: topHeight - baseHeight,
    heightMode: 'absolute',
  };
  if (buildingId !== undefined) properties.buildingId = buildingId;
  if (measured !== undefined) properties.measuredHeight = measured;

  const geometry =
    rings.length === 1
      ? {
          type: 'Polygon' as const,
          coordinates: [rings[0]!.coords],
        }
      : {
          type: 'MultiPolygon' as const,
          coordinates: rings.map((ring) => [ring.coords]),
        };

  return {
    type: 'Feature',
    id,
    properties,
    geometry,
  };
}

function ringsFromFootprint(
  buildingXml: string,
  budget: CityGmlImportBudget,
): BuildingRing[] {
  // PLATEAU LOD1 は lod0FootPrint を持つデータと lod0RoofEdge だけのデータがある。
  // MPM 障害物では水平 footprint が必要なので、FootPrint 優先、なければ RoofEdge を使う。
  const footprint =
    elementByLocalName(buildingXml, 'lod0FootPrint') ??
    elementByLocalName(buildingXml, 'lod0RoofEdge');
  if (!footprint) return [];

  const rings: BuildingRing[] = [];
  const posListRe =
    /<(?:[\w.-]+:)?posList(?=[\s>])([^>]*)>([\s\S]*?)<\/(?:[\w.-]+:)?posList>/g;
  for (const match of footprint.matchAll(posListRe)) {
    const triples = parsePositions(
      match[2] ?? '',
      Number(attr(match[1] ?? '', 'srsDimension')),
      budget,
    );
    if (triples.length < 4) continue;
    if (triples.length > MAX_CITYGML_RING_VERTICES) {
      throw new Error(`GML ring vertex 数が上限 ${MAX_CITYGML_RING_VERTICES} を超えています`);
    }
    const coords = closeRing(
      triples.map((point) =>
        point.height === undefined
          ? ([point.lon, point.lat] as [number, number])
          : ([point.lon, point.lat, point.height] as [number, number, number]),
      ),
    );
    const heights = triples
      .map((point) => point.height)
      .filter((height): height is number => height !== undefined);
    budget.rings++;
    budget.vertices += coords.length;
    if (budget.rings > MAX_CITYGML_RINGS) {
      throw new Error(`GML ring 数が上限 ${MAX_CITYGML_RINGS} を超えています`);
    }
    if (budget.vertices > MAX_CITYGML_TOTAL_VERTICES) {
      throw new Error(`GML vertex 総数が上限 ${MAX_CITYGML_TOTAL_VERTICES} を超えています`);
    }
    rings.push({ coords, heights });
  }
  return rings;
}

function heightRangeFromSolid(
  buildingXml: string,
  budget: CityGmlImportBudget,
): { min: number; max: number } | undefined {
  const solid = elementByLocalName(buildingXml, 'lod1Solid');
  if (!solid) return undefined;

  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  const posListRe =
    /<(?:[\w.-]+:)?posList(?=[\s>])([^>]*)>([\s\S]*?)<\/(?:[\w.-]+:)?posList>/g;
  for (const match of solid.matchAll(posListRe)) {
    for (const point of parsePositions(
      match[2] ?? '',
      Number(attr(match[1] ?? '', 'srsDimension')),
      budget,
    )) {
      if (point.height === undefined) continue;
      min = Math.min(min, point.height);
      max = Math.max(max, point.height);
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return undefined;
  return { min, max };
}

function parsePositions(
  posListText: string,
  explicitDimension: number,
  budget: CityGmlImportBudget,
): CityGmlPoint[] {
  if (posListText.length > MAX_POSLIST_TEXT_BYTES) {
    throw new Error(`GML posList が上限 ${formatBytes(MAX_POSLIST_TEXT_BYTES)} を超えています`);
  }
  const trimmed = posListText.trim();
  if (!trimmed) return [];
  const values = trimmed
    .split(/\s+/)
    .map((value) => Number(value));
  if (values.length === 0) return [];
  if (!values.every((value) => Number.isFinite(value))) {
    throw new Error('GML posList に非数値が含まれています');
  }
  budget.positionValues += values.length;
  if (budget.positionValues > MAX_CITYGML_POSITION_VALUES) {
    throw new Error(`GML posList 数値総数が上限 ${MAX_CITYGML_POSITION_VALUES} を超えています`);
  }

  const dimension =
    explicitDimension === 2 || explicitDimension === 3
      ? explicitDimension
      : values.length % 3 === 0
        ? 3
        : 2;
  const points: CityGmlPoint[] = [];
  for (let i = 0; i + dimension - 1 < values.length; i += dimension) {
    const lat = values[i]!;
    const lon = values[i + 1]!;
    const height = dimension >= 3 ? values[i + 2] : undefined;
    validateCityGmlPoint(lat, lon, height);
    points.push(height === undefined ? { lat, lon } : { lat, lon, height });
  }
  return points;
}

function validateCityGmlPoint(lat: number, lon: number, height: number | undefined): void {
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
    throw new Error(`GML 座標が緯度経度の範囲外です: lat=${lat}, lon=${lon}`);
  }
  if (height !== undefined && !Number.isFinite(height)) {
    throw new Error('GML height に非有限値が含まれています');
  }
}

function closeRing<T extends [number, number, number?]>(coords: T[]): T[] {
  if (coords.length === 0) return coords;
  const first = coords[0]!;
  const last = coords[coords.length - 1]!;
  if (first[0] === last[0] && first[1] === last[1]) return coords;
  return [...coords, first];
}

function measuredHeight(buildingXml: string): number | undefined {
  const value = textContentByLocalName(buildingXml, 'measuredHeight');
  if (value === undefined) return undefined;
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}

function finiteMin(values: number[]): number | undefined {
  let min = Number.POSITIVE_INFINITY;
  for (const value of values) {
    if (Number.isFinite(value)) min = Math.min(min, value);
  }
  return Number.isFinite(min) ? min : undefined;
}

function elementByLocalName(xml: string, localName: string): string | undefined {
  const re = new RegExp(
    `<(?:[\\w.-]+:)?${localName}\\b[\\s\\S]*?<\\/(?:[\\w.-]+:)?${localName}>`,
  );
  return re.exec(xml)?.[0];
}

function textContentByLocalName(xml: string, localName: string): string | undefined {
  const re = new RegExp(
    `<(?:[\\w.-]+:)?${localName}(?=[\\s>])[^>]*>([\\s\\S]*?)<\\/(?:[\\w.-]+:)?${localName}>`,
  );
  return re.exec(xml)?.[1]?.trim();
}

function attr(xml: string, attrName: string): string | undefined {
  const re = new RegExp(`${attrName}="([^"]+)"`);
  return re.exec(xml)?.[1];
}

function formatBytes(value: number): string {
  if (value >= 1024 * 1024 * 1024) return `${(value / (1024 * 1024 * 1024)).toFixed(1)}GiB`;
  if (value >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(0)}MiB`;
  if (value >= 1024) return `${(value / 1024).toFixed(0)}KiB`;
  return `${value}B`;
}
