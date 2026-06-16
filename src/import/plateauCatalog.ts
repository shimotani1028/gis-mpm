// =============================================================================
// PLATEAU CKAN catalog helper
// -----------------------------------------------------------------------------
// G空間情報センター CKAN API から PLATEAU CityGML の一覧だけを取得する。
// ZIP 本体はここでは落とさず、UI で選ばれた 1 件だけを main.ts 側で取得する。
// =============================================================================

import { validatePlateauZipUrl } from './plateauZipUrlPolicy';

const CKAN_ACTION_BASE = 'https://www.geospatial.jp/ckan/api/3/action/';
const PLATEAU_CATALOG_ROWS = 1000;
const PREFECTURE_ORDER = [
  '北海道',
  '青森県',
  '岩手県',
  '宮城県',
  '秋田県',
  '山形県',
  '福島県',
  '茨城県',
  '栃木県',
  '群馬県',
  '埼玉県',
  '千葉県',
  '東京都',
  '神奈川県',
  '新潟県',
  '富山県',
  '石川県',
  '福井県',
  '山梨県',
  '長野県',
  '岐阜県',
  '静岡県',
  '愛知県',
  '三重県',
  '滋賀県',
  '京都府',
  '大阪府',
  '兵庫県',
  '奈良県',
  '和歌山県',
  '鳥取県',
  '島根県',
  '岡山県',
  '広島県',
  '山口県',
  '徳島県',
  '香川県',
  '愛媛県',
  '高知県',
  '福岡県',
  '佐賀県',
  '長崎県',
  '熊本県',
  '大分県',
  '宮崎県',
  '鹿児島県',
  '沖縄県',
] as const;
const PREFECTURE_ORDER_INDEX = new Map<string, number>(
  PREFECTURE_ORDER.map((prefecture, index) => [prefecture, index]),
);

interface CkanActionResponse<T> {
  success: boolean;
  result?: T;
  error?: {
    message?: string;
    __type?: string;
  };
}

interface CkanPackageSearchResult {
  count: number;
  results: unknown[];
}

interface CkanPackageShowResult {
  name?: unknown;
  title?: unknown;
  area?: unknown;
  resources?: unknown[];
}

interface RawCkanResource {
  id?: unknown;
  name?: unknown;
  format?: unknown;
  url?: unknown;
  size?: unknown;
  description?: unknown;
  position?: unknown;
}

export interface PlateauCatalogDataset {
  name: string;
  title: string;
  label: string;
  cityKey: string;
  cityLabel: string;
  prefectureLabel: string;
  municipalityLabel: string;
  area: string;
  yearLabel: string;
  year?: string;
  metadataModified?: string;
}

export interface PlateauCatalogSearchResult {
  total: number;
  datasets: PlateauCatalogDataset[];
}

export interface PlateauCatalogResource {
  datasetName: string;
  datasetTitle: string;
  datasetLabel: string;
  name: string;
  label: string;
  url: string;
  fileName: string;
  sizeBytes?: number;
}

export async function loadPlateauCatalogDatasets(): Promise<PlateauCatalogSearchResult> {
  const url = new URL('package_search', CKAN_ACTION_BASE);
  // PLATEAU + CityGML tag に絞る。検索欄は設けず、都道府県/市町村/年度 select で選ばせる。
  url.searchParams.set('q', 'PLATEAU');
  url.searchParams.set('fq', 'tags:CityGML');
  url.searchParams.set('rows', String(PLATEAU_CATALOG_ROWS));
  // resources や thumbnail_url を含めると一覧だけで重くなるため、検索結果は最小限にする。
  url.searchParams.set('fl', 'name,title,area,metadata_modified');

  const result = await fetchCkanAction<CkanPackageSearchResult>(url);
  const datasets = result.results
    .map(readCatalogDataset)
    .filter((dataset): dataset is PlateauCatalogDataset => dataset !== undefined)
    .sort(compareDatasets);

  return {
    total: result.count,
    datasets,
  };
}

export async function loadPlateauCatalogResources(
  datasetName: string,
): Promise<PlateauCatalogResource[]> {
  const url = new URL('package_show', CKAN_ACTION_BASE);
  url.searchParams.set('id', datasetName);

  const result = await fetchCkanAction<CkanPackageShowResult>(url);
  const dataset = readCatalogDataset(result) ?? {
    name: datasetName,
    title: datasetName,
    cityKey: datasetName,
    cityLabel: datasetName,
    prefectureLabel: '都道府県不明',
    municipalityLabel: datasetName,
    area: '',
    yearLabel: '',
    label: datasetName,
  };
  const resources = Array.isArray(result.resources) ? result.resources : [];

  return resources
    .map((raw) => readCityGmlZipResource(raw, dataset))
    .filter((resource): resource is PlateauCatalogResource => resource !== undefined)
    .sort(compareCityGmlResources);
}

function readCatalogDataset(raw: unknown): PlateauCatalogDataset | undefined {
  if (!isRecord(raw)) return undefined;
  const name = readString(raw.name);
  const title = readString(raw.title) ?? name;
  if (!name || !title) return undefined;

  const area = readString(raw.area) ?? '';
  if (hasPartialUpdateMarker(name, title, area)) return undefined;
  const year = yearFromDataset(name, title);
  const location = datasetLocation(title, area);
  const cityLabel = `${location.prefectureLabel}_${location.municipalityLabel}`;
  const yearLabel = year ?? '年度不明';
  const cityKey = cityLabel;
  const label = `${cityLabel} ${yearLabel}`;
  const dataset: PlateauCatalogDataset = {
    name,
    title,
    label,
    cityKey,
    cityLabel,
    prefectureLabel: location.prefectureLabel,
    municipalityLabel: location.municipalityLabel,
    area,
    yearLabel,
  };
  const metadataModified = readString(raw.metadata_modified);
  if (year) dataset.year = year;
  if (metadataModified) dataset.metadataModified = metadataModified;
  return dataset;
}

function readCityGmlZipResource(
  raw: unknown,
  dataset: PlateauCatalogDataset,
): PlateauCatalogResource | undefined {
  if (!isRecord(raw)) return undefined;
  const resource = raw as RawCkanResource;
  const name = readString(resource.name);
  const rawUrl = readString(resource.url);
  const url = validatePlateauZipUrl(rawUrl);
  if (!name || !url || !isCityGmlZipResource(resource, name, url)) return undefined;
  const description = readString(resource.description);
  if (hasPartialUpdateMarker(name, url, description)) return undefined;

  const sizeBytes = readFiniteNumber(resource.size);
  const fileName = fileNameFromUrl(url, `${dataset.name}-${name}`);
  const label = sizeBytes === undefined
    ? name
    : `${name} (${formatBytes(sizeBytes)})`;

  const item: PlateauCatalogResource = {
    datasetName: dataset.name,
    datasetTitle: dataset.title,
    datasetLabel: dataset.label,
    name,
    label,
    url,
    fileName,
  };
  if (sizeBytes !== undefined) item.sizeBytes = sizeBytes;
  return item;
}

function isCityGmlZipResource(
  resource: RawCkanResource,
  name: string,
  url: string,
): boolean {
  const format = readString(resource.format)?.toLowerCase() ?? '';
  const isZip = format === 'zip' || /\.zip(?:$|[?#])/i.test(url);
  return isZip && /citygml/i.test(name);
}

function hasPartialUpdateMarker(...values: Array<string | undefined>): boolean {
  return values.some((value) => value !== undefined && /一部更新/.test(value));
}

function compareDatasets(a: PlateauCatalogDataset, b: PlateauCatalogDataset): number {
  const prefectureCompare = comparePrefectures(a.prefectureLabel, b.prefectureLabel);
  if (prefectureCompare !== 0) return prefectureCompare;
  const municipalityCompare = a.municipalityLabel.localeCompare(b.municipalityLabel, 'ja');
  if (municipalityCompare !== 0) return municipalityCompare;
  const yearCompare = (b.year ?? '').localeCompare(a.year ?? '');
  if (yearCompare !== 0) return yearCompare;
  return a.title.localeCompare(b.title, 'ja');
}

function compareCityGmlResources(
  a: PlateauCatalogResource,
  b: PlateauCatalogResource,
): number {
  const versionCompare = resourceVersionRank(b.name) - resourceVersionRank(a.name);
  if (versionCompare !== 0) return versionCompare;
  return a.name.localeCompare(b.name, 'ja');
}

function resourceVersionRank(name: string): number {
  const match = /v(\d+)/i.exec(name);
  if (!match) return 0;
  const value = Number(match[1]);
  return Number.isFinite(value) ? value : 0;
}

function datasetLocation(
  title: string,
  area: string,
): { prefectureLabel: string; municipalityLabel: string } {
  const cityFromTitle = cityNameFromTitle(title);
  const cityFromArea = cityNameFromArea(area);
  const rawCity = cityFromTitle ?? cityFromArea ?? title;
  const splitCity = splitPrefectureMunicipality(rawCity);
  const splitArea = splitPrefectureMunicipality(area);
  const prefectureLabel =
    splitCity?.prefectureLabel ??
    prefectureNameFromArea(area) ??
    splitArea?.prefectureLabel ??
    '都道府県不明';
  const municipalityLabel =
    splitCity?.municipalityLabel ??
    stripPrefecture(rawCity, prefectureLabel) ??
    cityFromArea ??
    title;

  return {
    prefectureLabel,
    municipalityLabel: municipalityLabel || rawCity,
  };
}

function yearFromDataset(name: string, title: string): string | undefined {
  return /(\d{4})年度/.exec(title)?.[1] ?? /-(\d{4})$/.exec(name)?.[1];
}

function cityNameFromTitle(title: string): string | undefined {
  return /Project PLATEAU）(.+?)（\d{4}年度）/.exec(title)?.[1];
}

function cityNameFromArea(area: string): string | undefined {
  const parts = area.split(/[,_、]/).map((part) => part.trim()).filter(Boolean);
  if (parts.length <= 1) {
    return splitPrefectureMunicipality(area)?.municipalityLabel;
  }
  return parts[parts.length - 1];
}

function prefectureNameFromArea(area: string): string | undefined {
  const parts = area.split(/[,_、]/).map((part) => part.trim()).filter(Boolean);
  const partPrefecture = parts.find((part) => PREFECTURE_ORDER_INDEX.has(part));
  return partPrefecture ?? splitPrefectureMunicipality(area)?.prefectureLabel;
}

function splitPrefectureMunicipality(
  value: string,
): { prefectureLabel: string; municipalityLabel: string } | undefined {
  const trimmed = value.trim();
  if (!trimmed) return undefined;

  const underscoreParts = trimmed.split(/[_＿]/).map((part) => part.trim()).filter(Boolean);
  if (underscoreParts.length >= 2 && PREFECTURE_ORDER_INDEX.has(underscoreParts[0]!)) {
    return {
      prefectureLabel: underscoreParts[0]!,
      municipalityLabel: underscoreParts.slice(1).join('_'),
    };
  }

  const prefecture = PREFECTURE_ORDER.find((item) => trimmed.startsWith(item));
  if (!prefecture) return undefined;
  const municipality = stripPrefecture(trimmed, prefecture);
  if (!municipality) return undefined;
  return {
    prefectureLabel: prefecture,
    municipalityLabel: municipality,
  };
}

function stripPrefecture(value: string, prefecture: string): string | undefined {
  const trimmed = value.trim();
  if (!trimmed.startsWith(prefecture)) return undefined;
  const municipality = trimmed
    .slice(prefecture.length)
    .replace(/^[_＿,\s、]+/, '')
    .trim();
  return municipality || undefined;
}

function comparePrefectures(a: string, b: string): number {
  const aIndex = PREFECTURE_ORDER_INDEX.get(a);
  const bIndex = PREFECTURE_ORDER_INDEX.get(b);
  if (aIndex !== undefined && bIndex !== undefined) return aIndex - bIndex;
  if (aIndex !== undefined) return -1;
  if (bIndex !== undefined) return 1;
  return a.localeCompare(b, 'ja');
}

function fileNameFromUrl(url: string, fallbackBase: string): string {
  try {
    const pathName = new URL(url).pathname;
    const lastSegment = pathName.split('/').filter(Boolean).pop();
    if (lastSegment && /\.zip$/i.test(lastSegment)) return decodeURIComponent(lastSegment);
  } catch {
    // URL として解釈できない場合だけ fallback を使う。
  }
  return `${safeFileName(fallbackBase)}.zip`;
}

function safeFileName(value: string): string {
  return value
    .trim()
    .replace(/[\\/:*?"<>|]+/g, '-')
    .replace(/\s+/g, '_')
    .slice(0, 120);
}

function formatBytes(value: number): string {
  if (value >= 1000 * 1000 * 1000) return `${(value / (1000 * 1000 * 1000)).toFixed(1)}GB`;
  if (value >= 1000 * 1000) return `${(value / (1000 * 1000)).toFixed(0)}MB`;
  if (value >= 1000) return `${(value / 1000).toFixed(0)}KB`;
  return `${value}B`;
}

async function fetchCkanAction<T>(url: URL): Promise<T> {
  const res = await fetch(url.toString(), { mode: 'cors' });
  if (!res.ok) {
    throw new Error(`CKAN API failed: ${res.status} ${res.statusText}`);
  }
  const payload = (await res.json()) as CkanActionResponse<T>;
  if (!payload.success || payload.result === undefined) {
    const message = payload.error?.message ?? payload.error?.__type ?? 'unknown CKAN error';
    throw new Error(`CKAN API error: ${message}`);
  }
  return payload.result;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function readString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() !== '' ? value.trim() : undefined;
}

function readFiniteNumber(value: unknown): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return undefined;
  return value;
}
