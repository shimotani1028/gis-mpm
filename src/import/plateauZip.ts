import type {
  BuildingGeoJson,
  BuildingGeoJsonMetadataBounds,
} from './BuildingObstacleLayer';
import {
  addCityGmlImportBudget,
  assertCityGmlFeatureCount,
  boundsForBuildingFeatures,
  emptyCityGmlImportBudget,
  parseCityGmlBuildingFeatures,
} from './plateauCityGmlParser';
import type {
  CityGmlImportBudget,
  ParsedCityGmlBuildingFeatures,
} from './plateauCityGmlParser';
import type { PlateauCityGmlWorkerResponse } from './plateauCityGmlWorkerTypes';

interface ZipEntry {
  name: string;
  compressionMethod: number;
  compressedSize: number;
  uncompressedSize: number;
  localHeaderOffset: number;
}

export interface PlateauZipImportOptions {
  onProgress?: (progress: PlateauZipProgress) => void;
  entrySelection?: PlateauZipEntrySelectionOptions;
  excludeEntryNames?: ReadonlySet<string>;
}

export interface PlateauZipEntrySelectionOptions {
  // ZIP 内の標準地域メッシュ群の中心付近から、初期表示用に読む矩形の一辺。
  // ここでは XML を読まず、GML entry 名の mesh code だけで候補を絞る。
  centerRegionLengthM?: number;
  // 将来、地図クリックや保存済み AOI から開始位置を指定したい場合の上書き用。
  center?: {
    lat: number;
    lon: number;
  };
  // カメラ移動時の追加読み込みでは、既に見えている ENU 範囲を緯度経度へ戻して指定する。
  bounds?: PlateauZipLatLonBounds;
  paddingM?: number;
  // 大都市では可視範囲に多数の GML entry が重なる。一度に展開すると UI が固まり、
  // safety budget にも当たるため、追加読み込みでは近い entry から小分けに読む。
  maxSelectedEntryCount?: number;
  maxSelectedUncompressedBytes?: number;
  maxSelectedCompressedBytes?: number;
  // 追加読み込みで「画面中央の GML mesh が未読」の場合だけ使う近傍予算。
  // 通常時は狭い読み込みを維持し、中央に穴がある時だけ 3x3 相当をまとめて埋める。
  centerUnloadedNeighborhoodEntryCount?: number;
  centerUnloadedNeighborhoodUncompressedBytes?: number;
  centerUnloadedNeighborhoodCompressedBytes?: number;
}

export interface PlateauZipLatLonBounds {
  minLat: number;
  maxLat: number;
  minLon: number;
  maxLon: number;
}

export interface PlateauZipSource {
  fileName: string;
  size: number;
  readRange(offset: number, length: number): Promise<ArrayBuffer>;
}

export interface PlateauZipProgress {
  message: string;
  loaded?: number;
  total?: number;
  indeterminate?: boolean;
}

export interface PlateauZipImportResult {
  manifest: {
    version: 1;
    name: string;
    description: string;
    crs: 'EPSG:6697';
    verticalDatum: 'TP';
    bounds: BuildingGeoJsonMetadataBounds;
    terrain: {
      type: 'raster-dem-tilejson';
      encoding: 'terrarium';
      tilejson: string;
      urlTemplate: string;
    };
    obstacles: Array<{
      type: 'geojson';
      path: string;
      heightMode: 'absolute';
      baseHeightProperty: 'baseHeight';
      topHeightProperty: 'topHeight';
    }>;
  };
  geojson: BuildingGeoJson;
  sourceFileName: string;
  sourceEntries: string[];
}

export interface PlateauZipBuildingChunk {
  geojson?: BuildingGeoJson;
  sourceFileName: string;
  sourceEntries: string[];
}

const EOCD_SIGNATURE = 0x06054b50;
const CENTRAL_DIRECTORY_SIGNATURE = 0x02014b50;
const LOCAL_FILE_HEADER_SIGNATURE = 0x04034b50;
const ZIP_STORED = 0;
const ZIP_DEFLATED = 8;
const ZIP64_SENTINEL_32 = 0xffffffff;
const ZIP64_SENTINEL_16 = 0xffff;
const ZIP64_EOCD_LOCATOR_SIGNATURE = 0x07064b50;
const ZIP64_EOCD_RECORD_SIGNATURE = 0x06064b50;
const ZIP64_EXTENDED_INFO_ID = 0x0001;

const textDecoder = new TextDecoder('utf-8');
const MIB = 1024 * 1024;

// Browser 上で扱う import なので、ZIP/GML は「正しい形式でも大きすぎるもの」を明示的に拒否する。
// PLATEAU の municipality ZIP は巨大でも、アプリが必要とするのは udx/bldg の GML だけ。
const MAX_CENTRAL_DIRECTORY_BYTES = 128 * MIB;
const MAX_ZIP_ENTRY_COUNT = 200_000;
const MAX_BUILDING_ENTRY_COUNT = 20_000;
const MAX_BUILDING_ENTRY_COMPRESSED_BYTES = 256 * MIB;
const MAX_BUILDING_ENTRY_UNCOMPRESSED_BYTES = 512 * MIB;
const MAX_TOTAL_BUILDING_COMPRESSED_BYTES = 1024 * MIB;
const MAX_TOTAL_BUILDING_UNCOMPRESSED_BYTES = 1024 * MIB;
const MAX_ZIP_COMPRESSION_RATIO = 200;
const zipDirectoryCache = new WeakMap<PlateauZipSource, Promise<ZipEntry[]>>();

interface BuildingEntryWithMeshBounds {
  entry: ZipEntry;
  bounds: PlateauZipLatLonBounds;
}

export async function importPlateauZipFile(
  file: File,
  options: PlateauZipImportOptions = {},
): Promise<PlateauZipImportResult> {
  return importPlateauZipSource(
    {
      fileName: file.name,
      size: file.size,
      readRange: (offset, length) => readBlobRange(file, offset, length),
    },
    options,
  );
}

export async function importPlateauZipSource(
  source: PlateauZipSource,
  options: PlateauZipImportOptions = {},
): Promise<PlateauZipImportResult> {
  const chunk = await importPlateauZipSourceChunk(source, options);
  if (!chunk?.geojson) {
    throw new Error('建物 footprint を抽出できませんでした');
  }

  const bounds = chunk.geojson.metadata!.bounds;
  return {
    manifest: {
      version: 1,
      name: displayNameForZip(source.fileName),
      description: `PLATEAU import generated in browser from ${source.fileName}.`,
      crs: 'EPSG:6697',
      verticalDatum: 'TP',
      bounds,
      terrain: {
        type: 'raster-dem-tilejson',
        encoding: 'terrarium',
        tilejson:
          'https://tile.plateauview.mlit.go.jp/terrarium/tilejson.json?format=png&geoid=none',
        urlTemplate:
          'https://tile.plateauview.mlit.go.jp/terrarium/{z}/{x}/{y}.png?geoid=none',
      },
      obstacles: [
        {
          type: 'geojson',
          path: 'drag-drop-buildings.geojson',
          heightMode: 'absolute',
          baseHeightProperty: 'baseHeight',
          topHeightProperty: 'topHeight',
        },
      ],
    },
    geojson: chunk.geojson,
    sourceFileName: source.fileName,
    sourceEntries: chunk.sourceEntries,
  };
}

export async function importPlateauZipSourceChunk(
  source: PlateauZipSource,
  options: PlateauZipImportOptions = {},
): Promise<PlateauZipBuildingChunk | undefined> {
  reportProgress(options, {
    message: `${source.fileName}: ZIP directory 読み込み中...`,
    indeterminate: true,
  });
  const entries = await readZipDirectoryCached(source);
  const buildingEntries = entries
    .filter((entry) => /^udx\/bldg\/[^/]+_bldg_[^/]+\.gml$/i.test(entry.name))
    .sort((a, b) => a.name.localeCompare(b.name));

  if (buildingEntries.length === 0) {
    throw new Error('ZIP 内に udx/bldg/*_bldg_*.gml が見つかりません');
  }
  const selectedBuildingEntries = selectBuildingEntriesForInitialRegion(
    buildingEntries,
    options,
    source.fileName,
  );
  if (selectedBuildingEntries.length === 0) {
    return undefined;
  }
  validateBuildingEntryBudgets(selectedBuildingEntries);

  const features: BuildingGeoJson['features'] = [];
  const budget: CityGmlImportBudget = emptyCityGmlImportBudget();
  for (let i = 0; i < selectedBuildingEntries.length; i++) {
    const entry = selectedBuildingEntries[i]!;
    reportProgress(options, {
      message: `${source.fileName}: 建物 GML ${i + 1}/${selectedBuildingEntries.length} 変換中...`,
      loaded: i,
      total: selectedBuildingEntries.length,
    });
    const gmlBuffer = await readZipEntry(source, entry, MAX_BUILDING_ENTRY_UNCOMPRESSED_BYTES);
    const parsed = await parseCityGmlBuildingFeaturesFromBuffer(
      gmlBuffer,
      entry.name,
      features.length,
    );
    addCityGmlImportBudget(budget, parsed.budget);
    assertCityGmlFeatureCount(features.length + parsed.features.length);
    features.push(...parsed.features);
    reportProgress(options, {
      message: `${source.fileName}: 建物 GML ${i + 1}/${selectedBuildingEntries.length} 変換済み`,
      loaded: i + 1,
      total: selectedBuildingEntries.length,
    });
  }

  const sourceEntries = selectedBuildingEntries.map((entry) => entry.name);
  if (features.length === 0) {
    return {
      sourceFileName: source.fileName,
      sourceEntries,
    };
  }
  const geojson = buildBuildingGeoJson(source.fileName, features, sourceEntries);
  return {
    sourceFileName: source.fileName,
    geojson,
    sourceEntries,
  };
}

function reportProgress(
  options: PlateauZipImportOptions,
  progress: PlateauZipProgress,
): void {
  options.onProgress?.(progress);
}

function buildBuildingGeoJson(
  sourceFileName: string,
  features: BuildingGeoJson['features'],
  sourceEntries: string[],
): BuildingGeoJson {
  const bounds = boundsForBuildingFeatures(features);
  return {
    type: 'FeatureCollection',
    name: `${stripZipExtension(sourceFileName)}-buildings`,
    crs: {
      type: 'name',
      properties: { name: 'EPSG:6697' },
    },
    metadata: {
      sourceZip: sourceFileName,
      sourceEntries,
      verticalDatum: 'TP',
      coordinateOrder: 'GeoJSON lon/lat/height, source CityGML lat/lon/height',
      featureCount: features.length,
      bounds,
    },
    features,
  };
}

interface CityGmlWorkerCallbacks {
  resolve(parsed: ParsedCityGmlBuildingFeatures): void;
  reject(error: Error): void;
}

let plateauCityGmlWorker: Worker | undefined;
let nextPlateauCityGmlWorkerRequestId = 1;
const plateauCityGmlWorkerRequests = new Map<number, CityGmlWorkerCallbacks>();

async function parseCityGmlBuildingFeaturesFromBuffer(
  gmlBuffer: ArrayBuffer,
  sourceEntry: string,
  featureIndexOffset: number,
): Promise<ParsedCityGmlBuildingFeatures> {
  if (typeof Worker === 'undefined') {
    // 古い環境やテスト環境向けの fallback。通常のブラウザでは worker 経由で実行する。
    return parseCityGmlBuildingFeatures(
      textDecoder.decode(gmlBuffer),
      sourceEntry,
      featureIndexOffset,
    );
  }

  return parseCityGmlBuildingFeaturesInWorker(gmlBuffer, sourceEntry, featureIndexOffset);
}

function parseCityGmlBuildingFeaturesInWorker(
  gmlBuffer: ArrayBuffer,
  sourceEntry: string,
  featureIndexOffset: number,
): Promise<ParsedCityGmlBuildingFeatures> {
  const worker = cityGmlWorker();
  const id = nextPlateauCityGmlWorkerRequestId++;

  return new Promise((resolve, reject) => {
    plateauCityGmlWorkerRequests.set(id, { resolve, reject });
    try {
      // ArrayBuffer は transfer して、巨大な GML bytes を main thread 側に残さない。
      worker.postMessage(
        {
          id,
          sourceEntry,
          featureIndexOffset,
          gmlBuffer,
        },
        [gmlBuffer],
      );
    } catch (error) {
      plateauCityGmlWorkerRequests.delete(id);
      reject(error instanceof Error ? error : new Error(String(error)));
    }
  });
}

function cityGmlWorker(): Worker {
  if (plateauCityGmlWorker) return plateauCityGmlWorker;

  const worker = new Worker(new URL('./plateauCityGmlWorker.ts', import.meta.url), {
    type: 'module',
  });
  worker.addEventListener('message', onCityGmlWorkerMessage);
  worker.addEventListener('messageerror', () => {
    resetCityGmlWorker(new Error('CityGML worker からの応答を読み取れませんでした'));
  });
  worker.addEventListener('error', (event) => {
    const message =
      event instanceof ErrorEvent && event.message
        ? event.message
        : 'CityGML worker でエラーが発生しました';
    resetCityGmlWorker(new Error(message));
  });
  plateauCityGmlWorker = worker;
  return worker;
}

function onCityGmlWorkerMessage(
  event: MessageEvent<PlateauCityGmlWorkerResponse>,
): void {
  const response = event.data;
  const callbacks = plateauCityGmlWorkerRequests.get(response.id);
  if (!callbacks) return;
  plateauCityGmlWorkerRequests.delete(response.id);

  if (response.type === 'error') {
    callbacks.reject(new Error(`${response.sourceEntry}: ${response.message}`));
    return;
  }

  callbacks.resolve({
    features: response.features,
    budget: response.budget,
  });
}

function resetCityGmlWorker(error: Error): void {
  const worker = plateauCityGmlWorker;
  plateauCityGmlWorker = undefined;
  worker?.terminate();

  for (const callbacks of plateauCityGmlWorkerRequests.values()) {
    callbacks.reject(error);
  }
  plateauCityGmlWorkerRequests.clear();
}

async function readZipDirectory(source: PlateauZipSource): Promise<ZipEntry[]> {
  const tailLength = Math.min(source.size, 22 + 0xffff);
  const tailOffset = source.size - tailLength;
  validateZipReadRange(source, tailOffset, tailLength, 'ZIP tail');
  const tail = await source.readRange(tailOffset, tailLength);
  const eocdOffsetInTail = findEndOfCentralDirectory(tail);
  if (eocdOffsetInTail < 0) {
    throw new Error('ZIP end of central directory が見つかりません');
  }

  const eocd = new DataView(tail, eocdOffsetInTail);
  const entryCount = eocd.getUint16(10, true);
  let centralDirectorySize = eocd.getUint32(12, true);
  let centralDirectoryOffset = eocd.getUint32(16, true);

  // 4GiB を超える PLATEAU ZIP は Zip64 になる。EOCD の sentinel を見たら
  // Zip64 locator/record から central directory の実 offset/size を読む。
  if (
    entryCount === ZIP64_SENTINEL_16 ||
    centralDirectorySize === ZIP64_SENTINEL_32 ||
    centralDirectoryOffset === ZIP64_SENTINEL_32
  ) {
    const zip64 = await readZip64CentralDirectoryInfo(source, tail, tailOffset, eocdOffsetInTail);
    centralDirectorySize = zip64.centralDirectorySize;
    centralDirectoryOffset = zip64.centralDirectoryOffset;
  }

  enforceMaxBytes(
    centralDirectorySize,
    MAX_CENTRAL_DIRECTORY_BYTES,
    'ZIP central directory',
  );
  validateZipReadRange(
    source,
    centralDirectoryOffset,
    centralDirectorySize,
    'ZIP central directory',
  );
  const directory = await source.readRange(
    centralDirectoryOffset,
    centralDirectorySize,
  );
  const view = new DataView(directory);
  const entries: ZipEntry[] = [];
  let offset = 0;
  while (offset + 46 <= directory.byteLength) {
    if (view.getUint32(offset, true) !== CENTRAL_DIRECTORY_SIGNATURE) {
      throw new Error('ZIP central directory header が壊れています');
    }
    const compressionMethod = view.getUint16(offset + 10, true);
    const compressedSize32 = view.getUint32(offset + 20, true);
    const uncompressedSize32 = view.getUint32(offset + 24, true);
    const fileNameLength = view.getUint16(offset + 28, true);
    const extraLength = view.getUint16(offset + 30, true);
    const commentLength = view.getUint16(offset + 32, true);
    const localHeaderOffset32 = view.getUint32(offset + 42, true);
    const nameStart = offset + 46;
    const nameEnd = nameStart + fileNameLength;
    const extraStart = nameEnd;
    const extraEnd = extraStart + extraLength;
    const recordEnd = extraEnd + commentLength;
    if (recordEnd > directory.byteLength) {
      throw new Error('ZIP central directory header length が壊れています');
    }
    const name = textDecoder.decode(new Uint8Array(directory, nameStart, fileNameLength));
    const zip64Entry = readZip64EntryInfo(
      directory,
      extraStart,
      extraEnd,
      uncompressedSize32,
      compressedSize32,
      localHeaderOffset32,
    );

    entries.push({
      name,
      compressionMethod,
      compressedSize: zip64Entry.compressedSize,
      uncompressedSize: zip64Entry.uncompressedSize,
      localHeaderOffset: zip64Entry.localHeaderOffset,
    });
    if (entries.length > MAX_ZIP_ENTRY_COUNT) {
      throw new Error(`ZIP entry 数が上限 ${MAX_ZIP_ENTRY_COUNT} を超えています`);
    }
    offset = recordEnd;
  }
  return entries;
}

function readZipDirectoryCached(source: PlateauZipSource): Promise<ZipEntry[]> {
  const cached = zipDirectoryCache.get(source);
  if (cached) return cached;

  // 追加読み込みでは同じ remote ZIP source を何度も使う。
  // ZIP の目次は source が同じなら変わらないため、tail / central directory の
  // Range 取得結果を Promise 単位で共有して、同時呼び出しと再読み込みを抑える。
  let task!: Promise<ZipEntry[]>;
  task = readZipDirectory(source).catch((err) => {
    if (zipDirectoryCache.get(source) === task) {
      zipDirectoryCache.delete(source);
    }
    throw err;
  });
  zipDirectoryCache.set(source, task);
  return task;
}

async function readZip64CentralDirectoryInfo(
  source: PlateauZipSource,
  tail: ArrayBuffer,
  tailOffset: number,
  eocdOffsetInTail: number,
): Promise<{ centralDirectorySize: number; centralDirectoryOffset: number }> {
  const locatorOffsetInTail = eocdOffsetInTail - 20;
  if (locatorOffsetInTail < 0) {
    throw new Error('Zip64 EOCD locator が tail に収まりません');
  }
  const locator = new DataView(tail, locatorOffsetInTail, 20);
  if (locator.getUint32(0, true) !== ZIP64_EOCD_LOCATOR_SIGNATURE) {
    throw new Error('Zip64 EOCD locator が見つかりません');
  }
  const zip64RecordOffset = readUint64(locator, 8);
  validateZipReadRange(source, zip64RecordOffset, 56, 'Zip64 EOCD record');
  const zip64RecordHead = await source.readRange(zip64RecordOffset, 56);
  const record = new DataView(zip64RecordHead);
  if (record.getUint32(0, true) !== ZIP64_EOCD_RECORD_SIGNATURE) {
    throw new Error('Zip64 EOCD record が見つかりません');
  }
  const centralDirectorySize = readUint64(record, 40);
  const centralDirectoryOffset = readUint64(record, 48);
  validateZipOffset(centralDirectorySize, 'Zip64 central directory size');
  validateZipOffset(centralDirectoryOffset, 'Zip64 central directory offset');
  if (zip64RecordOffset >= tailOffset + eocdOffsetInTail) {
    throw new Error('Zip64 EOCD record offset が不正です');
  }
  return { centralDirectorySize, centralDirectoryOffset };
}

function readZip64EntryInfo(
  directory: ArrayBuffer,
  extraStart: number,
  extraEnd: number,
  uncompressedSize32: number,
  compressedSize32: number,
  localHeaderOffset32: number,
): { uncompressedSize: number; compressedSize: number; localHeaderOffset: number } {
  let uncompressedSize = uncompressedSize32;
  let compressedSize = compressedSize32;
  let localHeaderOffset = localHeaderOffset32;
  let offset = extraStart;
  while (offset + 4 <= extraEnd) {
    const view = new DataView(directory, offset, 4);
    const headerId = view.getUint16(0, true);
    const dataSize = view.getUint16(2, true);
    const dataStart = offset + 4;
    const dataEnd = dataStart + dataSize;
    if (dataEnd > extraEnd) break;
    if (headerId === ZIP64_EXTENDED_INFO_ID) {
      const data = new DataView(directory, dataStart, dataSize);
      let dataOffset = 0;
      if (uncompressedSize32 === ZIP64_SENTINEL_32) {
        uncompressedSize = readUint64(data, dataOffset);
        dataOffset += 8;
      }
      if (compressedSize32 === ZIP64_SENTINEL_32) {
        compressedSize = readUint64(data, dataOffset);
        dataOffset += 8;
      }
      if (localHeaderOffset32 === ZIP64_SENTINEL_32) {
        localHeaderOffset = readUint64(data, dataOffset);
      }
      break;
    }
    offset = dataEnd;
  }

  validateZipOffset(uncompressedSize, 'ZIP entry uncompressed size');
  validateZipOffset(compressedSize, 'ZIP entry compressed size');
  validateZipOffset(localHeaderOffset, 'ZIP entry local header offset');
  return { uncompressedSize, compressedSize, localHeaderOffset };
}

function findEndOfCentralDirectory(buffer: ArrayBuffer): number {
  const view = new DataView(buffer);
  for (let offset = buffer.byteLength - 22; offset >= 0; offset--) {
    if (view.getUint32(offset, true) === EOCD_SIGNATURE) return offset;
  }
  return -1;
}

async function readZipEntry(
  source: PlateauZipSource,
  entry: ZipEntry,
  maxUncompressedBytes: number,
): Promise<ArrayBuffer> {
  validateZipReadRange(source, entry.localHeaderOffset, 30, `${entry.name} local header`);
  const localHeader = await source.readRange(entry.localHeaderOffset, 30);
  const view = new DataView(localHeader);
  if (view.getUint32(0, true) !== LOCAL_FILE_HEADER_SIGNATURE) {
    throw new Error(`ZIP local header が壊れています: ${entry.name}`);
  }
  const fileNameLength = view.getUint16(26, true);
  const extraLength = view.getUint16(28, true);
  const dataOffset = entry.localHeaderOffset + 30 + fileNameLength + extraLength;
  validateZipReadRange(source, dataOffset, entry.compressedSize, `${entry.name} compressed data`);
  const compressed = await source.readRange(dataOffset, entry.compressedSize);

  if (entry.compressionMethod === ZIP_STORED) return compressed;
  if (entry.compressionMethod === ZIP_DEFLATED) {
    if (entry.uncompressedSize > maxUncompressedBytes) {
      throw new Error(`${entry.name}: 展開後サイズが上限を超えています`);
    }
    const inflated = await inflateRawDeflate(compressed, maxUncompressedBytes, entry.name);
    if (
      entry.uncompressedSize > 0 &&
      inflated.byteLength !== entry.uncompressedSize
    ) {
      console.warn(
        '[Import] ZIP entry size mismatch',
        entry.name,
        inflated.byteLength,
        entry.uncompressedSize,
      );
    }
    return inflated;
  }
  throw new Error(
    `ZIP compression method ${entry.compressionMethod} は未対応です: ${entry.name}`,
  );
}

async function inflateRawDeflate(
  buffer: ArrayBuffer,
  maxBytes: number,
  label: string,
): Promise<ArrayBuffer> {
  const DecompressionStreamCtor = globalThis.DecompressionStream;
  if (!DecompressionStreamCtor) {
    throw new Error('このブラウザは ZIP 展開用 DecompressionStream に対応していません');
  }

  // ZIP の deflate payload は zlib header なしの raw deflate。
  // TypeScript の DOM 型が deflate-raw をまだ列挙していない環境があるため、ここだけ型を緩める。
  const stream = new Blob([buffer])
    .stream()
    .pipeThrough(new DecompressionStreamCtor('deflate-raw' as CompressionFormat));
  return readStreamWithByteLimit(stream, maxBytes, label);
}

async function readBlobRange(
  blob: Blob,
  offset: number,
  length: number,
): Promise<ArrayBuffer> {
  return blob.slice(offset, offset + length).arrayBuffer();
}

async function readStreamWithByteLimit(
  stream: ReadableStream<Uint8Array>,
  maxBytes: number,
  label: string,
): Promise<ArrayBuffer> {
  const reader = stream.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maxBytes) {
        await reader.cancel();
        throw new Error(`${label}: 展開後サイズが上限 ${formatBytes(maxBytes)} を超えています`);
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }

  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out.buffer;
}

function readUint64(view: DataView, offset: number): number {
  if (offset + 8 > view.byteLength) {
    throw new Error('Zip64 field が壊れています');
  }
  const value = view.getBigUint64(offset, true);
  const asNumber = Number(value);
  validateZipOffset(asNumber, 'Zip64 uint64');
  return asNumber;
}

function validateZipOffset(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${label} が JavaScript の安全な整数範囲を超えています`);
  }
}

function validateZipReadRange(
  source: PlateauZipSource,
  offset: number,
  length: number,
  label: string,
): void {
  validateZipOffset(offset, `${label} offset`);
  validateZipOffset(length, `${label} length`);
  const end = offset + length;
  if (!Number.isSafeInteger(end) || end > source.size) {
    throw new Error(`${label} range が ZIP サイズを超えています`);
  }
}

function enforceMaxBytes(value: number, maxBytes: number, label: string): void {
  if (value > maxBytes) {
    throw new Error(`${label} が上限 ${formatBytes(maxBytes)} を超えています`);
  }
}

function validateBuildingEntryBudgets(entries: ZipEntry[]): void {
  if (entries.length > MAX_BUILDING_ENTRY_COUNT) {
    throw new Error(`建物 GML entry 数が上限 ${MAX_BUILDING_ENTRY_COUNT} を超えています`);
  }

  let totalCompressed = 0;
  let totalUncompressed = 0;
  for (const entry of entries) {
    enforceMaxBytes(
      entry.compressedSize,
      MAX_BUILDING_ENTRY_COMPRESSED_BYTES,
      `${entry.name} compressed size`,
    );
    enforceMaxBytes(
      entry.uncompressedSize,
      MAX_BUILDING_ENTRY_UNCOMPRESSED_BYTES,
      `${entry.name} uncompressed size`,
    );
    if (
      entry.compressedSize > 0 &&
      entry.uncompressedSize / entry.compressedSize > MAX_ZIP_COMPRESSION_RATIO
    ) {
      throw new Error(`${entry.name}: ZIP compression ratio が上限を超えています`);
    }
    totalCompressed += entry.compressedSize;
    totalUncompressed += entry.uncompressedSize;
  }

  enforceMaxBytes(
    totalCompressed,
    MAX_TOTAL_BUILDING_COMPRESSED_BYTES,
    '建物 GML compressed total',
  );
  enforceMaxBytes(
    totalUncompressed,
    MAX_TOTAL_BUILDING_UNCOMPRESSED_BYTES,
    '建物 GML uncompressed total',
  );
}

function selectBuildingEntriesForInitialRegion(
  entries: ZipEntry[],
  options: PlateauZipImportOptions,
  sourceFileName: string,
): ZipEntry[] {
  const selection = options.entrySelection;
  if (!selection) return excludeBuildingEntries(entries, options.excludeEntryNames);

  const entriesWithBounds: BuildingEntryWithMeshBounds[] = [];
  for (const entry of entries) {
    const bounds = buildingEntryMeshBounds(entry.name);
    if (bounds) entriesWithBounds.push({ entry, bounds });
  }

  // PLATEAU の標準命名でない ZIP はここで無理に間引かず、従来互換の全読み込みに戻す。
  if (entriesWithBounds.length === 0) {
    reportProgress(options, {
      message: `${sourceFileName}: GML mesh code を読めないため全建物 GML を読み込みます`,
      indeterminate: true,
    });
    return excludeBuildingEntries(entries, options.excludeEntryNames);
  }

  const target = targetLatLonBoundsForSelection(selection, entriesWithBounds);
  const centerNeighborhoodWithBounds =
    selection.bounds === undefined
      ? []
      : centerUnloadedNeighborhoodEntries(
          entriesWithBounds,
          options.excludeEntryNames,
          target,
          selection,
        );
  if (centerNeighborhoodWithBounds.length > 0) {
    const selected = limitSelectedBuildingEntriesByBudget(
      centerNeighborhoodWithBounds,
      centerNeighborhoodBudgetSelection(selection),
      target,
    );
    reportProgress(options, {
      message:
        `${sourceFileName}: 中央近傍で建物 GML ${selected.length}/${entries.length} entries を選択`,
      loaded: selected.length,
      total: entries.length,
    });
    return selected;
  }

  let selectedWithBounds = entriesWithBounds
    .filter((item) => latLonBoundsIntersect(item.bounds, target));

  // 市域の形が細長い場合、mesh union の中心が建物 mesh のない場所になることがある。
  // その場合は中心に最も近い 1 entry を読み、空 import で止まらないようにする。
  if (selectedWithBounds.length === 0 && !selection.bounds) {
    const center = selectionCenterForEntries(selection, entriesWithBounds);
    selectedWithBounds = [nearestEntryToLatLon(entriesWithBounds, center)];
  }
  // 既読 entry を予算適用の前に外す。
  // 先に budget で近い entry だけを選ぶと、近い entry が全て既読だった場合に
  // 未読候補が周辺に残っていても 0 件になり、カメラ移動時の追加読み込みが止まる。
  let remainingWithBounds = excludeBuildingEntriesWithBounds(
    selectedWithBounds,
    options.excludeEntryNames,
  );
  let selectionLabel = selection.bounds ? '追加範囲' : '初期範囲';

  if (
    selection.bounds &&
    selectedWithBounds.length > 0 &&
    remainingWithBounds.length === 0
  ) {
    // 画面内の GML entry がすべて読み込み済みでも、画面端の先に未読 entry が
    // 残っていることがある。地形タイルと同じように周辺を埋めるため、次は
    // 可視範囲中心に近い未読 entry を近傍 chunk として読む。
    remainingWithBounds = excludeBuildingEntriesWithBounds(
      entriesWithBounds,
      options.excludeEntryNames,
    );
    if (remainingWithBounds.length > 0) {
      selectionLabel = '追加近傍';
    }
  }
  const selected = limitSelectedBuildingEntriesByBudget(
    remainingWithBounds,
    selection,
    target,
  );

  reportProgress(options, {
    message:
      `${sourceFileName}: ${selectionLabel}で建物 GML ${selected.length}/${entries.length} entries を選択`,
    loaded: selected.length,
    total: entries.length,
  });
  return selected;
}

function centerUnloadedNeighborhoodEntries(
  entriesWithBounds: BuildingEntryWithMeshBounds[],
  excludeEntryNames: ReadonlySet<string> | undefined,
  target: PlateauZipLatLonBounds,
  selection: PlateauZipEntrySelectionOptions,
): BuildingEntryWithMeshBounds[] {
  const maxCount = selection.centerUnloadedNeighborhoodEntryCount;
  if (maxCount === undefined || maxCount <= 0) return [];

  const targetCenter = latLonBoundsCenter(target);
  const centerEntry = entryContainingLatLon(entriesWithBounds, targetCenter);
  if (!centerEntry) return [];
  if (excludeEntryNames?.has(centerEntry.entry.name)) return [];

  const centerBounds = centerEntry.bounds;
  const latStep = centerBounds.maxLat - centerBounds.minLat;
  const lonStep = centerBounds.maxLon - centerBounds.minLon;
  if (latStep <= 0 || lonStep <= 0) return [];

  const neighborhoodBounds: PlateauZipLatLonBounds = {
    minLat: centerBounds.minLat - latStep,
    maxLat: centerBounds.maxLat + latStep,
    minLon: centerBounds.minLon - lonStep,
    maxLon: centerBounds.maxLon + lonStep,
  };
  const center = latLonBoundsCenter(centerBounds);
  return excludeBuildingEntriesWithBounds(entriesWithBounds, excludeEntryNames)
    .filter((item) => latLonPointInBounds(latLonBoundsCenter(item.bounds), neighborhoodBounds))
    .sort((a, b) => {
      const da = squaredDistanceMeters(latLonBoundsCenter(a.bounds), center);
      const db = squaredDistanceMeters(latLonBoundsCenter(b.bounds), center);
      return da - db;
    })
    .slice(0, maxCount);
}

function centerNeighborhoodBudgetSelection(
  selection: PlateauZipEntrySelectionOptions,
): PlateauZipEntrySelectionOptions {
  const next: PlateauZipEntrySelectionOptions = { ...selection };
  if (selection.centerUnloadedNeighborhoodEntryCount !== undefined) {
    next.maxSelectedEntryCount = selection.centerUnloadedNeighborhoodEntryCount;
  }
  if (selection.centerUnloadedNeighborhoodUncompressedBytes !== undefined) {
    next.maxSelectedUncompressedBytes =
      selection.centerUnloadedNeighborhoodUncompressedBytes;
  }
  if (selection.centerUnloadedNeighborhoodCompressedBytes !== undefined) {
    next.maxSelectedCompressedBytes =
      selection.centerUnloadedNeighborhoodCompressedBytes;
  }
  return next;
}

function entryContainingLatLon(
  entriesWithBounds: BuildingEntryWithMeshBounds[],
  point: { lat: number; lon: number },
): BuildingEntryWithMeshBounds | undefined {
  const containing = entriesWithBounds
    .filter((item) => latLonPointInBounds(point, item.bounds))
    .sort((a, b) => {
      // 同じ点を複数粒度の mesh が含む場合は、より細かい mesh を画面中央 tile とみなす。
      const areaA = latLonBoundsArea(a.bounds);
      const areaB = latLonBoundsArea(b.bounds);
      if (areaA !== areaB) return areaA - areaB;
      const da = squaredDistanceMeters(latLonBoundsCenter(a.bounds), point);
      const db = squaredDistanceMeters(latLonBoundsCenter(b.bounds), point);
      return da - db;
    });
  return containing[0];
}

function limitSelectedBuildingEntriesByBudget(
  entriesWithBounds: BuildingEntryWithMeshBounds[],
  selection: PlateauZipEntrySelectionOptions,
  target: PlateauZipLatLonBounds,
): ZipEntry[] {
  const maxEntryCount = selection.maxSelectedEntryCount;
  const maxUncompressedBytes = selection.maxSelectedUncompressedBytes;
  const maxCompressedBytes = selection.maxSelectedCompressedBytes;
  if (
    maxEntryCount === undefined &&
    maxUncompressedBytes === undefined &&
    maxCompressedBytes === undefined
  ) {
    return entriesWithBounds.map((item) => item.entry);
  }

  const targetCenter = latLonBoundsCenter(target);
  const sorted = [...entriesWithBounds].sort((a, b) => {
    const da = squaredDistanceMeters(latLonBoundsCenter(a.bounds), targetCenter);
    const db = squaredDistanceMeters(latLonBoundsCenter(b.bounds), targetCenter);
    return da - db;
  });

  const selected: ZipEntry[] = [];
  let totalUncompressed = 0;
  let totalCompressed = 0;
  for (const item of sorted) {
    const entry = item.entry;
    if (maxEntryCount !== undefined && selected.length >= maxEntryCount) break;

    const nextUncompressed = totalUncompressed + entry.uncompressedSize;
    const nextCompressed = totalCompressed + entry.compressedSize;
    const exceedsUncompressed =
      maxUncompressedBytes !== undefined && nextUncompressed > maxUncompressedBytes;
    const exceedsCompressed =
      maxCompressedBytes !== undefined && nextCompressed > maxCompressedBytes;
    if ((exceedsUncompressed || exceedsCompressed) && selected.length > 0) {
      continue;
    }

    // 1 entry だけで soft budget を超える場合でも、通常の per-entry hard limit 内なら
    // 1 つは読む。そうしないと巨大 mesh の地域で永久に追加読み込みが進まない。
    selected.push(entry);
    totalUncompressed = nextUncompressed;
    totalCompressed = nextCompressed;
  }
  return selected;
}

function excludeBuildingEntriesWithBounds(
  entriesWithBounds: BuildingEntryWithMeshBounds[],
  excludeEntryNames: ReadonlySet<string> | undefined,
): BuildingEntryWithMeshBounds[] {
  if (!excludeEntryNames || excludeEntryNames.size === 0) return entriesWithBounds;
  return entriesWithBounds.filter((item) => !excludeEntryNames.has(item.entry.name));
}

function excludeBuildingEntries(
  entries: ZipEntry[],
  excludeEntryNames: ReadonlySet<string> | undefined,
): ZipEntry[] {
  if (!excludeEntryNames || excludeEntryNames.size === 0) return entries;
  return entries.filter((entry) => !excludeEntryNames.has(entry.name));
}

function targetLatLonBoundsForSelection(
  selection: PlateauZipEntrySelectionOptions,
  entriesWithBounds: BuildingEntryWithMeshBounds[],
): PlateauZipLatLonBounds {
  if (selection.bounds) {
    const target = expandLatLonBounds(selection.bounds, selection.paddingM ?? 0);
    if (!isValidLatLonBounds(target)) {
      throw new Error('PLATEAU ZIP entry 追加範囲 bounds が不正です');
    }
    return target;
  }

  if (
    !Number.isFinite(selection.centerRegionLengthM) ||
    selection.centerRegionLengthM === undefined ||
    selection.centerRegionLengthM <= 0
  ) {
    throw new Error('PLATEAU ZIP entry 初期範囲 length が不正です');
  }
  const center = selectionCenterForEntries(selection, entriesWithBounds);
  return squareLatLonBoundsAround(center, selection.centerRegionLengthM);
}

function selectionCenterForEntries(
  selection: PlateauZipEntrySelectionOptions,
  entriesWithBounds: BuildingEntryWithMeshBounds[],
): { lat: number; lon: number } {
  const union = unionLatLonBounds(entriesWithBounds.map((item) => item.bounds));
  const center = selection.center ?? latLonBoundsCenter(union);
  validateLatLonCenter(center, 'PLATEAU ZIP entry 初期範囲 center');
  return center;
}

function buildingEntryMeshBounds(entryName: string): PlateauZipLatLonBounds | undefined {
  const baseName = entryName.split('/').pop() ?? entryName;
  const meshCode = /^(\d{8,})_bldg_/i.exec(baseName)?.[1];
  if (!meshCode) return undefined;
  return japaneseMeshCodeBounds(meshCode);
}

function japaneseMeshCodeBounds(meshCode: string): PlateauZipLatLonBounds | undefined {
  if (!/^\d{8,}$/.test(meshCode)) return undefined;

  const firstLatCode = Number(meshCode.slice(0, 2));
  const firstLonCode = Number(meshCode.slice(2, 4));
  const secondLatCode = Number(meshCode[4]);
  const secondLonCode = Number(meshCode[5]);
  const thirdLatCode = Number(meshCode[6]);
  const thirdLonCode = Number(meshCode[7]);
  if (
    !isIntegerInRange(firstLatCode, 0, 99) ||
    !isIntegerInRange(firstLonCode, 0, 99) ||
    !isIntegerInRange(secondLatCode, 0, 7) ||
    !isIntegerInRange(secondLonCode, 0, 7) ||
    !isIntegerInRange(thirdLatCode, 0, 9) ||
    !isIntegerInRange(thirdLonCode, 0, 9)
  ) {
    return undefined;
  }

  let minLat = firstLatCode * (2 / 3);
  let minLon = 100 + firstLonCode;
  let latSize = 2 / 3;
  let lonSize = 1;

  // 2 次 mesh は 1 次 mesh を緯度 8 分割、経度 8 分割したもの。
  latSize /= 8;
  lonSize /= 8;
  minLat += secondLatCode * latSize;
  minLon += secondLonCode * lonSize;

  // 3 次 mesh は 2 次 mesh を緯度 10 分割、経度 10 分割したもの。
  latSize /= 10;
  lonSize /= 10;
  minLat += thirdLatCode * latSize;
  minLon += thirdLonCode * lonSize;

  // 4 次以降の mesh code は 1=南西, 2=南東, 3=北西, 4=北東の 2 分割を繰り返す。
  for (const digit of meshCode.slice(8)) {
    latSize /= 2;
    lonSize /= 2;
    if (digit === '2' || digit === '4') minLon += lonSize;
    if (digit === '3' || digit === '4') minLat += latSize;
    if (digit < '1' || digit > '4') return undefined;
  }

  const bounds = {
    minLat,
    maxLat: minLat + latSize,
    minLon,
    maxLon: minLon + lonSize,
  };
  return isValidLatLonBounds(bounds) ? bounds : undefined;
}

function isIntegerInRange(value: number, min: number, max: number): boolean {
  return Number.isInteger(value) && value >= min && value <= max;
}

function unionLatLonBounds(boundsList: PlateauZipLatLonBounds[]): PlateauZipLatLonBounds {
  const union = {
    minLat: Number.POSITIVE_INFINITY,
    maxLat: Number.NEGATIVE_INFINITY,
    minLon: Number.POSITIVE_INFINITY,
    maxLon: Number.NEGATIVE_INFINITY,
  };
  for (const bounds of boundsList) {
    union.minLat = Math.min(union.minLat, bounds.minLat);
    union.maxLat = Math.max(union.maxLat, bounds.maxLat);
    union.minLon = Math.min(union.minLon, bounds.minLon);
    union.maxLon = Math.max(union.maxLon, bounds.maxLon);
  }
  if (!isValidLatLonBounds(union)) {
    throw new Error('GML mesh code bounds を計算できません');
  }
  return union;
}

function latLonBoundsCenter(bounds: PlateauZipLatLonBounds): { lat: number; lon: number } {
  return {
    lat: (bounds.minLat + bounds.maxLat) / 2,
    lon: (bounds.minLon + bounds.maxLon) / 2,
  };
}

function squareLatLonBoundsAround(
  center: { lat: number; lon: number },
  lengthM: number,
): PlateauZipLatLonBounds {
  const halfLengthM = lengthM / 2;
  const latHalfDegrees = halfLengthM / metersPerLatitudeDegree();
  const lonHalfDegrees = halfLengthM / metersPerLongitudeDegree(center.lat);
  return {
    minLat: center.lat - latHalfDegrees,
    maxLat: center.lat + latHalfDegrees,
    minLon: center.lon - lonHalfDegrees,
    maxLon: center.lon + lonHalfDegrees,
  };
}

function expandLatLonBounds(
  bounds: PlateauZipLatLonBounds,
  paddingM: number,
): PlateauZipLatLonBounds {
  const minLat = Math.min(bounds.minLat, bounds.maxLat);
  const maxLat = Math.max(bounds.minLat, bounds.maxLat);
  const minLon = Math.min(bounds.minLon, bounds.maxLon);
  const maxLon = Math.max(bounds.minLon, bounds.maxLon);
  const normalized = { minLat, maxLat, minLon, maxLon };
  if (paddingM <= 0) return normalized;
  const center = latLonBoundsCenter(normalized);
  const latPadding = paddingM / metersPerLatitudeDegree();
  const lonPadding = paddingM / metersPerLongitudeDegree(center.lat);
  return {
    minLat: normalized.minLat - latPadding,
    maxLat: normalized.maxLat + latPadding,
    minLon: normalized.minLon - lonPadding,
    maxLon: normalized.maxLon + lonPadding,
  };
}

function metersPerLatitudeDegree(): number {
  return 111_320;
}

function metersPerLongitudeDegree(lat: number): number {
  // 極域では経度 1 度あたりの距離が小さくなる。PLATEAU は日本国内だが、保険として下限を置く。
  return metersPerLatitudeDegree() * Math.max(0.2, Math.cos((lat * Math.PI) / 180));
}

function latLonBoundsIntersect(a: PlateauZipLatLonBounds, b: PlateauZipLatLonBounds): boolean {
  return (
    a.minLat <= b.maxLat &&
    a.maxLat >= b.minLat &&
    a.minLon <= b.maxLon &&
    a.maxLon >= b.minLon
  );
}

function latLonPointInBounds(
  point: { lat: number; lon: number },
  bounds: PlateauZipLatLonBounds,
): boolean {
  const epsilon = 1e-12;
  return (
    point.lat >= bounds.minLat - epsilon &&
    point.lat <= bounds.maxLat + epsilon &&
    point.lon >= bounds.minLon - epsilon &&
    point.lon <= bounds.maxLon + epsilon
  );
}

function latLonBoundsArea(bounds: PlateauZipLatLonBounds): number {
  return Math.max(0, bounds.maxLat - bounds.minLat) * Math.max(0, bounds.maxLon - bounds.minLon);
}

function nearestEntryToLatLon(
  entries: BuildingEntryWithMeshBounds[],
  center: { lat: number; lon: number },
): BuildingEntryWithMeshBounds {
  let nearest = entries[0];
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (const entry of entries) {
    const distance = squaredDistanceMeters(latLonBoundsCenter(entry.bounds), center);
    if (distance < nearestDistance) {
      nearest = entry;
      nearestDistance = distance;
    }
  }
  if (!nearest) {
    throw new Error('GML mesh code entry を選択できません');
  }
  return nearest;
}

function squaredDistanceMeters(
  a: { lat: number; lon: number },
  b: { lat: number; lon: number },
): number {
  const midLat = (a.lat + b.lat) / 2;
  const dx = (a.lon - b.lon) * metersPerLongitudeDegree(midLat);
  const dz = (a.lat - b.lat) * metersPerLatitudeDegree();
  return dx * dx + dz * dz;
}

function validateLatLonCenter(center: { lat: number; lon: number }, label: string): void {
  if (
    !Number.isFinite(center.lat) ||
    !Number.isFinite(center.lon) ||
    center.lat < -90 ||
    center.lat > 90 ||
    center.lon < -180 ||
    center.lon > 180
  ) {
    throw new Error(`${label} が緯度経度の範囲外です`);
  }
}

function isValidLatLonBounds(bounds: PlateauZipLatLonBounds): boolean {
  return (
    Number.isFinite(bounds.minLat) &&
    Number.isFinite(bounds.maxLat) &&
    Number.isFinite(bounds.minLon) &&
    Number.isFinite(bounds.maxLon) &&
    bounds.minLat >= -90 &&
    bounds.maxLat <= 90 &&
    bounds.minLon >= -180 &&
    bounds.maxLon <= 180 &&
    bounds.minLat < bounds.maxLat &&
    bounds.minLon < bounds.maxLon
  );
}

function formatBytes(value: number): string {
  if (value >= 1024 * 1024 * 1024) return `${(value / (1024 * 1024 * 1024)).toFixed(1)}GiB`;
  if (value >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(0)}MiB`;
  if (value >= 1024) return `${(value / 1024).toFixed(0)}KiB`;
  return `${value}B`;
}

function stripZipExtension(fileName: string): string {
  return fileName.replace(/\.zip$/i, '');
}

function displayNameForZip(fileName: string): string {
  const base = stripZipExtension(fileName);
  if (/shinagawa/i.test(base)) return 'Shinagawa PLATEAU ZIP import';
  return `${base} PLATEAU ZIP import`;
}
