// =============================================================================
// タイルロード Web Worker
// -----------------------------------------------------------------------------
// 役割:
//   - メインスレッドから {z, x, y} を受け取って GSI から PNG を fetch する
//   - PNG を OffscreenCanvas で RGBA バイトに展開する
//   - decodeGsiPng で標高 [m] の Float32Array に変換する
//   - 結果を transferable オブジェクトとしてメインに送り返す
//
// なぜワーカーで動かすか:
//   - PNG デコードと標高計算（256x256 = 65,536 ピクセル）はメインスレッドで
//     やると 1〜2ms 食う。これが視点移動中に毎フレーム何枚も来るとカクつく。
//   - ワーカーで並列化することで、レンダリングと完全に分離できる。
//
// Vite の解決:
//   - メインスレッドからは `import TileWorker from './tileWorker?worker'` で
//     コンストラクタが取れる。型は VitePluginWorker が自動生成する。
// =============================================================================

import {
  decodeGsiPng,
  decodeMapboxTerrainRgbPng,
  decodeTerrariumPng,
} from '../geo/elevationDecoder';
import {
  DEM5_MIN_ZOOM,
  GSI_DEM5_URLS,
  GSI_DEM_URL,
  TILE_PIXELS,
} from '../config';
import type { TerrainTileSource } from './TileSource';

// -----------------------------------------------------------------------------
// メッセージ型定義
// -----------------------------------------------------------------------------

/** メイン → ワーカー: タイルロード要求 */
export interface WorkerRequest {
  type: 'load';
  /** 重複抑制と結果対応に使う一意キー（"z/x/y"） */
  reqId: string;
  z: number;
  x: number;
  y: number;
  /** DEM タイルの取得元。省略時は従来互換の GSI として扱う。 */
  source?: TerrainTileSource;
}

/** ワーカー → メイン: 成功レスポンス */
export interface WorkerSuccess {
  type: 'success';
  reqId: string;
  /** 標高 [m] の TILE_PIXELS x TILE_PIXELS グリッド。row-major（Y 先行） */
  elevations: Float32Array;
  /** raw DEM で標高として使えなかった画素。1 = no-data / invalid。 */
  invalidMask?: Uint8Array;
  width: number;
  height: number;
}

/** ワーカー → メイン: エラーレスポンス */
export interface WorkerError {
  type: 'error';
  reqId: string;
  message: string;
}

/** ワーカー → メイン に流れる全メッセージ型 */
export type WorkerResponse = WorkerSuccess | WorkerError;

// -----------------------------------------------------------------------------
// fetch & デコード本体
// -----------------------------------------------------------------------------

interface DecodedTile {
  elevations: Float32Array;
  invalidMask?: Uint8Array;
}

const MAX_TILE_IMAGE_BYTES = 4 * 1024 * 1024;

/** URL テンプレートの {z}/{x}/{y} を実数値で置換する */
function buildUrl(template: string, z: number, x: number, y: number): string {
  return template
    .replace('{z}', String(z))
    .replace('{x}', String(x))
    .replace('{y}', String(y));
}

async function fetchPngImageData(
  url: string,
  missingAsNull: boolean,
): Promise<ImageData | null> {
  const res = await fetch(url, { mode: 'cors' });
  if (!res.ok) {
    if (missingAsNull && res.status === 404) return null;
    throw new Error(`fetch failed: ${res.status} ${res.statusText} for ${url}`);
  }

  const contentType = res.headers.get('content-type')?.split(';')[0]?.trim().toLowerCase();
  if (
    contentType &&
    contentType !== 'image/png' &&
    contentType !== 'application/octet-stream'
  ) {
    throw new Error(`unexpected tile content-type: ${contentType} for ${url}`);
  }

  const bytes = await readResponseBytesWithLimit(res, MAX_TILE_IMAGE_BYTES, url);
  validatePngTileHeader(bytes, url);

  const pngBuffer = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(pngBuffer).set(bytes);
  const bitmap = await createImageBitmap(new Blob([pngBuffer], { type: 'image/png' }));
  try {
    if (bitmap.width !== TILE_PIXELS || bitmap.height !== TILE_PIXELS) {
      throw new Error(`unexpected tile bitmap size: ${bitmap.width}x${bitmap.height} for ${url}`);
    }

    const canvas = new OffscreenCanvas(TILE_PIXELS, TILE_PIXELS);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('failed to create OffscreenCanvas 2D context');
    ctx.drawImage(bitmap, 0, 0, TILE_PIXELS, TILE_PIXELS);
    return ctx.getImageData(0, 0, TILE_PIXELS, TILE_PIXELS);
  } finally {
    bitmap.close();
  }
}

async function readResponseBytesWithLimit(
  res: Response,
  maxBytes: number,
  label: string,
): Promise<Uint8Array> {
  const contentLength = Number(res.headers.get('content-length'));
  if (Number.isFinite(contentLength) && contentLength > maxBytes) {
    throw new Error(`tile response is too large: ${contentLength}/${maxBytes} for ${label}`);
  }

  if (!res.body) {
    const buffer = await res.arrayBuffer();
    if (buffer.byteLength > maxBytes) {
      throw new Error(`tile response is too large: ${buffer.byteLength}/${maxBytes} for ${label}`);
    }
    return new Uint8Array(buffer);
  }

  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > maxBytes) {
        await reader.cancel();
        throw new Error(`tile response is too large: ${total}/${maxBytes} for ${label}`);
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
  return out;
}

function validatePngTileHeader(bytes: Uint8Array, label: string): void {
  if (bytes.byteLength < 24) {
    throw new Error(`tile PNG is too short for ${label}`);
  }
  const signature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  for (let i = 0; i < signature.length; i++) {
    if (bytes[i] !== signature[i]) {
      throw new Error(`tile response is not PNG for ${label}`);
    }
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const width = view.getUint32(16, false);
  const height = view.getUint32(20, false);
  if (width !== TILE_PIXELS || height !== TILE_PIXELS) {
    throw new Error(`unexpected tile PNG size: ${width}x${height} for ${label}`);
  }
}

/**
 * PNG を fetch → ImageBitmap → OffscreenCanvas に描画して標高 [m] にデコードする。
 * - 404 の場合は null を返す（呼び出し側でフォールバックを判断）
 * - その他のエラーは throw
 * - GSI 無効値は NaN として残す。ここで 0m に潰すと、DEM5 の部分欠損が
 *   高標高地形の巨大な穴としてメッシュ化されてしまうため。
 */
async function tryFetchDem(
  template: string,
  z: number,
  x: number,
  y: number,
): Promise<Float32Array | null> {
  const url = buildUrl(template, z, x, y);
  const imageData = await fetchPngImageData(url, true);
  if (!imageData) return null;
  return decodeGsiPng(imageData.data, { invalidValue: Number.NaN });
}

/**
 * 一般的な raster-dem タイルを取得して標高 [m] にデコードする。
 * PLATEAU-Terrain の Terrarium / Mapbox Terrain-RGB はここを通る。
 */
async function fetchRasterDem(
  source: Extract<TerrainTileSource, { kind: 'raster-dem' }>,
  z: number,
  x: number,
  y: number,
): Promise<DecodedTile> {
  const url = buildUrl(source.urlTemplate, z, x, y);
  const imageData = await fetchPngImageData(url, false);
  if (!imageData) {
    throw new Error(`fetch failed: 404 for ${url}`);
  }
  const elevations =
    source.encoding === 'terrarium'
      ? decodeTerrariumPng(imageData.data)
      : decodeMapboxTerrainRgbPng(imageData.data);
  const invalidMask = createInvalidMask(elevations);

  // Mesh 生成に NaN を渡すと geometry 全体が壊れるため、描画 fallback としては
  // 有限値へ直す。ただし raw invalid は invalidMask に残し、権威ある標高サンプルは
  // Terrain 側で隣接 tile を含めて読み直す。
  fillInvalidElevationsFromNeighbors(elevations);
  return invalidMask ? { elevations, invalidMask } : { elevations };
}

function createInvalidMask(elevations: Float32Array): Uint8Array | undefined {
  let mask: Uint8Array | undefined;
  for (let i = 0; i < elevations.length; i++) {
    if (Number.isFinite(elevations[i]!)) continue;
    if (!mask) mask = new Uint8Array(elevations.length);
    mask[i] = 1;
  }
  return mask;
}

function hasInvalidElevation(elevations: Float32Array): boolean {
  for (let i = 0; i < elevations.length; i++) {
    if (!Number.isFinite(elevations[i]!)) return true;
  }
  return false;
}

function copyValidElevationsIntoMissing(target: Float32Array, source: Float32Array): void {
  for (let i = 0; i < target.length; i++) {
    if (Number.isFinite(target[i]!)) continue;
    const value = source[i]!;
    if (Number.isFinite(value)) target[i] = value;
  }
}

function fillInvalidElevationsWithZero(elevations: Float32Array): void {
  for (let i = 0; i < elevations.length; i++) {
    if (!Number.isFinite(elevations[i]!)) elevations[i] = 0;
  }
}

/**
 * raster-dem の欠損ピクセルを同一タイル内の近傍値で埋める。
 * PLATEAU Terrarium ではタイル境界や海域に no-data 黒が入ることがあり、
 * そのまま 0m へ落とすとタイル境界が直線の溝として見える。
 */
function fillInvalidElevationsFromNeighbors(elevations: Float32Array): void {
  if (!hasInvalidElevation(elevations)) return;

  const N = TILE_PIXELS;
  let src = elevations;
  let dst = new Float32Array(src);

  for (let pass = 0; pass < 8; pass++) {
    let changed = 0;
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const index = y * N + x;
        if (Number.isFinite(src[index]!)) {
          dst[index] = src[index]!;
          continue;
        }

        let sum = 0;
        let count = 0;
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = x + dx;
            const ny = y + dy;
            if (nx < 0 || nx >= N || ny < 0 || ny >= N) continue;
            const value = src[ny * N + nx]!;
            if (!Number.isFinite(value)) continue;
            sum += value;
            count++;
          }
        }
        if (count > 0) {
          dst[index] = sum / count;
          changed++;
        } else {
          dst[index] = Number.NaN;
        }
      }
    }

    src = dst;
    dst = new Float32Array(src);
    if (changed === 0 || !hasInvalidElevation(src)) break;
  }

  elevations.set(src);
  fillInvalidElevationsWithZero(elevations);
}

function sampleBilinearIgnoreInvalid(
  elevations: Float32Array,
  x: number,
  y: number,
): number {
  const N = TILE_PIXELS;
  const clampedX = Math.max(0, Math.min(N - 1, x));
  const clampedY = Math.max(0, Math.min(N - 1, y));
  const x0 = Math.floor(clampedX);
  const y0 = Math.floor(clampedY);
  const x1 = Math.max(0, Math.min(N - 1, x0 + 1));
  const y1 = Math.max(0, Math.min(N - 1, y0 + 1));
  const tx = clampedX - x0;
  const ty = clampedY - y0;

  // 親 DEM10B 側にも無効値が残ることがある。通常の bilinear だと隣接 1 点の
  // NaN が出力全体へ伝播するため、有効な近傍だけで重み付き平均する。
  let total = 0;
  let weightSum = 0;

  let value = elevations[y0 * N + x0]!;
  let weight = (1 - tx) * (1 - ty);
  if (Number.isFinite(value) && weight > 0) {
    total += value * weight;
    weightSum += weight;
  }

  value = elevations[y0 * N + x1]!;
  weight = tx * (1 - ty);
  if (Number.isFinite(value) && weight > 0) {
    total += value * weight;
    weightSum += weight;
  }

  value = elevations[y1 * N + x0]!;
  weight = (1 - tx) * ty;
  if (Number.isFinite(value) && weight > 0) {
    total += value * weight;
    weightSum += weight;
  }

  value = elevations[y1 * N + x1]!;
  weight = tx * ty;
  if (Number.isFinite(value) && weight > 0) {
    total += value * weight;
    weightSum += weight;
  }
  return weightSum > 0 ? total / weightSum : Number.NaN;
}

/**
 * 親タイル (z-1) の DEM10B から該当象限を切り出して bilinear で TILE_PIXELS まで拡大。
 * - 5m データが無い地域では 10m データから補間して z=15 の見た目を保つ
 * - 子タイル (x, y) の親は (x>>1, y>>1)、象限は (x&1, y&1)
 */
async function upsampleFromParent(z: number, x: number, y: number): Promise<Float32Array> {
  const pz = z - 1;
  const px = x >> 1;
  const py = y >> 1;
  const qx = x & 1;
  const qy = y & 1;

  const parent = await tryFetchDem(GSI_DEM_URL, pz, px, py);
  if (!parent) {
    // 親も無い（海上等）→ ゼロ標高
    return new Float32Array(TILE_PIXELS * TILE_PIXELS);
  }

  const N = TILE_PIXELS;
  const half = N / 2;                 // 128: 親の中で対応する象限の幅
  const out = new Float32Array(N * N);

  // 子の各ピクセル (i,j) は親の連続座標 (qx*half + (i+0.5)*half/N, qy*half + (j+0.5)*half/N) - 0.5
  // をサンプリングする。bilinear で 2x 拡大。
  for (let j = 0; j < N; j++) {
    // 親側の y 座標（中心オフセット込み）
    const sy = qy * half + (j + 0.5) * (half / N) - 0.5;
    for (let i = 0; i < N; i++) {
      const sx = qx * half + (i + 0.5) * (half / N) - 0.5;
      out[j * N + i] = sampleBilinearIgnoreInvalid(parent, sx, sy);
    }
  }
  return out;
}

/**
 * z, x, y のタイル標高を取得する。
 * - z >= DEM5_MIN_ZOOM: DEM5A → DEM5B → DEM5C → 親 DEM10B を upsample の順でフォールバック
 * - z <  DEM5_MIN_ZOOM: DEM10B（404 はゼロ標高）
 */
async function fetchAndDecode(
  z: number,
  x: number,
  y: number,
  source: TerrainTileSource | undefined,
): Promise<DecodedTile> {
  if (source?.kind === 'raster-dem') {
    return fetchRasterDem(source, z, x, y);
  }

  if (z >= DEM5_MIN_ZOOM) {
    const merged = new Float32Array(TILE_PIXELS * TILE_PIXELS);
    merged.fill(Number.NaN);

    for (const tmpl of GSI_DEM5_URLS) {
      const r = await tryFetchDem(tmpl, z, x, y);
      if (!r) continue;
      copyValidElevationsIntoMissing(merged, r);
      if (!hasInvalidElevation(merged)) return { elevations: merged };
    }

    // DEM5A/B/C のタイルが存在しても、内部に 0x800000 の無効ピクセルを含むことがある。
    // 残った欠損だけを親 DEM10B の対応象限から補い、精細 DEM の有効値は上書きしない。
    const parent = await upsampleFromParent(z, x, y);
    copyValidElevationsIntoMissing(merged, parent);
    fillInvalidElevationsWithZero(merged);
    return { elevations: merged };
  }

  const r = await tryFetchDem(GSI_DEM_URL, z, x, y);
  if (r) {
    fillInvalidElevationsWithZero(r);
    return { elevations: r };
  }
  // 海上などはゼロで埋めて返す
  return { elevations: new Float32Array(TILE_PIXELS * TILE_PIXELS) };
}

// -----------------------------------------------------------------------------
// ワーカーのメッセージハンドラ
// -----------------------------------------------------------------------------

// `self` をワーカーグローバルとして型付け（lib に WebWorker を入れている）
const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
  const req = ev.data;
  if (req.type !== 'load') return;

  try {
    const result = await fetchAndDecode(req.z, req.x, req.y, req.source);
    const response: WorkerSuccess = {
      type: 'success',
      reqId: req.reqId,
      elevations: result.elevations,
      width: TILE_PIXELS,
      height: TILE_PIXELS,
    };
    const transfer: Transferable[] = [result.elevations.buffer];
    if (result.invalidMask) {
      response.invalidMask = result.invalidMask;
      transfer.push(result.invalidMask.buffer);
    }
    // TypedArray.buffer を transferable にしてコピーを避ける
    ctx.postMessage(response, transfer);
  } catch (e) {
    const response: WorkerError = {
      type: 'error',
      reqId: req.reqId,
      message: e instanceof Error ? e.message : String(e),
    };
    ctx.postMessage(response);
  }
};
