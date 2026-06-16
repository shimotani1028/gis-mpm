// =============================================================================
// XYZ タイル座標 ↔ 経緯度 ↔ ローカル ENU(m) の変換ユーティリティ
// -----------------------------------------------------------------------------
// 用語:
//   - XYZ タイル: Web Mercator 投影で世界を 2^z * 2^z 枚に区切ったタイル番号 (x, y)
//                 z=0 の時 1 枚（地球全体）、z=1 で 4 枚、…
//                 仕様: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
//   - ENU: East-North-Up。原点（=AOI_CENTER）からの東(+x)/北(+y)メートル座標
//          地球の曲率を無視した接平面近似。半径数十 km なら誤差 cm オーダーで実用上 OK。
// =============================================================================

import { AOI_CENTER } from '../config';

export interface AoiCenter {
  lat: number;
  lon: number;
}

let currentAoiCenter: AoiCenter = { ...AOI_CENTER };

/** 現在の ENU 原点を返す。Import モードでは読み込みデータの中心へ切り替わる。 */
export function getAoiCenter(): AoiCenter {
  return currentAoiCenter;
}

/** ENU 原点を切り替える。既存タイルや建物は、この後に呼び出し側で破棄して作り直す。 */
export function setAoiCenter(center: AoiCenter): void {
  currentAoiCenter = { lat: center.lat, lon: center.lon };
}

/**
 * 座標変換を一時的な ENU 原点で実行する。
 *
 * GSI と PLATEAU を別 Terrain として同時保持する場合でも、既存の変換関数は
 * グローバルな currentAoiCenter を参照する。処理単位で原点を差し替えて戻すことで、
 * Terrain ごとの cache を別原点で安全に作れる。
 */
export function withAoiCenter<T>(center: AoiCenter, fn: () => T): T {
  const previous = currentAoiCenter;
  currentAoiCenter = { lat: center.lat, lon: center.lon };
  try {
    return fn();
  } finally {
    currentAoiCenter = previous;
  }
}

/** 既定の富士山原点へ戻す。GSI mode へ戻るときに使う。 */
export function resetAoiCenter(): void {
  currentAoiCenter = { ...AOI_CENTER };
}

/** 地球の半径（WGS84 ではなく、Web Mercator が前提とする球の半径 [m]） */
export const EARTH_RADIUS_M = 6378137;

/** 度 → ラジアン */
const DEG2RAD = Math.PI / 180;
/** ラジアン → 度 */
const RAD2DEG = 180 / Math.PI;

// -----------------------------------------------------------------------------
// XYZ タイル ⇄ 経緯度
// -----------------------------------------------------------------------------

/**
 * XYZ タイルの「左上隅」の経緯度を返す。
 * Slippy Map の標準計算式そのまま。
 */
export function tileTopLeftLonLat(z: number, x: number, y: number) {
  const n = 2 ** z;
  const lon = (x / n) * 360 - 180;
  const latRad = Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / n)));
  const lat = latRad * RAD2DEG;
  return { lon, lat };
}

/**
 * XYZ タイルの矩形範囲（北西〜南東）を返す。
 * - 経緯度の minLon < maxLon、minLat < maxLat になるように整える。
 */
export function tileBoundsLonLat(z: number, x: number, y: number) {
  // タイルの左上 (北西)
  const nw = tileTopLeftLonLat(z, x, y);
  // タイルの右下 (南東) = 次のタイル (x+1, y+1) の左上
  const se = tileTopLeftLonLat(z, x + 1, y + 1);
  return {
    minLon: nw.lon,
    maxLon: se.lon,
    minLat: se.lat,        // se が南なので min
    maxLat: nw.lat,        // nw が北なので max
  };
}

/**
 * 経緯度から XYZ タイル座標（小数）を返す。
 * - 整数化は呼び出し側で `Math.floor` する想定。
 * - z は整数。
 */
export function lonLatToTileXY(lon: number, lat: number, z: number) {
  const n = 2 ** z;
  const x = ((lon + 180) / 360) * n;
  const latRad = lat * DEG2RAD;
  const y = ((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * n;
  return { x, y };
}

// -----------------------------------------------------------------------------
// 経緯度 ⇄ ローカル ENU (m)
// -----------------------------------------------------------------------------

/**
 * AOI_CENTER を原点とする「東 / 北」方向のメートル座標に変換する。
 * 接平面近似:
 *   x = R * cos(originLat) * (lon - originLon) [rad]
 *   y = R * (lat - originLat) [rad]
 *
 * - 富士山周辺数十 km スケールでは誤差 cm 級。
 * - 数百 km を超える広域だと正しくない（必要になったら平面投影を切り替える）。
 */
export function lonLatToENU(lon: number, lat: number) {
  const origin = currentAoiCenter;
  const dLon = (lon - origin.lon) * DEG2RAD;
  const dLat = (lat - origin.lat) * DEG2RAD;
  const cosOrigin = Math.cos(origin.lat * DEG2RAD);
  return {
    x: EARTH_RADIUS_M * cosOrigin * dLon,   // East
    y: EARTH_RADIUS_M * dLat,               // North
  };
}

/**
 * AOI_CENTER 原点の ENU メートル座標を経緯度へ戻す。
 * `lonLatToENU()` と同じ接平面近似の逆変換なので、地形描画・タイル選択・
 * MPM coverage が同じ前提で tile key を求められる。
 */
export function enuToLonLat(eastM: number, northM: number) {
  const origin = currentAoiCenter;
  const cosOrigin = Math.cos(origin.lat * DEG2RAD);
  const dLon = eastM / (EARTH_RADIUS_M * cosOrigin);
  const dLat = northM / EARTH_RADIUS_M;
  return {
    lon: origin.lon + dLon * RAD2DEG,
    lat: origin.lat + dLat * RAD2DEG,
  };
}

/**
 * 1 タイル分のサイズ（メートル）と中心位置（ローカル ENU）を返すヘルパー。
 * TileMesh が DEM 頂点のローカル座標とメッシュ位置を決めるのに使う。
 */
export function tileSizeAndCenterENU(z: number, x: number, y: number) {
  const b = tileBoundsLonLat(z, x, y);
  const sw = lonLatToENU(b.minLon, b.minLat);
  const ne = lonLatToENU(b.maxLon, b.maxLat);

  // DEM メッシュのサイズ（東-西方向の幅 / 南-北方向の高さ）
  const widthM = ne.x - sw.x;
  const heightM = ne.y - sw.y;

  // タイル中心座標（ENU 平面上）
  const centerX = (sw.x + ne.x) * 0.5;
  const centerY = (sw.y + ne.y) * 0.5;

  return { widthM, heightM, centerX, centerY };
}

// -----------------------------------------------------------------------------
// タイルキー（Map のキーとして文字列化）
// -----------------------------------------------------------------------------

/** タイル識別子の構造体型 */
export interface TileKey {
  z: number;
  x: number;
  y: number;
}

export interface EnuBounds {
  minE: number;
  maxE: number;
  minN: number;
  maxN: number;
}

export interface TileKeyRange {
  z: number;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

/** ENU x/z 位置を指定 zoom の XYZ タイルキーへ変換する。範囲外なら undefined。 */
export function enuToTileKey(eastM: number, northM: number, z: number): TileKey | undefined {
  const lonLat = enuToLonLat(eastM, northM);
  const xy = lonLatToTileXY(lonLat.lon, lonLat.lat, z);
  const max = 2 ** z;
  const x = Math.floor(xy.x);
  const y = Math.floor(xy.y);
  if (x < 0 || x >= max || y < 0 || y >= max) return undefined;
  return { z, x, y };
}

/**
 * ENU 境界を覆う XYZ タイル一覧を返す。
 * MPM coverage と通常の視錐台タイル選択で同じ計算を使い、境界の取りこぼしを避ける。
 */
export function tilesCoveringEnuBounds(bounds: EnuBounds, z: number): TileKey[] {
  const range = tileRangeCoveringEnuBounds(bounds, z);
  const out: TileKey[] = [];
  for (let ty = range.yMin; ty <= range.yMax; ty++) {
    for (let tx = range.xMin; tx <= range.xMax; tx++) {
      out.push({ z, x: tx, y: ty });
    }
  }
  return out;
}

export function tileCountCoveringEnuBounds(
  bounds: EnuBounds,
  z: number,
  paddingTiles = 0,
): number {
  const range = tileRangeCoveringEnuBounds(bounds, z);
  const maxTile = 2 ** z;
  const xMin = clampInt(range.xMin - paddingTiles, 0, maxTile - 1);
  const xMax = clampInt(range.xMax + paddingTiles, 0, maxTile - 1);
  const yMin = clampInt(range.yMin - paddingTiles, 0, maxTile - 1);
  const yMax = clampInt(range.yMax + paddingTiles, 0, maxTile - 1);
  return (xMax - xMin + 1) * (yMax - yMin + 1);
}

function tileRangeCoveringEnuBounds(bounds: EnuBounds, z: number): TileKeyRange {
  const minE = Math.min(bounds.minE, bounds.maxE);
  const maxE = Math.max(bounds.minE, bounds.maxE);
  const minN = Math.min(bounds.minN, bounds.maxN);
  const maxN = Math.max(bounds.minN, bounds.maxN);

  // 4 隅を経緯度に直してタイル番号レンジを作る。
  // Web Mercator の y は緯度に対して反転するので min/max を取り直す。
  const sw = enuToLonLat(minE, minN);
  const ne = enuToLonLat(maxE, maxN);
  const swXY = lonLatToTileXY(sw.lon, sw.lat, z);
  const neXY = lonLatToTileXY(ne.lon, ne.lat, z);

  const maxTile = 2 ** z;
  const xMin = clampInt(Math.floor(Math.min(swXY.x, neXY.x)), 0, maxTile - 1);
  const xMax = clampInt(Math.floor(Math.max(swXY.x, neXY.x)), 0, maxTile - 1);
  const yMin = clampInt(Math.floor(Math.min(swXY.y, neXY.y)), 0, maxTile - 1);
  const yMax = clampInt(Math.floor(Math.max(swXY.y, neXY.y)), 0, maxTile - 1);

  return { z, xMin, xMax, yMin, yMax };
}

/** Map のキー用に "z/x/y" の文字列に変換 */
export function tileKeyString(k: TileKey): string {
  return `${k.z}/${k.x}/${k.y}`;
}

/** 文字列から TileKey に戻す（必要なら） */
export function parseTileKey(s: string): TileKey {
  const [z, x, y] = s.split('/').map((v) => Number(v));
  return { z: z!, x: x!, y: y! };
}

function clampInt(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
