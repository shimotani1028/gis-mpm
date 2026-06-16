// =============================================================================
// 国土地理院 DEM PNG → 標高 [m] へのデコード
// -----------------------------------------------------------------------------
// 国土地理院の標高タイル仕様:
//   x = 2^16 * R + 2^8 * G + B   （24bit signed int をビッグエンディアンで RGB に）
//   x が 0x800000 のときは「無効値」（海域・データなし）
//   それ以外は、x が 0x800000 以上なら 2^24 を引いて負数として扱う
//   標高 [m] = x * 0.01      （= 1cm 単位で 24bit 表現）
//
// 参考: https://maps.gsi.go.jp/development/demtile.html
// =============================================================================

/** 無効値を表すマジック値。タイル仕様上、この RGB は「標高なし」を意味する。 */
const GSI_INVALID = 0x800000;

/** ビット幅 24bit の符号反転境界（0x800000 以上は負として扱う） */
const SIGN_BOUNDARY = 0x800000;
const TWO_POW_24 = 0x1000000;

/** スケール係数。1cm 単位で格納されているので 0.01 倍で m に直す。 */
const SCALE_M_PER_UNIT = 0.01;

export interface DecodeGsiPngOptions {
  /**
   * GSI 無効値を見つけたときに出力する値。
   * - 既定は従来互換の 0m
   * - worker の DEM 合成処理では NaN を指定し、後段で欠損ピクセルだけを別 DEM から補完する
   */
  invalidValue?: number;
}

/**
 * RGBA 画素配列（ImageData.data 相当）から標高 [m] の配列に変換する。
 *
 * @param rgba - PNG をデコードして得た RGBA バイト列。長さは width * height * 4。
 * @returns Float32Array of length (width * height)。値は標高 [m]、無効値は options.invalidValue。
 *
 * 実装メモ:
 * - 画素単位ループは比較的ホットになりやすいので、indexed access で
 *   オーバーヘッドを抑えている。
 * - Float32Array を返すのは、後段で DEM 原寸 BufferGeometry の頂点生成に使うため。
 */
export function decodeGsiPng(
  rgba: Uint8ClampedArray,
  options: DecodeGsiPngOptions = {},
): Float32Array {
  const pixelCount = rgba.length >>> 2;       // 4 で割る = >>> 2
  const out = new Float32Array(pixelCount);
  const invalidValue = options.invalidValue ?? 0;

  for (let i = 0; i < pixelCount; i++) {
    const base = i << 2;                      // i * 4
    const r = rgba[base]!;                    // noUncheckedIndexedAccess 対策で ! を付与
    const g = rgba[base + 1]!;
    const b = rgba[base + 2]!;

    // RGB を 24bit 整数に詰める（A は使わない）
    const x = (r << 16) | (g << 8) | b;

    // 無効値は呼び出し側の方針に任せる。
    // 描画直前に 0m へ落とすだけだと、標高の高い湖沼・未整備領域が深い穴に見える。
    // そのため worker では NaN として残し、DEM5B/5C/親 DEM10B でピクセル単位に埋める。
    if (x === GSI_INVALID) {
      out[i] = invalidValue;
      continue;
    }

    // 24bit signed として復元: 0x800000 以上は負側
    const signed = x >= SIGN_BOUNDARY ? x - TWO_POW_24 : x;

    // 1cm 単位 → m
    out[i] = signed * SCALE_M_PER_UNIT;
  }

  return out;
}

/**
 * Mapzen Terrarium PNG/WebP/AVIF の RGBA 画素列を標高 [m] に変換する。
 *
 * 仕様:
 * - height = R * 256 + G + B / 256 - 32768
 * - PLATEAU-Terrain の `/terrarium/` はこの encoding で配信される。
 */
export function decodeTerrariumPng(rgba: Uint8ClampedArray): Float32Array {
  const pixelCount = rgba.length >>> 2;
  const out = new Float32Array(pixelCount);

  for (let i = 0; i < pixelCount; i++) {
    const base = i << 2;
    const r = rgba[base]!;
    const g = rgba[base + 1]!;
    const b = rgba[base + 2]!;
    // PLATEAU-Terrain の海域・欠損部では黒が返ることがあり、
    // Terrarium 仕様どおりに読むと -32768m の巨大な穴になる。
    // ここでは NaN にして、worker 側で近傍の有効値から補間する。
    if (r === 0 && g === 0 && b === 0) {
      out[i] = Number.NaN;
      continue;
    }
    out[i] = r * 256 + g + b / 256 - 32768;
  }

  return out;
}

/**
 * Mapbox Terrain-RGB v1 の RGBA 画素列を標高 [m] に変換する。
 *
 * 仕様:
 * - height = -10000 + (R * 256 * 256 + G * 256 + B) * 0.1
 */
export function decodeMapboxTerrainRgbPng(rgba: Uint8ClampedArray): Float32Array {
  const pixelCount = rgba.length >>> 2;
  const out = new Float32Array(pixelCount);

  for (let i = 0; i < pixelCount; i++) {
    const base = i << 2;
    const r = rgba[base]!;
    const g = rgba[base + 1]!;
    const b = rgba[base + 2]!;
    out[i] = -10000 + ((r << 16) + (g << 8) + b) * 0.1;
  }

  return out;
}
