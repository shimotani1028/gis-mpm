// =============================================================================
// TileSelector: クアッドツリー LOD によるタイル選択
// -----------------------------------------------------------------------------
// 戦略（Google Earth 風）:
//   1. 視野フラスタムが地表平面 Y=0 と交わる範囲を「最大 200km」に切って求める
//   2. その範囲を MIN_ZOOM のタイルで覆う（=ルートタイル）
//   3. 各ルートタイルを再帰的に細分化:
//        - 画面に大きく映りそうなら 4 つの子タイル（z+1）に分割
//        - 十分小さい or MAX_ZOOM 到達 → 採用
//      指標は「Screen-Space Error (SSE)」=（タイル幅[m] × pixelsPerRad / 距離[m]）
//      これが TARGET_SSE を超えていれば「ピクセル比で粗すぎる」と判断して分割
//
// 利点:
//   - カメラ近くは高 LOD、遠景は低 LOD で「Google Earth らしい」見え方
//   - 視野外のタイルは展開しないのでロード数を抑えられる
//
// 既知の制約:
//   - LOD 境界では頂点密度が一致しない。TileMesh 側の skirt で隙間を隠す
//   - 1 フレームで作られるタイル数の上限は HARD_LIMIT で抑制
// =============================================================================

import {
  Camera,
  PerspectiveCamera,
  Vector3,
  Plane,
  Ray,
  Frustum,
  Matrix4,
  Sphere,
} from 'three/webgpu';

import {
  MAX_ZOOM,
  MIN_ZOOM,
} from '../config';

import {
  tileKeyString,
  tileSizeAndCenterENU,
  tilesCoveringEnuBounds,
  type TileKey,
} from '../geo/tileMath';

// 海面平面（標高 0m）。フラスタム射影に使う
const GROUND_PLANE = new Plane(new Vector3(0, 1, 0), 0);

// 1 ピクセルあたり何「テクセル」分の許容誤差か。
// 256 だと「タイル全体が画面で 256px ぴったり」を理想とする = 1 ピクセル 1 テクセル相当。
// 小さくすると分割が深くなって近景がより鮮明、大きくするとロードが軽くなる。
const TARGET_SSE = 384;

// フラスタム交点の最大距離（m）。これより遠くは「視野外」として扱う。
// 200km あれば十分な広さの遠景が見える。
const MAX_VIEW_DISTANCE_M = 200_000;

// 1 フレームで採用する最大タイル数（暴発防止）。
// DEM 原寸メッシュは 1 タイルあたり 256x256 頂点なので、旧テクスチャ変位版より
// 少なめに抑える。近傍優先の切り詰めは下で行う。
const HARD_LIMIT = 96;

const DEG2RAD = Math.PI / 180;

// -----------------------------------------------------------------------------
// NDC からのレイ生成 + 地表交点（max distance クランプ付き）
// -----------------------------------------------------------------------------

const _near = new Vector3();
const _far = new Vector3();
const _dir = new Vector3();
const _hit = new Vector3();

/**
 * NDC (ndcX, ndcY) ∈ [-1,1] からレイを作って地表平面 Y=0 との交点を返す。
 * 交点が無い（空を見ている）or 遠すぎる場合は、レイ方向に MAX_VIEW_DISTANCE 進んだ
 * 点を「擬似交点」として返す（後段の bounding box 計算が壊れないように）。
 */
function pickGround(camera: PerspectiveCamera, ndcX: number, ndcY: number): Vector3 {
  _near.set(ndcX, ndcY, -1).unproject(camera);
  _far.set(ndcX, ndcY, 1).unproject(camera);
  _dir.copy(_far).sub(_near).normalize();

  const ray = new Ray(camera.position.clone(), _dir.clone());

  if (ray.intersectPlane(GROUND_PLANE, _hit)) {
    const dist = _hit.distanceTo(camera.position);
    if (dist <= MAX_VIEW_DISTANCE_M) {
      return _hit.clone();
    }
  }
  // 空を向いていて交点なし、もしくは遠すぎる → カメラから MAX_VIEW_DISTANCE 進んだ点
  return camera.position.clone().addScaledVector(_dir, MAX_VIEW_DISTANCE_M);
}

/**
 * 視錐台（中央 + 四隅）から地表交点を求めて、その XZ 範囲のバウンディングを返す。
 * - x = East, z = North
 */
function frustumBoundsOnGround(camera: PerspectiveCamera) {
  const ndcs: Array<[number, number]> = [
    [ 0,  0],
    [-1, -1],
    [ 1, -1],
    [-1,  1],
    [ 1,  1],
  ];

  let minE = Infinity, maxE = -Infinity;
  let minN = Infinity, maxN = -Infinity;
  for (const [nx, ny] of ndcs) {
    const p = pickGround(camera, nx, ny);
    if (p.x < minE) minE = p.x;
    if (p.x > maxE) maxE = p.x;
    if (p.z < minN) minN = p.z;
    if (p.z > maxN) maxN = p.z;
  }
  return { minE, maxE, minN, maxN };
}

// -----------------------------------------------------------------------------
// Screen-Space Error 計算
// -----------------------------------------------------------------------------

/**
 * タイル中心 → カメラまでの距離と、画面 1 ラジアンあたりのピクセル数から、
 * このタイルが「画面上で何ピクセルくらいの大きさで映るか」を概算する。
 *
 * SSE = タイル幅[m] × (pixelsPerRad / 距離[m])
 *
 * - SSE が大きい = 画面上で大きく映る → 細分化して詳細にすべき
 * - SSE が小さい = 画面上で小さい → このまま採用してよい
 */
function screenSpaceError(tile: TileKey, camera: PerspectiveCamera): number {
  const { widthM, centerX, centerY } = tileSizeAndCenterENU(tile.z, tile.x, tile.y);

  // タイル中心の標高は不明だが、海抜 0m で十分（誤差は距離が遠いほど影響小）
  const dx = camera.position.x - centerX;
  const dy = camera.position.y;            // 標高 0 を仮定
  const dz = camera.position.z - centerY;
  const distance = Math.max(Math.sqrt(dx * dx + dy * dy + dz * dz), 1);

  const fovRad = camera.fov * DEG2RAD;
  const screenH = window.innerHeight;
  const pixelsPerRad = screenH / (2 * Math.tan(fovRad / 2));

  return (widthM * pixelsPerRad) / distance;
}

// -----------------------------------------------------------------------------
// 再帰的細分化
// -----------------------------------------------------------------------------

/**
 * 視錐台に含まれるかを大まかに判定（タイル中心 + 半径での球判定）。
 * 厳密な AABB 判定ではないが、視野外タイルの大半を弾けて十分。
 *
 * - 高さは未知（標高にバラつき）なので、半径方向に余裕を持たせて
 *   球の半径 = タイル対角 × 0.5 + 標高余裕 4000m とする
 *   （日本の最高峰 3776m を包含するための保険）
 */
const _tileSphere = new Sphere();
function tileLikelyInFrustum(tile: TileKey, frustum: Frustum): boolean {
  const { widthM, heightM, centerX, centerY } = tileSizeAndCenterENU(tile.z, tile.x, tile.y);
  const horizontalRadius = Math.hypot(widthM, heightM) * 0.5;
  // 標高方向の不確かさを吸収（Y=2000 を中心に半径 +4000）
  _tileSphere.center.set(centerX, 2000, centerY);
  _tileSphere.radius = horizontalRadius + 4000;
  return frustum.intersectsSphere(_tileSphere);
}

// 1 フレームの間だけ使う共有バッファ。selectVisibleTiles の冒頭で更新する
const _frustum = new Frustum();
const _projViewMat = new Matrix4();

function refine(
  tile: TileKey,
  camera: PerspectiveCamera,
  out: TileKey[],
): void {
  // 視錐台外ならスキップ（球判定で大半を弾く）
  if (!tileLikelyInFrustum(tile, _frustum)) return;

  // 最深部 or 十分小さい → 採用
  if (tile.z >= MAX_ZOOM) {
    out.push(tile);
    return;
  }
  const sse = screenSpaceError(tile, camera);
  if (sse < TARGET_SSE) {
    out.push(tile);
    return;
  }

  // 4 つの子に分割して再帰
  const z = tile.z + 1;
  const xb = tile.x * 2;
  const yb = tile.y * 2;
  refine({ z, x: xb,     y: yb     }, camera, out);
  refine({ z, x: xb + 1, y: yb     }, camera, out);
  refine({ z, x: xb,     y: yb + 1 }, camera, out);
  refine({ z, x: xb + 1, y: yb + 1 }, camera, out);
}

// -----------------------------------------------------------------------------
// 公開 API
// -----------------------------------------------------------------------------

export interface SelectedTiles {
  /** 結果に含まれる最大ズーム（HUD 表示用）。複数 LOD が混在する点に注意 */
  zoom: number;
  keys: TileKey[];
}

export interface TileSelectionOptions {
  fixedZoom?: number;
}

/**
 * 1 フレームに表示すべきタイル集合を返す。
 * クアッドツリー LOD で、近景は高 LOD、遠景は低 LOD のタイルを混在させる。
 */
export function selectVisibleTiles(
  camera: Camera,
  options: TileSelectionOptions = {},
): SelectedTiles {
  if (!(camera instanceof PerspectiveCamera)) {
    return { zoom: MIN_ZOOM, keys: [] };
  }

  // 視錐台を更新
  _projViewMat.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  _frustum.setFromProjectionMatrix(_projViewMat);

  if (options.fixedZoom !== undefined) {
    return selectFixedZoomTiles(camera, options.fixedZoom);
  }

  // 1. ルート（MIN_ZOOM）タイル候補
  const bounds = frustumBoundsOnGround(camera);
  const roots = tilesCoveringEnuBounds(bounds, MIN_ZOOM);

  // 2. 再帰細分化
  const result: TileKey[] = [];
  for (const root of roots) {
    refine(root, camera, result);
  }

  // 3. 暴発抑制: 多すぎたらカメラ近傍を優先して切り詰め
  if (result.length > HARD_LIMIT) {
    result.sort((a, b) => {
      const ca = tileSizeAndCenterENU(a.z, a.x, a.y);
      const cb = tileSizeAndCenterENU(b.z, b.x, b.y);
      const da = (camera.position.x - ca.centerX) ** 2 + (camera.position.z - ca.centerY) ** 2;
      const db = (camera.position.x - cb.centerX) ** 2 + (camera.position.z - cb.centerY) ** 2;
      return da - db;
    });
    result.length = HARD_LIMIT;
  }

  // HUD 用に「最大ズーム」を返す（実際は混在）
  let maxZ = MIN_ZOOM;
  for (const k of result) if (k.z > maxZ) maxZ = k.z;

  return { zoom: maxZ, keys: result };
}

function selectFixedZoomTiles(camera: PerspectiveCamera, zoom: number): SelectedTiles {
  const safeZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Math.round(zoom)));
  const bounds = frustumBoundsOnGround(camera);
  const candidates = tilesCoveringEnuBounds(bounds, safeZoom).filter((tile) =>
    tileLikelyInFrustum(tile, _frustum),
  );

  // fixed zoom は LOD 境界を作らない代わりに候補が増えやすい。
  // 既存の HARD_LIMIT と同じくカメラ近傍を優先し、全て同じ zoom のまま切り詰める。
  if (candidates.length > HARD_LIMIT) {
    candidates.sort((a, b) => {
      const ca = tileSizeAndCenterENU(a.z, a.x, a.y);
      const cb = tileSizeAndCenterENU(b.z, b.x, b.y);
      const da = (camera.position.x - ca.centerX) ** 2 + (camera.position.z - ca.centerY) ** 2;
      const db = (camera.position.x - cb.centerX) ** 2 + (camera.position.z - cb.centerY) ** 2;
      return da - db;
    });
    candidates.length = HARD_LIMIT;
  }

  return { zoom: safeZoom, keys: candidates };
}

/** 表記揺れ防止: keys を Set<string> 化 */
export function asKeyStrings(sel: SelectedTiles): Set<string> {
  return new Set(sel.keys.map((k) => tileKeyString(k)));
}
