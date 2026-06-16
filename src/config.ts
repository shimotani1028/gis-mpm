// =============================================================================
// アプリ全体で参照する定数
// -----------------------------------------------------------------------------
// ここを書き換えるだけで「対象地域」「ズーム範囲」「キャッシュ容量」などを
// 1 ヶ所で調整できる。コード本体は基本ここを import する。
// =============================================================================

/**
 * 表示の中心となる経緯度（Area Of Interest = AOI の中心）。
 * - lat: 緯度（北が +）
 * - lon: 経度（東が +）
 *
 * デフォルトは富士山頂（35.3606°N, 138.7274°E）。
 * 別の山に切り替えたい場合は、ここを書き換えるだけで初期視点と
 * ローカル ENU 座標系の原点が一気に変わる。
 */
export const AOI_CENTER = { lat: 35.3606, lon: 138.7274 } as const;

/**
 * クアッドツリー LOD の「ルート」になる最小ズームレベル。
 * - z=6 だと 1 タイル ≒ 600km 四方。日本全体を視野に入れる場合の起点として十分。
 * - 大きすぎる（z=4 等）と無駄に粗い遠景まで読みに行くので注意
 */
export const MIN_ZOOM = 6;

/**
 * 国土地理院 DEM タイルの最大ズームレベル。
 * - 14: DEM10B（10m メッシュ）→ 1 タイル ≒ 2.4km 四方を 256x256 標高点で表現
 * - 15: DEM5A/5B/5C（5m メッシュ）→ より細かい地表起伏が見える
 * - z=15 のリクエストでは tileWorker が 5A→5B→5C→親 DEM10B の順にピクセル単位でフォールバックする
 */
export const MAX_ZOOM = 15;

/**
 * z=15 で DEM5A→5B→5C の順に試す URL テンプレート。
 * - 5A: 航空レーザ測量（最も高精度、主に山地）
 * - 5B: 写真測量（平野部・市街地）
 * - 5C: 5m メッシュ（その他）
 * - 404 またはタイル内の無効ピクセルは GSI_DEM_URL（DEM10B）の親タイルで補完する
 */
export const GSI_DEM5_URLS = [
  'https://cyberjapandata.gsi.go.jp/xyz/dem5a_png/{z}/{x}/{y}.png',
  'https://cyberjapandata.gsi.go.jp/xyz/dem5b_png/{z}/{x}/{y}.png',
  'https://cyberjapandata.gsi.go.jp/xyz/dem5c_png/{z}/{x}/{y}.png',
] as const;

/** DEM5 系が提供される最大ズーム。これ未満では DEM10B（GSI_DEM_URL）を使う。 */
export const DEM5_MIN_ZOOM = 15;

/**
 * LRU キャッシュに保持するタイル数の上限。
 * - DEM 原寸メッシュでは 1 タイルあたりの頂点・インデックスが大きい。
 * - 160 枚で視点を戻した時の再ロードを抑えつつ、GPU メモリの膨張を避ける。
 */
export const TILE_CACHE_LIMIT = 160;

/**
 * Web Worker の並列数。
 * - GSI に対する同時 fetch 数の上限になる
 * - 多すぎると GSI 側に迷惑、少なすぎるとロードが詰まる
 * - 4 がブラウザ HTTP 並列数と整合的で実用的
 */
export const WORKER_POOL_SIZE = 4;

/**
 * タイル外周に追加する skirt の深さ [m]。
 * - 地表面自体は隣接タイルと重ねず、外周から下向きの壁だけを足す。
 * - LOD 境界で頂点密度が違っても、隙間から空色が見えるのを防ぐための値。
 * - 大きすぎるとタイル境界が断層状の壁として見えるため、必要最小限に抑える。
 */
export const TILE_SKIRT_DEPTH_M = 30;

/**
 * 国土地理院 DEM PNG タイルの URL テンプレート。
 * - {z}/{x}/{y} に数値を埋めて使う
 * - DEM10B（10m メッシュ・標高）が標準
 * - 利用規約: https://maps.gsi.go.jp/development/ichiran.html
 */
export const GSI_DEM_URL =
  'https://cyberjapandata.gsi.go.jp/xyz/dem_png/{z}/{x}/{y}.png';

/**
 * 1 タイル PNG の解像度（ピクセル）。
 * - GSI 標準は 256x256
 * - DEM 原寸メッシュの頂点数を決める
 */
export const TILE_PIXELS = 256;

/**
 * 1 フレームで新規発行できるタイルリクエスト数の上限。
 * - 視点が急に動いたとき、欲しいタイルが何十枚も同時に出るのを防ぐ
 * - 実装上は TileSelector → Terrain で「足りない分のうち、近いものから N 枚だけ」採用する
 * - クアッドツリー LOD ではフレーム当たりの新規タイルが増えるので 12 に拡張
 */
export const MAX_REQUESTS_PER_FRAME = 12;

// =============================================================================
// MLS-MPM ダムブレイク設定
// -----------------------------------------------------------------------------
// ここでは、地形統合前の検証用として「直方体ボックス内の粘性流体」を扱う。
// 座標系は MPM の計算格子単位で、描画側では別途 MPM_RENDER_SCALE_M を掛けて
// ローカル ENU メートルへ写像する。
// =============================================================================

/** 計算格子のセル数。数十万粒子での解析を優先し、粒子密度に見合う 64^3 を既定にする。 */
export const MPM_GRID_SIZE = { x: 64, y: 64, z: 64 } as const;

/** 初期粒子数。WebGPU-Ocean と同じ桁の粒子数を最初から見られる既定値にする。 */
export const MPM_PARTICLE_COUNT = 500_000;

/** UI から指定できる粒子数の上限。GPU メモリを食い潰さないための安全弁。 */
export const MPM_PARTICLE_COUNT_MAX = 500_000;

/** UI から指定できる MPM 計算領域の X/Z 最大長 [m]。過大な DEM coverage fetch を防ぐ。 */
export const MPM_MAX_BOX_XZ_LENGTH_M = 20_000;

/** MPM 計算領域の DEM coverage として一度に読む z=MAX_ZOOM タイル数の上限。 */
export const MPM_MAX_COVERAGE_TILE_COUNT = 256;

/**
 * 地形衝突用 heightfield の 1 辺サンプル数。
 * 描画用 DEM は高精細のまま維持し、MPM が毎 step 読む接触面だけを粗くして
 * Reset 時の CPU sampling と GPU storage read の局所性を改善する。
 */
export const MPM_TERRAIN_BOUNDARY_RESOLUTION = 384;

/** grid 側で地形近傍セルを減速する厚み [MPM grid cell]。最終補正は粒子側で行う。 */
export const MPM_TERRAIN_GRID_COLLISION_THICKNESS_CELLS = 0.05;

/** g2p で粒子を地形面へ戻すときに残す最小隙間 [MPM grid cell]。 */
export const MPM_TERRAIN_PARTICLE_CLEARANCE_CELLS = 0.02;

/** 初期粒子底面を「各粒子地点の DEM 地形 + この小さな隙間」に置く既定値 [m]。 */
export const MPM_INITIAL_TERRAIN_LIFT_M = 0;

/** 地表面の水層観察で過大な浮きを作らないための surface gap 上限 [m]。 */
export const MPM_INITIAL_TERRAIN_LIFT_MAX_M = 50;

/** DEM に沿って初期配置する水層の既定厚み [m]。大粒子数で流れの厚みが見える値にする。 */
export const MPM_INITIAL_WATER_DEPTH_M = 500;

/** 1 回の GPU command buffer にまとめる MPM substep 数。描画フレームとは独立して実行する。 */
export const MPM_STEPS_PER_BATCH = 4;

/** MLS-MPM の時間刻み。計算格子単位の速度に掛かる値。 */
export const MPM_DT = 0.05;

/** 弱圧縮性流体の Tait 型圧力項に掛ける剛性。 */
export const MPM_STIFFNESS = 3.0;

/** 参照密度。Zenn 記事の参照実装と同じ値を初期値にする。 */
export const MPM_REST_DENSITY = 4.0;

/** 流速コンターの初期下限。速度 0 は青として表示する。 */
export const MPM_FLUID_CONTOUR_SPEED_MIN = 0;

/** 流速コンターの初期上限。地形上の表面流で色差が読める経験値。 */
export const MPM_FLUID_CONTOUR_SPEED_MAX = 0.5;

/** 密度比コンターの初期下限。基準密度の半分を青側の基準にする。 */
export const MPM_FLUID_CONTOUR_DENSITY_MIN = 0.5;

/** 密度比コンターの初期上限。基準密度の 1.5 倍を赤側の基準にする。 */
export const MPM_FLUID_CONTOUR_DENSITY_MAX = 1.5;

/** 粘性係数。速度勾配の対称成分に掛ける。 */
export const MPM_DYNAMIC_VISCOSITY = 0.01;

/** 右上パネルで初期表示する、地形上の局所計算領域 [m]。 */
export const MPM_DEFAULT_BOX_M = {
  originX: 930.94,
  originZ: -5_736.27,
  xLength: 4_831.4,
  yLength: 1_466.8,
  zLength: 4_749.78,
} as const;

/** 初期流体 footprint [m]。計算領域を広げても水の初期範囲はこの値に固定する。 */
export const MPM_DEFAULT_FLUID_REGION_M = {
  originX: 930.94,
  originZ: -3_813.48,
  xLength: 2_731.5,
  zLength: 2_826.99,
} as const;

/**
 * 初期流体の物理粒子間隔 [m]。
 * - active particles は `流体体積 / spacing^3` で決め、`MPM_PARTICLE_COUNT` を上限にする。
 * - 既定値は、スクリーンショットで調整済みの初期状態に合わせる。
 * - 描画上の見た目の大きさは `MPM_PARTICLE_SIZE_CELLS` で別に調整する。
 */
export const MPM_PARTICLE_SPACING_M = 30;

/** 小さい建物まわりを解像できるよう、UI で許す物理粒子間隔の下限 [m]。 */
export const MPM_PARTICLE_SPACING_MIN_M = 0.02;

/** 計算領域外周の壁に掛ける初期摩擦係数。 */
export const MPM_DOMAIN_FRICTION = 0.04;

/** 地形高さ場の壁に掛ける初期摩擦係数。 */
export const MPM_TERRAIN_FRICTION = 0.04;

/** 建物など、地形ではない障害物 heightfield に掛ける初期摩擦係数。 */
export const MPM_OBJECT_FRICTION = 0.25;

/** 建物などの障害物用反射率。流体境界では既定で跳ね返らせず、法線侵入を止める。 */
export const MPM_OBJECT_RESTITUTION = 0.0;

/** 地形境界の反射率。既定値は建物側面と同じく非反発にする。 */
export const MPM_BOUNDARY_RESTITUTION = MPM_OBJECT_RESTITUTION;

/** 計算領域外周の反射率。既定値は建物側面と同じく非反発にする。 */
export const MPM_DOMAIN_RESTITUTION = MPM_OBJECT_RESTITUTION;

/** WebGPU の integer atomicAdd で float 量を扱うための固定小数点倍率。 */
export const MPM_FIXED_POINT_MULTIPLIER = 1e7;

/** y 方向重力。MPM 計算座標では y+ が上。 */
export const MPM_GRAVITY_Y = -0.12;

/** 粒子を Three.js の ENU メートル座標へ写像する倍率。 */
export const MPM_RENDER_SCALE_M = 140;

/** MPM ボックスを初期視点の中心近くに置く描画オフセット。 */
export const MPM_RENDER_ORIGIN_M = { x: 930.94, y: 1_745.56, z: -5_736.27 } as const;

/** 初期カメラ位置 [m]。 */
export const VIEWER_INITIAL_CAMERA_POSITION_M = {
  x: 8_100,
  y: 11_150,
  z: -8_200,
} as const;

/** 初期カメラの注視点 [m]。 */
export const VIEWER_INITIAL_CAMERA_TARGET_M = {
  x: 2_500,
  y: 2_650,
  z: -2_900,
} as const;

/** 粒子を小さなキューブで描くときの一辺長 [MPM grid cell]。 */
export const MPM_PARTICLE_SIZE_CELLS = 0.06;

/** 起動時にスクリーンスペース流体レンダリングを有効にするか。 */
export const MPM_FLUID_RENDER_ENABLED = true;

/**
 * 流体レンダリング用の粒子半径 [MPM grid cell]。
 * 物理粒子間隔より少し大きめの billboard sphere として描き、depth を平滑化して連続面に見せる。
 * 実際の描画では固定値としてではなく、実効 particle spacing に基づく上限として使う。
 */
export const MPM_FLUID_RENDER_RADIUS_CELLS = 1.05;

/** water render の球半径を、実効 particle spacing の何倍にするか。 */
export const MPM_FLUID_RENDER_RADIUS_SPACING_MULTIPLIER = 0.95;

/**
 * spacing をかなり小さくした時でも水面として読める最低半径 [m]。
 * water render は UI の `particle spacing [m]` へ直接追従させるため、下限も
 * grid cell ではなく物理メートルで持つ。これにより小さな街区 import でも、
 * box サイズに依存せず spacing 変更が見た目へ出る。
 */
export const MPM_FLUID_RENDER_MIN_RADIUS_M = 0.9;

/**
 * 流体用 offscreen texture の解像度倍率。
 * `devicePixelRatio * scale` で描くため、Retina では 0.5 でも CSS pixel 相当の解像度になる。
 */
export const MPM_FLUID_RENDER_RESOLUTION_SCALE = 0.5;

/** 流体 overlay の DPR 上限。高 DPI で depth/thickness/blur pass が重くなりすぎるのを防ぐ。 */
export const MPM_FLUID_RENDER_MAX_DPR = 2;

/** 合成時の最大不透明度。地形を残しつつ、水塊として読める値にする。 */
export const MPM_FLUID_RENDER_ALPHA = 0.78;

/** thickness pass で 1 particle が足す厚み量。大きくすると深い青と不透明感が強くなる。 */
export const MPM_FLUID_RENDER_THICKNESS_SCALE = 0.032;

/** 何 update frame ごとに 1 回描画するか。1 なら毎 frame 描画。 */
export const RENDER_INTERVAL_FRAMES = 2;
