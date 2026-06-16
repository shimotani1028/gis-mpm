// =============================================================================
// MapControls ラッパ
// -----------------------------------------------------------------------------
// - マウス左ドラッグ: 移動（パン）
// - マウス右ドラッグ: 回転（画面中心まわり）
// - Shift + 左ドラッグ: 回転（画面中心まわり）
// - ホイール: 拡大縮小（カーソル位置へ寄るズーム）
//
// 回転については、shift+左ドラッグ / 右ドラッグの開始時に、画面中心 ray が
// 最初に当たる地点を `controls.target` にセットする。アプリ側から地形交差点を
// 返す callback が登録されていればそれを優先し、地形未ロードや ray miss 時だけ
// 地面 (y=0) との交点へ fallback する。
//
// ズームは `zoomToCursor` を有効にすることで、ターゲットがカーソル位置に向けて
// 移動する。これによりズーム倍率が距離に対して相対値（一定）のまま、地表
// すれすれまで連続的に寄ることができる（従来は target が高高度に取り残されて
// 途中で寄れなくなっていた）。
// =============================================================================

import { Vector3, type PerspectiveCamera } from 'three/webgpu';
import { MapControls } from 'three/addons/controls/MapControls.js';

import { VIEWER_INITIAL_CAMERA_TARGET_M } from '../config';

/** 回転 pivot に使うワールド座標を返す callback。undefined なら fallback に任せる。 */
export type CameraOrbitPivotResolver = () => Vector3 | undefined;

export class CameraController {
  /** 内部の MapControls インスタンス（毎フレーム update を呼ぶ） */
  readonly controls: MapControls;

  private readonly camera: PerspectiveCamera;
  private orbitPivotResolver: CameraOrbitPivotResolver | undefined = undefined;

  constructor(camera: PerspectiveCamera, dom: HTMLElement) {
    this.camera = camera;
    this.controls = new MapControls(camera, dom);

    // ---- 操作感の調整 ----
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    // ズーム: カーソル位置に寄りながらドリー。これにより target が手前へ
    // 引き寄せられ、距離に対する一定倍率のままで地表近くまで連続的に寄れる。
    this.controls.zoomToCursor = true;
    this.controls.zoomSpeed = 1.2;

    // ズーム範囲: 地表 10m 〜 上空 400km。
    this.controls.minDistance = 10;
    this.controls.maxDistance = 400_000;

    // 初期ターゲット: 富士山頂（標高 3776m）
    this.controls.target.set(
      VIEWER_INITIAL_CAMERA_TARGET_M.x,
      VIEWER_INITIAL_CAMERA_TARGET_M.y,
      VIEWER_INITIAL_CAMERA_TARGET_M.z,
    );

    // 上下回転の制限
    this.controls.minPolarAngle = 0.1;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.05;

    // ---- 回転ピボットを「画面中心の地表点」に合わせる ----
    // 回転ジェスチャ開始時 (shift+左 / 右) に target を再計算する。
    // OrbitControls 既定では左=パン / 右=回転、左+修飾キー=回転となる。
    // Ctrl は MPM 計算領域の全体移動に使うため、ここでは Shift / Meta だけを
    // 左ドラッグ回転の pivot 更新対象にする。
    dom.addEventListener('pointerdown', (e) => {
      const isRightRotate = e.button === 2;
      const isShiftLeftRotate =
        e.button === 0 && (e.shiftKey || e.metaKey);
      if (isRightRotate || isShiftLeftRotate) {
        this.recenterTargetToScreenCenter();
      }
    });
  }

  /**
   * 回転開始時の pivot を外部から解決する callback を登録する。
   *
   * CameraController は Terrain を直接知らない。これにより Viewer 単体の責務を
   * 軽く保ったまま、main.ts 側で「現在ロード済み地形への raycast」を差し込める。
   */
  setOrbitPivotResolver(resolver: CameraOrbitPivotResolver | undefined): void {
    this.orbitPivotResolver = resolver;
  }

  /**
   * 計算領域のマウスリサイズ中は、領域面の選択を優先するため pan/rotate を止める。
   * wheel zoom は残し、範囲調整中でも視点距離だけは変えられるようにする。
   */
  setZoomOnlyMode(enabled: boolean): void {
    this.controls.enablePan = !enabled;
    this.controls.enableRotate = !enabled;
    this.controls.enableZoom = true;
  }

  /** 外部からカメラ注視点を明示的に設定する。Import サンプル読み込み時に使う。 */
  setTarget(target: Vector3): void {
    this.controls.target.copy(target);
    this.controls.update();
  }

  /**
   * 回転 pivot を画面中心に見えている地点へ寄せる。
   *
   * 優先順位:
   * 1. アプリ側 callback が返す地形交差点
   * 2. 従来の y=0 平面交差点
   *
   * どちらも視線方向上の点なので、カメラ姿勢は変えずに pivot だけを差し替えられる。
   */
  private recenterTargetToScreenCenter(): void {
    const resolvedPivot = this.orbitPivotResolver?.();
    if (resolvedPivot !== undefined && isFiniteVector3(resolvedPivot)) {
      this.controls.target.copy(resolvedPivot);
      return;
    }

    this.recenterTargetToGroundPlane();
  }

  /**
   * 従来の fallback。地形がまだ読めていない初期ロード中や、画面中心 ray が
   * 可視タイルへ当たらないケースでは、y=0 平面との交点を pivot にする。
   */
  private recenterTargetToGroundPlane(): void {
    const pos = this.camera.position;
    const fwd = new Vector3().subVectors(this.controls.target, pos);
    const len = fwd.length();
    if (len < 1e-6) return;
    fwd.divideScalar(len);

    // 前方が下を向いていない（または水平）なら、地面と交わらないので skip
    if (fwd.y >= -1e-4) return;

    const t = -pos.y / fwd.y; // y=0 平面までの距離
    if (t <= 0) return;

    this.controls.target.copy(pos).addScaledVector(fwd, t);
  }

  /** レンダループから毎フレーム呼ぶ */
  update(): void {
    this.controls.update();
  }

  /** 解放（必要なら） */
  dispose(): void {
    this.controls.dispose();
  }
}

function isFiniteVector3(v: Vector3): boolean {
  return Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z);
}
