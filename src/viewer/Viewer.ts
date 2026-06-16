// =============================================================================
// Viewer: WebGPURenderer / Scene / Camera を統括する薄いラッパ
// -----------------------------------------------------------------------------
// 役割:
//   - WebGPU の初期化（async）
//   - Scene/Camera/Lighting/Controls の生成
//   - レンダループ（setAnimationLoop）
//   - ウィンドウリサイズ追従
//
// 使い方:
//   const viewer = new Viewer(document.body);
//   await viewer.init();
//   viewer.scene.add(...);
//   viewer.onUpdate = (camera) => terrain.update(camera);
//   viewer.start();
// =============================================================================

import {
  Scene,
  PerspectiveCamera,
  WebGPURenderer,
  Color,
  Vector3,
  type Camera,
} from 'three/webgpu';

import { setupLighting } from './lighting';
import {
  CameraController,
  type CameraOrbitPivotResolver,
} from './CameraController';
import {
  VIEWER_INITIAL_CAMERA_POSITION_M,
  VIEWER_INITIAL_CAMERA_TARGET_M,
} from '../config';

export interface ViewerRenderStats {
  updateFrames: number;
  renderedFrames: number;
  renderIntervalFrames: number;
}

export interface ViewerCameraViewState {
  position: Vector3;
  target: Vector3;
}

export class Viewer {
  /** Three.js シーン（外から add してもらう） */
  readonly scene = new Scene();

  /** 透視カメラ */
  readonly camera: PerspectiveCamera;

  /** WebGPU レンダラ */
  readonly renderer: WebGPURenderer;

  /** マウス操作 */
  private readonly cameraController: CameraController;

  /** 親 DOM（canvas をぶら下げる場所） */
  private readonly container: HTMLElement;

  /** レンダループから呼ばれるユーザフック */
  onUpdate: ((camera: Camera) => void) | undefined = undefined;

  /** Three.js の scene 描画が終わった直後に呼ぶ追加描画フック。overlay pass などに使う。 */
  onAfterRender: ((camera: Camera) => void) | undefined = undefined;

  /** `renderer.render()` を何 update frame ごとに実行するか。1 なら毎 frame 描画。 */
  private renderIntervalFrames = 1;

  /** 間引き判定用の update frame 累積数。 */
  private updateFrameCounter = 0;

  /** HUD 用の累積 update frame 数。 */
  private totalUpdateFrames = 0;

  /** HUD 用の累積 render 実行数。 */
  private totalRenderedFrames = 0;

  constructor(container: HTMLElement, device?: GPUDevice) {
    this.container = container;

    // ---- カメラ ----
    // FOV 50, near 1m, far 500km（広域から地表近接まで通せる範囲）
    this.camera = new PerspectiveCamera(50, 1, 1, 500_000);

    // 初期視点: Google Earth 風に「富士山を中心に俯瞰する高高度視点」。
    // ENU では x=East+, y=Up+, z=North+。富士山頂は y=3776m。
    // - 東 25km / 上空 18km / 南 25km から見下ろす（距離 ≒ 40km）
    //   → 富士山と周囲の湖沼・南アルプス・駿河湾までが視野に入る
    this.camera.position.set(
      VIEWER_INITIAL_CAMERA_POSITION_M.x,
      VIEWER_INITIAL_CAMERA_POSITION_M.y,
      VIEWER_INITIAL_CAMERA_POSITION_M.z,
    );

    // ---- レンダラ ----
    this.renderer = new WebGPURenderer({ antialias: true, device });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    // setSize は後段の updateSize() で確定させるが、最低限の初期化はここで
    // 空の色（薄い水色）。地表が無い箇所は空が見える
    this.renderer.setClearColor(new Color(0x9fc5e8), 1.0);

    // canvas を DOM に貼る
    container.appendChild(this.renderer.domElement);

    // ---- 操作 ----
    this.cameraController = new CameraController(this.camera, this.renderer.domElement);

    // ---- ライト ----
    setupLighting(this.scene);

    // ---- 背景色（薄い水色）----
    this.scene.background = new Color(0x9fc5e8);
  }

  /**
   * shift+左ドラッグ / 右ドラッグ回転で使う pivot 解決 callback を登録する。
   * Viewer は Terrain を生成しないため、main.ts から地形 raycast を差し込む。
   */
  setCameraOrbitPivotResolver(resolver: CameraOrbitPivotResolver | undefined): void {
    this.cameraController.setOrbitPivotResolver(resolver);
  }

  /** 計算領域リサイズ中だけ、カメラ操作を wheel zoom のみに絞る。 */
  setCameraZoomOnlyMode(enabled: boolean): void {
    this.cameraController.setZoomOnlyMode(enabled);
  }

  /**
   * 指定した中心と半径を画面へ収める。
   * Import モードでは既定の富士山視点からデータ範囲へ一気に移動する必要がある。
   */
  focusOn(center: Vector3, radius: number): void {
    const distance = Math.max(300, radius * 2.4);
    this.camera.position.set(
      center.x + distance * 0.72,
      center.y + distance * 0.62,
      center.z - distance * 0.72,
    );
    this.cameraController.setTarget(center);
    this.camera.updateProjectionMatrix();
  }

  /** タブ切替で視点を復元できるよう、カメラ位置と MapControls target を一緒に保存する。 */
  getCameraViewState(): ViewerCameraViewState {
    return {
      position: this.camera.position.clone(),
      target: this.cameraController.controls.target.clone(),
    };
  }

  /** 保存済みのカメラ位置と注視点を復元する。target も戻さないと旋回中心がずれる。 */
  setCameraViewState(state: ViewerCameraViewState): void {
    this.camera.position.copy(state.position);
    this.cameraController.setTarget(state.target);
    this.camera.updateProjectionMatrix();
    this.camera.updateMatrixWorld(true);
  }

  /**
   * 描画間引きの倍率を設定する。
   * - 1: 毎 update frame 描画
   * - 2: 2 update frame に 1 回だけ描画
   *
   * MPM 計算や Terrain.update は毎 update frame 継続し、ここでは GPU render だけを間引く。
   */
  setRenderIntervalFrames(value: number): void {
    this.renderIntervalFrames = Math.min(120, Math.max(1, Math.floor(value)));
    // 値を変えた直後に古い余りで長く描画が止まらないよう、次 frame で描画される状態へ戻す。
    this.updateFrameCounter = 0;
  }

  getRenderStats(): ViewerRenderStats {
    return {
      updateFrames: this.totalUpdateFrames,
      renderedFrames: this.totalRenderedFrames,
      renderIntervalFrames: this.renderIntervalFrames,
    };
  }

  /**
   * WebGPU の初期化（非同期）。
   * - WebGPU 非対応ブラウザは内部で例外を投げるので、呼び出し側で catch 推奨
   */
  async init(): Promise<void> {
    // WebGPU 機能チェック
    if (!('gpu' in navigator)) {
      throw new Error('WebGPU not supported in this browser');
    }
    await this.renderer.init();

    // 初期サイズと、ウィンドウサイズ変更追従
    this.updateSize();
    window.addEventListener('resize', () => this.updateSize());
  }

  /** レンダラとカメラのアスペクトを window 全体に揃える */
  private updateSize(): void {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** レンダループ開始 */
  start(): void {
    this.renderer.setAnimationLoop(() => this.frame());
  }

  /** 1 フレーム分の更新と描画 */
  private frame(): void {
    this.totalUpdateFrames++;
    this.updateFrameCounter++;

    // OrbitControls を回す（damping を進める）
    this.cameraController.update();

    // ユーザフック（Terrain.update を呼ぶ想定）
    this.onUpdate?.(this.camera);

    if (this.updateFrameCounter < this.renderIntervalFrames) {
      return;
    }
    this.updateFrameCounter = 0;
    this.totalRenderedFrames++;

    // r177 以降、WebGPURenderer は同期的な render() を推奨
    // （renderAsync は deprecated。await renderer.init() 後は render() で OK）
    this.renderer.render(this.scene, this.camera);

    this.onAfterRender?.(this.camera);
  }
}
