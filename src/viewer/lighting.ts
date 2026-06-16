// =============================================================================
// 環境光・指向光のセットアップ
// -----------------------------------------------------------------------------
// 山の凹凸が見えるように、AmbientLight（全方位）と DirectionalLight（太陽光）を
// 1 つずつ用意。MeshStandardNodeMaterial のシェーディングが活きる。
// =============================================================================

import {
  Scene,
  AmbientLight,
  DirectionalLight,
} from 'three/webgpu';

/**
 * 与えられた Scene にライトを追加する。
 * - AmbientLight: 谷の影が真っ黒にならないよう、全体を弱く持ち上げる
 * - DirectionalLight: 山肌に陰影を出すための「太陽」。南東上空からあてる
 */
export function setupLighting(scene: Scene): void {
  // ---- 環境光（弱め）----
  const ambient = new AmbientLight(0xffffff, 0.45);
  scene.add(ambient);

  // ---- 指向光（太陽）----
  const sun = new DirectionalLight(0xffffff, 1.4);
  // 南東上空からあてる（ENU では x=East+, z=North+, y=Up+）
  sun.position.set(8000, 12000, -6000);
  sun.target.position.set(0, 0, 0);
  scene.add(sun);
  scene.add(sun.target);
}
