// =============================================================================
// 地形タイル用マテリアル
// -----------------------------------------------------------------------------
// DEM 原寸メッシュ化後は、標高による頂点変位も色分けも CPU 側で済ませる。
// ここでは Three.js 標準マテリアルに vertex color とライト反応だけを任せる。
// =============================================================================

import {
  FrontSide,
  MeshStandardMaterial,
} from 'three/webgpu';

/**
 * タイル 1 枚分の地形マテリアルを生成する。
 *
 * 設計メモ:
 * - `vertexColors: true` により、TileMesh が焼き込んだ標高カラーを使う。
 * - `FrontSide` のみ描画する。CPU 側で正しい winding を作るので、旧実装の
 *   `DoubleSide` による裏面ちらつき回避は不要になる。
 */
export function buildTerrainMaterial(): MeshStandardMaterial {
  return new MeshStandardMaterial({
    vertexColors: true,
    metalness: 0.0,
    roughness: 1.0,
    flatShading: false,
    side: FrontSide,
  });
}
