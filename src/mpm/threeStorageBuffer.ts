import type {
  StorageInstancedBufferAttribute,
  WebGPURenderer,
} from 'three/webgpu';

interface WebGPUBackendData {
  buffer?: GPUBuffer;
}

interface WebGPUBackendWithDataMap {
  get(object: object): WebGPUBackendData;
}

/**
 * Three.js の StorageInstancedBufferAttribute に、こちらで作った raw GPUBuffer を割り当てる。
 *
 * Three.js WebGPU backend は attribute ごとの GPUBuffer を内部 DataMap に保持する。
 * 先に同じ場所へ buffer を登録しておくと、描画パイプラインはその buffer を
 * VERTEX/STORAGE attribute として使う。これにより compute shader が書いた位置を
 * CPU readback なしでそのまま粒子描画に渡せる。
 */
export function bindExternalStorageBuffer(
  renderer: WebGPURenderer,
  attribute: StorageInstancedBufferAttribute,
  buffer: GPUBuffer,
): void {
  const backend = renderer.backend as unknown as WebGPUBackendWithDataMap;
  backend.get(attribute).buffer = buffer;
}
