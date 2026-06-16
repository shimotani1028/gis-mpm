// Vite が提供するクライアント側アンビエント型を取り込む。
// - import.meta.env など Vite 固有 API
// - "*?worker" / "*?worker&inline" / "*?url" 等のサブパス import の型
/// <reference types="vite/client" />

// WGSL を Vite の `?raw` import で文字列として読み込む。
// MLS-MPM の compute shader は raw WebGPU API へそのまま渡す。
declare module '*.wgsl?raw' {
  const source: string;
  export default source;
}
