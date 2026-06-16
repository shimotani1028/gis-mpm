// Vite 設定ファイル
// - TypeScript ワーカー（`new Worker(new URL('./tileWorker.ts', import.meta.url), { type: 'module' })`）を有効化
// - 開発サーバの host / port を明示
// - WebGPU テスト用に COOP/COEP ヘッダは特に不要（同一オリジンで動作）
import { defineConfig } from 'vite';

const ALLOW_LAN_DEV_SERVER = process.env.LOCAL_MPM_ALLOW_LAN === '1';
const ENABLE_PRODUCTION_SOURCEMAP = process.env.LOCAL_MPM_SOURCEMAP === '1';

export default defineConfig({
  // dev サーバは既定 localhost 限定。LAN 確認が必要な時だけ LOCAL_MPM_ALLOW_LAN=1 で公開する。
  server: {
    host: ALLOW_LAN_DEV_SERVER ? true : '127.0.0.1',
    port: 5173,
    strictPort: true,
  },
  // ビルド設定：WebGPU は ES2022 が前提なので target を高めに
  build: {
    target: 'es2022',
    // source map は公開配布時の source disclosure になるため、必要な private debug 時だけ出す。
    sourcemap: ENABLE_PRODUCTION_SOURCEMAP,
  },
  // Worker のデフォルトタイプを ESM に（`type: "module"` 指定の手間を減らす）
  worker: {
    format: 'es',
  },
});
