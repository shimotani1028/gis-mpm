// =============================================================================
// TileLoader: Worker プールでタイルを並列ロードするマネージャ
// -----------------------------------------------------------------------------
// 機能:
//   - WORKER_POOL_SIZE 個のワーカーを起動して保持
//   - request(key) で Promise<TileData> を返す。同じキーへの重複要求は同じ
//     Promise を返してネットワーク・デコードを節約
//   - abort(key) で「もう要らない」と通知し、Promise は必ず reject して settle させる
//   - worker reqId は tile key とは別の一意 ID にし、古い応答を安全に無視する
//
// なぜプールにするか:
//   - 1 ワーカーだとリクエストが直列化されて 60fps を割る
//   - ブラウザの HTTP 並列数（同一オリジン 6 程度）と整合的に 4〜6 が妥当
// =============================================================================

import TileWorker from './tileWorker?worker';
import type { WorkerRequest, WorkerResponse, WorkerSuccess } from './tileWorker';
import { tileKeyString, type TileKey } from '../geo/tileMath';
import { WORKER_POOL_SIZE } from '../config';
import { GSI_TERRAIN_SOURCE, type TerrainTileSource } from './TileSource';

/** ロード結果。標高グリッドそのもの */
export interface TileData {
  /** 描画用の有限値標高 [m]。raw no-data は worker 内で fallback 値に置換済み。 */
  elevations: Float32Array;
  /** raw DEM で標高として使えなかった画素。1 = no-data / invalid。 */
  invalidMask?: Uint8Array;
  width: number;
  height: number;
}

/** 視点移動などで不要になったロードを表す通常系エラー */
export class TileLoadAbortError extends Error {
  constructor(key: string) {
    super(`tile load aborted: ${key}`);
    this.name = 'TileLoadAbortError';
  }
}

/** 内部用: 進行中リクエストの状態 */
interface InFlight {
  /** tileKeyString(key)。同一タイルの重複抑制に使う */
  keyId: string;
  /** 同一キーへは同じ Promise を返す */
  promise: Promise<TileData>;
  resolve: (data: TileData) => void;
  reject: (err: Error) => void;
}

export class TileLoader {
  /** プール内の全ワーカー */
  private readonly workers: Worker[] = [];

  /** ラウンドロビン用のカーソル */
  private cursor = 0;

  /** worker reqId 採番用。tile key と切り離して古い応答を識別する */
  private nextReqId = 1;

  /** worker reqId → 進行中リクエスト */
  private readonly inFlightByReqId = new Map<string, InFlight>();

  /** tile key 文字列 → worker reqId。重複要求の統合に使う */
  private readonly inFlightByKey = new Map<string, string>();

  constructor(private readonly source: TerrainTileSource = GSI_TERRAIN_SOURCE) {
    // ワーカーを WORKER_POOL_SIZE 個起動
    for (let i = 0; i < WORKER_POOL_SIZE; i++) {
      // Vite の `?worker` 解決でコンストラクタが手に入る
      const w = new TileWorker();
      // 共通のメッセージハンドラを付与
      w.onmessage = (ev: MessageEvent<WorkerResponse>) => this.handleMessage(ev.data);
      // ワーカー側で例外が出たら error 扱い
      w.onerror = (ev) => {
        console.error('[TileLoader] worker error', ev);
      };
      this.workers.push(w);
    }
  }

  /**
   * タイルロードを要求する。
   * - 同一 key への重複は 1 本に統合（同じ Promise を返す）
   * - 結果は Float32Array なのでメインスレッドでそのまま DEM メッシュ化できる
   */
  request(key: TileKey): Promise<TileData> {
    const keyId = this.sourceKeyString(key);

    // 既に飛んでいるなら、その Promise を再利用
    const existingReqId = this.inFlightByKey.get(keyId);
    if (existingReqId !== undefined) {
      const existing = this.inFlightByReqId.get(existingReqId);
      if (existing) return existing.promise;
      // 片方だけ残ることは通常ないが、破損状態を見つけたら復旧する。
      this.inFlightByKey.delete(keyId);
    }

    // 新規リクエスト
    let resolveTile!: (data: TileData) => void;
    let rejectTile!: (err: Error) => void;
    const promise = new Promise<TileData>((resolve, reject) => {
      resolveTile = resolve;
      rejectTile = reject;
    });

    const reqId = `${keyId}#${this.nextReqId++}`;
    this.inFlightByReqId.set(reqId, {
      keyId,
      promise,
      resolve: resolveTile,
      reject: rejectTile,
    });
    this.inFlightByKey.set(keyId, reqId);

    // ラウンドロビンで送る先のワーカーを選ぶ
    const w = this.workers[this.cursor % this.workers.length]!;
    this.cursor++;

    const req: WorkerRequest = { type: 'load', reqId, source: this.source, ...key };
    w.postMessage(req);
    return promise;
  }

  /**
   * 「結果が来ても捨てたい」と通知する。
   * - ネットワーク自体は止められないが、対応する Promise はここで reject する。
   * - 後から worker 応答が来ても reqId が in-flight から消えているので無視される。
   */
  abort(key: TileKey): void {
    const keyId = this.sourceKeyString(key);
    const reqId = this.inFlightByKey.get(keyId);
    if (reqId === undefined) return;

    this.inFlightByKey.delete(keyId);
    const f = this.inFlightByReqId.get(reqId);
    if (!f) return;

    this.inFlightByReqId.delete(reqId);
    f.reject(new TileLoadAbortError(keyId));
  }

  /**
   * ワーカーから戻ってきたメッセージを inFlight に紐付けて resolve/reject。
   */
  private handleMessage(msg: WorkerResponse): void {
    const f = this.inFlightByReqId.get(msg.reqId);
    if (!f) return;

    // 1 回限りで取り出して消す
    this.inFlightByReqId.delete(msg.reqId);
    if (this.inFlightByKey.get(f.keyId) === msg.reqId) {
      this.inFlightByKey.delete(f.keyId);
    }

    if (msg.type === 'success') {
      const ok = msg as WorkerSuccess;
      const data: TileData = {
        elevations: ok.elevations,
        width: ok.width,
        height: ok.height,
      };
      if (ok.invalidMask) data.invalidMask = ok.invalidMask;
      f.resolve(data);
    } else {
      f.reject(new Error(msg.message));
    }
  }

  /** 終了処理（必要なら） */
  dispose(): void {
    for (const f of this.inFlightByReqId.values()) {
      f.reject(new TileLoadAbortError(f.keyId));
    }
    for (const w of this.workers) w.terminate();
    this.workers.length = 0;
    this.inFlightByReqId.clear();
    this.inFlightByKey.clear();
  }

  private sourceKeyString(key: TileKey): string {
    return `${this.source.id}:${tileKeyString(key)}`;
  }
}
