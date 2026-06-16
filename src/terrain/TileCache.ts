// =============================================================================
// LRU タイルキャッシュ
// -----------------------------------------------------------------------------
// JavaScript の Map は挿入順序を保つので、それを利用して LRU を実装する:
//   - get で「touch」する場合は delete → set し直して末尾に移動
//   - サイズ超過時は、保護されていない最古のタイルを pop して dispose
//
// なぜキャッシュするか:
//   - 視点を戻したときに再ロードを避ける
//   - パン中に視野外になったタイルも、すぐ戻ってくる可能性が高い
// =============================================================================

import { TILE_CACHE_LIMIT } from '../config';
import { TileMesh } from './TileMesh';

export class TileCache {
  /** key string → TileMesh */
  private readonly map = new Map<string, TileMesh>();

  /** scene に attach 中など、dispose してはいけないキー */
  private protectedKeys = new Set<string>();

  /** 現在の保持数（map.size と同義） */
  get size(): number {
    return this.map.size;
  }

  /**
   * 取得。存在すれば「最近使った」扱いにして末尾へ移動。
   * 戻り値の有無で「ロード済かどうか」を判定する。
   */
  get(key: string): TileMesh | undefined {
    const m = this.map.get(key);
    if (m === undefined) return undefined;

    // touch: delete してから再 set することで Map の挿入順末尾に移す
    this.map.delete(key);
    this.map.set(key, m);
    return m;
  }

  /**
   * 保持。既存があれば dispose してから差し替える（同一キーの二重保持を防ぐ）。
   * 容量超過なら最も古いものを 1 つ evict + dispose。
   */
  set(key: string, mesh: TileMesh): void {
    // 既存差し替え（古い方は GPU 解放）
    const existing = this.map.get(key);
    if (existing) existing.dispose();

    this.map.set(key, mesh);

    this.trimToLimit();
  }

  /**
   * scene に表示中のタイルを保護する。
   *
   * LOD 差し替えでは「旧タイルを表示したまま新タイルをロード」するため、
   * 一時的に cache 上限を超えることがある。その間も表示中タイルは evict せず、
   * 差し替え後に保護対象から外れた古いタイルを trim する。
   */
  setProtectedKeys(keys: Iterable<string>): void {
    this.protectedKeys = new Set(keys);
    this.trimToLimit();
  }

  private trimToLimit(): void {
    while (this.map.size > TILE_CACHE_LIMIT) {
      let evicted = false;

      // Map のイテレータは挿入順なので、先頭から「保護されていない最古」を探す。
      for (const oldestKey of this.map.keys()) {
        if (this.protectedKeys.has(oldestKey)) continue;

        const oldestMesh = this.map.get(oldestKey)!;
        this.map.delete(oldestKey);
        oldestMesh.dispose();
        evicted = true;
        break;
      }

      // すべて protected の場合は、描画中タイルを壊さないことを優先して一時超過する。
      if (!evicted) break;
    }
  }

  /** キャッシュに入っているかだけを確認（touch しない） */
  has(key: string): boolean {
    return this.map.has(key);
  }

  /** 取得だけ行い、LRU の順序は変えない。大量サンプル時の Map 更新を避けるために使う。 */
  peek(key: string): TileMesh | undefined {
    return this.map.get(key);
  }

  /** 全てのメッシュを順次返す（イテレーション用） */
  *entries(): IterableIterator<[string, TileMesh]> {
    for (const e of this.map.entries()) yield e;
  }

  /** 全クリア（dispose 込み） */
  clear(): void {
    for (const m of this.map.values()) m.dispose();
    this.map.clear();
    this.protectedKeys.clear();
  }
}
