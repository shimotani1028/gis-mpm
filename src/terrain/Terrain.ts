// =============================================================================
// Terrain: タイル表示を統括するルート Object3D
// -----------------------------------------------------------------------------
// 役割:
//   - update(camera) で「いまフレームに必要なタイル」を選び、ロード/表示/破棄
//   - TileLoader にロードを依頼、結果を TileMesh にして TileCache に格納
//   - 視野外になったタイルは scene から外す（cache には残す）
//   - cache 容量超過分は cache 側で自動 evict + dispose
//
// この Terrain を Scene に add すれば、毎フレーム update を呼ぶだけで
// ストリーミング地形ができあがる。
// =============================================================================

import { Group, type Camera } from 'three/webgpu';

import { TileLoadAbortError, TileLoader, type TileData } from './TileLoader';
import { TileCache } from './TileCache';
import {
  GSI_TERRAIN_SOURCE,
  type TerrainTileSource,
} from './TileSource';
import {
  TileBridgeMesh,
  TileCornerBridgeMesh,
  TileLodBridgeMesh,
  TileMesh,
  type TerrainSurfaceSample,
  type TileEdge,
  type TileLodBridgeAxis,
  type TileLodBridgeSpan,
  type TileMeshBoundaryNeighbors,
} from './TileMesh';
import {
  selectVisibleTiles,
  asKeyStrings,
  type SelectedTiles,
} from './TileSelector';
import { TerrainSurfaceTile } from './TerrainSurfaceTile';
import {
  getAoiCenter,
  enuToTileKey,
  enuToLonLat,
  lonLatToTileXY,
  tileSizeAndCenterENU,
  tileKeyString,
  parseTileKey,
  tilesCoveringEnuBounds,
  tileCountCoveringEnuBounds,
  withAoiCenter,
  type AoiCenter,
  type TileKey,
} from '../geo/tileMath';
import {
  MAX_REQUESTS_PER_FRAME,
  MAX_ZOOM,
  MIN_ZOOM,
  MPM_MAX_COVERAGE_TILE_COUNT,
  TILE_PIXELS,
  WORKER_POOL_SIZE,
} from '../config';

const NO_DATA_SEARCH_RADIUS_PIXELS = 2;

/**
 * シンプルな統計用カウンタ。HUD 表示などに使えるよう公開する。
 */
export interface TerrainStats {
  zoom: number;
  visible: number;       // いま scene に居るタイル数
  cached: number;        // cache 全体の保持数
  pending: number;       // ロード中のタイル数
  coverage: number;      // MPM 計算領域用に CPU 標高だけ保持しているタイル数
  coveragePending: number;
}

export interface TerrainSample extends TerrainSurfaceSample {
  zoom: number;
}

export interface TerrainCoverageBox {
  originX: number;
  originZ: number;
  xLength: number;
  zLength: number;
}

interface TerrainSourceOptions {
  force?: boolean;
}

interface TerrainOptions {
  aoiCenter?: AoiCenter;
}

interface TerrainAoiOptions {
  clear?: boolean;
}

interface AttachedTileEntry {
  keyStr: string;
  key: TileKey;
  tile: TileMesh;
}

interface TileRangeAtZoom {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

interface LodBridgeConnection {
  a: AttachedTileEntry;
  b: AttachedTileEntry;
  aEdge: TileEdge;
  bEdge: TileEdge;
  span: TileLodBridgeSpan;
}

export class Terrain extends Group {
  private loader: TileLoader;
  private readonly cache = new TileCache();
  private source: TerrainTileSource;
  private aoiCenter: AoiCenter;

  /** いま「scene に add 済み」のキー集合 */
  private readonly attached = new Set<string>();

  /** pixel-center mode で隣接タイル間の実セルを張る bridge mesh。 */
  private readonly bridges = new Map<string, TileBridgeMesh | TileCornerBridgeMesh | TileLodBridgeMesh>();

  /** いま「ロード中」のキー集合（重複リクエスト抑制 + 統計用） */
  private readonly pending = new Set<string>();

  /** MPM 計算領域の厳密サンプル用。描画メッシュは作らず DEM 標高だけを保持する。 */
  private readonly coverageCache = new Map<string, TerrainSurfaceTile>();

  /** 現在の MPM 計算領域に必要な coverage tile。古い async 結果を捨てるためにも使う。 */
  private coverageWantedKeys = new Set<string>();

  /** ensureSurfaceCoverage() が現在ロード中の tile。HUD と多重呼び出し抑制に使う。 */
  private readonly coveragePending = new Map<string, Promise<void>>();

  /** 直近の選択結果（HUD 用） */
  private lastSelection: SelectedTiles = { zoom: 0, keys: [] };

  constructor(source: TerrainTileSource = GSI_TERRAIN_SOURCE, options: TerrainOptions = {}) {
    super();
    this.source = source;
    this.aoiCenter = options.aoiCenter ?? getAoiCenter();
    this.loader = new TileLoader(source);
    this.name = 'Terrain';
  }

  getAoiCenter(): AoiCenter {
    return { ...this.aoiCenter };
  }

  setAoiCenter(center: AoiCenter, options: TerrainAoiOptions = {}): void {
    const changed =
      center.lat !== this.aoiCenter.lat ||
      center.lon !== this.aoiCenter.lon;
    this.aoiCenter = { lat: center.lat, lon: center.lon };
    if (changed || options.clear) {
      this.clearTileState();
    }
  }

  /**
   * 表示・サンプリング対象の DEM タイルソースを切り替える。
   * 既存の GSI タイルと Import/PLATEAU タイルは標高基準も URL も異なるため、
   * 切り替え時は描画 cache と MPM coverage cache をまとめて破棄する。
   */
  setTileSource(source: TerrainTileSource, options: TerrainSourceOptions = {}): void {
    if (!options.force && source.id === this.source.id) return;

    this.loader.dispose();
    this.loader = new TileLoader(source);
    this.source = source;
    this.clearTileState();
  }

  private clearTileState(): void {
    for (const keyStr of this.attached) {
      const mesh = this.cache.peek(keyStr);
      if (mesh) this.remove(mesh.mesh);
    }
    this.attached.clear();
    this.clearBridgeMeshes();
    this.pending.clear();
    this.coverageCache.clear();
    this.coveragePending.clear();
    this.coverageWantedKeys.clear();
    this.cache.clear();
    this.lastSelection = { zoom: 0, keys: [] };
  }

  /** 1 フレーム分の更新。呼び出し側はレンダループから毎フレーム呼ぶ */
  update(camera: Camera): void {
    this.withTerrainAoi(() => this.updateWithCurrentAoi(camera));
  }

  private withTerrainAoi<T>(fn: () => T): T {
    return withAoiCenter(this.aoiCenter, fn);
  }

  private updateWithCurrentAoi(camera: Camera): void {
    // 1. 必要タイル集合
    this.lastSelection =
      this.source.fixedZoom === undefined
        ? selectVisibleTiles(camera)
        : selectVisibleTiles(camera, { fixedZoom: this.source.fixedZoom });
    const wantedKeys = asKeyStrings(this.lastSelection);

    // 表示中タイルを touch して、LOD 切り替え待ちの間に LRU evict されにくくする。
    this.touchAttachedTiles();

    // 2. wanted のうち cache に無く、ロード中でもないものをリクエスト
    //    ただし 1 フレームで投げる本数は MAX_REQUESTS_PER_FRAME に絞る
    const toRequest: TileKey[] = [];
    for (const keyStr of wantedKeys) {
      if (this.cache.has(keyStr) || this.pending.has(keyStr)) continue;
      toRequest.push(parseTileKey(keyStr));
      if (toRequest.length >= MAX_REQUESTS_PER_FRAME) break;
    }
    for (const key of toRequest) {
      this.requestTile(key);
    }

    // 3. 視野外になったタイルのリクエストはキャンセル通知
    //    （ネットワークは止められないが、結果が来ても捨てる扱い）
    for (const keyStr of this.pending) {
      if (!wantedKeys.has(keyStr)) {
        this.loader.abort(parseTileKey(keyStr));
        this.pending.delete(keyStr);
      }
    }

    // 4. 表示の差し替え
    //    新しい選択が全部揃うまでは旧タイルを保持する。これによりズーム・回転中に
    //    旧タイルを先に外して空色の穴が見える状態を避ける。
    this.reconcileAttachedTiles(wantedKeys);
  }

  /**
   * MPM 計算領域を覆う DEM タイルを、描画 LOD とは独立に事前ロードする。
   *
   * 以前は scene に attach 済みのタイルだけから `TerrainBoundary` を作っていたため、
   * 視野外や cache 未保持の地点は originY へフォールバックし、接触面が描画地形から
   * ずれていた。ここでは計算ボックス全体の DEM coverage を CPU 標高キャッシュとして
   * 揃えてから MPM heightfield を生成する。
   */
  async ensureSurfaceCoverage(box: TerrainCoverageBox, zoom = MAX_ZOOM): Promise<void> {
    const { keys, wantedKeys } = this.withTerrainAoi(() => {
      const bounds = {
        minE: box.originX,
        maxE: box.originX + box.xLength,
        minN: box.originZ,
        maxN: box.originZ + box.zLength,
      };
      // bilinear sample はタイル境界で隣接タイルの画素も読むため、計算 box を覆う
      // タイルに 1 タイル分の padding を足しておく。これで box 端や境界直上でも
      // fallback なしで連続した DEM 高さを作れる。
      const coverageTileCount = tileCountCoveringEnuBounds(bounds, zoom, 1);
      if (coverageTileCount > MPM_MAX_COVERAGE_TILE_COUNT) {
        throw new Error(
          `MPM DEM coverage tile 数 ${coverageTileCount} が上限 ${MPM_MAX_COVERAGE_TILE_COUNT} を超えています`,
        );
      }
      const coverageKeys = expandTileKeys(tilesCoveringEnuBounds(bounds, zoom), 1);
      return {
        keys: coverageKeys,
        wantedKeys: new Set(coverageKeys.map((key) => tileKeyString(key))),
      };
    });
    this.coverageWantedKeys = wantedKeys;

    // 古い MPM 領域の CPU DEM は捨てる。描画用 TileCache とは分離しているので、
    // 視点操作の LRU と MPM coverage が互いに evict し合わない。
    for (const keyStr of this.coverageCache.keys()) {
      if (!wantedKeys.has(keyStr)) this.coverageCache.delete(keyStr);
    }

    const missingKeys = keys.filter((key) => !this.coverageCache.has(tileKeyString(key)));
    const concurrency = Math.max(1, WORKER_POOL_SIZE * 2);
    for (let i = 0; i < missingKeys.length; i += concurrency) {
      const batch = missingKeys.slice(i, i + concurrency);
      await Promise.all(batch.map((key) => this.loadCoverageTile(key)));
    }

    // ensure 中に別領域へ切り替わった場合に、遅れて入った古い tile を最後に掃除する。
    for (const keyStr of this.coverageCache.keys()) {
      if (!this.coverageWantedKeys.has(keyStr)) this.coverageCache.delete(keyStr);
    }
  }

  /** TileLoader にリクエストを投げて、完了したら TileMesh 化して cache + scene に追加 */
  private requestTile(key: TileKey): void {
    const keyStr = tileKeyString(key);
    this.pending.add(keyStr);

    this.loader
      .request(key)
      .then((data: TileData) => {
        this.withTerrainAoi(() => {
          this.pending.delete(keyStr);
          // 既にキャッシュ済（同時複数要求で先に入った）なら捨てる
          if (this.cache.has(keyStr)) return;

          // TileMesh を作って cache に保存。直近で必要なら attach も
          const mesh =
            this.source.skirtDepthM === undefined
              ? new TileMesh(key, data, {
                  geometryMode: this.tileMeshGeometryMode(),
                })
              : new TileMesh(key, data, {
                  geometryMode: this.tileMeshGeometryMode(),
                  skirtDepthM: this.source.skirtDepthM,
                });
          this.cache.set(keyStr, mesh);

          // ロード完了により選択タイルが揃ったなら、このタイミングでまとめて差し替える。
          this.reconcileAttachedTiles(asKeyStrings(this.lastSelection));
        });
      })
      .catch((err) => {
        // エラーは pending を解除して、次フレームで再要求できる状態に戻す。
        // abort は視点移動中の通常系なので、警告としては扱わない。
        this.pending.delete(keyStr);
        if (err instanceof TileLoadAbortError) return;
        console.warn(`[Terrain] tile load failed: ${keyStr}`, err);
      });
  }

  /** MPM coverage 用に DEM data だけをロードして CPU サンプラとして保持する。 */
  private loadCoverageTile(key: TileKey): Promise<void> {
    const keyStr = tileKeyString(key);
    if (this.coverageCache.has(keyStr)) return Promise.resolve();

    const existing = this.coveragePending.get(keyStr);
    if (existing) return existing;

    const task = this.loadCoverageTileData(key)
      .finally(() => {
        this.coveragePending.delete(keyStr);
      });
    this.coveragePending.set(keyStr, task);
    return task;
  }

  private async loadCoverageTileData(key: TileKey): Promise<void> {
    const keyStr = tileKeyString(key);
    let data: TileData;
    try {
      data = await this.loader.request(key);
    } catch (err) {
      // 同じ tile が通常描画の pending と共有されていて、視点移動により abort される
      // ことがある。MPM coverage は必要性が別なので、abort だけは一度リトライする。
      if (!(err instanceof TileLoadAbortError)) throw err;
      data = await this.loader.request(key);
    }

    if (!this.coverageWantedKeys.has(keyStr) || this.coverageCache.has(keyStr)) return;
    this.coverageCache.set(keyStr, new TerrainSurfaceTile(key, data));
  }

  /** 現在 scene に残しているタイルを LRU 上も最近使用扱いにする */
  private touchAttachedTiles(): void {
    for (const keyStr of this.attached) {
      this.cache.get(keyStr);
    }
    this.cache.setProtectedKeys(this.attached);
  }

  /**
   * 表示タイルを現在選択へ近づける。
   *
   * - 選択タイルがすべて cache 済みなら、同一フレーム内で attach/detach して
   *   視覚的に原子的な LOD 差し替えにする。
   * - 初期ロード時だけは、まだ全部揃っていなくても読み終わった分から表示する。
   */
  private reconcileAttachedTiles(wantedKeys: Set<string>): void {
    const selectionReady = this.isSelectionCached(wantedKeys);
    const allowPartialInitialAttach = this.attached.size === 0;
    if (!selectionReady && !allowPartialInitialAttach) return;

    const addedKeys: string[] = [];
    for (const keyStr of wantedKeys) {
      const cached = this.cache.get(keyStr);
      if (cached && !this.attached.has(keyStr)) {
        this.add(cached.mesh);
        this.attached.add(keyStr);
        addedKeys.push(keyStr);
      }
    }

    if (!selectionReady) {
      this.updateTileConnections(addedKeys, []);
      return;
    }

    // wanted から外れたタイルは scene から detach（cache には残す）
    // ※ Set を直接イテレートして delete するので一度配列に固定
    const toDetach: string[] = [];
    for (const keyStr of this.attached) {
      if (!wantedKeys.has(keyStr)) toDetach.push(keyStr);
    }
    for (const keyStr of toDetach) {
      const m = this.cache.get(keyStr);          // touch（再利用候補にする）
      if (m) this.remove(m.mesh);
      this.attached.delete(keyStr);
    }

    this.cache.setProtectedKeys(this.attached);
    this.updateTileConnections(addedKeys, toDetach);
  }

  private tileMeshGeometryMode(): 'cell-boundary' | 'pixel-center' {
    return this.source.meshMode === 'pixel-center-bridged'
      ? 'pixel-center'
      : 'cell-boundary';
  }

  private updateTileConnections(addedKeys: string[], removedKeys: string[]): void {
    if (this.source.meshMode === 'pixel-center-bridged') {
      this.syncBridgeCells();
      return;
    }
    this.updateImpactedBoundarySampling(addedKeys, removedKeys);
  }

  private syncBridgeCells(): void {
    const desired = new Set<string>();

    for (const keyStr of this.attached) {
      const tile = this.cache.peek(keyStr);
      if (!tile) continue;
      const key = parseTileKey(keyStr);

      const eastKey = { z: key.z, x: key.x + 1, y: key.y };
      const east = this.attachedNeighbor(eastKey);
      if (east) {
        const bridgeKey = this.bridgeKey(key, 'east');
        desired.add(bridgeKey);
        if (!this.bridges.has(bridgeKey)) {
          const bridge = new TileBridgeMesh(tile, east, 'east');
          this.bridges.set(bridgeKey, bridge);
          this.add(bridge.mesh);
        }
      }

      const southKey = { z: key.z, x: key.x, y: key.y + 1 };
      const south = this.attachedNeighbor(southKey);
      if (south) {
        const bridgeKey = this.bridgeKey(key, 'south');
        desired.add(bridgeKey);
        if (!this.bridges.has(bridgeKey)) {
          const bridge = new TileBridgeMesh(tile, south, 'south');
          this.bridges.set(bridgeKey, bridge);
          this.add(bridge.mesh);
        }
      }

      const southeastKey = { z: key.z, x: key.x + 1, y: key.y + 1 };
      const southeast = this.attachedNeighbor(southeastKey);
      if (east && south && southeast) {
        const bridgeKey = this.bridgeKey(key, 'corner');
        desired.add(bridgeKey);
        if (!this.bridges.has(bridgeKey)) {
          const bridge = new TileCornerBridgeMesh(tile, east, south, southeast);
          this.bridges.set(bridgeKey, bridge);
          this.add(bridge.mesh);
        }
      }
    }

    this.syncMixedLodBridgeCells(desired);

    const staleBridgeKeys = [...this.bridges.keys()].filter((bridgeKey) => !desired.has(bridgeKey));
    for (const bridgeKey of staleBridgeKeys) {
      const bridge = this.bridges.get(bridgeKey);
      if (!bridge) continue;
      this.remove(bridge.mesh);
      bridge.dispose();
      this.bridges.delete(bridgeKey);
    }
  }

  private syncMixedLodBridgeCells(desired: Set<string>): void {
    const entries = this.attachedTileEntries();
    for (let i = 0; i < entries.length; i++) {
      const a = entries[i]!;
      for (let j = i + 1; j < entries.length; j++) {
        const b = entries[j]!;
        const connection = this.mixedLodBridgeConnection(a, b);
        if (!connection) continue;

        const bridgeKey = this.lodBridgeKey(connection);
        desired.add(bridgeKey);
        if (this.bridges.has(bridgeKey)) continue;

        const bridge = new TileLodBridgeMesh(
          { tile: connection.a.tile, edge: connection.aEdge },
          { tile: connection.b.tile, edge: connection.bEdge },
          connection.span,
        );
        this.bridges.set(bridgeKey, bridge);
        this.add(bridge.mesh);
      }
    }
  }

  private attachedTileEntries(): AttachedTileEntry[] {
    const entries: AttachedTileEntry[] = [];
    for (const keyStr of this.attached) {
      const tile = this.cache.peek(keyStr);
      if (!tile) continue;
      entries.push({ keyStr, key: parseTileKey(keyStr), tile });
    }
    return entries;
  }

  private mixedLodBridgeConnection(
    a: AttachedTileEntry,
    b: AttachedTileEntry,
  ): LodBridgeConnection | undefined {
    if (a.key.z === b.key.z) return undefined;

    const commonZoom = Math.max(a.key.z, b.key.z);
    const ar = tileRangeAtZoom(a.key, commonZoom);
    const br = tileRangeAtZoom(b.key, commonZoom);

    if (ar.x1 === br.x0) {
      return this.verticalLodBridgeConnection(a, b, 'east', 'west', commonZoom, ar, br);
    }
    if (br.x1 === ar.x0) {
      return this.verticalLodBridgeConnection(a, b, 'west', 'east', commonZoom, ar, br);
    }
    if (ar.y1 === br.y0) {
      return this.horizontalLodBridgeConnection(a, b, 'south', 'north', commonZoom, ar, br);
    }
    if (br.y1 === ar.y0) {
      return this.horizontalLodBridgeConnection(a, b, 'north', 'south', commonZoom, ar, br);
    }
    return undefined;
  }

  private verticalLodBridgeConnection(
    a: AttachedTileEntry,
    b: AttachedTileEntry,
    aEdge: TileEdge,
    bEdge: TileEdge,
    commonZoom: number,
    ar: TileRangeAtZoom,
    br: TileRangeAtZoom,
  ): LodBridgeConnection | undefined {
    const start = Math.max(ar.y0, br.y0);
    const end = Math.min(ar.y1, br.y1);
    if (end <= start) return undefined;
    return {
      a,
      b,
      aEdge,
      bEdge,
      span: { axis: 'y', commonZoom, start, end },
    };
  }

  private horizontalLodBridgeConnection(
    a: AttachedTileEntry,
    b: AttachedTileEntry,
    aEdge: TileEdge,
    bEdge: TileEdge,
    commonZoom: number,
    ar: TileRangeAtZoom,
    br: TileRangeAtZoom,
  ): LodBridgeConnection | undefined {
    const start = Math.max(ar.x0, br.x0);
    const end = Math.min(ar.x1, br.x1);
    if (end <= start) return undefined;
    return {
      a,
      b,
      aEdge,
      bEdge,
      span: { axis: 'x', commonZoom, start, end },
    };
  }

  private clearBridgeMeshes(): void {
    for (const bridge of this.bridges.values()) {
      this.remove(bridge.mesh);
      bridge.dispose();
    }
    this.bridges.clear();
  }

  private bridgeKey(key: TileKey, direction: 'east' | 'south' | 'corner'): string {
    return `${direction}:${tileKeyString(key)}`;
  }

  private lodBridgeKey(connection: LodBridgeConnection): string {
    return [
      'lod',
      `${connection.a.keyStr}:${connection.aEdge}`,
      `${connection.b.keyStr}:${connection.bEdge}`,
      connection.span.axis,
      connection.span.commonZoom,
      connection.span.start,
      connection.span.end,
    ].join(':');
  }

  private isSelectionCached(wantedKeys: Set<string>): boolean {
    if (wantedKeys.size === 0) return true;
    for (const keyStr of wantedKeys) {
      if (!this.cache.has(keyStr)) return false;
    }
    return true;
  }

  /**
   * 表示中の隣接 DEM を使って、外周頂点を画素中心 bilinear で再サンプルする。
   *
   * 旧実装のように両側の端画素を平均して見た目だけ縫うのではなく、CPU sampler と
   * 同じ座標式でタイル境界の高さを計算する。隣接 tile が未ロードの外周だけは、
   * 単体 tile 内 clamp のままにして、データが揃った時点で再計算する。
   */
  private updateImpactedBoundarySampling(addedKeys: string[], removedKeys: string[]): void {
    const impacted = new Set<string>();

    // 新しく表示した tile は、自分と周囲 tile の境界サンプルが変わる。
    for (const keyStr of addedKeys) {
      this.collectAttachedBoundaryNeighbors(keyStr, impacted, true);
    }

    // 外れた tile の周囲は、隣接 DEM を使わない境界サンプルへ戻す必要がある。
    for (const keyStr of removedKeys) {
      this.collectAttachedBoundaryNeighbors(keyStr, impacted, false);
    }

    for (const keyStr of impacted) {
      const tile = this.cache.peek(keyStr);
      if (!tile) continue;
      const key = parseTileKey(keyStr);
      tile.updateBoundarySampling(this.attachedBoundaryNeighbors(key));
    }
  }

  private collectAttachedBoundaryNeighbors(
    keyStr: string,
    impacted: Set<string>,
    includeSelf: boolean,
  ): void {
    if (includeSelf && this.attached.has(keyStr)) impacted.add(keyStr);

    const key = parseTileKey(keyStr);
    const neighborKeys = this.boundaryNeighborKeys(key);

    for (const neighborKey of neighborKeys) {
      const neighborKeyStr = tileKeyString(neighborKey);
      if (this.attached.has(neighborKeyStr)) impacted.add(neighborKeyStr);
    }
  }

  private attachedBoundaryNeighbors(key: TileKey): TileMeshBoundaryNeighbors {
    const neighbors: TileMeshBoundaryNeighbors = {};
    const north = this.attachedNeighbor({ z: key.z, x: key.x, y: key.y - 1 });
    const south = this.attachedNeighbor({ z: key.z, x: key.x, y: key.y + 1 });
    const west = this.attachedNeighbor({ z: key.z, x: key.x - 1, y: key.y });
    const east = this.attachedNeighbor({ z: key.z, x: key.x + 1, y: key.y });
    const northwest = this.attachedNeighbor({ z: key.z, x: key.x - 1, y: key.y - 1 });
    const northeast = this.attachedNeighbor({ z: key.z, x: key.x + 1, y: key.y - 1 });
    const southwest = this.attachedNeighbor({ z: key.z, x: key.x - 1, y: key.y + 1 });
    const southeast = this.attachedNeighbor({ z: key.z, x: key.x + 1, y: key.y + 1 });
    if (north) neighbors.north = north;
    if (south) neighbors.south = south;
    if (west) neighbors.west = west;
    if (east) neighbors.east = east;
    if (northwest) neighbors.northwest = northwest;
    if (northeast) neighbors.northeast = northeast;
    if (southwest) neighbors.southwest = southwest;
    if (southeast) neighbors.southeast = southeast;
    return neighbors;
  }

  private boundaryNeighborKeys(key: TileKey): TileKey[] {
    return [
      { z: key.z, x: key.x, y: key.y - 1 },
      { z: key.z, x: key.x, y: key.y + 1 },
      { z: key.z, x: key.x - 1, y: key.y },
      { z: key.z, x: key.x + 1, y: key.y },
      { z: key.z, x: key.x - 1, y: key.y - 1 },
      { z: key.z, x: key.x + 1, y: key.y - 1 },
      { z: key.z, x: key.x - 1, y: key.y + 1 },
      { z: key.z, x: key.x + 1, y: key.y + 1 },
    ];
  }

  private attachedNeighbor(key: TileKey): TileMesh | undefined {
    const maxTile = 2 ** key.z;
    if (key.x < 0 || key.x >= maxTile || key.y < 0 || key.y >= maxTile) return undefined;
    const keyStr = tileKeyString(key);
    if (!this.attached.has(keyStr)) return undefined;
    return this.cache.peek(keyStr);
  }

  /** 統計情報（HUD 表示用） */
  getStats(): TerrainStats {
    return {
      zoom: this.lastSelection.zoom,
      visible: this.attached.size,
      cached: this.cache.size,
      pending: this.pending.size,
      coverage: this.coverageCache.size,
      coveragePending: this.coveragePending.size,
    };
  }

  /**
   * ワールド ENU の x/z から、現在ロード済みの地形高さと法線をサンプルする。
   * LOD 差し替え中は複数ズームのタイルが重なることがあるため、一番細かい zoom を優先する。
   */
  sampleSurface(worldX: number, worldZ: number): TerrainSample | undefined {
    return this.withTerrainAoi(() => this.sampleSurfaceWithCurrentAoi(worldX, worldZ));
  }

  private sampleSurfaceWithCurrentAoi(worldX: number, worldZ: number): TerrainSample | undefined {
    // まず MPM coverage 用 CPU DEM を見る。これは計算領域全体を MAX_ZOOM で
    // 揃えてあるため、描画されているタイルの有無やカメラ位置に左右されない。
    for (let z = MAX_ZOOM; z >= MIN_ZOOM; z--) {
      const sample = this.sampleContinuousSurface(worldX, worldZ, z, 'coverage');
      if (sample) return sample;
    }

    // coverage 外の通常用途では、描画用 cache から tile key で直接引く。
    // 全 cache を総当たりしないので、MPM heightfield 生成時の大量サンプルでも重くなりにくい。
    for (let z = MAX_ZOOM; z >= MIN_ZOOM; z--) {
      const sample = this.sampleContinuousSurface(worldX, worldZ, z, 'cache');
      if (sample) return sample;

      // 画面端などで隣接 tile が cache に無い場合だけ、単体 tile 内サンプルへ落とす。
      // MPM coverage は上で padding 済みなので、計算用 heightfield は通常ここへ来ない。
      const fallback = this.sampleSingleTileSurface(worldX, worldZ, z);
      if (fallback) return fallback;
    }
    return undefined;
  }

  private sampleContinuousSurface(
    worldX: number,
    worldZ: number,
    zoom: number,
    source: 'coverage' | 'cache',
  ): TerrainSample | undefined {
    const pixel = worldToGlobalPixel(worldX, worldZ, zoom);
    const height = this.sampleGlobalHeight(pixel.x, pixel.y, zoom, source);
    if (height === undefined) return undefined;

    const hW = this.sampleGlobalHeight(pixel.x - 1, pixel.y, zoom, source);
    const hE = this.sampleGlobalHeight(pixel.x + 1, pixel.y, zoom, source);
    const hN = this.sampleGlobalHeight(pixel.x, pixel.y - 1, zoom, source);
    const hS = this.sampleGlobalHeight(pixel.x, pixel.y + 1, zoom, source);
    if (
      hW === undefined ||
      hE === undefined ||
      hN === undefined ||
      hS === undefined
    ) {
      return undefined;
    }

    const key = globalPixelToTileKey(pixel.x, pixel.y, zoom);
    if (!key) return undefined;
    const { widthM, heightM } = this.tileSizeForKey(key);
    const dx = Math.max(1e-6, widthM / TILE_PIXELS);
    const dz = Math.max(1e-6, heightM / TILE_PIXELS);
    const dhdx = (hE - hW) / (2 * dx);
    // global pixel y は南向きに増える。world z は北向き正なので N-S で勾配を作る。
    const dhdz = (hN - hS) / (2 * dz);
    const nx = -dhdx;
    const ny = 1;
    const nz = -dhdz;
    const len = Math.hypot(nx, ny, nz) || 1;

    return {
      height,
      normal: { x: nx / len, y: ny / len, z: nz / len },
      zoom,
    };
  }

  private sampleGlobalHeight(
    globalPixelX: number,
    globalPixelY: number,
    zoom: number,
    source: 'coverage' | 'cache',
  ): number | undefined {
    const bilinear = this.sampleGlobalHeightBilinear(globalPixelX, globalPixelY, zoom, source);
    if (bilinear !== undefined) return bilinear;
    return this.nearestValidGlobalHeight(
      globalPixelX,
      globalPixelY,
      zoom,
      source,
      NO_DATA_SEARCH_RADIUS_PIXELS,
    );
  }

  private sampleGlobalHeightBilinear(
    globalPixelX: number,
    globalPixelY: number,
    zoom: number,
    source: 'coverage' | 'cache',
  ): number | undefined {
    const x0 = Math.floor(globalPixelX);
    const y0 = Math.floor(globalPixelY);
    const x1 = x0 + 1;
    const y1 = y0 + 1;
    const tx = globalPixelX - x0;
    const ty = globalPixelY - y0;

    const h00 = this.heightAtGlobalPixel(x0, y0, zoom, source);
    const h10 = this.heightAtGlobalPixel(x1, y0, zoom, source);
    const h01 = this.heightAtGlobalPixel(x0, y1, zoom, source);
    const h11 = this.heightAtGlobalPixel(x1, y1, zoom, source);

    let total = 0;
    let weightSum = 0;
    const add = (height: number | undefined, weight: number): void => {
      if (height === undefined || weight <= 0) return;
      total += height * weight;
      weightSum += weight;
    };
    add(h00, (1 - tx) * (1 - ty));
    add(h10, tx * (1 - ty));
    add(h01, (1 - tx) * ty);
    add(h11, tx * ty);
    if (weightSum <= 0) {
      return undefined;
    }

    return total / weightSum;
  }

  private nearestValidGlobalHeight(
    globalPixelX: number,
    globalPixelY: number,
    zoom: number,
    source: 'coverage' | 'cache',
    radius: number,
  ): number | undefined {
    const centerX = Math.round(globalPixelX);
    const centerY = Math.round(globalPixelY);
    let bestDistSq = Number.POSITIVE_INFINITY;
    let total = 0;
    let count = 0;

    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const sampleX = centerX + dx;
        const sampleY = centerY + dy;
        const height = this.heightAtGlobalPixel(sampleX, sampleY, zoom, source);
        if (height === undefined) continue;

        const distSq =
          (sampleX - globalPixelX) ** 2 +
          (sampleY - globalPixelY) ** 2;
        if (distSq + 1e-9 < bestDistSq) {
          bestDistSq = distSq;
          total = height;
          count = 1;
        } else if (Math.abs(distSq - bestDistSq) <= 1e-9) {
          total += height;
          count++;
        }
      }
    }

    return count > 0 ? total / count : undefined;
  }

  private heightAtGlobalPixel(
    globalPixelX: number,
    globalPixelY: number,
    zoom: number,
    source: 'coverage' | 'cache',
  ): number | undefined {
    const key = globalPixelToTileKey(globalPixelX, globalPixelY, zoom);
    if (!key) return undefined;
    const tile = this.surfaceTileForKey(key, source);
    if (!tile) return undefined;

    const col = positiveModulo(globalPixelX, TILE_PIXELS);
    const row = positiveModulo(globalPixelY, TILE_PIXELS);
    return tile.heightAtValidPixel(col, row);
  }

  private surfaceTileForKey(
    key: TileKey,
    source: 'coverage' | 'cache',
  ): TerrainSurfaceTile | undefined {
    const keyStr = tileKeyString(key);
    if (source === 'coverage') return this.coverageCache.get(keyStr);
    return this.cache.peek(keyStr);
  }

  private sampleSingleTileSurface(
    worldX: number,
    worldZ: number,
    zoom: number,
  ): TerrainSample | undefined {
    const key = enuToTileKey(worldX, worldZ, zoom);
    if (!key) return undefined;
    const cached = this.cache.peek(tileKeyString(key));
    if (!cached) return undefined;
    const sample = cached.sampleSurface(worldX, worldZ);
    return sample ? { ...sample, zoom: cached.key.z } : undefined;
  }

  private tileSizeForKey(key: TileKey): { widthM: number; heightM: number } {
    return tileSizeAndCenterENU(key.z, key.x, key.y);
  }

  /** ライフサイクル終了時のクリーンアップ */
  dispose(): void {
    this.loader.dispose();
    this.clearBridgeMeshes();
    this.cache.clear();
    this.attached.clear();
    this.pending.clear();
    this.coverageCache.clear();
    this.coveragePending.clear();
    this.coverageWantedKeys.clear();
  }
}

function worldToGlobalPixel(worldX: number, worldZ: number, zoom: number): { x: number; y: number } {
  const lonLat = enuToLonLat(worldX, worldZ);
  const tileXY = lonLatToTileXY(lonLat.lon, lonLat.lat, zoom);
  return {
    // tileXY * TILE_PIXELS はタイル左上の画像境界。画素値は中心値なので
    // index 座標は -0.5 して、境界上では左右/上下タイルを bilinear で混ぜる。
    x: tileXY.x * TILE_PIXELS - 0.5,
    y: tileXY.y * TILE_PIXELS - 0.5,
  };
}

function globalPixelToTileKey(
  globalPixelX: number,
  globalPixelY: number,
  zoom: number,
): TileKey | undefined {
  const maxTile = 2 ** zoom;
  const tileX = Math.floor(globalPixelX / TILE_PIXELS);
  const tileY = Math.floor(globalPixelY / TILE_PIXELS);
  if (tileX < 0 || tileX >= maxTile || tileY < 0 || tileY >= maxTile) return undefined;
  return { z: zoom, x: tileX, y: tileY };
}

function positiveModulo(value: number, modulus: number): number {
  const r = value % modulus;
  return r < 0 ? r + modulus : r;
}

function tileRangeAtZoom(key: TileKey, zoom: number): TileRangeAtZoom {
  const scale = 2 ** (zoom - key.z);
  return {
    x0: key.x * scale,
    x1: (key.x + 1) * scale,
    y0: key.y * scale,
    y1: (key.y + 1) * scale,
  };
}

function expandTileKeys(keys: TileKey[], radius: number): TileKey[] {
  const out = new Map<string, TileKey>();
  for (const key of keys) {
    const maxTile = 2 ** key.z;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const x = key.x + dx;
        const y = key.y + dy;
        if (x < 0 || x >= maxTile || y < 0 || y >= maxTile) continue;
        const expanded = { z: key.z, x, y };
        out.set(tileKeyString(expanded), expanded);
      }
    }
  }
  return [...out.values()];
}
