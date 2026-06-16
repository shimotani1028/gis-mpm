// =============================================================================
// アプリのエントリ
// -----------------------------------------------------------------------------
// 流れ:
//   1. WebGPU 対応確認
//   2. Viewer を初期化（async）
//   3. Terrain を Scene に追加
//   4. レンダループでカメラ更新を Terrain に渡す
//   5. HUD（FPS / タイル数）を 1Hz で更新
// =============================================================================

import { PerspectiveCamera, Raycaster, Vector2, Vector3 } from 'three/webgpu';

import { Viewer, type ViewerCameraViewState } from './viewer/Viewer';
import { Terrain } from './terrain/Terrain';
import {
  enuToLonLat,
  getAoiCenter,
  lonLatToENU,
  resetAoiCenter,
  setAoiCenter,
  tileCountCoveringEnuBounds,
  withAoiCenter,
  type AoiCenter,
} from './geo/tileMath';
import {
  BuildingObstacleLayer,
  type BuildingGeoJson,
  type BuildingLayerHorizontalBounds,
  type BuildingLayerBounds,
} from './import/BuildingObstacleLayer';
import {
  loadPlateauCatalogDatasets,
  loadPlateauCatalogResources,
  type PlateauCatalogDataset,
  type PlateauCatalogResource,
} from './import/plateauCatalog';
import {
  importPlateauZipSource,
  importPlateauZipSourceChunk,
  type PlateauZipEntrySelectionOptions,
  type PlateauZipImportResult,
  type PlateauZipLatLonBounds,
  type PlateauZipProgress,
  type PlateauZipSource,
} from './import/plateauZip';
import {
  validatePlateauZipUrl,
} from './import/plateauZipUrlPolicy';
import {
  GSI_TERRAIN_SOURCE,
  PLATEAU_TERRARIUM_TP_SOURCE,
} from './terrain/TileSource';
import {
  MpmParticleObject,
  type MpmFluidRegionMeters,
  type MpmResizeFace,
} from './mpm/MpmParticleObject';
import type { MpmDivergenceCheckResult } from './mpm/MlsMpmSimulator';
import {
  ScreenSpaceFluidRenderer,
  type FluidDisplayType,
} from './mpm/render/ScreenSpaceFluidRenderer';
import {
  MPM_BOUNDARY_RESTITUTION,
  MPM_DEFAULT_BOX_M,
  MPM_DEFAULT_FLUID_REGION_M,
  MPM_DOMAIN_FRICTION,
  MPM_DOMAIN_RESTITUTION,
  MPM_DT,
  MPM_DYNAMIC_VISCOSITY,
  MPM_FLUID_CONTOUR_DENSITY_MAX,
  MPM_FLUID_CONTOUR_DENSITY_MIN,
  MPM_FLUID_CONTOUR_SPEED_MAX,
  MPM_FLUID_CONTOUR_SPEED_MIN,
  MPM_FLUID_RENDER_ENABLED,
  MPM_FLUID_RENDER_MIN_RADIUS_M,
  MPM_FLUID_RENDER_RADIUS_CELLS,
  MPM_FLUID_RENDER_RADIUS_SPACING_MULTIPLIER,
  MPM_FLUID_RENDER_THICKNESS_SCALE,
  MPM_INITIAL_TERRAIN_LIFT_M,
  MPM_INITIAL_WATER_DEPTH_M,
  MAX_ZOOM,
  MPM_MAX_BOX_XZ_LENGTH_M,
  MPM_MAX_COVERAGE_TILE_COUNT,
  MPM_OBJECT_FRICTION,
  MPM_OBJECT_RESTITUTION,
  MPM_PARTICLE_COUNT,
  MPM_PARTICLE_COUNT_MAX,
  MPM_PARTICLE_SPACING_M,
  MPM_PARTICLE_SPACING_MIN_M,
  MPM_PARTICLE_SIZE_CELLS,
  MPM_RENDER_ORIGIN_M,
  MPM_STEPS_PER_BATCH,
  MPM_TERRAIN_BOUNDARY_RESOLUTION,
  MPM_TERRAIN_FRICTION,
  RENDER_INTERVAL_FRAMES,
  VIEWER_INITIAL_CAMERA_POSITION_M,
  VIEWER_INITIAL_CAMERA_TARGET_M,
} from './config';
import {
  BuildingSideBoundary,
  DomainBoundary,
  TerrainBoundary,
  type MpmHeightfieldObstacle,
  type MpmSideBoundaryObstacle,
  type MpmBoxInputMeters,
  type MpmBoxMeters,
} from './mpm/boundary/MpmBoundary';

// HUD 用 DOM の取得（index.html 側で <div id="hud"> と <div id="unsupported"> を用意）
const hud = document.getElementById('hud') as HTMLDivElement | null;
const unsupported = document.getElementById('unsupported') as HTMLDivElement | null;
const importStatus = document.getElementById('import-status') as HTMLDivElement | null;
const loadingModal = document.getElementById('loading-modal') as HTMLDivElement | null;
const loadingTitle = document.getElementById('loading-title') as HTMLDivElement | null;
const loadingMessage = document.getElementById('loading-message') as HTMLDivElement | null;
const loadingProgressBar = document.getElementById('loading-progress-bar') as HTMLDivElement | null;
const loadingProgressLabel = document.getElementById('loading-progress-label') as HTMLDivElement | null;
const loadingCancelButton = document.getElementById('cancel-loading') as HTMLButtonElement | null;
const dataAttribution = document.getElementById('data-attribution') as HTMLDivElement | null;
const fluidColorbar = document.getElementById('fluid-colorbar') as HTMLDivElement | null;
const fluidColorbarTitle = document.getElementById('fluid-colorbar-title') as HTMLDivElement | null;
const fluidColorbarMin = document.getElementById('fluid-colorbar-min') as HTMLSpanElement | null;
const fluidColorbarMax = document.getElementById('fluid-colorbar-max') as HTMLSpanElement | null;
const simulationTooltip = document.getElementById('simulation-tooltip') as HTMLDivElement | null;
const interactionTooltip = document.getElementById('interaction-tooltip') as HTMLDivElement | null;

type DataAttributionMode = 'gsi' | 'plateau';
type ImportSourceMode = 'gsi' | 'plateau';

interface PreparedPlateauTemplate {
  name: string;
  aoiCenter: AoiCenter;
  box: MpmBoxInputMeters;
  source: PlateauBuildingImportSource;
  focusCenter: Vector3;
  focusRadius: number;
}

interface PreparedPlateauScene extends PreparedPlateauTemplate {
  buildingLayer: BuildingObstacleLayer;
}

interface DataAttributionContent {
  linkText: string;
  url: string;
  suffix: string;
}

const DATA_ATTRIBUTION_BY_MODE: Record<DataAttributionMode, DataAttributionContent> = {
  gsi: {
    linkText: '国土地理院',
    url: 'https://www.gsi.go.jp/kikakuchousei/kikakuchousei40182.html',
    suffix: '地理院タイル（標高タイル）を加工して作成',
  },
  plateau: {
    linkText: '国土交通省 Project PLATEAU',
    url: 'https://www.mlit.go.jp/plateau/site-policy/',
    suffix: 'の PLATEAU-Terrain / CityGML を加工して作成',
  },
};
const MPM_DIVERGENCE_SPEED_LIMIT = 9999;
const MPM_DIVERGENCE_CHECK_INTERVAL_MS = 1000;

function setHudStatus(message: string): void {
  if (hud) hud.textContent = message;
}

function setDataAttribution(mode: DataAttributionMode): void {
  if (!dataAttribution) return;
  const content = DATA_ATTRIBUTION_BY_MODE[mode];
  const link = document.createElement('a');
  link.href = content.url;
  link.target = '_blank';
  link.rel = 'noopener noreferrer';
  link.textContent = content.linkText;

  // 出典は使用中の地形/建物データソースに連動させる。ユーザーが
  // スクリーンショットを撮った場合も、画面右下だけで出所が読めるようにする。
  dataAttribution.replaceChildren('出典: ', link, ` ${content.suffix}`);
}

function attachCollapsiblePanel(panelEl: HTMLElement | null, toggleButton: HTMLButtonElement): void {
  toggleButton.addEventListener('click', () => {
    if (!panelEl) return;
    const collapsed = panelEl.classList.toggle('collapsed');
    toggleButton.textContent = collapsed ? '+' : '−';
    toggleButton.title = collapsed ? '展開' : '折り畳み';
    toggleButton.setAttribute('aria-expanded', String(!collapsed));
  });
}

function attachSimulationTooltips(panelEl: HTMLElement | null): void {
  if (!panelEl || !simulationTooltip) return;

  const icons = Array.from(panelEl.querySelectorAll<HTMLElement>('.sim-info[data-tooltip]'));
  let activeIcon: HTMLElement | undefined;

  const hideTooltip = (): void => {
    activeIcon = undefined;
    simulationTooltip.hidden = true;
  };

  const positionTooltip = (icon: HTMLElement): void => {
    const iconRect = icon.getBoundingClientRect();
    const tooltipRect = simulationTooltip.getBoundingClientRect();
    const gap = 8;
    const margin = 8;
    let left = iconRect.left - tooltipRect.width - gap;
    if (left < margin) {
      left = iconRect.right + gap;
    }
    left = Math.min(left, window.innerWidth - tooltipRect.width - margin);
    left = Math.max(margin, left);

    const centeredTop = iconRect.top + iconRect.height * 0.5 - tooltipRect.height * 0.5;
    const top = Math.max(
      margin,
      Math.min(centeredTop, window.innerHeight - tooltipRect.height - margin),
    );

    simulationTooltip.style.left = `${Math.round(left)}px`;
    simulationTooltip.style.top = `${Math.round(top)}px`;
  };

  const showTooltip = (icon: HTMLElement): void => {
    const text = icon.dataset.tooltip;
    if (!text) return;
    activeIcon = icon;
    simulationTooltip.textContent = text;
    simulationTooltip.hidden = false;
    // パネル本体はスクロールするため、tooltip は fixed で別要素に出して clipping を避ける。
    positionTooltip(icon);
  };

  for (const icon of icons) {
    icon.addEventListener('mouseenter', () => showTooltip(icon));
    icon.addEventListener('mouseleave', hideTooltip);
    icon.addEventListener('focus', () => showTooltip(icon));
    icon.addEventListener('blur', hideTooltip);
    icon.addEventListener('keydown', (event) => {
      if (event.key !== 'Escape') return;
      hideTooltip();
      icon.blur();
    });
  }

  window.addEventListener(
    'scroll',
    () => {
      if (activeIcon && !simulationTooltip.hidden) positionTooltip(activeIcon);
    },
    true,
  );
  window.addEventListener('resize', () => {
    if (activeIcon && !simulationTooltip.hidden) positionTooltip(activeIcon);
  });
}

interface SimulationPanel {
  originX: HTMLInputElement;
  originY: HTMLInputElement;
  originZ: HTMLInputElement;
  xLength: HTMLInputElement;
  yLength: HTMLInputElement;
  zLength: HTMLInputElement;
  terrainLift: HTMLInputElement;
  waterDepth: HTMLInputElement;
  particleCount: HTMLInputElement;
  activeParticleCount: HTMLInputElement;
  particleSpacing: HTMLInputElement;
  viscosity: HTMLInputElement;
  domainFriction: HTMLInputElement;
  terrainFriction: HTMLInputElement;
  objectFriction: HTMLInputElement;
  domainRestitution: HTMLInputElement;
  terrainRestitution: HTMLInputElement;
  objectRestitution: HTMLInputElement;
  dt: HTMLInputElement;
  stepsPerBatch: HTMLInputElement;
  renderInterval: HTMLInputElement;
  waterRender: HTMLInputElement;
  fluidDisplayType: HTMLSelectElement;
  fluidDisplayMin: HTMLInputElement;
  fluidDisplayMax: HTMLInputElement;
  particleSize: HTMLInputElement;
  start: HTMLButtonElement;
  stop: HTMLButtonElement;
  reset: HTMLButtonElement;
  toggle: HTMLButtonElement;
}

interface SimulationPanelValues extends MpmBoxInputMeters {
  viscosity: number;
  domainFriction: number;
  terrainFriction: number;
  objectFriction: number;
  domainRestitution: number;
  terrainRestitution: number;
  objectRestitution: number;
  terrainLift: number;
  waterDepth: number;
  particleCount: number;
  particleSpacing: number;
  dt: number;
  stepsPerBatch: number;
  renderInterval: number;
  waterRender: boolean;
  fluidDisplayType: FluidDisplayType;
  fluidDisplayMin: number;
  fluidDisplayMax: number;
  particleSize: number;
}

interface ApplySimulationPanelOptions {
  resetParticles: boolean;
  stopAfterReset: boolean;
}

interface ImportManifestBounds {
  minLon: number;
  minLat: number;
  maxLon: number;
  maxLat: number;
  minHeight: number;
  maxHeight: number;
}

interface ImportManifest {
  name: string;
  bounds: ImportManifestBounds;
  obstacles: Array<{
    type: 'geojson';
    path: string;
  }>;
}

interface PlateauBuildingImportSource {
  name: string;
  bounds: ImportManifestBounds;
  loadTerrainDrapedLayer(
    terrainHeightSampler: (worldX: number, worldZ: number) => number | undefined,
    aoiCenter: AoiCenter,
  ): Promise<BuildingObstacleLayer>;
}

interface PlateauBuildingImportOptions {
  skipPrepare?: boolean;
  onProgress?: (progress: PlateauImportProgress) => void;
  preserveCamera?: boolean;
}

interface PlateauImportProgress {
  message: string;
  progress?: number;
  indeterminate?: boolean;
}

interface PlateauImportFailureOptions {
  showModal?: boolean;
  modalLabel?: string;
}

interface PlateauZipFetchOptions {
  signal: AbortSignal;
  onProgress: (progress: PlateauImportProgress) => void;
}

type ProgressAwarePlateauZipSource = PlateauZipSource & {
  setProgressReporter?: (onProgress: (progress: PlateauImportProgress) => void) => void;
};

interface PlateauBuildingExpansionSource {
  label: string;
}

interface ActivePlateauBuildingExpansion {
  imported: PlateauZipImportResult;
  source: PlateauBuildingExpansionSource;
  zipSource: PlateauZipSource;
  aoiCenter: AoiCenter;
  geoJsonChunks: BuildingGeoJson[];
  loadedEntryNames: Set<string>;
  loading: boolean;
  lastCheckAt: number;
  lastCenter?: Vector3;
}

interface PlateauVisibleGroundBounds {
  center: Vector3;
  latLonBounds: PlateauZipLatLonBounds;
}

interface TerrainHeightRange {
  minY: number;
  maxY: number;
  missingSamples: number;
  totalSamples: number;
}

interface ResizeFaceHit {
  face: MpmResizeFace;
  point: Vector3;
}

interface FluidRegionHit {
  point: Vector3;
}

interface DomainMoveHit {
  point: Vector3;
}

interface DomainResizeSession {
  face: MpmResizeFace;
  startBox: MpmBoxMeters;
  previewBox: MpmBoxMeters;
  startHitPoint: Vector3;
  normal: Vector3;
  horizontalY: number;
}

interface DomainMoveSession {
  pointerId: number | undefined;
  startBox: MpmBoxMeters;
  previewBox: MpmBoxMeters;
  startFluidRegion: MpmFluidRegionMeters;
  previewFluidRegion: MpmFluidRegionMeters;
  startHitPoint: Vector3;
  horizontalY: number;
}

type FluidMoveMode = 'free' | 'x' | 'y' | 'z' | 'scale' | 'depth';

interface FluidMoveSession {
  startRegion: MpmFluidRegionMeters;
  previewRegion: MpmFluidRegionMeters;
  startHitPoint: Vector3;
  horizontalY: number;
  mode: FluidMoveMode;
  startTerrainLiftM: number;
  previewTerrainLiftM: number;
  startWaterDepthM: number;
  previewWaterDepthM: number;
}

interface EditClickCandidate {
  pointerId: number;
  startX: number;
  startY: number;
  dragged: boolean;
}

const MIN_DOMAIN_LENGTH_M = 10;
const PLATEAU_IMPORT_INITIAL_BOX_MAX_LENGTH_M = 360;
const PLATEAU_IMPORT_INITIAL_BOX_MIN_LENGTH_M = 180;
const PLATEAU_IMPORT_INITIAL_BOX_PADDING_M = 40;
const PLATEAU_IMPORT_INITIAL_BOX_XZ_LENGTH_MULTIPLIER = 2;
const PLATEAU_IMPORT_INITIAL_Y_PADDING_M = 40;
const PLATEAU_IMPORT_INITIAL_Y_LENGTH_MIN_M = 70;
const PLATEAU_IMPORT_INITIAL_WATER_DEPTH_M = 5.0;
// PLATEAU import 後も、粒子 buffer の上限は通常起動（国土地理院）と同じ値を使う。
const PLATEAU_IMPORT_INITIAL_PARTICLE_COUNT = MPM_PARTICLE_COUNT;
const PLATEAU_IMPORT_INITIAL_PARTICLE_SPACING_M = 2.0;
const PLATEAU_IMPORT_INITIAL_ENTRY_SELECTION_LENGTH_M = 4000;
const PLATEAU_IMPORT_INITIAL_ENTRY_MAX_SELECTED_COUNT = 9;
const PLATEAU_IMPORT_INITIAL_ENTRY_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024;
const PLATEAU_IMPORT_INITIAL_ENTRY_MAX_COMPRESSED_BYTES = 96 * 1024 * 1024;
const PLATEAU_IMPORT_CAMERA_ENTRY_SELECTION_LENGTH_M = 3200;
const PLATEAU_IMPORT_CAMERA_ENTRY_SELECTION_PADDING_M = 200;
const PLATEAU_IMPORT_CAMERA_ENTRY_CHECK_INTERVAL_MS = 2500;
const PLATEAU_IMPORT_CAMERA_ENTRY_MOVE_THRESHOLD_M = 1400;
const PLATEAU_IMPORT_CAMERA_ENTRY_MAX_RAY_DISTANCE_M = 14_000;
const PLATEAU_IMPORT_CAMERA_ENTRY_MAX_BOUNDS_LENGTH_M = 5200;
const PLATEAU_IMPORT_CAMERA_ENTRY_MAX_SELECTED_COUNT = 2;
const PLATEAU_IMPORT_CAMERA_ENTRY_MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024;
const PLATEAU_IMPORT_CAMERA_ENTRY_MAX_COMPRESSED_BYTES = 18 * 1024 * 1024;
const PLATEAU_IMPORT_CAMERA_CENTER_NEIGHBOR_ENTRY_COUNT = 9;
const PLATEAU_IMPORT_CAMERA_CENTER_NEIGHBOR_UNCOMPRESSED_BYTES = 192 * 1024 * 1024;
const PLATEAU_IMPORT_CAMERA_CENTER_NEIGHBOR_COMPRESSED_BYTES = 64 * 1024 * 1024;
const PLATEAU_MPM_BUILDING_CLIP_MARGIN_M = 80;
const PLATEAU_IMPORT_INITIAL_ENTRY_SELECTION = {
  // 初回は municipality 全域ではなく、中央付近の数 mesh だけを読んで scene を作る。
  // 最終的なシミュレーション box は import 後の建物 bounds から別途 720m 程度に収める。
  // 東京区部のように 4km 内だけでも GML 展開後合計が 1GiB を超える地域があるため、
  // 初回も 3x3 程度に soft budget を掛け、残りはカメラ移動時の追加読み込みで補う。
  centerRegionLengthM: PLATEAU_IMPORT_INITIAL_ENTRY_SELECTION_LENGTH_M,
  maxSelectedEntryCount: PLATEAU_IMPORT_INITIAL_ENTRY_MAX_SELECTED_COUNT,
  maxSelectedUncompressedBytes: PLATEAU_IMPORT_INITIAL_ENTRY_MAX_UNCOMPRESSED_BYTES,
  maxSelectedCompressedBytes: PLATEAU_IMPORT_INITIAL_ENTRY_MAX_COMPRESSED_BYTES,
} satisfies PlateauZipEntrySelectionOptions;
const DEFAULT_PLATEAU_PREFECTURE_LABEL = '静岡県';
const DEFAULT_PLATEAU_MUNICIPALITY_LABEL = '下田市';
const DEFAULT_SHIMODA_SAMPLE_MANIFEST_URL = '/samples/shimoda/manifest.json';
const RESIZE_TERRAIN_HEIGHT_MARGIN_M = 0.5;
const EDIT_CLICK_DRAG_THRESHOLD_PX = 6;
const EDIT_CLICK_DRAG_THRESHOLD_SQ_PX =
  EDIT_CLICK_DRAG_THRESHOLD_PX * EDIT_CLICK_DRAG_THRESHOLD_PX;
const MIN_FLUID_REGION_LENGTH_M = 1;
const MAX_FLUID_REGION_SCALE = 100;
// `d` キーの水深変更は `y` キーの surface gap 移動と同じ world meter 感度にする。
const FLUID_DEPTH_POINTER_SCALE = 1;

const RESIZE_FACE_NORMALS: Record<MpmResizeFace, Vector3> = {
  xMin: new Vector3(-1, 0, 0),
  xMax: new Vector3(1, 0, 0),
  zMin: new Vector3(0, 0, -1),
  zMax: new Vector3(0, 0, 1),
};

function inputById(id: string): HTMLInputElement {
  const el = document.getElementById(id);
  if (!(el instanceof HTMLInputElement)) {
    throw new Error(`Missing input: ${id}`);
  }
  return el;
}

function buttonById(id: string): HTMLButtonElement {
  const el = document.getElementById(id);
  if (!(el instanceof HTMLButtonElement)) {
    throw new Error(`Missing button: ${id}`);
  }
  return el;
}

function selectById(id: string): HTMLSelectElement {
  const el = document.getElementById(id);
  if (!(el instanceof HTMLSelectElement)) {
    throw new Error(`Missing select: ${id}`);
  }
  return el;
}

function getSimulationPanel(): SimulationPanel {
  return {
    originX: inputById('sim-origin-x'),
    originY: inputById('sim-origin-y'),
    originZ: inputById('sim-origin-z'),
    xLength: inputById('sim-x-length'),
    yLength: inputById('sim-y-length'),
    zLength: inputById('sim-z-length'),
    terrainLift: inputById('sim-terrain-lift'),
    waterDepth: inputById('sim-water-depth'),
    particleCount: inputById('sim-particle-count'),
    activeParticleCount: inputById('sim-active-particle-count'),
    particleSpacing: inputById('sim-particle-spacing'),
    viscosity: inputById('sim-viscosity'),
    domainFriction: inputById('sim-domain-friction'),
    terrainFriction: inputById('sim-terrain-friction'),
    objectFriction: inputById('sim-object-friction'),
    domainRestitution: inputById('sim-domain-restitution'),
    terrainRestitution: inputById('sim-terrain-restitution'),
    objectRestitution: inputById('sim-object-restitution'),
    dt: inputById('sim-dt'),
    stepsPerBatch: inputById('sim-substeps'),
    renderInterval: inputById('sim-render-interval'),
    waterRender: inputById('sim-water-render'),
    fluidDisplayType: selectById('sim-fluid-display-type'),
    fluidDisplayMin: inputById('sim-fluid-display-min'),
    fluidDisplayMax: inputById('sim-fluid-display-max'),
    particleSize: inputById('sim-particle-size'),
    start: buttonById('sim-start'),
    stop: buttonById('sim-stop'),
    reset: buttonById('sim-reset'),
    toggle: buttonById('panel-toggle'),
  };
}

function setInitialPanelValues(panel: SimulationPanel): void {
  panel.originX.value = String(MPM_DEFAULT_BOX_M.originX);
  panel.originY.value = String(MPM_RENDER_ORIGIN_M.y);
  panel.originZ.value = String(MPM_DEFAULT_BOX_M.originZ);
  panel.xLength.value = String(MPM_DEFAULT_BOX_M.xLength);
  panel.yLength.value = String(MPM_DEFAULT_BOX_M.yLength);
  panel.zLength.value = String(MPM_DEFAULT_BOX_M.zLength);
  panel.terrainLift.value = String(MPM_INITIAL_TERRAIN_LIFT_M);
  panel.waterDepth.value = String(MPM_INITIAL_WATER_DEPTH_M);
  panel.particleCount.value = String(MPM_PARTICLE_COUNT);
  panel.activeParticleCount.value = '0';
  panel.particleSpacing.value = MPM_PARTICLE_SPACING_M.toFixed(2);
  panel.viscosity.value = String(MPM_DYNAMIC_VISCOSITY);
  panel.domainFriction.value = String(MPM_DOMAIN_FRICTION);
  panel.terrainFriction.value = String(MPM_TERRAIN_FRICTION);
  panel.objectFriction.value = String(MPM_OBJECT_FRICTION);
  panel.domainRestitution.value = String(MPM_DOMAIN_RESTITUTION);
  panel.terrainRestitution.value = String(MPM_BOUNDARY_RESTITUTION);
  panel.objectRestitution.value = String(MPM_OBJECT_RESTITUTION);
  panel.dt.value = String(MPM_DT);
  panel.stepsPerBatch.value = String(MPM_STEPS_PER_BATCH);
  panel.renderInterval.value = String(RENDER_INTERVAL_FRAMES);
  panel.waterRender.checked = MPM_FLUID_RENDER_ENABLED;
  panel.fluidDisplayType.value = 'none';
  panel.fluidDisplayMin.value = String(MPM_FLUID_CONTOUR_SPEED_MIN);
  panel.fluidDisplayMax.value = String(MPM_FLUID_CONTOUR_SPEED_MAX);
  panel.particleSize.value = String(MPM_PARTICLE_SIZE_CELLS);
}

function setImportStatus(message: string): void {
  if (importStatus) importStatus.textContent = message;
}

let loadingModalVisible = false;
let loadingModalDismissible = false;
let loadingCancelHandler: (() => void) | undefined = undefined;

interface LoadingModalOptions {
  opaque?: boolean;
}

function errorMessageFromUnknown(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function isLimitFailureMessage(message: string): boolean {
  return /上限|最大|too large|exceed|limit/i.test(message);
}

function setLoadingCancelHandler(
  handler: (() => void) | undefined,
  label = 'キャンセル',
): void {
  loadingCancelHandler = handler;
  if (loadingCancelButton) {
    loadingCancelButton.textContent = label;
    loadingCancelButton.disabled = handler === undefined;
  }
}

function showLoadingModal(
  title: string,
  message: string,
  cancelHandler?: () => void,
  options: LoadingModalOptions = {},
): void {
  loadingModalVisible = true;
  loadingModalDismissible = false;
  document.body.classList.add('loading-blocked');
  if (loadingModal) {
    loadingModal.hidden = false;
    loadingModal.setAttribute('role', 'dialog');
    loadingModal.setAttribute('aria-busy', 'true');
    loadingModal.classList.toggle('loading-opaque', options.opaque === true);
    loadingModal.classList.remove('loading-error');
  }
  if (loadingTitle) loadingTitle.textContent = title;
  setLoadingCancelHandler(cancelHandler, 'キャンセル');
  updateLoadingModal({ message, indeterminate: true });
}

function hideLoadingModal(): void {
  loadingModalVisible = false;
  loadingModalDismissible = false;
  setLoadingCancelHandler(undefined);
  document.body.classList.remove('loading-blocked');
  if (loadingModal) {
    loadingModal.hidden = true;
    loadingModal.setAttribute('role', 'dialog');
    loadingModal.setAttribute('aria-busy', 'false');
    loadingModal.classList.remove('loading-opaque');
    loadingModal.classList.remove('loading-error');
  }
}

function showErrorModal(title: string, message: string): void {
  loadingModalVisible = true;
  loadingModalDismissible = true;
  document.body.classList.add('loading-blocked');
  if (loadingModal) {
    loadingModal.hidden = false;
    loadingModal.setAttribute('role', 'alertdialog');
    loadingModal.setAttribute('aria-busy', 'false');
    loadingModal.classList.remove('loading-opaque');
    loadingModal.classList.add('loading-error');
  }
  if (loadingTitle) loadingTitle.textContent = title;
  setLoadingCancelHandler(() => {
    hideLoadingModal();
  }, '閉じる');
  updateLoadingModal({ message, progress: 1 });
  loadingCancelButton?.focus();
}

function updateLoadingModal(progress: PlateauImportProgress): void {
  if (loadingMessage) loadingMessage.textContent = progress.message;

  const rawProgress = progress.progress;
  const hasProgress =
    rawProgress !== undefined &&
    Number.isFinite(rawProgress) &&
    !progress.indeterminate;
  const percent = hasProgress
    ? Math.round(Math.min(1, Math.max(0, rawProgress)) * 100)
    : undefined;

  if (loadingProgressBar) {
    loadingProgressBar.classList.toggle('indeterminate', !hasProgress);
    loadingProgressBar.style.width = percent === undefined ? '' : `${percent}%`;
    loadingProgressBar.setAttribute('aria-valuemin', '0');
    loadingProgressBar.setAttribute('aria-valuemax', '100');
    if (percent === undefined) {
      loadingProgressBar.removeAttribute('aria-valuenow');
    } else {
      loadingProgressBar.setAttribute('aria-valuenow', String(percent));
    }
  }
  if (loadingProgressLabel) {
    loadingProgressLabel.textContent = percent === undefined ? '処理中...' : `${percent}%`;
  }
}

loadingCancelButton?.addEventListener('click', () => {
  loadingCancelHandler?.();
});

window.addEventListener(
  'keydown',
  (event) => {
    if (!loadingModalVisible) return;
    if (
      loadingModalDismissible &&
      (event.key === 'Escape' || event.key === 'Enter' || event.key === ' ')
    ) {
      event.preventDefault();
      event.stopImmediatePropagation();
      loadingCancelHandler?.();
      return;
    }
    event.preventDefault();
    event.stopImmediatePropagation();
  },
  { capture: true },
);

function readNumber(input: HTMLInputElement, fallback: number): number {
  const value = Number(input.value);
  return Number.isFinite(value) ? value : fallback;
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

function importedBoundsToBox(bounds: BuildingLayerBounds): MpmBoxInputMeters {
  const centerX = (bounds.minX + bounds.maxX) * 0.5;
  const centerZ = (bounds.minZ + bounds.maxZ) * 0.5;
  const xLength =
    clampedImportInitialLength(bounds.maxX - bounds.minX) *
    PLATEAU_IMPORT_INITIAL_BOX_XZ_LENGTH_MULTIPLIER;
  const zLength =
    clampedImportInitialLength(bounds.maxZ - bounds.minZ) *
    PLATEAU_IMPORT_INITIAL_BOX_XZ_LENGTH_MULTIPLIER;

  // PLATEAU ZIP は自治体全域を含むことがある。初回から全域を MPM 計算領域にすると、
  // DEM coverage と粒子初期化が過大になるため、中心付近の街区サイズだけを初期値にする。
  return {
    originX: centerX - xLength * 0.5,
    originZ: centerZ - zLength * 0.5,
    xLength,
    yLength: Math.max(
      PLATEAU_IMPORT_INITIAL_Y_LENGTH_MIN_M,
      bounds.maxY - bounds.minY + PLATEAU_IMPORT_INITIAL_Y_PADDING_M,
    ),
    zLength,
  };
}

function clampedImportInitialLength(sourceLengthM: number): number {
  const padded = Math.max(0, sourceLengthM) + PLATEAU_IMPORT_INITIAL_BOX_PADDING_M * 2;
  return Math.min(
    PLATEAU_IMPORT_INITIAL_BOX_MAX_LENGTH_M,
    Math.max(PLATEAU_IMPORT_INITIAL_BOX_MIN_LENGTH_M, padded),
  );
}

function importSampleCenter(bounds: BuildingLayerBounds): Vector3 {
  return new Vector3(
    (bounds.minX + bounds.maxX) * 0.5,
    (bounds.minY + bounds.maxY) * 0.5,
    (bounds.minZ + bounds.maxZ) * 0.5,
  );
}

function importSampleRadius(bounds: BuildingLayerBounds): number {
  return Math.max(
    bounds.maxX - bounds.minX,
    bounds.maxY - bounds.minY,
    bounds.maxZ - bounds.minZ,
  );
}

function importBoxCenter(box: MpmBoxInputMeters, bounds: BuildingLayerBounds): Vector3 {
  return new Vector3(
    box.originX + box.xLength * 0.5,
    (bounds.minY + bounds.maxY) * 0.5,
    box.originZ + box.zLength * 0.5,
  );
}

function importBoxRadius(box: MpmBoxInputMeters): number {
  return Math.max(box.xLength, box.yLength, box.zLength);
}

function resolveManifestPath(manifestUrl: string, relativePath: string): string {
  return new URL(relativePath, new URL(manifestUrl, window.location.href)).toString();
}

function manifestBoundsToLayerBounds(bounds: ImportManifestBounds): BuildingLayerBounds {
  const sw = lonLatToENU(bounds.minLon, bounds.minLat);
  const ne = lonLatToENU(bounds.maxLon, bounds.maxLat);
  return {
    minX: Math.min(sw.x, ne.x),
    maxX: Math.max(sw.x, ne.x),
    minY: bounds.minHeight,
    maxY: bounds.maxHeight,
    minZ: Math.min(sw.y, ne.y),
    maxZ: Math.max(sw.y, ne.y),
  };
}

function manifestBoundsCenter(bounds: ImportManifestBounds): { lat: number; lon: number } {
  return {
    lat: (bounds.minLat + bounds.maxLat) * 0.5,
    lon: (bounds.minLon + bounds.maxLon) * 0.5,
  };
}

function readIntegerInRange(
  input: HTMLInputElement,
  fallback: number,
  min: number,
  max: number,
): number {
  return Math.min(max, Math.max(min, Math.floor(readNumber(input, fallback))));
}

function fluidDisplayTypeFromValue(value: string): FluidDisplayType {
  if (value === 'velocity' || value === 'density') return value;
  return 'none';
}

function fluidContourDefaults(type: FluidDisplayType): { min: number; max: number } {
  if (type === 'density') {
    return {
      min: MPM_FLUID_CONTOUR_DENSITY_MIN,
      max: MPM_FLUID_CONTOUR_DENSITY_MAX,
    };
  }
  return {
    min: MPM_FLUID_CONTOUR_SPEED_MIN,
    max: MPM_FLUID_CONTOUR_SPEED_MAX,
  };
}

function readFluidContourRange(
  panel: SimulationPanel,
  type: FluidDisplayType,
): { min: number; max: number } {
  const defaults = fluidContourDefaults(type);
  const min = readNumber(panel.fluidDisplayMin, defaults.min);
  const requestedMax = readNumber(panel.fluidDisplayMax, defaults.max);
  return {
    min,
    max: requestedMax > min ? requestedMax : min + 1e-6,
  };
}

function formatFluidContourValue(value: number): string {
  if (!Number.isFinite(value)) return '0';
  const abs = Math.abs(value);
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  if (abs >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

function formatDivergenceValue(value: number): string {
  if (Number.isNaN(value)) return 'NaN';
  if (value === Infinity) return 'Infinity';
  if (value === -Infinity) return '-Infinity';
  if (!Number.isFinite(value)) return String(value);
  const abs = Math.abs(value);
  if (abs >= 1000) return value.toExponential(3);
  if (abs >= 1) return value.toFixed(3);
  return value.toExponential(3);
}

function mpmDivergenceModalMessage(result: MpmDivergenceCheckResult): string {
  const issue = result.issue;
  if (!issue) {
    return 'MPM 計算が発散したため停止しました。粒子状態に異常値が検出されました。';
  }

  if (issue.kind === 'speed') {
    return (
      `MPM 計算が発散したため停止しました。粒子 #${issue.particleIndex} の流速 ` +
      `${formatDivergenceValue(issue.value)} が上限 ${issue.maxSpeed} を超えています。` +
      `dt、steps / batch、反発係数を下げて Reset してください。`
    );
  }

  return (
    `MPM 計算が発散したため停止しました。粒子 #${issue.particleIndex} の ` +
    `${issue.component} が ${formatDivergenceValue(issue.value)} になっています。` +
    `dt、steps / batch、反発係数を下げて Reset してください。`
  );
}

function readPanelValues(panel: SimulationPanel): SimulationPanelValues {
  const yLength = Math.max(10, readNumber(panel.yLength, MPM_DEFAULT_BOX_M.yLength));
  const waterDepth = Math.max(1, readNumber(panel.waterDepth, MPM_INITIAL_WATER_DEPTH_M));
  const terrainLift = clampTerrainLiftM(
    readNumber(panel.terrainLift, MPM_INITIAL_TERRAIN_LIFT_M),
    yLength,
    waterDepth,
  );
  const fluidDisplayType = fluidDisplayTypeFromValue(panel.fluidDisplayType.value);
  const fluidContourRange = readFluidContourRange(panel, fluidDisplayType);
  return {
    originX: readNumber(panel.originX, MPM_DEFAULT_BOX_M.originX),
    originZ: readNumber(panel.originZ, MPM_DEFAULT_BOX_M.originZ),
    xLength: Math.max(10, readNumber(panel.xLength, MPM_DEFAULT_BOX_M.xLength)),
    yLength,
    zLength: Math.max(10, readNumber(panel.zLength, MPM_DEFAULT_BOX_M.zLength)),
    viscosity: Math.max(0, readNumber(panel.viscosity, MPM_DYNAMIC_VISCOSITY)),
    domainFriction: clamp01(readNumber(panel.domainFriction, MPM_DOMAIN_FRICTION)),
    terrainFriction: clamp01(readNumber(panel.terrainFriction, MPM_TERRAIN_FRICTION)),
    objectFriction: clamp01(readNumber(panel.objectFriction, MPM_OBJECT_FRICTION)),
    domainRestitution: Math.max(0, readNumber(panel.domainRestitution, MPM_DOMAIN_RESTITUTION)),
    terrainRestitution: Math.max(0, readNumber(panel.terrainRestitution, MPM_BOUNDARY_RESTITUTION)),
    objectRestitution: Math.max(0, readNumber(panel.objectRestitution, MPM_OBJECT_RESTITUTION)),
    terrainLift,
    waterDepth,
    particleCount: readIntegerInRange(
      panel.particleCount,
      MPM_PARTICLE_COUNT,
      1,
      MPM_PARTICLE_COUNT_MAX,
    ),
    // 物理的な粒子代表間隔。小さくすると必要 active 数が増え、上限で打ち切られる。
    particleSpacing: Math.max(
      MPM_PARTICLE_SPACING_MIN_M,
      readNumber(panel.particleSpacing, MPM_PARTICLE_SPACING_M),
    ),
    dt: Math.max(1e-5, readNumber(panel.dt, MPM_DT)),
    // steps/batch は 1 回の command buffer にまとめる substep 数。
    // HTML の max=3200 は手入力やスクリプト代入では実行時制限にならないため、読み取り時にも上限を掛ける。
    stepsPerBatch: readIntegerInRange(panel.stepsPerBatch, MPM_STEPS_PER_BATCH, 1, 3200),
    // 描画間引きは render だけを減らす設定。操作不能に近い値を避けるため 120 を上限にする。
    renderInterval: readIntegerInRange(panel.renderInterval, RENDER_INTERVAL_FRAMES, 1, 120),
    waterRender: panel.waterRender.checked,
    fluidDisplayType,
    fluidDisplayMin: fluidContourRange.min,
    fluidDisplayMax: fluidContourRange.max,
    particleSize: Math.max(0.01, readNumber(panel.particleSize, MPM_PARTICLE_SIZE_CELLS)),
  };
}

function mpmBoxLengthLimitError(
  box: Pick<MpmBoxInputMeters, 'xLength' | 'zLength'>,
): string | undefined {
  if (box.xLength > MPM_MAX_BOX_XZ_LENGTH_M || box.zLength > MPM_MAX_BOX_XZ_LENGTH_M) {
    return (
      `MPM 計算領域 xL/zL は最大 ${MPM_MAX_BOX_XZ_LENGTH_M.toFixed(0)}m です ` +
      `(現在 ${box.xLength.toFixed(0)}m x ${box.zLength.toFixed(0)}m)`
    );
  }
  return undefined;
}

function mpmCoverageTileCountForBox(
  box: Pick<MpmBoxInputMeters, 'originX' | 'originZ' | 'xLength' | 'zLength'>,
): number {
  const bounds = {
    minE: box.originX,
    maxE: box.originX + box.xLength,
    minN: box.originZ,
    maxN: box.originZ + box.zLength,
  };
  // Terrain.ensureSurfaceCoverage() と同じ z / padding で、適用前に同じ上限を再現する。
  return tileCountCoveringEnuBounds(bounds, MAX_ZOOM, 1);
}

function mpmCoverageTileLimitError(
  box: Pick<MpmBoxInputMeters, 'originX' | 'originZ' | 'xLength' | 'zLength'>,
): string | undefined {
  const coverageTileCount = mpmCoverageTileCountForBox(box);
  if (coverageTileCount > MPM_MAX_COVERAGE_TILE_COUNT) {
    return (
      `MPM DEM coverage tile 数 ${coverageTileCount} が上限 ` +
      `${MPM_MAX_COVERAGE_TILE_COUNT} を超えています`
    );
  }
  return undefined;
}

function mpmPanelLimitError(values: SimulationPanelValues): string | undefined {
  return mpmBoxLengthLimitError(values) ?? mpmCoverageTileLimitError(values);
}

function horizontalScaleFromMatrixWorld(object: MpmParticleObject): number {
  object.updateMatrixWorld(true);
  const e = object.matrixWorld.elements;
  const sx = Math.hypot(e[0], e[1], e[2]);
  const sz = Math.hypot(e[8], e[9], e[10]);
  return Math.max(1e-6, Math.min(sx, sz));
}

function fluidRenderScaleFromParticleSpacing(
  particleSpacingM: number,
  horizontalCellSizeM: number,
): {
  radiusWorld: number;
  thicknessScale: number;
} {
  // water render の見かけ半径は、ユーザーが直接操作する物理 spacing に追従させる。
  // particleMass から逆算すると、粒子上限で打ち切られた時に spacing 変更が見た目へ
  // 出にくくなるため、描画半径だけは spacing [m] を主入力にする。
  const requestedRadiusWorld =
    Math.max(
      MPM_FLUID_RENDER_MIN_RADIUS_M,
      particleSpacingM * MPM_FLUID_RENDER_RADIUS_SPACING_MULTIPLIER,
    );
  const maxRadiusWorld =
    Math.max(1e-6, horizontalCellSizeM) * MPM_FLUID_RENDER_RADIUS_CELLS;
  const radiusWorld = Math.min(maxRadiusWorld, requestedRadiusWorld);

  // 半径は spacing に追従させる一方、thickness は水面の視認性として扱う。
  // ここまで spacing に連動して落とすと、地形上で alpha がほぼ 0 になり、
  // 流体が存在していても見えなくなる。
  const thicknessScale = MPM_FLUID_RENDER_THICKNESS_SCALE;
  return { radiusWorld, thicknessScale };
}

function isMpmResizeFace(value: unknown): value is MpmResizeFace {
  return value === 'xMin' || value === 'xMax' || value === 'zMin' || value === 'zMax';
}

function cursorForResizeFace(face: MpmResizeFace | undefined): string {
  if (face === 'xMin' || face === 'xMax') return 'ew-resize';
  if (face === 'zMin' || face === 'zMax') return 'ns-resize';
  return '';
}

function labelForResizeFace(face: MpmResizeFace): string {
  if (face === 'xMin') return 'X- 側面';
  if (face === 'xMax') return 'X+ 側面';
  if (face === 'zMin') return 'Z- 側面';
  return 'Z+ 側面';
}

function isResizeHoverSuppressed(event: MouseEvent | PointerEvent): boolean {
  return event.shiftKey;
}

function hasFileDrop(dataTransfer: DataTransfer | null): boolean {
  if (!dataTransfer) return false;
  if (dataTransfer.items.length > 0) {
    return Array.from(dataTransfer.items).some((item) => item.kind === 'file');
  }
  return dataTransfer.files.length > 0;
}

function pickDroppedZipFile(dataTransfer: DataTransfer | null): File | undefined {
  if (!dataTransfer) return undefined;
  return Array.from(dataTransfer.files).find((file) => /\.zip$/i.test(file.name));
}

function zipProgressToModalProgressRange(
  progress: PlateauZipProgress,
  start: number,
  span: number,
): PlateauImportProgress {
  if (
    progress.indeterminate ||
    progress.loaded === undefined ||
    progress.total === undefined ||
    progress.total <= 0
  ) {
    return {
      message: progress.message,
      indeterminate: true,
    };
  }

  // ZIP/GML 変換の進捗を、呼び出し元が割り当てた modal 範囲へ写像する。
  // drag/drop は広め、URL import はダウンロード後なので狭めに使う。
  const fraction = Math.min(1, Math.max(0, progress.loaded / progress.total));
  return {
    message: progress.message,
    progress: start + fraction * span,
  };
}

async function createRemotePlateauZipSource(
  resource: PlateauCatalogResource,
  options: PlateauZipFetchOptions,
): Promise<PlateauZipSource> {
  const targetUrl = requireAllowedPlateauZipUrl(resource);
  const size = await fetchPlateauZipSize(resource, targetUrl, options);
  let fetchedBytes = 0;
  let onProgress = options.onProgress;
  const source: ProgressAwarePlateauZipSource = {
    fileName: resource.fileName,
    size,
    setProgressReporter: (nextOnProgress) => {
      onProgress = nextOnProgress;
    },
    readRange: async (offset, length) => {
      options.signal.throwIfAborted();
      const buffer = await fetchPlateauZipRange(resource, targetUrl, offset, length, options);
      fetchedBytes += buffer.byteLength;
      onProgress({
        message:
          `${resource.fileName}: 建物データ取得中... ` +
          formatBytesForStatus(fetchedBytes),
        indeterminate: true,
      });
      return buffer;
    },
  };
  return source;
}

function setPlateauZipSourceProgressReporter(
  source: PlateauZipSource,
  onProgress: (progress: PlateauImportProgress) => void,
): void {
  (source as ProgressAwarePlateauZipSource).setProgressReporter?.(onProgress);
}

function createLocalPlateauZipSource(file: File): PlateauZipSource {
  return {
    fileName: file.name,
    size: file.size,
    readRange: (offset, length) => file.slice(offset, offset + length).arrayBuffer(),
  };
}

function localPlateauBuildingExpansionSource(file: File): PlateauBuildingExpansionSource {
  return {
    label: file.name,
  };
}

function remotePlateauBuildingExpansionSource(
  resource: PlateauCatalogResource,
): PlateauBuildingExpansionSource {
  return {
    label: resource.datasetLabel,
  };
}

function requireAllowedPlateauZipUrl(resource: PlateauCatalogResource): string {
  const url = validatePlateauZipUrl(resource.url);
  if (!url) {
    throw new Error(`${resource.fileName}: 許可されていない PLATEAU ZIP URL です`);
  }
  return url;
}

async function fetchPlateauZipSize(
  resource: PlateauCatalogResource,
  targetUrl: string,
  options: PlateauZipFetchOptions,
): Promise<number> {
  try {
    const directSize = await fetchPlateauZipHead(targetUrl, options);
    if (directSize !== undefined) return directSize;
  } catch (err) {
    if (isAbortError(err)) throw err;
    console.warn('[Import] direct PLATEAU ZIP HEAD failed; using CKAN size if available', err);
  }

  if (resource.sizeBytes !== undefined) return resource.sizeBytes;
  throw new Error(`${resource.fileName}: ZIP サイズをブラウザから直接取得できません`);
}

async function fetchPlateauZipHead(
  url: string,
  options: PlateauZipFetchOptions,
): Promise<number | undefined> {
  const res = await fetch(url, {
    method: 'HEAD',
    mode: 'cors',
    signal: options.signal,
  });
  if (!res.ok) {
    throw new Error(`ZIP HEAD failed: ${res.status} ${res.statusText}`);
  }
  return finitePositiveNumber(Number(res.headers.get('content-length')));
}

async function fetchPlateauZipRange(
  resource: PlateauCatalogResource,
  targetUrl: string,
  offset: number,
  length: number,
  options: PlateauZipFetchOptions,
): Promise<ArrayBuffer> {
  const end = offset + length - 1;
  let res: Response;
  try {
    res = await fetch(targetUrl, {
      mode: 'cors',
      signal: options.signal,
      headers: {
        Range: `bytes=${offset}-${end}`,
      },
    });
  } catch (err) {
    if (isAbortError(err)) throw err;
    throw new Error(
      `${resource.fileName}: ZIP をブラウザから直接取得できません。` +
        `配信元の CORS / Range 設定を確認してください。`,
    );
  }
  if (res.status !== 206) {
    throw new Error(`ZIP range failed: ${res.status} ${res.statusText}`);
  }
  const buffer = await res.arrayBuffer();
  if (buffer.byteLength !== length) {
    throw new Error(`ZIP range size mismatch: ${buffer.byteLength}/${length}`);
  }
  return buffer;
}

function finitePositiveNumber(value: number): number | undefined {
  return Number.isFinite(value) && value > 0 ? value : undefined;
}

function formatBytesForStatus(value: number): string {
  if (value >= 1000 * 1000 * 1000) return `${(value / (1000 * 1000 * 1000)).toFixed(1)}GB`;
  if (value >= 1000 * 1000) return `${(value / (1000 * 1000)).toFixed(0)}MB`;
  if (value >= 1000) return `${(value / 1000).toFixed(0)}KB`;
  return `${value}B`;
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

function sampleTerrainHeightRange(
  terrain: Terrain,
  box: MpmBoxInputMeters,
  resolution: number,
  obstacles: MpmHeightfieldObstacle[] = [],
): TerrainHeightRange | null {
  const size = Math.max(2, Math.floor(resolution));
  const influenceRadiusM = heightRangeSampleRadiusM(box, size, size);
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let missingSamples = 0;

  for (let x = 0; x < size; x++) {
    const tx = size <= 1 ? 0 : x / (size - 1);
    const worldX = box.originX + box.xLength * tx;
    for (let z = 0; z < size; z++) {
      const tz = size <= 1 ? 0 : z / (size - 1);
      const worldZ = box.originZ + box.zLength * tz;
      const sample = terrain.sampleSurface(worldX, worldZ);
      if (sample) {
        minY = Math.min(minY, sample.height);
        maxY = Math.max(maxY, sample.height);
      } else {
        missingSamples++;
      }
      // yLength は現在の計算 box に実際に掛かる障害物だけで広げる。
      // BuildingObstacleLayer.bounds.maxY は自治体全域の最大値なので、ここで使うと
      // 小さな初期 box でも遠方の高い建物に引きずられて過大な yLength になる。
      for (const obstacle of obstacles) {
        const obstacleHeight = obstacle.sampleHeight(worldX, worldZ, influenceRadiusM);
        if (
          obstacleHeight === undefined ||
          !Number.isFinite(obstacleHeight)
        ) {
          continue;
        }
        minY = Math.min(minY, obstacleHeight);
        maxY = Math.max(maxY, obstacleHeight);
      }
    }
  }

  if (!Number.isFinite(minY) || !Number.isFinite(maxY)) {
    return null;
  }

  return {
    minY,
    maxY,
    missingSamples,
    totalSamples: size * size,
  };
}

function sampleFluidBaseHeight(
  terrain: Terrain,
  worldX: number,
  worldZ: number,
  obstacles: MpmHeightfieldObstacle[] = [],
  obstacleInfluenceRadiusM = 0,
): number | undefined {
  const terrainSample = terrain.sampleSurface(worldX, worldZ);
  let height = terrainSample?.height;
  for (const obstacle of obstacles) {
    const obstacleHeight = obstacle.sampleHeight(
      worldX,
      worldZ,
      obstacleInfluenceRadiusM,
    );
    if (
      obstacleHeight === undefined ||
      !Number.isFinite(obstacleHeight)
    ) {
      continue;
    }
    height = Math.max(height ?? Number.NEGATIVE_INFINITY, obstacleHeight);
  }
  return height;
}

function heightRangeSampleRadiusM(
  box: MpmBoxInputMeters,
  width: number,
  depth: number,
): number {
  const dx = box.xLength / Math.max(1, width - 1);
  const dz = box.zLength / Math.max(1, depth - 1);
  return Math.hypot(dx, dz) * 0.5;
}

function resizedBoxFromDelta(session: DomainResizeSession, delta: number): MpmBoxMeters {
  const start = session.startBox;
  const box = { ...start };
  const maxX = start.originX + start.xLength;
  const maxZ = start.originZ + start.zLength;

  switch (session.face) {
    case 'xMin': {
      const nextMin = Math.min(maxX - MIN_DOMAIN_LENGTH_M, start.originX - delta);
      box.originX = nextMin;
      box.xLength = maxX - nextMin;
      break;
    }
    case 'xMax': {
      const nextMax = Math.max(start.originX + MIN_DOMAIN_LENGTH_M, maxX + delta);
      box.xLength = nextMax - start.originX;
      break;
    }
    case 'zMin': {
      const nextMin = Math.min(maxZ - MIN_DOMAIN_LENGTH_M, start.originZ - delta);
      box.originZ = nextMin;
      box.zLength = maxZ - nextMin;
      break;
    }
    case 'zMax': {
      const nextMax = Math.max(start.originZ + MIN_DOMAIN_LENGTH_M, maxZ + delta);
      box.zLength = nextMax - start.originZ;
      break;
    }
  }

  return box;
}

function fluidRegionForBox(box: MpmBoxMeters): MpmFluidRegionMeters {
  return {
    originX: box.originX,
    originZ: box.originZ,
    xLength: box.xLength,
    zLength: box.zLength,
  };
}

function plateauInitialFluidRegionForBox(box: MpmBoxMeters): MpmFluidRegionMeters {
  // PLATEAU は計算領域を広めに取るが、初期水量は X 方向を半分にして x- 側境界へ寄せる。
  // `currentBox` 自体は変えず、粒子初期化と流体 proxy の footprint だけを狭める。
  const xLength = box.xLength * 0.5;
  return {
    originX: box.originX,
    originZ: box.originZ,
    xLength,
    zLength: box.zLength,
  };
}

function defaultFluidRegion(): MpmFluidRegionMeters {
  return {
    originX: MPM_DEFAULT_FLUID_REGION_M.originX,
    originZ: MPM_DEFAULT_FLUID_REGION_M.originZ,
    xLength: MPM_DEFAULT_FLUID_REGION_M.xLength,
    zLength: MPM_DEFAULT_FLUID_REGION_M.zLength,
  };
}

function fluidRegionIntersectsBoxXZ(
  region: MpmFluidRegionMeters,
  box: MpmBoxMeters,
): boolean {
  const regionMaxX = region.originX + region.xLength;
  const regionMaxZ = region.originZ + region.zLength;
  const boxMaxX = box.originX + box.xLength;
  const boxMaxZ = box.originZ + box.zLength;
  const overlapX = Math.min(regionMaxX, boxMaxX) - Math.max(region.originX, box.originX);
  const overlapZ = Math.min(regionMaxZ, boxMaxZ) - Math.max(region.originZ, box.originZ);
  return overlapX > 0 && overlapZ > 0;
}

function translatedBoxXZ(box: MpmBoxMeters, deltaX: number, deltaZ: number): MpmBoxMeters {
  return {
    ...box,
    originX: box.originX + deltaX,
    originZ: box.originZ + deltaZ,
  };
}

function translatedFluidRegionXZ(
  region: MpmFluidRegionMeters,
  deltaX: number,
  deltaZ: number,
): MpmFluidRegionMeters {
  return {
    ...region,
    originX: region.originX + deltaX,
    originZ: region.originZ + deltaZ,
  };
}

function scaledFluidRegionXZ(
  region: MpmFluidRegionMeters,
  scale: number,
): MpmFluidRegionMeters {
  const centerX = region.originX + region.xLength * 0.5;
  const centerZ = region.originZ + region.zLength * 0.5;
  const xLength = Math.max(MIN_FLUID_REGION_LENGTH_M, region.xLength * scale);
  const zLength = Math.max(MIN_FLUID_REGION_LENGTH_M, region.zLength * scale);

  // 流体 footprint は計算領域から外れた分も full size で保持する。
  // ここでは中心を固定したまま X/Z の長さだけを変え、表示側で現在 box へ clip する。
  return {
    originX: centerX - xLength * 0.5,
    originZ: centerZ - zLength * 0.5,
    xLength,
    zLength,
  };
}

function fluidRegionScaleFromPointer(
  session: FluidMoveSession,
  currentPoint: Vector3,
): number {
  const region = session.startRegion;
  const centerX = region.originX + region.xLength * 0.5;
  const centerZ = region.originZ + region.zLength * 0.5;
  const startDx = session.startHitPoint.x - centerX;
  const startDz = session.startHitPoint.z - centerZ;
  const currentDx = currentPoint.x - centerX;
  const currentDz = currentPoint.z - centerZ;
  const startDistance = Math.hypot(startDx, startDz);
  const currentDistance = Math.hypot(currentDx, currentDz);
  const referenceDistance = Math.max(
    MIN_FLUID_REGION_LENGTH_M,
    Math.hypot(region.xLength, region.zLength) * 0.5,
  );
  const rawScale =
    startDistance > 1e-6
      ? currentDistance / startDistance
      : 1 + currentDistance / referenceDistance;
  const minScale = Math.max(
    MIN_FLUID_REGION_LENGTH_M / Math.max(1e-6, region.xLength),
    MIN_FLUID_REGION_LENGTH_M / Math.max(1e-6, region.zLength),
  );

  // Ray が中心付近をまたいだ時に負値や極端値が入らないように、UI 操作用の範囲へ抑える。
  return Math.min(
    MAX_FLUID_REGION_SCALE,
    Math.max(minScale, Number.isFinite(rawScale) ? rawScale : 1),
  );
}

function clampTerrainLiftM(
  value: number,
  boxYLengthM: number,
  waterDepthM: number,
): number {
  if (!Number.isFinite(value)) return MPM_INITIAL_TERRAIN_LIFT_M;
  // y 移動は流体上面が現在の計算領域 y 上面を越えない範囲まで許可する。
  // 確定時は terrainLift を含めて yLength を再計算するため、高い位置へ動かした後も
  // simulation box 側が追従できる。
  const maxLiftM = Math.max(0, boxYLengthM - Math.max(0, waterDepthM));
  return Math.min(maxLiftM, Math.max(0, value));
}

function clampWaterDepthM(value: number): number {
  if (!Number.isFinite(value)) return MPM_INITIAL_WATER_DEPTH_M;
  return Math.max(1, value);
}

function cursorForFluidMoveMode(mode: FluidMoveMode): string {
  if (mode === 'x') return 'ew-resize';
  if (mode === 'z') return 'ns-resize';
  if (mode === 'scale') return 'nwse-resize';
  if (mode === 'y' || mode === 'depth') return 'ns-resize';
  return 'move';
}

/**
 * WebGPU が使えるか先に判定。使えなければ案内を出して終了。
 * - navigator.gpu の有無だけでなく、requestAdapter() で実アダプタが返るかも見る
 *   （Linux + 一部 Chromium で gpu はあるがアダプタが取れないことがある）
 */
async function requestWebGPUDevice(): Promise<GPUDevice | null> {
  if (!('gpu' in navigator)) return null;
  try {
    setHudStatus('WebGPU adapter 取得中...');
    // powerPreference を固定すると環境によって requestDevice() が返らないことがある。
    // ブラウザ既定の adapter 選択に任せ、まず確実に device を得る。
    const adapter = await navigator.gpu.requestAdapter();
    if (adapter === null) return null;

    // このアプリの compute/render 経路は追加 feature を要求しない。
    // adapter.features を全要求すると環境によって device 取得で止まることがあるため、
    // 最小構成の device を Three.js と MPM で共有する。
    setHudStatus('WebGPU device 取得中...');
    const device = await adapter.requestDevice();
    attachWebGPUErrorLogger(device);
    setHudStatus('WebGPU device 取得完了');
    return device;
  } catch {
    return null;
  }
}

function attachWebGPUErrorLogger(device: GPUDevice): void {
  device.addEventListener('uncapturederror', (ev) => {
    console.error('[WebGPU] uncaptured error', ev.error.message);
    if (hud) hud.textContent = `WebGPU error: ${ev.error.message}`;
  });
  device.lost.then((info) => {
    console.error('[WebGPU] device lost', info.reason, info.message);
    if (hud) hud.textContent = `WebGPU device lost: ${info.message}`;
  }).catch((err) => {
    console.error('[WebGPU] device lost handler failed', err);
  });
}

/** メイン処理 */
async function main(): Promise<void> {
  const panel = getSimulationPanel();
  const importGsiTabButton = buttonById('import-tab-gsi');
  const importPlateauTabButton = buttonById('import-tab-plateau');
  const importToggleButton = buttonById('import-toggle');
  const plateauPrefectureSelect = selectById('plateau-prefecture-select');
  const plateauCitySelect = selectById('plateau-city-select');
  const plateauYearSelect = selectById('plateau-year-select');
  const plateauResourceSelect = selectById('plateau-resource-select');
  const applyPlateauCatalogButton = buttonById('apply-plateau-catalog');
  const importPanelEl = document.getElementById('import-panel') as HTMLDivElement | null;
  const gsiImportPane = document.getElementById('gsi-import-pane') as HTMLDivElement | null;
  const plateauImportPane = document.getElementById('plateau-import-pane') as HTMLDivElement | null;
  const simPanelEl = document.getElementById('simulation-panel') as HTMLDivElement | null;
  attachCollapsiblePanel(importPanelEl, importToggleButton);
  attachCollapsiblePanel(simPanelEl, panel.toggle);
  attachSimulationTooltips(simPanelEl);
  const setImportSourceMode = (mode: ImportSourceMode): void => {
    const gsiActive = mode === 'gsi';
    importGsiTabButton.classList.toggle('active', gsiActive);
    importPlateauTabButton.classList.toggle('active', !gsiActive);
    importGsiTabButton.setAttribute('aria-selected', String(gsiActive));
    importPlateauTabButton.setAttribute('aria-selected', String(!gsiActive));
    if (gsiImportPane) gsiImportPane.hidden = !gsiActive;
    if (plateauImportPane) plateauImportPane.hidden = gsiActive;
  };
  const setSimulationRunningUi = (running: boolean): void => {
    simPanelEl?.classList.toggle('running', running);
    simPanelEl?.setAttribute('aria-busy', String(running));
  };
  setInitialPanelValues(panel);
  setImportSourceMode('gsi');
  setDataAttribution('gsi');

  // ---- WebGPU チェック ----
  const device = await requestWebGPUDevice();
  if (device === null) {
    if (unsupported) unsupported.style.display = 'flex';
    if (hud) hud.textContent = 'WebGPU 非対応';
    return;
  }

  // ---- Viewer 初期化 ----
  setHudStatus('Viewer 初期化中...');
  const viewer = new Viewer(document.body, device);
  await viewer.init();
  setHudStatus('Viewer 初期化完了');

  // ---- Terrain を Scene に追加 ----
  // GSI と PLATEAU は別 Terrain として保持し、タブ切替では表示対象だけを入れ替える。
  const gsiTerrain = new Terrain(GSI_TERRAIN_SOURCE, { aoiCenter: getAoiCenter() });
  const plateauTerrain = new Terrain(PLATEAU_TERRARIUM_TP_SOURCE, { aoiCenter: getAoiCenter() });
  plateauTerrain.visible = false;
  viewer.scene.add(gsiTerrain);
  viewer.scene.add(plateauTerrain);
  let terrain = gsiTerrain;
  let buildingLayers: BuildingObstacleLayer[] = [];
  let plateauBuildingLayer: BuildingObstacleLayer | undefined = undefined;
  let plateauMpmBuildingLayers: BuildingObstacleLayer[] = [];
  const plateauBuildingChunkLayers = new Map<string, BuildingObstacleLayer>();
  let preparedPlateauTemplate: PreparedPlateauTemplate | undefined = undefined;
  let preparedPlateauScene: PreparedPlateauScene | undefined = undefined;
  let activePlateauBuildingExpansion: ActivePlateauBuildingExpansion | undefined = undefined;
  let plateauCatalogDatasets: PlateauCatalogDataset[] = [];
  let plateauCatalogResources: PlateauCatalogResource[] = [];
  let plateauCatalogBusy = false;
  let activeTerrainMode: ImportSourceMode = 'gsi';
  const cameraViewByMode = new Map<ImportSourceMode, ViewerCameraViewState>();

  const disposeBuildingLayer = (
    layer: BuildingObstacleLayer,
    options: { removeFromScene?: boolean } = {},
  ): void => {
    if (options.removeFromScene) {
      viewer.scene.remove(layer);
    }
    layer.dispose();
  };

  const clearPlateauMpmBuildingLayers = (): void => {
    for (const layer of plateauMpmBuildingLayers) {
      // MPM 用 layer は scene に追加しない。geometry/index のメモリだけ破棄する。
      disposeBuildingLayer(layer);
    }
    plateauMpmBuildingLayers = [];
    buildingLayers = [];
  };

  const clearPlateauBuildingLayer = (): void => {
    if (plateauBuildingLayer) {
      disposeBuildingLayer(plateauBuildingLayer, { removeFromScene: true });
      plateauBuildingLayer = undefined;
    }
    for (const layer of plateauBuildingChunkLayers.values()) {
      disposeBuildingLayer(layer, { removeFromScene: true });
    }
    plateauBuildingChunkLayers.clear();
    clearPlateauMpmBuildingLayers();
  };

  const clearPlateauImportDataForReload = (): void => {
    activePlateauBuildingExpansion = undefined;
    clearPlateauBuildingLayer();
    preparedPlateauTemplate = undefined;
    preparedPlateauScene = undefined;
    // 同じ PLATEAU source への再 import でも force で描画 tile / coverage tile を捨てる。
    // これにより旧都市の DEM cache と新 ZIP 変換結果を同時に長く保持しない。
    plateauTerrain.setTileSource(PLATEAU_TERRARIUM_TP_SOURCE, { force: true });
  };

  const setActivePlateauBuildingExpansion = (
    imported: PlateauZipImportResult,
    source: PlateauBuildingExpansionSource,
    zipSource: PlateauZipSource,
  ): void => {
    activePlateauBuildingExpansion = {
      imported,
      source,
      zipSource,
      // 追加読み込みの ENU -> 緯度経度変換は、建物 layer / PLATEAU terrain と
      // 同じ AOI 原点で行う必要がある。この関数は terrain の AOI 切替より前に
      // 呼ばれる経路があるため、現在の plateauTerrain ではなく import manifest
      // そのものから原点を決める。
      aoiCenter: manifestBoundsCenter(imported.manifest.bounds),
      geoJsonChunks: [imported.geojson],
      loadedEntryNames: new Set(imported.sourceEntries),
      loading: false,
      lastCheckAt: 0,
    };
  };

  const latLonBoundsFromEnuBounds = (
    bounds: BuildingLayerHorizontalBounds,
    aoiCenter: AoiCenter,
  ): PlateauZipLatLonBounds =>
    withAoiCenter(aoiCenter, () => {
      // ENU -> 緯度経度はほぼ線形だが、bounds は常に 4 隅から作る。
      // 2 点だけだと、将来の投影変更や符号の取り違えで「見えているのに
      // GML mesh と交差しない」状態を見逃しやすい。
      const corners = [
        enuToLonLat(bounds.minX, bounds.minZ),
        enuToLonLat(bounds.minX, bounds.maxZ),
        enuToLonLat(bounds.maxX, bounds.minZ),
        enuToLonLat(bounds.maxX, bounds.maxZ),
      ];
      return {
        minLat: Math.min(...corners.map((corner) => corner.lat)),
        maxLat: Math.max(...corners.map((corner) => corner.lat)),
        minLon: Math.min(...corners.map((corner) => corner.lon)),
        maxLon: Math.max(...corners.map((corner) => corner.lon)),
      };
    });

  const normalizedEnuBoundsAroundCenter = (
    center: Vector3,
    sourceBounds: BuildingLayerHorizontalBounds,
  ): BuildingLayerHorizontalBounds => {
    const sourceWidth = Math.max(0, sourceBounds.maxX - sourceBounds.minX);
    const sourceDepth = Math.max(0, sourceBounds.maxZ - sourceBounds.minZ);
    const length = Math.min(
      PLATEAU_IMPORT_CAMERA_ENTRY_MAX_BOUNDS_LENGTH_M,
      Math.max(
        PLATEAU_IMPORT_CAMERA_ENTRY_SELECTION_LENGTH_M,
        sourceWidth,
        sourceDepth,
      ),
    );
    const half = length / 2;
    return {
      minX: center.x - half,
      maxX: center.x + half,
      minZ: center.z - half,
      maxZ: center.z + half,
    };
  };

  const visibleGroundBoundsForPlateauBuildings = (
    camera: PerspectiveCamera,
    targetTerrain: Terrain,
    aoiCenter: AoiCenter,
  ): PlateauVisibleGroundBounds | undefined => {
    // 追加建物は「画面に少し入った遠景」まで先読みすると、大阪市のような
    // 高密度地域で常に chunk import が走り続ける。中央〜手前寄りの地表だけを
    // sample し、カメラを実際に移動した時だけ次の小範囲を読む。
    const ndcSamples: Array<[number, number]> = [
      [0, -0.05],
      [-0.45, -0.55],
      [0, -0.55],
      [0.45, -0.55],
      [-0.35, 0.1],
      [0.35, 0.1],
    ];
    const points: Vector3[] = [];
    const groundRaycaster = new Raycaster();
    const sampleNdc = new Vector2();
    let centerHitPoint: Vector3 | undefined;

    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
    for (let sampleIndex = 0; sampleIndex < ndcSamples.length; sampleIndex++) {
      const [ndcX, ndcY] = ndcSamples[sampleIndex]!;
      // 地形タイルと同じ実 mesh に ray を当てる。水平 0m 平面で代用すると、
      // 斜め視点や高低差のある地形で画面上の DEM と別地点を追加読み込み対象にしてしまう。
      sampleNdc.set(ndcX, ndcY);
      groundRaycaster.setFromCamera(sampleNdc, camera);
      const hit = groundRaycaster.intersectObject(targetTerrain, true)[0];
      if (!hit) continue;
      if (hit.distance > PLATEAU_IMPORT_CAMERA_ENTRY_MAX_RAY_DISTANCE_M) continue;
      const hitPoint = hit.point.clone();
      if (sampleIndex === 0) centerHitPoint = hitPoint.clone();
      points.push(hitPoint);
    }

    if (points.length === 0) return undefined;
    const rawBounds: BuildingLayerHorizontalBounds = {
      minX: Number.POSITIVE_INFINITY,
      maxX: Number.NEGATIVE_INFINITY,
      minZ: Number.POSITIVE_INFINITY,
      maxZ: Number.NEGATIVE_INFINITY,
    };
    const center = new Vector3();
    for (const point of points) {
      rawBounds.minX = Math.min(rawBounds.minX, point.x);
      rawBounds.maxX = Math.max(rawBounds.maxX, point.x);
      rawBounds.minZ = Math.min(rawBounds.minZ, point.z);
      rawBounds.maxZ = Math.max(rawBounds.maxZ, point.z);
      center.add(point);
    }
    center.multiplyScalar(1 / points.length);
    const selectionCenter = centerHitPoint ?? center;
    const normalizedBounds = normalizedEnuBoundsAroundCenter(selectionCenter, rawBounds);
    return {
      center: selectionCenter,
      latLonBounds: latLonBoundsFromEnuBounds(normalizedBounds, aoiCenter),
    };
  };

  const syncPlateauLoadedEntryMetadata = (
    imported: PlateauZipImportResult,
    loadedEntryNames: Set<string>,
  ): void => {
    const sourceEntries = Array.from(loadedEntryNames).sort((a, b) => a.localeCompare(b));
    imported.sourceEntries = sourceEntries;
    if (imported.geojson.metadata) {
      imported.geojson.metadata.sourceEntries = sourceEntries;
      imported.geojson.metadata.featureCount = imported.geojson.features.length;
    }
  };

  const addPlateauDisplayBuildingChunk = (
    state: ActivePlateauBuildingExpansion,
    geojson: BuildingGeoJson,
    key: string,
  ): BuildingObstacleLayer | undefined => {
    if (plateauBuildingChunkLayers.has(key)) return undefined;
    const nextLayer = BuildingObstacleLayer.fromGeoJson(geojson, {
      aoiCenter: state.aoiCenter,
      placement: 'terrain-draped',
      terrainHeightSampler: (worldX, worldZ) => plateauTerrain.sampleSurface(worldX, worldZ)?.height,
      buildMpmObstacles: false,
    });
    nextLayer.visible = activeTerrainMode === 'plateau';
    viewer.scene.add(nextLayer);
    plateauBuildingChunkLayers.set(key, nextLayer);
    return nextLayer;
  };

  const plateauChunkLayerKey = (sourceEntries: string[]): string =>
    sourceEntries.length === 0 ? `chunk-${plateauBuildingChunkLayers.size}` : sourceEntries.join('\n');

  const setPlateauDisplayLayersVisible = (visible: boolean): void => {
    if (plateauBuildingLayer) {
      plateauBuildingLayer.visible = visible;
    }
    for (const layer of plateauBuildingChunkLayers.values()) {
      layer.visible = visible;
    }
  };

  const buildingClipBoundsForBox = (
    box: MpmBoxInputMeters,
    marginM: number,
  ): BuildingLayerHorizontalBounds => ({
    minX: box.originX - marginM,
    maxX: box.originX + box.xLength + marginM,
    minZ: box.originZ - marginM,
    maxZ: box.originZ + box.zLength + marginM,
  });

  const rebuildPlateauMpmBuildingLayersForBox = (box: MpmBoxInputMeters): void => {
    clearPlateauMpmBuildingLayers();
    if (activeTerrainMode !== 'plateau') return;

    const state = activePlateauBuildingExpansion;
    if (!state) {
      // Cached sample など chunk state が無い経路は、既存 layer を従来通り MPM に使う。
      buildingLayers = plateauBuildingLayer ? [plateauBuildingLayer] : [];
      return;
    }

    const includeBounds = buildingClipBoundsForBox(
      box,
      PLATEAU_MPM_BUILDING_CLIP_MARGIN_M,
    );
    for (const geojson of state.geoJsonChunks) {
      const layer = BuildingObstacleLayer.fromGeoJson(geojson, {
        aoiCenter: state.aoiCenter,
        placement: 'terrain-draped',
        terrainHeightSampler: (worldX, worldZ) =>
          plateauTerrain.sampleSurface(worldX, worldZ)?.height,
        buildMpmObstacles: true,
        includeBounds,
      });
      if (layer.buildingCount === 0) {
        layer.dispose();
        continue;
      }
      layer.visible = false;
      plateauMpmBuildingLayers.push(layer);
    }
    buildingLayers = plateauMpmBuildingLayers;
  };

  const loadPlateauBuildingsForVisibleGround = async (
    state: ActivePlateauBuildingExpansion,
    visibleBounds: PlateauVisibleGroundBounds,
  ): Promise<void> => {
    state.loading = true;
    const report = (progress: PlateauImportProgress): void => {
      if (activePlateauBuildingExpansion !== state) return;
      setImportStatus(progress.message);
    };

    try {
      const source = state.zipSource;
      setPlateauZipSourceProgressReporter(source, report);
      if (activePlateauBuildingExpansion !== state) return;
      const chunk = await importPlateauZipSourceChunk(source, {
        excludeEntryNames: state.loadedEntryNames,
        entrySelection: {
          bounds: visibleBounds.latLonBounds,
          paddingM: PLATEAU_IMPORT_CAMERA_ENTRY_SELECTION_PADDING_M,
          // 地形タイルに近い挙動にするため、建物も 1 回で可視範囲全体を読まない。
          // 大都市では可視範囲に数百 GML entry が重なるので、近い未読 entry から
          // 小さな chunk として足していく。
          maxSelectedEntryCount: PLATEAU_IMPORT_CAMERA_ENTRY_MAX_SELECTED_COUNT,
          maxSelectedUncompressedBytes:
            PLATEAU_IMPORT_CAMERA_ENTRY_MAX_UNCOMPRESSED_BYTES,
          maxSelectedCompressedBytes:
            PLATEAU_IMPORT_CAMERA_ENTRY_MAX_COMPRESSED_BYTES,
          // 画面中央の GML mesh が未読の場合だけ、中心 tile + 周辺 8 tile 相当を
          // まとめて読む。中央が既読なら通常の 2 entry 読み込みのままにして、
          // 停止中に遠方まで掘り続けない。
          centerUnloadedNeighborhoodEntryCount:
            PLATEAU_IMPORT_CAMERA_CENTER_NEIGHBOR_ENTRY_COUNT,
          centerUnloadedNeighborhoodUncompressedBytes:
            PLATEAU_IMPORT_CAMERA_CENTER_NEIGHBOR_UNCOMPRESSED_BYTES,
          centerUnloadedNeighborhoodCompressedBytes:
            PLATEAU_IMPORT_CAMERA_CENTER_NEIGHBOR_COMPRESSED_BYTES,
        },
        onProgress: (zipProgress) => report(zipProgress),
      });
      if (activePlateauBuildingExpansion !== state || !chunk) return;
      for (const entryName of chunk.sourceEntries) {
        state.loadedEntryNames.add(entryName);
      }
      // 以前は 1 chunk 読めたら `lastCenter` を消し、同じカメラ位置でも次の
      // interval で残り候補を読んでいた。しかし都市部ではこれが連続 import になり、
      // 操作中ずっと mesh 追加のスパイクが出る。ここでは現在位置の読み足しは
      // 1 回に留め、カメラが閾値以上動いた時だけ次の小範囲を読む。
      syncPlateauLoadedEntryMetadata(state.imported, state.loadedEntryNames);

      if (!chunk.geojson || chunk.geojson.features.length === 0) return;
      state.geoJsonChunks.push(chunk.geojson);
      const layer = addPlateauDisplayBuildingChunk(
        state,
        chunk.geojson,
        plateauChunkLayerKey(chunk.sourceEntries),
      );
      const addedCount = layer?.buildingCount ?? chunk.geojson.features.length;
      setImportStatus(
        `${state.imported.manifest.name}: 周辺建物 ${addedCount} buildings を追加表示`,
      );
    } catch (err) {
      if (activePlateauBuildingExpansion !== state || isAbortError(err)) return;
      console.warn('[Import] PLATEAU building expansion failed', err);
      setImportStatus(
        `${state.source.label}: 周辺建物の追加読み込みに失敗しました: ${errorMessageFromUnknown(err)}`,
      );
    } finally {
      if (activePlateauBuildingExpansion === state) {
        state.loading = false;
      }
    }
  };

  const maybeLoadPlateauBuildingsForCamera = (camera: unknown): void => {
    const state = activePlateauBuildingExpansion;
    if (
      activeTerrainMode !== 'plateau' ||
      !state ||
      state.loading ||
      !plateauBuildingLayer ||
      !(camera instanceof PerspectiveCamera)
    ) {
      return;
    }

    const now = performance.now();
    if (now - state.lastCheckAt < PLATEAU_IMPORT_CAMERA_ENTRY_CHECK_INTERVAL_MS) return;
    state.lastCheckAt = now;

    const visibleBounds = visibleGroundBoundsForPlateauBuildings(
      camera,
      plateauTerrain,
      state.aoiCenter,
    );
    if (!visibleBounds) return;
    const target = visibleBounds.center;
    if (
      !Number.isFinite(target.x) ||
      !Number.isFinite(target.y) ||
      !Number.isFinite(target.z)
    ) {
      return;
    }
    if (
      state.lastCenter &&
      target.distanceToSquared(state.lastCenter) <
        PLATEAU_IMPORT_CAMERA_ENTRY_MOVE_THRESHOLD_M *
          PLATEAU_IMPORT_CAMERA_ENTRY_MOVE_THRESHOLD_M
    ) {
      return;
    }
    state.lastCenter = target.clone();
    void loadPlateauBuildingsForVisibleGround(state, visibleBounds);
  };

  const selectedPlateauCatalogResource = (): PlateauCatalogResource | undefined => {
    if (plateauResourceSelect.value === '') return undefined;
    const index = Number(plateauResourceSelect.value);
    if (!Number.isInteger(index) || index < 0) return undefined;
    return plateauCatalogResources[index];
  };

  const updatePlateauResourceStatus = (): void => {
    const resource = selectedPlateauCatalogResource();
    if (!resource) return;
    if (resource.sizeBytes === undefined) {
      setImportStatus(`${resource.datasetLabel}: ZIP 全体サイズ不明 / 建物だけ取得`);
      return;
    }
    const sizeLabel = formatBytesForStatus(resource.sizeBytes);
    setImportStatus(`${resource.datasetLabel}: ZIP 全体 ${sizeLabel} / 建物だけ取得`);
  };

  const updatePlateauCatalogControls = (): void => {
    plateauPrefectureSelect.disabled = plateauCatalogBusy || plateauCatalogDatasets.length === 0;
    plateauCitySelect.disabled =
      plateauCatalogBusy || datasetsForPlateauPrefecture(plateauPrefectureSelect.value).length === 0;
    plateauYearSelect.disabled =
      plateauCatalogBusy ||
      datasetsForPlateauCity(plateauPrefectureSelect.value, plateauCitySelect.value).length === 0;
    plateauResourceSelect.disabled = plateauCatalogBusy || plateauCatalogResources.length === 0;
    const selectedResource = selectedPlateauCatalogResource();
    applyPlateauCatalogButton.disabled =
      plateauCatalogBusy ||
      selectedResource === undefined;
  };

  const setPlateauSelectPlaceholder = (
    select: HTMLSelectElement,
    label: string,
  ): void => {
    select.replaceChildren(new Option(label, ''));
  };

  const datasetsForPlateauPrefecture = (prefectureLabel: string): PlateauCatalogDataset[] =>
    plateauCatalogDatasets.filter((dataset) => dataset.prefectureLabel === prefectureLabel);

  const datasetsForPlateauCity = (
    prefectureLabel: string,
    cityKey: string,
  ): PlateauCatalogDataset[] =>
    plateauCatalogDatasets.filter(
      (dataset) =>
        dataset.prefectureLabel === prefectureLabel &&
        dataset.cityKey === cityKey,
    );

  const defaultPlateauDataset = (datasets: PlateauCatalogDataset[]): PlateauCatalogDataset | undefined =>
    datasets.find(
      (dataset) =>
        dataset.prefectureLabel === DEFAULT_PLATEAU_PREFECTURE_LABEL &&
        dataset.municipalityLabel === DEFAULT_PLATEAU_MUNICIPALITY_LABEL,
    );

  const populatePlateauPrefectureSelect = (
    datasets: PlateauCatalogDataset[],
    preferredPrefectureLabel: string | undefined,
    preferredCityKey: string | undefined,
    preferredDatasetName: string | undefined,
  ): void => {
    plateauCatalogDatasets = datasets;
    plateauCatalogResources = [];
    setPlateauSelectPlaceholder(
      plateauCitySelect,
      datasets.length === 0 ? '候補なし' : '都道府県を選択',
    );
    setPlateauSelectPlaceholder(
      plateauYearSelect,
      datasets.length === 0 ? '候補なし' : '市町村を選択',
    );
    setPlateauSelectPlaceholder(
      plateauResourceSelect,
      datasets.length === 0 ? '候補なし' : '年度を選択',
    );
    const seenPrefectures = new Set<string>();
    const prefectureOptions: HTMLOptionElement[] = [];
    for (const dataset of datasets) {
      if (seenPrefectures.has(dataset.prefectureLabel)) continue;
      seenPrefectures.add(dataset.prefectureLabel);
      prefectureOptions.push(new Option(dataset.prefectureLabel, dataset.prefectureLabel));
    }
    const options = [
      new Option(datasets.length === 0 ? '候補なし' : '都道府県を選択', ''),
      ...prefectureOptions,
    ];
    plateauPrefectureSelect.replaceChildren(...options);
    if (
      preferredPrefectureLabel &&
      datasets.some((dataset) => dataset.prefectureLabel === preferredPrefectureLabel)
    ) {
      plateauPrefectureSelect.value = preferredPrefectureLabel;
    } else if (prefectureOptions.length === 1) {
      plateauPrefectureSelect.value = prefectureOptions[0]!.value;
    }
    populatePlateauCitySelect(
      plateauPrefectureSelect.value,
      preferredCityKey,
      preferredDatasetName,
    );
  };

  const populatePlateauCitySelect = (
    prefectureLabel: string,
    preferredCityKey: string | undefined,
    preferredDatasetName: string | undefined,
  ): void => {
    plateauCatalogResources = [];
    setPlateauSelectPlaceholder(plateauYearSelect, '市町村を選択');
    setPlateauSelectPlaceholder(plateauResourceSelect, '年度を選択');
    const datasets = datasetsForPlateauPrefecture(prefectureLabel);
    const seenCityKeys = new Set<string>();
    const cityOptions: HTMLOptionElement[] = [];
    for (const dataset of datasets) {
      if (seenCityKeys.has(dataset.cityKey)) continue;
      seenCityKeys.add(dataset.cityKey);
      const option = new Option(dataset.municipalityLabel, dataset.cityKey);
      option.title = dataset.cityLabel;
      cityOptions.push(option);
    }
    const options = [
      new Option(datasets.length === 0 ? '都道府県を選択' : '市町村を選択', ''),
      ...cityOptions,
    ];
    plateauCitySelect.replaceChildren(...options);
    if (preferredCityKey && datasets.some((dataset) => dataset.cityKey === preferredCityKey)) {
      plateauCitySelect.value = preferredCityKey;
    } else if (cityOptions.length === 1) {
      plateauCitySelect.value = cityOptions[0]!.value;
    }
    populatePlateauYearSelect(
      prefectureLabel,
      plateauCitySelect.value,
      preferredDatasetName,
    );
  };

  const populatePlateauYearSelect = (
    prefectureLabel: string,
    cityKey: string,
    preferredDatasetName: string | undefined,
  ): void => {
    plateauCatalogResources = [];
    setPlateauSelectPlaceholder(plateauResourceSelect, '年度を選択');
    const datasets = datasetsForPlateauCity(prefectureLabel, cityKey);
    const options = [
      new Option(datasets.length === 0 ? '市町村を選択' : '年度を選択', ''),
      ...datasets.map((dataset) => {
        const option = new Option(dataset.yearLabel, dataset.name);
        option.title = dataset.title;
        return option;
      }),
    ];
    plateauYearSelect.replaceChildren(...options);
    if (
      preferredDatasetName &&
      datasets.some((dataset) => dataset.name === preferredDatasetName)
    ) {
      plateauYearSelect.value = preferredDatasetName;
    } else if (datasets.length > 0) {
      // `loadPlateauCatalogDatasets()` は同一市町村内を年度降順で返す。
      // 既存選択が無い場合は、常に最新年度を既定値にする。
      plateauYearSelect.value = datasets[0]!.name;
    }
    updatePlateauCatalogControls();
  };

  const populatePlateauResourceSelect = (
    resources: PlateauCatalogResource[],
  ): void => {
    plateauCatalogResources = resources;
    const options = [
      new Option(resources.length === 0 ? 'CityGML ZIP なし' : 'ZIP を選択', ''),
      ...resources.map((resource, index) => {
        const option = new Option(resource.label, String(index));
        option.title = resource.url;
        return option;
      }),
    ];
    plateauResourceSelect.replaceChildren(...options);
    if (resources.length > 0) {
      plateauResourceSelect.value = '0';
    }
    updatePlateauCatalogControls();
    updatePlateauResourceStatus();
  };

  const loadPlateauCatalog = async (
    options: { preferDefaultShimoda?: boolean } = {},
  ): Promise<PlateauCatalogDataset | undefined> => {
    const currentPreferredPrefectureLabel = plateauPrefectureSelect.value || undefined;
    const currentPreferredCityKey = plateauCitySelect.value || undefined;
    const currentPreferredDatasetName = plateauYearSelect.value || undefined;
    plateauCatalogBusy = true;
    setImportStatus('PLATEAU CityGML 一覧取得中...');
    setPlateauSelectPlaceholder(plateauPrefectureSelect, '取得中...');
    setPlateauSelectPlaceholder(plateauCitySelect, '都道府県を選択');
    setPlateauSelectPlaceholder(plateauYearSelect, '市町村を選択');
    setPlateauSelectPlaceholder(plateauResourceSelect, '年度を選択');
    updatePlateauCatalogControls();
    try {
      const result = await loadPlateauCatalogDatasets();
      const defaultDataset = options.preferDefaultShimoda
        ? defaultPlateauDataset(result.datasets)
        : undefined;
      const preferredPrefectureLabel =
        defaultDataset?.prefectureLabel ?? currentPreferredPrefectureLabel;
      const preferredCityKey =
        defaultDataset?.cityKey ?? currentPreferredCityKey;
      const preferredDatasetName =
        defaultDataset?.name ?? currentPreferredDatasetName;
      populatePlateauPrefectureSelect(
        result.datasets,
        preferredPrefectureLabel,
        preferredCityKey,
        preferredDatasetName,
      );
      setImportStatus(`PLATEAU CityGML: ${result.datasets.length}/${result.total} datasets`);
      return plateauCatalogDatasets.find((dataset) => dataset.name === plateauYearSelect.value);
    } catch (err) {
      console.error('[Import] PLATEAU catalog load failed', err);
      plateauCatalogDatasets = [];
      plateauCatalogResources = [];
      setPlateauSelectPlaceholder(plateauPrefectureSelect, '取得失敗');
      setPlateauSelectPlaceholder(plateauCitySelect, '都道府県を選択');
      setPlateauSelectPlaceholder(plateauYearSelect, '市町村を選択');
      setPlateauSelectPlaceholder(plateauResourceSelect, '年度を選択');
      setImportStatus(`PLATEAU catalog failed: ${String(err)}`);
      return undefined;
    } finally {
      plateauCatalogBusy = false;
      updatePlateauCatalogControls();
    }
  };

  const loadPlateauResourceOptions = async (
    datasetName: string,
  ): Promise<PlateauCatalogResource[]> => {
    if (datasetName === '') {
      plateauCatalogResources = [];
      setPlateauSelectPlaceholder(plateauResourceSelect, '年度を選択');
      updatePlateauCatalogControls();
      return [];
    }
    const dataset = plateauCatalogDatasets.find((item) => item.name === datasetName);
    const datasetLabel = dataset?.label ?? datasetName;
    plateauCatalogBusy = true;
    plateauCatalogResources = [];
    setImportStatus(`${datasetLabel}: CityGML ZIP 確認中...`);
    setPlateauSelectPlaceholder(plateauResourceSelect, '取得中...');
    updatePlateauCatalogControls();
    try {
      const resources = await loadPlateauCatalogResources(datasetName);
      if (plateauYearSelect.value !== datasetName) return [];
      populatePlateauResourceSelect(resources);
      setImportStatus(`${datasetLabel}: ${resources.length} CityGML ZIP`);
      updatePlateauResourceStatus();
      return resources;
    } catch (err) {
      console.error('[Import] PLATEAU resource load failed', err);
      if (plateauYearSelect.value === datasetName) {
        plateauCatalogResources = [];
        setPlateauSelectPlaceholder(plateauResourceSelect, '取得失敗');
        setImportStatus(`PLATEAU resource failed: ${String(err)}`);
      }
      return [];
    } finally {
      plateauCatalogBusy = false;
      updatePlateauCatalogControls();
    }
  };

  // ---- カメラ回転 pivot を「画面中心の地形交差点」にする ----
  // CameraController 自体は Terrain を知らないので、ここでアプリ固有の raycast を渡す。
  // 回転開始時だけ評価されるため、DEM 原寸メッシュへの raycast でも毎フレーム負荷はない。
  const terrainRaycaster = new Raycaster();
  const screenCenterNdc = new Vector2(0, 0);
  const pointerNdc = new Vector2();
  viewer.setCameraOrbitPivotResolver(() => {
    terrainRaycaster.setFromCamera(screenCenterNdc, viewer.camera);
    const hits = terrainRaycaster.intersectObject(terrain, true);
    return hits[0]?.point;
  });

  // ---- MLS-MPM ダムブレイクを Scene に追加 ----
  // 計算領域境界と地形高さ場境界は、右上パネルの Reset 時に simulator へ渡す。
  const createMpm = (particleCount: number): MpmParticleObject => {
    const object = new MpmParticleObject(device, viewer.renderer, particleCount);
    viewer.scene.add(object);
    return object;
  };

  let mpm = createMpm(readPanelValues(panel).particleCount);
  const fluidRenderer = new ScreenSpaceFluidRenderer(
    device,
    document.body,
    mpm.simulator.particleBuffer,
  );
  let waterRenderEnabled = panel.waterRender.checked;
  let waterRenderParticleSpacingM = readPanelValues(panel).particleSpacing;
  let fluidDisplayType = fluidDisplayTypeFromValue(panel.fluidDisplayType.value);
  const fluidDisplayRanges: Record<FluidDisplayType, { min: number; max: number }> = {
    none: fluidContourDefaults('none'),
    velocity: fluidContourDefaults('velocity'),
    density: fluidContourDefaults('density'),
  };
  let fluidDisplayRange = fluidDisplayRanges[fluidDisplayType];

  const updateFluidColorbar = (): void => {
    if (!fluidColorbar) return;
    const shouldShow = fluidDisplayType !== 'none';
    fluidColorbar.hidden = !shouldShow;
    fluidColorbar.setAttribute('aria-hidden', String(!shouldShow));
    if (!shouldShow) return;

    if (fluidColorbarTitle) {
      fluidColorbarTitle.textContent = fluidDisplayType === 'velocity' ? '流速' : '密度比';
    }
    if (fluidColorbarMin) {
      fluidColorbarMin.textContent = formatFluidContourValue(fluidDisplayRange.min);
    }
    if (fluidColorbarMax) {
      fluidColorbarMax.textContent = formatFluidContourValue(fluidDisplayRange.max);
    }
  };

  const saveCurrentFluidDisplayRange = (): void => {
    if (fluidDisplayType === 'none') return;
    fluidDisplayRanges[fluidDisplayType] = readFluidContourRange(panel, fluidDisplayType);
  };

  const syncFluidDisplayControls = (
    options: { nextType?: FluidDisplayType; writeBack?: boolean } = {},
  ): void => {
    if (options.nextType !== undefined && options.nextType !== fluidDisplayType) {
      // type を切り替える直前に、現在の min/max を type 別 state へ退避する。
      // これにより `流速 -> 密度 -> 流速` と戻しても入力済み range が消えない。
      saveCurrentFluidDisplayRange();
      fluidDisplayType = options.nextType;
      panel.fluidDisplayType.value = fluidDisplayType;
    } else {
      fluidDisplayType = fluidDisplayTypeFromValue(panel.fluidDisplayType.value);
      saveCurrentFluidDisplayRange();
    }

    fluidDisplayRange = fluidDisplayRanges[fluidDisplayType];
    if (options.writeBack !== false) {
      panel.fluidDisplayMin.value = String(fluidDisplayRange.min);
      panel.fluidDisplayMax.value = String(fluidDisplayRange.max);
    }
    const contourDisabled = fluidDisplayType === 'none';
    panel.fluidDisplayMin.disabled = contourDisabled;
    panel.fluidDisplayMax.disabled = contourDisabled;
    mpm.setParticleContour(
      fluidDisplayType,
      fluidDisplayRange.min,
      fluidDisplayRange.max,
    );
    updateFluidColorbar();
  };

  const setWaterRenderEnabled = (enabled: boolean): void => {
    waterRenderEnabled = enabled;
    panel.waterRender.checked = enabled;
    fluidRenderer.setEnabled(enabled);
    mpm.setParticleMeshVisible(!enabled);
    updateFluidColorbar();
  };
  syncFluidDisplayControls();
  setWaterRenderEnabled(waterRenderEnabled);

  const syncActiveParticleCountPanel = (): void => {
    panel.activeParticleCount.value = String(mpm.getStats().activeParticles);
  };
  syncActiveParticleCountPanel();

  let mpmRunning = false;
  let mpmRunToken = 0;
  let lastMpmSubsteps = 0;
  let lastMpmSimTime = 0;
  let lastMpmDivergenceCheckAt =
    performance.now() - MPM_DIVERGENCE_CHECK_INTERVAL_MS;
  let currentBox: MpmBoxMeters = {
    originX: MPM_DEFAULT_BOX_M.originX,
    originY: MPM_RENDER_ORIGIN_M.y,
    originZ: MPM_DEFAULT_BOX_M.originZ,
    xLength: MPM_DEFAULT_BOX_M.xLength,
    yLength: MPM_DEFAULT_BOX_M.yLength,
    zLength: MPM_DEFAULT_BOX_M.zLength,
  };
  let currentFluidRegion: MpmFluidRegionMeters = defaultFluidRegion();

  const stopMpmCompute = (): number => {
    mpmRunning = false;
    setSimulationRunningUi(false);
    mpmRunToken++;
    return mpmRunToken;
  };

  const handleMpmDivergence = (result: MpmDivergenceCheckResult): void => {
    const message = mpmDivergenceModalMessage(result);
    stopMpmCompute();
    console.error('[MPM] divergence detected', result);
    setHudStatus(message);
    showErrorModal('MPM 計算を停止しました', message);
  };

  const checkMpmDivergenceIfNeeded = async (token: number): Promise<boolean> => {
    const now = performance.now();
    if (now - lastMpmDivergenceCheckAt < MPM_DIVERGENCE_CHECK_INTERVAL_MS) {
      return true;
    }
    lastMpmDivergenceCheckAt = now;

    const checkedMpm = mpm;
    const result = await checkedMpm.simulator.checkDivergence(
      MPM_DIVERGENCE_SPEED_LIMIT,
    );
    if (token !== mpmRunToken || checkedMpm !== mpm) {
      return false;
    }
    if (result.ok) {
      return true;
    }

    handleMpmDivergence(result);
    return false;
  };

  const waitForSubmittedMpmWork = async (): Promise<void> => {
    try {
      await device.queue.onSubmittedWorkDone();
    } catch (err) {
      console.warn('[MPM] queue wait failed', err);
    }
  };

  const runMpmComputeLoop = async (token: number): Promise<void> => {
    try {
      while (mpmRunning && token === mpmRunToken) {
        // 描画 frame とは独立に、GPU が直前 batch を処理し終えたら次 batch を投げる。
        // 無制限 submit は UI と GPU memory を圧迫するため、queue 完了を待って背圧を掛ける。
        mpm.stepBatch();
        await device.queue.onSubmittedWorkDone();
        if (token !== mpmRunToken) return;
        if (!(await checkMpmDivergenceIfNeeded(token))) return;
      }
    } catch (err) {
      if (token !== mpmRunToken) return;
      mpmRunning = false;
      setSimulationRunningUi(false);
      mpmRunToken++;
      console.error('[MPM] compute loop failed', err);
      const message = `MPM compute failed: ${errorMessageFromUnknown(err)}`;
      setHudStatus(message);
      showErrorModal('MPM 計算エラー', message);
    }
  };

  const startMpmCompute = (): void => {
    if (mpmRunning) return;
    mpmRunning = true;
    setSimulationRunningUi(true);
    lastMpmDivergenceCheckAt =
      performance.now() - MPM_DIVERGENCE_CHECK_INTERVAL_MS;
    const token = ++mpmRunToken;
    void runMpmComputeLoop(token);
  };

  const replaceMpm = (particleCount: number): void => {
    viewer.scene.remove(mpm);
    mpm.dispose();
    mpm = createMpm(particleCount);
    fluidRenderer.setParticleBuffer(mpm.simulator.particleBuffer);
    mpm.setParticleContour(
      fluidDisplayType,
      fluidDisplayRange.min,
      fluidDisplayRange.max,
    );
    setWaterRenderEnabled(waterRenderEnabled);
    lastMpmSubsteps = 0;
    lastMpmSimTime = 0;
  };

  let applySimulationPanelRun = 0;
  let lastSimulationApplyFailureMessage: string | undefined = undefined;
  const failSimulationPanelApply = (
    message: string,
    options: { importStatus?: boolean } = {},
  ): false => {
    lastSimulationApplyFailureMessage = message;
    setHudStatus(message);
    if (options.importStatus) {
      setImportStatus(message);
    }
    return false;
  };
  const applySimulationPanel = async (
    options: ApplySimulationPanelOptions,
  ): Promise<boolean> => {
    const runId = ++applySimulationPanelRun;
    lastSimulationApplyFailureMessage = undefined;
    const values = readPanelValues(panel);
    const limitError = mpmPanelLimitError(values);
    if (limitError) {
      return failSimulationPanelApply(limitError, { importStatus: true });
    }
    const wasRunning = mpmRunning;
    const stoppedToken = stopMpmCompute();

    // 計算 buffer / boundary buffer を書き換える前に、直前 batch の GPU work 完了を待つ。
    // これを挟まないと、実行中の compute が読んでいる buffer を CPU 側から差し替える可能性がある。
    await waitForSubmittedMpmWork();
    if (runId !== applySimulationPanelRun) return false;

    panel.terrainLift.value = String(values.terrainLift);
    panel.waterDepth.value = String(values.waterDepth);
    panel.particleCount.value = String(values.particleCount);
    panel.particleSpacing.value = String(values.particleSpacing);
    panel.objectFriction.value = String(values.objectFriction);
    panel.objectRestitution.value = String(values.objectRestitution);
    panel.stepsPerBatch.value = String(values.stepsPerBatch);
    panel.waterRender.checked = values.waterRender;
    panel.fluidDisplayType.value = values.fluidDisplayType;
    panel.fluidDisplayMin.value = String(values.fluidDisplayMin);
    panel.fluidDisplayMax.value = String(values.fluidDisplayMax);
    syncFluidDisplayControls();
    waterRenderParticleSpacingM = values.particleSpacing;
    const particleCountChanged = values.particleCount !== mpm.simulator.particleCount;
    if (particleCountChanged) {
      replaceMpm(values.particleCount);
    }

    setHudStatus(
      `MPM 領域 DEM 読み込み中: ${values.xLength.toFixed(0)}m x ${values.zLength.toFixed(0)}m`,
    );
    try {
      await terrain.ensureSurfaceCoverage(values);
    } catch (err) {
      if (runId !== applySimulationPanelRun) return false;
      console.warn('[MPM] terrain coverage load failed', err);
      return failSimulationPanelApply(
        `MPM 領域 DEM 読み込みに失敗しました: ${errorMessageFromUnknown(err)}`,
        { importStatus: true },
      );
    }
    if (runId !== applySimulationPanelRun) return false;

    rebuildPlateauMpmBuildingLayersForBox(values);

    const heightfieldObstacles: MpmHeightfieldObstacle[] | undefined =
      buildingLayers.length > 0 ? buildingLayers : undefined;
    const sideBoundaryObstacles: MpmSideBoundaryObstacle[] | undefined =
      buildingLayers.length > 0 ? buildingLayers : undefined;
    const heightRange = sampleTerrainHeightRange(
      terrain,
      values,
      MPM_TERRAIN_BOUNDARY_RESOLUTION,
      heightfieldObstacles,
    );
    if (heightRange === null) {
      return failSimulationPanelApply(
        'MPM 地形高度範囲を計算できませんでした。DEM samples が不足しています。',
        { importStatus: true },
      );
    }
    if (heightRange.missingSamples > 0) {
      return failSimulationPanelApply(
        `MPM terrain missing: ${heightRange.missingSamples}/${heightRange.totalSamples} samples. DEM coverage が不足しています。`,
        { importStatus: true },
      );
    }
    const collisionMaxY = heightRange.maxY;
    values.yLength = Math.max(
      values.terrainLift + values.waterDepth,
      collisionMaxY + values.terrainLift + values.waterDepth - heightRange.minY,
    );
    panel.originY.value = heightRange.minY.toFixed(2);
    panel.yLength.value = values.yLength.toFixed(2);

    const fallbackOriginY = Number.isFinite(currentBox.originY)
      ? currentBox.originY
      : MPM_RENDER_ORIGIN_M.y;
    const terrainBoundaryMaterial = {
      friction: values.terrainFriction,
      restitution: values.terrainRestitution,
    };
    const objectBoundaryMaterial = {
      friction: values.objectFriction,
      restitution: values.objectRestitution,
    };
    const terrainBoundary = TerrainBoundary.fromTerrain({
      terrain,
      gridSize: mpm.simulator.gridSize,
      box: values,
      material: terrainBoundaryMaterial,
      // 建物側面は BuildingSideBoundary で横法線境界として扱う。
      // ここで屋上高さを DEM heightfield に混ぜると、建物内へ入った粒子が
      // 屋上へ押し上げられる既存挙動が残るため、地形 boundary は DEM だけにする。
      obstacles: undefined,
      fallbackOriginY,
      resolution: MPM_TERRAIN_BOUNDARY_RESOLUTION,
    });
    if (terrainBoundary.missingSamples > 0) {
      return failSimulationPanelApply(
        `MPM terrain missing: ${terrainBoundary.missingSamples} samples. DEM coverage が不足しています。`,
        { importStatus: true },
      );
    }
    const initialFluidBaseBoundary =
      heightfieldObstacles === undefined
        ? terrainBoundary
        : TerrainBoundary.fromTerrain({
            terrain,
            gridSize: mpm.simulator.gridSize,
            box: values,
            material: terrainBoundaryMaterial,
            objectMaterial: objectBoundaryMaterial,
            // Reset 時の粒子配置だけは、DEM と建物上面の高い方を水層底面にする。
            // 衝突用 terrainBoundary には混ぜず、建物内部を屋上へ押し上げる挙動は戻さない。
            obstacles: heightfieldObstacles,
            fallbackOriginY,
            resolution: MPM_TERRAIN_BOUNDARY_RESOLUTION,
          });
    if (initialFluidBaseBoundary.missingSamples > 0) {
      return failSimulationPanelApply(
        `MPM terrain missing: ${initialFluidBaseBoundary.missingSamples} samples. DEM coverage が不足しています。`,
        { importStatus: true },
      );
    }
    const domainBoundary = new DomainBoundary(
      mpm.simulator.gridSize,
      {
        friction: values.domainFriction,
        restitution: values.domainRestitution,
      },
    );
    const buildingSideBoundary = BuildingSideBoundary.fromObstacles({
      gridSize: mpm.simulator.gridSize,
      box: values,
      originY: terrainBoundary.originY,
      material: objectBoundaryMaterial,
      obstacles: sideBoundaryObstacles,
      resolution: MPM_TERRAIN_BOUNDARY_RESOLUTION,
    });

    const nextBox = {
      originX: values.originX,
      originY: terrainBoundary.originY,
      originZ: values.originZ,
      xLength: values.xLength,
      yLength: values.yLength,
      zLength: values.zLength,
    };
    // 流体の footprint は world 座標で保持する。
    // 計算領域の 4 面を動かしてもここでは移動・拡縮せず、
    // MpmParticleObject / simulator 側の clipping で交差部分だけを有効にする。
    currentBox = nextBox;
    if (!fluidRegionIntersectsBoxXZ(currentFluidRegion, currentBox)) {
      // 流体 footprint が計算領域から完全に外れると、表示面も active 粒子も 0 になる。
      // その状態ではユーザーが面を掴んで戻せないため、全域を初期流体領域として復帰する。
      currentFluidRegion = fluidRegionForBox(currentBox);
    }
    panel.originY.value = terrainBoundary.originY.toFixed(2);
    mpm.configure({
      box: currentBox,
      initialFluidRegion: currentFluidRegion,
      initialFluidRegionHeightSampler: (worldX, worldZ) =>
        sampleFluidBaseHeight(
          terrain,
          worldX,
          worldZ,
          heightfieldObstacles,
        ),
      initialFluidBaseBoundary,
      viscosity: values.viscosity,
      domainBoundary,
      terrainBoundary,
      buildingSideBoundary,
      terrainLiftM: values.terrainLift,
      waterDepthM: values.waterDepth,
      particleSpacingM: values.particleSpacing,
    });
    // dt / steps per batch / 粒子描画サイズは runtime に直接反映 (リセット不要)
    mpm.simulator.setDt(values.dt);
    mpm.simulator.setStepsPerBatch(values.stepsPerBatch);
    viewer.setRenderIntervalFrames(values.renderInterval);
    setWaterRenderEnabled(values.waterRender);
    mpm.setParticleSize(values.particleSize);

    if (particleCountChanged || options.resetParticles) {
      mpm.resetDamBreak();
      syncActiveParticleCountPanel();
    }
    if (!options.stopAfterReset && wasRunning && mpmRunToken === stoppedToken) {
      startMpmCompute();
    }
    return true;
  };

  let geometryPanelApplyTimer: number | undefined = undefined;
  const clearGeometryPanelApplyTimer = (): void => {
    if (geometryPanelApplyTimer === undefined) return;
    window.clearTimeout(geometryPanelApplyTimer);
    geometryPanelApplyTimer = undefined;
  };
  const applyGeometryPanelChange = (): void => {
    clearGeometryPanelApplyTimer();
    void applySimulationPanel({ resetParticles: true, stopAfterReset: true });
  };
  const scheduleGeometryPanelApply = (): void => {
    clearGeometryPanelApplyTimer();
    geometryPanelApplyTimer = window.setTimeout(() => {
      geometryPanelApplyTimer = undefined;
      void applySimulationPanel({ resetParticles: true, stopAfterReset: true });
    }, 500);
  };

  const canvas = viewer.renderer.domElement;
  const domainResizeRaycaster = new Raycaster();
  const horizontalResizeHit = new Vector3();
  const horizontalDomainMoveHit = new Vector3();
  const horizontalFluidMoveHit = new Vector3();
  const fluidVerticalAxis = new Vector3(0, 1, 0);
  const lastPointerClient = new Vector2();
  let hasLastPointerClient = false;
  let hoveredDomainMove = false;
  let hoveredResizeFace: MpmResizeFace | undefined = undefined;
  let hoveredFluidRegion = false;
  let domainMoveSession: DomainMoveSession | undefined = undefined;
  let domainResizeSession: DomainResizeSession | undefined = undefined;
  let fluidMoveSession: FluidMoveSession | undefined = undefined;
  let lastResizeInteractionTime = 0;
  let editClickCandidate: EditClickCandidate | undefined = undefined;
  let suppressNextEditClick = false;

  const hideInteractionTooltip = (): void => {
    if (!interactionTooltip) return;
    interactionTooltip.hidden = true;
  };

  const showInteractionTooltip = (
    text: string,
    clientX: number,
    clientY: number,
  ): void => {
    if (!interactionTooltip) return;
    interactionTooltip.textContent = text;
    interactionTooltip.hidden = false;

    const tooltipRect = interactionTooltip.getBoundingClientRect();
    const gap = 14;
    const margin = 8;
    let left = clientX + gap;
    if (left + tooltipRect.width > window.innerWidth - margin) {
      left = clientX - tooltipRect.width - gap;
    }
    let top = clientY + gap;
    if (top + tooltipRect.height > window.innerHeight - margin) {
      top = clientY - tooltipRect.height - gap;
    }

    interactionTooltip.style.left = `${Math.round(
      Math.max(margin, Math.min(left, window.innerWidth - tooltipRect.width - margin)),
    )}px`;
    interactionTooltip.style.top = `${Math.round(
      Math.max(margin, Math.min(top, window.innerHeight - tooltipRect.height - margin)),
    )}px`;
  };

  const showInteractionTooltipAtLastPointer = (text: string): void => {
    if (!hasLastPointerClient) return;
    showInteractionTooltip(text, lastPointerClient.x, lastPointerClient.y);
  };

  const resizeHoverTooltipText = (face: MpmResizeFace): string =>
    `${labelForResizeFace(face)}\nクリック: この面の引き伸ばしを開始`;

  const resizeActiveTooltipText = (face: MpmResizeFace): string =>
    `${labelForResizeFace(face)}を編集中\nマウス移動: 面を引き伸ばし\nクリック: 確定\nEscキー: キャンセル`;

  const domainMoveHoverTooltipText = (): string =>
    '計算領域の移動\nCtrl+ドラッグ: 平行移動を開始';

  const domainMoveActiveTooltipText = (): string =>
    '計算領域を移動中\nドラッグ: 平行移動\nマウスを離す: 確定\nEscキー: キャンセル';

  const fluidHoverTooltipText = (): string =>
    '初期流体領域の編集\nクリック: 移動を開始';

  const fluidActiveTooltipText = (mode: FluidMoveMode): string => {
    const modeLine =
      mode === 'x'
        ? 'X 方向だけ移動中'
        : mode === 'y'
          ? '高さだけ移動中'
          : mode === 'z'
            ? 'Z 方向だけ移動中'
            : mode === 'scale'
              ? '中心固定で拡大縮小中'
              : mode === 'depth'
                ? '水深を変更中'
                : '自由移動中';
    return `初期流体領域を編集中\n${modeLine}\nXキー: X方向だけ移動\nYキー: 高さだけ移動\nZキー: Z方向だけ移動\nSキー: 中心固定で拡大縮小\nDキー: 水深を変更\nクリック: 確定\nEscキー: キャンセル`;
  };

  const updateEditClickCandidateDragState = (event: PointerEvent): void => {
    if (!editClickCandidate || event.pointerId !== editClickCandidate.pointerId) return;

    const dx = event.clientX - editClickCandidate.startX;
    const dy = event.clientY - editClickCandidate.startY;
    // カメラ移動や押しっぱなし調整の release で click が発火しても、面編集の開始/確定に使わない。
    if (dx * dx + dy * dy >= EDIT_CLICK_DRAG_THRESHOLD_SQ_PX) {
      editClickCandidate.dragged = true;
    }
  };

  const consumeSuppressedEditClick = (): boolean => {
    if (!suppressNextEditClick) return false;
    suppressNextEditClick = false;
    return true;
  };

  const updatePointerNdcFromClient = (clientX: number, clientY: number): void => {
    const rect = canvas.getBoundingClientRect();
    pointerNdc.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    pointerNdc.y = -((clientY - rect.top) / rect.height) * 2 + 1;
  };

  const updatePointerNdcFromEvent = (event: MouseEvent | PointerEvent): void => {
    updatePointerNdcFromClient(event.clientX, event.clientY);
  };

  const pickResizeFace = (event: MouseEvent | PointerEvent): ResizeFaceHit | undefined => {
    if (mpmRunning || isResizeHoverSuppressed(event)) return undefined;

    updatePointerNdcFromEvent(event);
    domainResizeRaycaster.setFromCamera(pointerNdc, viewer.camera);
    const resizeHits = domainResizeRaycaster.intersectObjects(
      mpm.getResizeFaceObjects(),
      false,
    );
    for (const hit of resizeHits) {
      const face = hit.object.userData.mpmResizeFace;
      if (!isMpmResizeFace(face)) continue;

      // pointermove ごとに DEM 原寸 mesh 全体へ raycast すると操作が重くなる。
      // ここでは resize 面の hit 点 x/z にある CPU DEM 高さだけを見て、
      // 操作面が地形面以下に埋まっている場合だけ hover/click を無効化する。
      const terrainSample = terrain.sampleSurface(hit.point.x, hit.point.z);
      if (
        terrainSample &&
        terrainSample.height >= hit.point.y - RESIZE_TERRAIN_HEIGHT_MARGIN_M
      ) {
        return undefined;
      }

      return { face, point: hit.point.clone() };
    }
    return undefined;
  };

  const pickFluidRegion = (event: MouseEvent | PointerEvent): FluidRegionHit | undefined => {
    if (mpmRunning || isResizeHoverSuppressed(event)) return undefined;

    updatePointerNdcFromEvent(event);
    domainResizeRaycaster.setFromCamera(pointerNdc, viewer.camera);
    const hits = domainResizeRaycaster.intersectObject(
      mpm.getFluidRegionObject(),
      false,
    );
    const hit = hits[0];
    if (!hit) return undefined;
    const handlePoint = mpm.snapPointToFluidRegionHandlePlane(hit.point);

    // 流体操作面も pointermove ごとに地形 mesh 全体へ raycast しない。
    // 操作面 hit の x/z にある CPU DEM 高さだけを見て、地形が面より手前なら pick しない。
    const terrainSample = terrain.sampleSurface(handlePoint.x, handlePoint.z);
    if (
      terrainSample &&
      terrainSample.height >= handlePoint.y - RESIZE_TERRAIN_HEIGHT_MARGIN_M
    ) {
      return undefined;
    }

    return { point: handlePoint };
  };

  const pickDomainMoveAtClient = (
    clientX: number,
    clientY: number,
    ctrlKey: boolean,
    shiftKey: boolean,
  ): DomainMoveHit | undefined => {
    // Ctrl 押下時だけ、計算領域全体の footprint を 1 枚の proxy 面として掴む。
    // 通常時は既存の流体移動・側面 resize を優先し、操作モードを混ぜない。
    if (mpmRunning || !ctrlKey || shiftKey) return undefined;

    updatePointerNdcFromClient(clientX, clientY);
    domainResizeRaycaster.setFromCamera(pointerNdc, viewer.camera);
    const hits = domainResizeRaycaster.intersectObjects(
      mpm.getDomainMoveObjects(),
      false,
    );
    const hit = hits[0];
    if (!hit) return undefined;

    // proxy は計算領域上面に置いている。地形より十分上なので通常は通るが、
    // 地形が box 上面へ届く破綻ケースでは「見えていない面」を掴まないようにする。
    const terrainSample = terrain.sampleSurface(hit.point.x, hit.point.z);
    if (
      terrainSample &&
      terrainSample.height >= hit.point.y - RESIZE_TERRAIN_HEIGHT_MARGIN_M
    ) {
      return undefined;
    }

    return { point: hit.point.clone() };
  };

  const pickDomainMove = (event: MouseEvent | PointerEvent): DomainMoveHit | undefined =>
    pickDomainMoveAtClient(event.clientX, event.clientY, event.ctrlKey, event.shiftKey);

  const clearResizeHover = (): void => {
    hoveredDomainMove = false;
    hoveredResizeFace = undefined;
    hoveredFluidRegion = false;
    hideInteractionTooltip();
    canvas.style.cursor = '';
    mpm.setDomainMoveState(false, false);
    mpm.setResizeFaceState(undefined, undefined);
    mpm.setFluidRegionState(false, false);
  };

  const writeHorizontalBoxToPanel = (box: MpmBoxMeters): void => {
    panel.originX.value = box.originX.toFixed(2);
    panel.originZ.value = box.originZ.toFixed(2);
    panel.xLength.value = box.xLength.toFixed(2);
    panel.zLength.value = box.zLength.toFixed(2);
  };

  const writeFluidVerticalToPanel = (terrainLiftM: number, waterDepthM: number): void => {
    panel.terrainLift.value = terrainLiftM.toFixed(2);
    panel.waterDepth.value = waterDepthM.toFixed(2);
  };

  const cancelRegionEdit = (): void => {
    const activeFluidSession = fluidMoveSession;
    const hadActiveSession =
      domainMoveSession !== undefined ||
      domainResizeSession !== undefined ||
      fluidMoveSession !== undefined;
    domainMoveSession = undefined;
    domainResizeSession = undefined;
    fluidMoveSession = undefined;
    if (hadActiveSession) {
      viewer.setCameraZoomOnlyMode(false);
      mpm.setDisplayBox(currentBox);
      mpm.setDisplayFluidRegion(currentFluidRegion, currentBox);
      writeHorizontalBoxToPanel(currentBox);
      if (activeFluidSession) {
        writeFluidVerticalToPanel(
          activeFluidSession.startTerrainLiftM,
          activeFluidSession.startWaterDepthM,
        );
      }
      lastResizeInteractionTime = performance.now();
    }
    clearResizeHover();
  };

  const applyImportedBoxToPanel = (box: MpmBoxInputMeters): void => {
    panel.originX.value = box.originX.toFixed(2);
    panel.originZ.value = box.originZ.toFixed(2);
    panel.xLength.value = box.xLength.toFixed(2);
    panel.yLength.value = box.yLength.toFixed(2);
    panel.zLength.value = box.zLength.toFixed(2);
  };

  const applyPlateauImportPresetToPanel = (box: MpmBoxInputMeters): void => {
    applyImportedBoxToPanel(box);
    panel.waterDepth.value = PLATEAU_IMPORT_INITIAL_WATER_DEPTH_M.toFixed(1);
    panel.particleCount.value = String(PLATEAU_IMPORT_INITIAL_PARTICLE_COUNT);
    panel.particleSpacing.value = PLATEAU_IMPORT_INITIAL_PARTICLE_SPACING_M.toFixed(1);
  };

  const prepareForImport = async (): Promise<void> => {
    cancelRegionEdit();
    stopMpmCompute();
    await waitForSubmittedMpmWork();
  };

  const sleep = (ms: number): Promise<void> =>
    new Promise((resolve) => window.setTimeout(resolve, ms));

  const focusPreviewCamera = (center: Vector3, radius: number): PerspectiveCamera => {
    const camera = new PerspectiveCamera(
      viewer.camera.fov,
      viewer.camera.aspect,
      viewer.camera.near,
      viewer.camera.far,
    );
    const distance = Math.max(300, radius * 2.4);
    camera.position.set(
      center.x + distance * 0.72,
      center.y + distance * 0.62,
      center.z - distance * 0.72,
    );
    camera.lookAt(center);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
    return camera;
  };

  const warmPlateauTerrainView = async (state: PreparedPlateauTemplate): Promise<void> => {
    const preloadCamera = focusPreviewCamera(state.focusCenter, state.focusRadius);
    const start = performance.now();
    while (performance.now() - start < 6_000) {
      plateauTerrain.update(preloadCamera);
      const stats = plateauTerrain.getStats();
      if (stats.visible > 0 && stats.pending === 0) return;
      await sleep(120);
    }
  };

  const warmTerrainViewForCamera = async (
    targetTerrain: Terrain,
    camera: PerspectiveCamera,
  ): Promise<void> => {
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
    const start = performance.now();
    while (performance.now() - start < 8_000) {
      targetTerrain.update(camera);
      const stats = targetTerrain.getStats();
      if (stats.visible > 0 && stats.pending === 0) return;
      await sleep(120);
    }
  };

  const initialCameraViewState = (): ViewerCameraViewState => ({
    position: new Vector3(
      VIEWER_INITIAL_CAMERA_POSITION_M.x,
      VIEWER_INITIAL_CAMERA_POSITION_M.y,
      VIEWER_INITIAL_CAMERA_POSITION_M.z,
    ),
    target: new Vector3(
      VIEWER_INITIAL_CAMERA_TARGET_M.x,
      VIEWER_INITIAL_CAMERA_TARGET_M.y,
      VIEWER_INITIAL_CAMERA_TARGET_M.z,
    ),
  });

  const saveActiveCameraView = (): void => {
    cameraViewByMode.set(activeTerrainMode, viewer.getCameraViewState());
  };

  const restoreCameraViewForMode = (
    mode: ImportSourceMode,
    fallback: () => void,
  ): void => {
    const state = cameraViewByMode.get(mode);
    if (state) {
      viewer.setCameraViewState(state);
      return;
    }
    fallback();
  };

  const setActiveTerrainMode = (mode: ImportSourceMode): void => {
    if (mode !== activeTerrainMode) {
      saveActiveCameraView();
    }
    const plateauActive = mode === 'plateau';
    terrain = plateauActive ? plateauTerrain : gsiTerrain;
    gsiTerrain.visible = !plateauActive;
    plateauTerrain.visible = plateauActive;
    setPlateauDisplayLayersVisible(plateauActive);
    buildingLayers = plateauActive ? plateauMpmBuildingLayers : [];
    setAoiCenter(terrain.getAoiCenter());
    setImportSourceMode(mode);
    setDataAttribution(mode);
    activeTerrainMode = mode;
  };

  const applyPreparedPlateauTemplateToPanel = (state: PreparedPlateauTemplate): void => {
    applyPlateauImportPresetToPanel(state.box);
    setWaterRenderEnabled(true);
    currentBox = {
      originX: state.box.originX,
      originY: currentBox.originY,
      originZ: state.box.originZ,
      xLength: state.box.xLength,
      yLength: state.box.yLength,
      zLength: state.box.zLength,
    };
    currentFluidRegion = plateauInitialFluidRegionForBox(currentBox);
  };

  const activatePreparedPlateauTemplate = (
    state: PreparedPlateauTemplate,
    options: { preserveCamera?: boolean } = {},
  ): void => {
    setActiveTerrainMode('plateau');
    applyPreparedPlateauTemplateToPanel(state);
    if (options.preserveCamera) {
      restoreCameraViewForMode('plateau', () => {
        viewer.focusOn(state.focusCenter, state.focusRadius);
      });
      return;
    }
    viewer.focusOn(state.focusCenter, state.focusRadius);
  };

  const activatePreparedPlateauScene = async (
    state: PreparedPlateauScene,
    options: PlateauBuildingImportOptions = {},
  ): Promise<void> => {
    const report = options.onProgress;
    if (!options.skipPrepare) {
      report?.({ message: `${state.name}: 実行中の解析を停止中...`, progress: 0.88 });
      await prepareForImport();
    }

    activatePreparedPlateauTemplate(state, {
      preserveCamera: options.preserveCamera === true,
    });

    report?.({ message: `${state.name}: MPM 境界を再構成中...`, progress: 0.92 });
    const ok = await applySimulationPanel({ resetParticles: true, stopAfterReset: true });
    if (!ok) {
      const reason = lastSimulationApplyFailureMessage ?? 'terrain coverage failed';
      const message = `${state.name}: ${reason}`;
      setImportStatus(message);
      report?.({ message, progress: 1 });
      throw new Error(message);
    }
    const message = `${state.name}: ${state.buildingLayer.buildingCount} buildings`;
    setImportStatus(message);
    report?.({ message, progress: 1 });
  };

  const preparePlateauTemplate = async (
    source: PlateauBuildingImportSource,
    options: PlateauBuildingImportOptions = {},
  ): Promise<PreparedPlateauTemplate> => {
    const report = options.onProgress;
    const aoiCenter = manifestBoundsCenter(source.bounds);
    report?.({ message: `${source.name}: import 範囲を初期化中...`, progress: 0.58 });

    clearPlateauBuildingLayer();
    preparedPlateauScene = undefined;
    plateauTerrain.setAoiCenter(aoiCenter, { clear: true });
    const { box, layerBounds } = withAoiCenter(aoiCenter, () => {
      const bounds = manifestBoundsToLayerBounds(source.bounds);
      return {
        box: importedBoundsToBox(bounds),
        layerBounds: bounds,
      };
    });

    setImportStatus(`${source.name}: terrain coverage 読み込み中...`);
    setHudStatus(`${source.name} terrain coverage 読み込み中...`);
    report?.({ message: `${source.name}: terrain coverage 読み込み中...`, progress: 0.68 });
    await plateauTerrain.ensureSurfaceCoverage(box);

    const template: PreparedPlateauTemplate = {
      name: source.name,
      aoiCenter,
      box,
      source,
      focusCenter: importBoxCenter(box, layerBounds),
      focusRadius: importBoxRadius(box),
    };
    preparedPlateauTemplate = template;

    report?.({ message: `${source.name}: 表示タイル事前読み込み中...`, progress: 0.76 });
    await warmPlateauTerrainView(template);
    return template;
  };

  const preparePlateauBuildingImport = async (
    source: PlateauBuildingImportSource,
    options: PlateauBuildingImportOptions = {},
  ): Promise<PreparedPlateauScene> => {
    const template =
      preparedPlateauTemplate?.source === source
        ? preparedPlateauTemplate
        : await preparePlateauTemplate(source, options);
    const report = options.onProgress;

    setImportStatus(`${source.name}: 建物配置中...`);
    report?.({ message: `${source.name}: 建物配置中...`, progress: 0.78 });
    const nextBuildingLayer = await template.source.loadTerrainDrapedLayer(
      (worldX, worldZ) => plateauTerrain.sampleSurface(worldX, worldZ)?.height,
      template.aoiCenter,
    );
    nextBuildingLayer.visible = false;
    viewer.scene.add(nextBuildingLayer);
    plateauBuildingLayer = nextBuildingLayer;

    const state: PreparedPlateauScene = {
      ...template,
      buildingLayer: nextBuildingLayer,
    };
    preparedPlateauScene = state;
    return state;
  };

  const loadPlateauBuildingImport = async (
    source: PlateauBuildingImportSource,
    options: PlateauBuildingImportOptions = {},
  ): Promise<void> => {
    if (!options.skipPrepare) {
      options.onProgress?.({ message: `${source.name}: 実行中の解析を停止中...`, progress: 0.02 });
      await prepareForImport();
    }
    const state = await preparePlateauBuildingImport(source, options);
    await activatePreparedPlateauScene(state, {
      ...options,
      skipPrepare: true,
    });
  };

  const loadCachedPlateauSample = async (
    manifestUrl: string,
    label: string,
    options: PlateauBuildingImportOptions = {},
  ): Promise<PreparedPlateauTemplate | undefined> => {
    setImportStatus(`${label} 読み込み中...`);
    setHudStatus(`${label} 読み込み中...`);

    const manifestRes = await fetch(manifestUrl);
    if (manifestRes.status === 404) return undefined;
    if (!manifestRes.ok) {
      throw new Error(`manifest fetch failed: ${manifestRes.status} ${manifestRes.statusText}`);
    }
    const manifest = (await manifestRes.json()) as ImportManifest;
    const obstacle = manifest.obstacles.find((item) => item.type === 'geojson');
    if (!obstacle) {
      throw new Error('manifest に GeoJSON obstacle がありません');
    }

    const obstacleUrl = resolveManifestPath(manifestUrl, obstacle.path);
    return preparePlateauTemplate({
      name: manifest.name,
      bounds: manifest.bounds,
      loadTerrainDrapedLayer: (terrainHeightSampler, aoiCenter) =>
        BuildingObstacleLayer.load(obstacleUrl, {
          aoiCenter,
          placement: 'terrain-draped',
          terrainHeightSampler,
        }),
    }, options);
  };

  const loadHokotaSample = async (): Promise<void> => {
    const template = await loadCachedPlateauSample(
      '/samples/hokota/manifest.json',
      'Hokota PLATEAU sample',
    );
    if (!template) throw new Error('Hokota sample manifest が見つかりません');
    const state = await preparePlateauBuildingImport(template.source);
    await activatePreparedPlateauScene(state);
  };

  const activateGsiMap = async (): Promise<void> => {
    cancelRegionEdit();
    stopMpmCompute();
    await waitForSubmittedMpmWork();

    resetAoiCenter();
    setActiveTerrainMode('gsi');
    setInitialPanelValues(panel);
    setWaterRenderEnabled(panel.waterRender.checked);
    currentBox = {
      originX: MPM_DEFAULT_BOX_M.originX,
      originY: MPM_RENDER_ORIGIN_M.y,
      originZ: MPM_DEFAULT_BOX_M.originZ,
      xLength: MPM_DEFAULT_BOX_M.xLength,
      yLength: MPM_DEFAULT_BOX_M.yLength,
      zLength: MPM_DEFAULT_BOX_M.zLength,
    };
    currentFluidRegion = defaultFluidRegion();
    restoreCameraViewForMode('gsi', () => {
      viewer.setCameraViewState(initialCameraViewState());
    });
    setImportStatus('');
    await applySimulationPanel({ resetParticles: true, stopAfterReset: true });
  };

  const closestDeltaOnResizeAxis = (session: DomainResizeSession): number => {
    const ray = domainResizeRaycaster.ray;
    const b = session.normal.dot(ray.direction);
    const denom = 1 - b * b;
    if (Math.abs(denom) < 1e-6) return 0;

    const axisToRay = new Vector3().subVectors(session.startHitPoint, ray.origin);
    const c = session.normal.dot(axisToRay);
    const f = ray.direction.dot(axisToRay);
    return (b * f - c) / denom;
  };

  const resizeDeltaFromPointer = (
    event: MouseEvent | PointerEvent,
    session: DomainResizeSession,
  ): number => {
    updatePointerNdcFromEvent(event);
    domainResizeRaycaster.setFromCamera(pointerNdc, viewer.camera);

    const ray = domainResizeRaycaster.ray;
    if (Math.abs(ray.direction.y) > 1e-6) {
      const t = (session.horizontalY - ray.origin.y) / ray.direction.y;
      if (t > 0) {
        horizontalResizeHit.copy(ray.origin).addScaledVector(ray.direction, t);
        const delta = horizontalResizeHit.sub(session.startHitPoint).dot(session.normal);
        return Number.isFinite(delta) ? delta : 0;
      }
    }

    const fallbackDelta = closestDeltaOnResizeAxis(session);
    return Number.isFinite(fallbackDelta) ? fallbackDelta : 0;
  };

  const updateDomainResizePreview = (event: MouseEvent | PointerEvent): void => {
    if (!domainResizeSession) return;
    const delta = resizeDeltaFromPointer(event, domainResizeSession);
    const previewBox = resizedBoxFromDelta(domainResizeSession, delta);
    domainResizeSession.previewBox = previewBox;
    writeHorizontalBoxToPanel(previewBox);
    mpm.setDisplayBox(previewBox);
    mpm.setDisplayFluidRegion(
      fluidRegionIntersectsBoxXZ(currentFluidRegion, previewBox)
        ? currentFluidRegion
        : fluidRegionForBox(previewBox),
      previewBox,
    );
    mpm.setResizeFaceState(domainResizeSession.face, domainResizeSession.face);
  };

  const domainMoveFromPointer = (
    event: MouseEvent | PointerEvent,
    session: DomainMoveSession,
  ): {
    box: MpmBoxMeters;
    fluidRegion: MpmFluidRegionMeters;
  } => {
    updatePointerNdcFromEvent(event);
    domainResizeRaycaster.setFromCamera(pointerNdc, viewer.camera);

    const ray = domainResizeRaycaster.ray;
    if (Math.abs(ray.direction.y) <= 1e-6) {
      return {
        box: session.previewBox,
        fluidRegion: session.previewFluidRegion,
      };
    }

    const t = (session.horizontalY - ray.origin.y) / ray.direction.y;
    if (t <= 0) {
      return {
        box: session.previewBox,
        fluidRegion: session.previewFluidRegion,
      };
    }

    horizontalDomainMoveHit.copy(ray.origin).addScaledVector(ray.direction, t);
    const deltaX = horizontalDomainMoveHit.x - session.startHitPoint.x;
    const deltaZ = horizontalDomainMoveHit.z - session.startHitPoint.z;
    return {
      box: translatedBoxXZ(session.startBox, deltaX, deltaZ),
      fluidRegion: translatedFluidRegionXZ(session.startFluidRegion, deltaX, deltaZ),
    };
  };

  const updateDomainMovePreview = (event: MouseEvent | PointerEvent): void => {
    if (!domainMoveSession) return;
    const preview = domainMoveFromPointer(event, domainMoveSession);
    domainMoveSession.previewBox = preview.box;
    domainMoveSession.previewFluidRegion = preview.fluidRegion;
    writeHorizontalBoxToPanel(preview.box);
    mpm.setDisplayBox(preview.box);
    mpm.setDisplayFluidRegion(preview.fluidRegion, preview.box);
    mpm.setDomainMoveState(true, true);
    mpm.setResizeFaceState(undefined, undefined);
    mpm.setFluidRegionState(false, false);
  };

  const closestDeltaOnFluidAxis = (
    axisOrigin: Vector3,
    axis: Vector3,
  ): number => {
    const ray = domainResizeRaycaster.ray;
    const b = axis.dot(ray.direction);
    const denom = 1 - b * b;
    if (Math.abs(denom) < 1e-6) return 0;

    const axisToRay = new Vector3().subVectors(axisOrigin, ray.origin);
    const c = axis.dot(axisToRay);
    const f = ray.direction.dot(axisToRay);
    const delta = (b * f - c) / denom;
    return Number.isFinite(delta) ? delta : 0;
  };

  const fluidRegionFromPointerClient = (
    clientX: number,
    clientY: number,
    session: FluidMoveSession,
  ): {
    region: MpmFluidRegionMeters;
    terrainLiftM: number;
    waterDepthM: number;
  } => {
    updatePointerNdcFromClient(clientX, clientY);
    domainResizeRaycaster.setFromCamera(pointerNdc, viewer.camera);

    if (session.mode === 'y' || session.mode === 'depth') {
      const deltaY = closestDeltaOnFluidAxis(session.startHitPoint, fluidVerticalAxis);
      return {
        region: session.startRegion,
        terrainLiftM:
          session.mode === 'y'
            ? clampTerrainLiftM(
              session.startTerrainLiftM + deltaY,
              currentBox.yLength,
              session.startWaterDepthM,
            )
            : session.startTerrainLiftM,
        waterDepthM:
          session.mode === 'depth'
            ? clampWaterDepthM(
              session.startWaterDepthM + deltaY * FLUID_DEPTH_POINTER_SCALE,
            )
            : session.startWaterDepthM,
      };
    }

    const ray = domainResizeRaycaster.ray;
    if (Math.abs(ray.direction.y) <= 1e-6) {
      return {
        region: session.previewRegion,
        terrainLiftM: session.previewTerrainLiftM,
        waterDepthM: session.previewWaterDepthM,
      };
    }

    const t = (session.horizontalY - ray.origin.y) / ray.direction.y;
    if (t <= 0) {
      return {
        region: session.previewRegion,
        terrainLiftM: session.previewTerrainLiftM,
        waterDepthM: session.previewWaterDepthM,
      };
    }

    horizontalFluidMoveHit.copy(ray.origin).addScaledVector(ray.direction, t);
    if (session.mode === 'scale') {
      return {
        region: scaledFluidRegionXZ(
          session.startRegion,
          fluidRegionScaleFromPointer(session, horizontalFluidMoveHit),
        ),
        terrainLiftM: session.startTerrainLiftM,
        waterDepthM: session.startWaterDepthM,
      };
    }

    let deltaX = horizontalFluidMoveHit.x - session.startHitPoint.x;
    let deltaZ = horizontalFluidMoveHit.z - session.startHitPoint.z;
    if (session.mode === 'x') {
      deltaZ = 0;
    } else if (session.mode === 'z') {
      deltaX = 0;
    }

    return {
      region: {
        ...session.startRegion,
        originX: session.startRegion.originX + deltaX,
        originZ: session.startRegion.originZ + deltaZ,
      },
      terrainLiftM: session.startTerrainLiftM,
      waterDepthM: session.startWaterDepthM,
    };
  };

  const updateFluidMovePreviewFromClient = (clientX: number, clientY: number): void => {
    if (!fluidMoveSession) return;
    const preview = fluidRegionFromPointerClient(clientX, clientY, fluidMoveSession);
    fluidMoveSession.previewRegion = preview.region;
    fluidMoveSession.previewTerrainLiftM = preview.terrainLiftM;
    fluidMoveSession.previewWaterDepthM = preview.waterDepthM;
    writeFluidVerticalToPanel(preview.terrainLiftM, preview.waterDepthM);
    mpm.setDisplayFluidRegion(
      fluidRegionIntersectsBoxXZ(preview.region, currentBox)
        ? preview.region
        : fluidRegionForBox(currentBox),
      currentBox,
      {
        terrainLiftM: preview.terrainLiftM,
        waterDepthM: preview.waterDepthM,
      },
    );
    mpm.setFluidRegionState(true, true, {
      depthMode: fluidMoveSession.mode === 'depth',
    });
    canvas.style.cursor = cursorForFluidMoveMode(fluidMoveSession.mode);
  };

  const updateFluidMovePreview = (event: MouseEvent | PointerEvent): void => {
    updateFluidMovePreviewFromClient(event.clientX, event.clientY);
  };

  const beginDomainMove = (hit: DomainMoveHit, pointerId?: number): void => {
    stopMpmCompute();
    lastResizeInteractionTime = performance.now();
    hoveredDomainMove = true;
    hoveredResizeFace = undefined;
    hoveredFluidRegion = false;
    domainMoveSession = {
      pointerId,
      startBox: { ...currentBox },
      previewBox: { ...currentBox },
      startFluidRegion: { ...currentFluidRegion },
      previewFluidRegion: { ...currentFluidRegion },
      startHitPoint: hit.point.clone(),
      horizontalY: hit.point.y,
    };
    viewer.setCameraZoomOnlyMode(true);
    canvas.style.cursor = 'move';
    mpm.setDomainMoveState(true, true);
    mpm.setResizeFaceState(undefined, undefined);
    mpm.setFluidRegionState(false, false);
    showInteractionTooltipAtLastPointer(domainMoveActiveTooltipText());
  };

  const beginDomainResize = (hit: ResizeFaceHit): void => {
    stopMpmCompute();
    lastResizeInteractionTime = performance.now();
    hoveredDomainMove = false;
    hoveredResizeFace = hit.face;
    hoveredFluidRegion = false;
    domainResizeSession = {
      face: hit.face,
      startBox: { ...currentBox },
      previewBox: { ...currentBox },
      startHitPoint: hit.point.clone(),
      normal: RESIZE_FACE_NORMALS[hit.face].clone(),
      horizontalY: hit.point.y,
    };
    viewer.setCameraZoomOnlyMode(true);
    canvas.style.cursor = cursorForResizeFace(hit.face);
    mpm.setDomainMoveState(false, false);
    mpm.setResizeFaceState(hit.face, hit.face);
    mpm.setFluidRegionState(false, false);
    showInteractionTooltipAtLastPointer(resizeActiveTooltipText(hit.face));
  };

  const beginFluidMove = (hit: FluidRegionHit): void => {
    stopMpmCompute();
    lastResizeInteractionTime = performance.now();
    const values = readPanelValues(panel);
    hoveredDomainMove = false;
    hoveredResizeFace = undefined;
    hoveredFluidRegion = true;
    fluidMoveSession = {
      startRegion: { ...currentFluidRegion },
      previewRegion: { ...currentFluidRegion },
      startHitPoint: hit.point.clone(),
      horizontalY: hit.point.y,
      mode: 'free',
      startTerrainLiftM: values.terrainLift,
      previewTerrainLiftM: values.terrainLift,
      startWaterDepthM: values.waterDepth,
      previewWaterDepthM: values.waterDepth,
    };
    viewer.setCameraZoomOnlyMode(true);
    canvas.style.cursor = 'move';
    mpm.setDomainMoveState(false, false);
    mpm.setResizeFaceState(undefined, undefined);
    mpm.setFluidRegionState(true, true);
    showInteractionTooltipAtLastPointer(fluidActiveTooltipText(fluidMoveSession.mode));
  };

  const finishDomainMove = (): void => {
    if (!domainMoveSession) return;
    const previousFluidRegion = currentFluidRegion;
    const previousBox = { ...currentBox };
    const confirmedBox = { ...domainMoveSession.previewBox };
    const confirmedFluidRegion = { ...domainMoveSession.previewFluidRegion };
    domainMoveSession = undefined;
    hoveredDomainMove = false;
    lastResizeInteractionTime = performance.now();
    viewer.setCameraZoomOnlyMode(false);
    canvas.style.cursor = '';
    hideInteractionTooltip();
    currentFluidRegion = confirmedFluidRegion;
    writeHorizontalBoxToPanel(confirmedBox);
    mpm.setDomainMoveState(false, false);
    mpm.setResizeFaceState(undefined, undefined);
    mpm.setDisplayBox(confirmedBox);
    mpm.setDisplayFluidRegion(confirmedFluidRegion, confirmedBox);
    void applySimulationPanel({ resetParticles: true, stopAfterReset: true })
      .then((ok) => {
        if (ok) return;
        currentFluidRegion = previousFluidRegion;
        writeHorizontalBoxToPanel(previousBox);
        mpm.setDisplayBox(previousBox);
        mpm.setDisplayFluidRegion(previousFluidRegion, previousBox);
      });
  };

  const finishDomainResize = (): void => {
    if (!domainResizeSession) return;
    const confirmedBox = { ...domainResizeSession.previewBox };
    domainResizeSession = undefined;
    hoveredResizeFace = undefined;
    lastResizeInteractionTime = performance.now();
    viewer.setCameraZoomOnlyMode(false);
    canvas.style.cursor = '';
    hideInteractionTooltip();
    mpm.setDomainMoveState(false, false);
    mpm.setResizeFaceState(undefined, undefined);
    writeHorizontalBoxToPanel(confirmedBox);
    void applySimulationPanel({ resetParticles: true, stopAfterReset: true })
      .then((ok) => {
        if (!ok) {
          mpm.setDisplayBox(currentBox);
        }
      });
  };

  const finishFluidMove = (): void => {
    if (!fluidMoveSession) return;
    const previousFluidRegion = currentFluidRegion;
    const previousTerrainLiftM = fluidMoveSession.startTerrainLiftM;
    const previousWaterDepthM = fluidMoveSession.startWaterDepthM;
    const confirmedRegion = { ...fluidMoveSession.previewRegion };
    const confirmedTerrainLiftM = fluidMoveSession.previewTerrainLiftM;
    const confirmedWaterDepthM = fluidMoveSession.previewWaterDepthM;
    fluidMoveSession = undefined;
    hoveredFluidRegion = false;
    lastResizeInteractionTime = performance.now();
    viewer.setCameraZoomOnlyMode(false);
    canvas.style.cursor = '';
    hideInteractionTooltip();
    currentFluidRegion = confirmedRegion;
    writeFluidVerticalToPanel(confirmedTerrainLiftM, confirmedWaterDepthM);
    mpm.setDisplayFluidRegion(currentFluidRegion, currentBox, {
      terrainLiftM: confirmedTerrainLiftM,
      waterDepthM: confirmedWaterDepthM,
    });
    mpm.setDomainMoveState(false, false);
    mpm.setFluidRegionState(false, false);
    void applySimulationPanel({ resetParticles: true, stopAfterReset: true })
      .then((ok) => {
        if (ok) return;
        currentFluidRegion = previousFluidRegion;
        writeFluidVerticalToPanel(previousTerrainLiftM, previousWaterDepthM);
        mpm.setDisplayFluidRegion(currentFluidRegion, currentBox, {
          terrainLiftM: previousTerrainLiftM,
          waterDepthM: previousWaterDepthM,
        });
      });
  };

  canvas.addEventListener('pointermove', (event) => {
    lastPointerClient.set(event.clientX, event.clientY);
    hasLastPointerClient = true;
    updateEditClickCandidateDragState(event);

    if (domainMoveSession) {
      updateDomainMovePreview(event);
      showInteractionTooltip(domainMoveActiveTooltipText(), event.clientX, event.clientY);
      return;
    }

    if (domainResizeSession) {
      updateDomainResizePreview(event);
      showInteractionTooltip(
        resizeActiveTooltipText(domainResizeSession.face),
        event.clientX,
        event.clientY,
      );
      return;
    }

    if (fluidMoveSession) {
      updateFluidMovePreview(event);
      showInteractionTooltip(
        fluidActiveTooltipText(fluidMoveSession.mode),
        event.clientX,
        event.clientY,
      );
      return;
    }

    if (event.buttons !== 0) {
      clearResizeHover();
      return;
    }

    if (isResizeHoverSuppressed(event)) {
      clearResizeHover();
      return;
    }

    const domainHit = pickDomainMove(event);
    if (domainHit) {
      hoveredDomainMove = true;
      hoveredResizeFace = undefined;
      hoveredFluidRegion = false;
      mpm.setDomainMoveState(true, false);
      mpm.setResizeFaceState(undefined, undefined);
      mpm.setFluidRegionState(false, false);
      canvas.style.cursor = 'move';
      showInteractionTooltip(domainMoveHoverTooltipText(), event.clientX, event.clientY);
      return;
    }

    const fluidHit = pickFluidRegion(event);
    if (fluidHit) {
      hoveredDomainMove = false;
      hoveredResizeFace = undefined;
      hoveredFluidRegion = true;
      mpm.setDomainMoveState(false, false);
      mpm.setResizeFaceState(undefined, undefined);
      mpm.setFluidRegionState(true, false);
      canvas.style.cursor = 'move';
      showInteractionTooltip(fluidHoverTooltipText(), event.clientX, event.clientY);
      return;
    }

    const hit = pickResizeFace(event);
    if (hit) {
      hoveredDomainMove = false;
      hoveredResizeFace = hit.face;
      hoveredFluidRegion = false;
      mpm.setDomainMoveState(false, false);
      mpm.setResizeFaceState(hoveredResizeFace, undefined);
      mpm.setFluidRegionState(false, false);
      canvas.style.cursor = cursorForResizeFace(hoveredResizeFace);
      showInteractionTooltip(resizeHoverTooltipText(hit.face), event.clientX, event.clientY);
      return;
    }

    hoveredDomainMove = false;
    hoveredResizeFace = undefined;
    hoveredFluidRegion = false;
    hideInteractionTooltip();
    mpm.setDomainMoveState(false, false);
    mpm.setResizeFaceState(undefined, undefined);
    mpm.setFluidRegionState(false, false);
    canvas.style.cursor = '';
  });

  canvas.addEventListener(
    'pointerdown',
    (event) => {
      if (event.button !== 0 || !event.ctrlKey) return;
      // MapControls は Ctrl+左ドラッグを Shift と同じ回転扱いにする。
      // Ctrl は計算領域移動の専用修飾キーにするため、領域 hit の有無に関係なく
      // pointerdown をここで止める。編集開始/確定は後段の click で処理する。
      lastPointerClient.set(event.clientX, event.clientY);
      hasLastPointerClient = true;
      suppressNextEditClick = false;
      editClickCandidate = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        dragged: false,
      };
      const domainHit = pickDomainMove(event);
      if (domainHit && !domainMoveSession && !mpmRunning) {
        beginDomainMove(domainHit, event.pointerId);
        updateDomainMovePreview(event);
      } else {
        clearResizeHover();
      }
      event.preventDefault();
      event.stopImmediatePropagation();
    },
    { capture: true },
  );

  canvas.addEventListener('contextmenu', (event) => {
    if (!event.ctrlKey && !domainMoveSession && !pickDomainMove(event)) return;
    event.preventDefault();
    event.stopPropagation();
  });

  canvas.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) return;

    lastPointerClient.set(event.clientX, event.clientY);
    hasLastPointerClient = true;
    suppressNextEditClick = false;
    editClickCandidate = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      dragged: false,
    };
  });

  canvas.addEventListener('pointerup', (event) => {
    lastPointerClient.set(event.clientX, event.clientY);
    hasLastPointerClient = true;

    if (
      domainMoveSession &&
      domainMoveSession.pointerId !== undefined &&
      domainMoveSession.pointerId === event.pointerId
    ) {
      updateDomainMovePreview(event);
      finishDomainMove();
      suppressNextEditClick = true;
      editClickCandidate = undefined;
      event.preventDefault();
      event.stopPropagation();
      return;
    }

    updateEditClickCandidateDragState(event);
    if (!editClickCandidate || event.pointerId !== editClickCandidate.pointerId) return;

    if (editClickCandidate.dragged) {
      suppressNextEditClick = true;
    }
    editClickCandidate = undefined;
  });

  canvas.addEventListener('pointercancel', (event) => {
    if (
      domainMoveSession &&
      domainMoveSession.pointerId !== undefined &&
      domainMoveSession.pointerId === event.pointerId
    ) {
      cancelRegionEdit();
      editClickCandidate = undefined;
      return;
    }
    if (!editClickCandidate || event.pointerId !== editClickCandidate.pointerId) return;
    editClickCandidate = undefined;
  });

  canvas.addEventListener('pointerleave', () => {
    if (domainMoveSession || domainResizeSession || fluidMoveSession) return;
    clearResizeHover();
  });

  const fluidMoveModeFromKey = (key: string): FluidMoveMode | undefined => {
    const lowerKey = key.toLowerCase();
    if (lowerKey === 'x') return 'x';
    if (lowerKey === 'y') return 'y';
    if (lowerKey === 'z') return 'z';
    if (lowerKey === 's') return 'scale';
    if (lowerKey === 'd') return 'depth';
    return undefined;
  };

  const toggleFluidMoveMode = (mode: FluidMoveMode): void => {
    if (!fluidMoveSession) return;
    fluidMoveSession.mode = fluidMoveSession.mode === mode ? 'free' : mode;
    canvas.style.cursor = cursorForFluidMoveMode(fluidMoveSession.mode);
    if (hasLastPointerClient) {
      updateFluidMovePreviewFromClient(lastPointerClient.x, lastPointerClient.y);
      showInteractionTooltipAtLastPointer(fluidActiveTooltipText(fluidMoveSession.mode));
    }
  };

  window.addEventListener('keydown', (event) => {
    if (
      event.key === 'Escape' &&
      (domainMoveSession || domainResizeSession || fluidMoveSession)
    ) {
      event.preventDefault();
      event.stopPropagation();
      cancelRegionEdit();
      return;
    }

    if (fluidMoveSession) {
      const mode = fluidMoveModeFromKey(event.key);
      if (mode) {
        event.preventDefault();
        event.stopPropagation();
        if (!event.repeat) toggleFluidMoveMode(mode);
        return;
      }
    }

    if (
      event.key === 'Control' &&
      !event.repeat &&
      !domainMoveSession &&
      !domainResizeSession &&
      !fluidMoveSession &&
      hasLastPointerClient
    ) {
      const hit = pickDomainMoveAtClient(
        lastPointerClient.x,
        lastPointerClient.y,
        true,
        false,
      );
      if (hit) {
        hoveredDomainMove = true;
        hoveredResizeFace = undefined;
        hoveredFluidRegion = false;
        mpm.setDomainMoveState(true, false);
        mpm.setResizeFaceState(undefined, undefined);
        mpm.setFluidRegionState(false, false);
        canvas.style.cursor = 'move';
        showInteractionTooltipAtLastPointer(domainMoveHoverTooltipText());
      }
      return;
    }

    if (
      event.key !== 'Shift' ||
      domainMoveSession ||
      domainResizeSession ||
      fluidMoveSession
    ) {
      return;
    }
    clearResizeHover();
  });

  window.addEventListener('keyup', (event) => {
    if (event.key !== 'Control' || domainMoveSession || domainResizeSession || fluidMoveSession) {
      return;
    }
    if (hoveredDomainMove) {
      clearResizeHover();
    }
  });

  canvas.addEventListener('click', (event) => {
    if (event.button !== 0) return;
    lastPointerClient.set(event.clientX, event.clientY);
    hasLastPointerClient = true;
    if (consumeSuppressedEditClick()) {
      event.preventDefault();
      event.stopPropagation();
      if (!domainMoveSession && !domainResizeSession && !fluidMoveSession) {
        clearResizeHover();
      }
      return;
    }

    if (isResizeHoverSuppressed(event)) {
      clearResizeHover();
      return;
    }

    if (domainMoveSession) {
      event.preventDefault();
      event.stopPropagation();
      updateDomainMovePreview(event);
      finishDomainMove();
      return;
    }

    if (domainResizeSession) {
      event.preventDefault();
      event.stopPropagation();
      updateDomainResizePreview(event);
      finishDomainResize();
      return;
    }

    if (fluidMoveSession) {
      event.preventDefault();
      event.stopPropagation();
      updateFluidMovePreview(event);
      finishFluidMove();
      return;
    }

    const domainHit = pickDomainMove(event);
    if (domainHit) {
      event.preventDefault();
      event.stopPropagation();
      beginDomainMove(domainHit);
      return;
    }

    const fluidHit = pickFluidRegion(event);
    if (fluidHit) {
      event.preventDefault();
      event.stopPropagation();
      beginFluidMove(fluidHit);
      return;
    }

    const hit = pickResizeFace(event);
    if (!hit) return;
    event.preventDefault();
    event.stopPropagation();
    beginDomainResize(hit);
  });

  const prepareInitialGsiMap = async (): Promise<void> => {
    const width = MPM_DEFAULT_BOX_M.xLength.toFixed(0);
    const depth = MPM_DEFAULT_BOX_M.zLength.toFixed(0);
    showLoadingModal(
      '国土地理院地図を読み込み中',
      `MPM 領域 DEM 読み込み中: ${width}m x ${depth}m`,
      undefined,
      { opaque: true },
    );
    updateLoadingModal({
      message: `MPM 領域 DEM 読み込み中: ${width}m x ${depth}m`,
      progress: 0.25,
    });

    try {
      const ok = await applySimulationPanel({ resetParticles: true, stopAfterReset: true });
      if (!ok) {
        updateLoadingModal({
          message: '国土地理院 DEM の準備に失敗しました。HUD のエラーを確認してください。',
          progress: 1,
        });
        return;
      }
      updateLoadingModal({
        message: '国土地理院の表示タイルを読み込み中...',
        progress: 0.82,
      });
      await warmTerrainViewForCamera(gsiTerrain, viewer.camera);
      updateLoadingModal({
        message: '国土地理院地図の表示準備が完了しました。',
        progress: 1,
      });
      hideLoadingModal();
    } catch (err) {
      console.error('[MPM] initial GSI apply failed', err);
      const message = `Initial GSI apply failed: ${String(err)}`;
      setHudStatus(message);
      updateLoadingModal({
        message,
        progress: 1,
      });
    }
  };

  await prepareInitialGsiMap();

  const setImportDragActive = (active: boolean): void => {
    importPanelEl?.classList.toggle('drag-over', active);
  };

  let plateauZipImportRun = 0;
  let plateauZipDownloadAbort: AbortController | undefined = undefined;
  const beginPlateauZipImportRun = (): number => {
    plateauZipDownloadAbort?.abort();
    plateauZipDownloadAbort = undefined;
    return ++plateauZipImportRun;
  };

  const loadPlateauImportResultIntoScene = async (
    imported: PlateauZipImportResult,
    zipSource: PlateauZipSource,
    runId: number,
    report: (progress: PlateauImportProgress) => void,
    expansionSource?: PlateauBuildingExpansionSource,
  ): Promise<void> => {
    if (runId !== plateauZipImportRun) return;
    if (expansionSource) {
      setActivePlateauBuildingExpansion(imported, expansionSource, zipSource);
    }
    await loadPlateauBuildingImport(
      {
        name: imported.manifest.name,
        bounds: imported.manifest.bounds,
        loadTerrainDrapedLayer: async (terrainHeightSampler, aoiCenter) =>
          BuildingObstacleLayer.fromGeoJson(imported.geojson, {
            aoiCenter,
            placement: 'terrain-draped',
            terrainHeightSampler,
            buildMpmObstacles: false,
          }),
      },
      {
        skipPrepare: true,
        onProgress: report,
      },
    );
  };

  const importPlateauZipIntoScene = async (
    file: File,
    runId: number,
    report: (progress: PlateauImportProgress) => void,
    zipProgressStart = 0.08,
    zipProgressSpan = 0.5,
  ): Promise<void> => {
    report({ message: `${file.name}: PLATEAU ZIP 読み込み中...`, progress: zipProgressStart });
    const zipSource = createLocalPlateauZipSource(file);
    const imported = await importPlateauZipSource(zipSource, {
      entrySelection: PLATEAU_IMPORT_INITIAL_ENTRY_SELECTION,
      onProgress: (zipProgress) => {
        report(zipProgressToModalProgressRange(
          zipProgress,
          zipProgressStart,
          zipProgressSpan,
        ));
      },
    });
    if (runId !== plateauZipImportRun) return;

    await loadPlateauImportResultIntoScene(
      imported,
      zipSource,
      runId,
      report,
      localPlateauBuildingExpansionSource(file),
    );
  };

  const loadDroppedPlateauZip = async (file: File): Promise<void> => {
    const runId = beginPlateauZipImportRun();
    setImportSourceMode('plateau');
    const report = (progress: PlateauImportProgress): void => {
      if (runId !== plateauZipImportRun) return;
      setImportStatus(progress.message);
      setHudStatus(progress.message);
      updateLoadingModal(progress);
    };

    showLoadingModal('PLATEAU ZIP import', `${file.name}: 準備中...`);
    try {
      report({ message: `${file.name}: 実行中の解析を停止中...`, progress: 0.02 });
      await prepareForImport();
      if (runId !== plateauZipImportRun) return;

      report({ message: `${file.name}: 旧 import データを破棄中...`, progress: 0.05 });
      clearPlateauImportDataForReload();
      if (runId !== plateauZipImportRun) return;

      await importPlateauZipIntoScene(file, runId, report);
    } finally {
      if (runId === plateauZipImportRun) hideLoadingModal();
    }
  };

  const loadCatalogPlateauZip = async (resource: PlateauCatalogResource): Promise<void> => {
    const runId = beginPlateauZipImportRun();
    const controller = new AbortController();
    plateauZipDownloadAbort = controller;
    const cancelDownload = (): void => {
      if (runId !== plateauZipImportRun) return;
      controller.abort();
      plateauZipDownloadAbort = undefined;
      plateauZipImportRun++;
      const message = `${resource.datasetLabel}: ダウンロードをキャンセルしました`;
      setImportStatus(message);
      setHudStatus(message);
      hideLoadingModal();
    };
    const report = (progress: PlateauImportProgress): void => {
      if (runId !== plateauZipImportRun) return;
      setImportStatus(progress.message);
      setHudStatus(progress.message);
      updateLoadingModal(progress);
    };

    showLoadingModal(
      'PLATEAU CityGML import',
      `${resource.datasetLabel}: 準備中...`,
      cancelDownload,
    );
    try {
      report({ message: `${resource.datasetLabel}: 実行中の解析を停止中...`, progress: 0.02 });
      await prepareForImport();
      if (runId !== plateauZipImportRun) return;

      report({ message: `${resource.datasetLabel}: 旧 import データを破棄中...`, progress: 0.05 });
      clearPlateauImportDataForReload();
      if (runId !== plateauZipImportRun) return;

      const zipSource = await createRemotePlateauZipSource(resource, {
        signal: controller.signal,
        onProgress: report,
      });
      if (runId !== plateauZipImportRun) return;

      report({ message: `${resource.datasetLabel}: 建物 GML を取得・変換中...`, progress: 0.12 });
      const imported = await importPlateauZipSource(zipSource, {
        entrySelection: PLATEAU_IMPORT_INITIAL_ENTRY_SELECTION,
        onProgress: (zipProgress) => {
          report(zipProgressToModalProgressRange(
            zipProgress,
            0.12,
            0.46,
          ));
        },
      });
      if (runId !== plateauZipImportRun) return;

      plateauZipDownloadAbort = undefined;
      setLoadingCancelHandler(undefined);
      await loadPlateauImportResultIntoScene(
        imported,
        zipSource,
        runId,
        report,
        remotePlateauBuildingExpansionSource(resource),
      );
    } finally {
      if (runId === plateauZipImportRun) {
        plateauZipDownloadAbort = undefined;
        hideLoadingModal();
      }
    }
  };

  const plateauImportFailureModalMessage = (label: string, err: unknown): string => {
    const reason = errorMessageFromUnknown(err);
    if (isLimitFailureMessage(reason)) {
      return (
        `${label} は安全上限に達したため終了しました。\n` +
        `${reason}\n` +
        'データが大きすぎるため、別の CityGML ZIP を選ぶか、より小さい範囲で読み込んでください。'
      );
    }
    return `${label} を完了できませんでした。\n${reason}`;
  };

  const handlePlateauImportFailure = (
    label: string,
    err: unknown,
    options: PlateauImportFailureOptions = {},
  ): void => {
    if (isAbortError(err)) return;
    console.error(`[Import] ${label} failed`, err);
    const message = `${label} failed: ${errorMessageFromUnknown(err)}`;
    setImportStatus(message);
    setHudStatus(message);
    if (options.showModal) {
      showErrorModal(
        'PLATEAU 読み込みを終了しました',
        plateauImportFailureModalMessage(options.modalLabel ?? label, err),
      );
    }
  };

  let plateauCatalogReadyPromise: Promise<void> | undefined = undefined;
  const ensurePlateauCatalogReady = (
    preferDefaultShimoda: boolean,
  ): Promise<void> => {
    if (plateauCatalogReadyPromise) return plateauCatalogReadyPromise;
    if (plateauCatalogDatasets.length > 0) {
      const datasetName = plateauYearSelect.value;
      if (datasetName !== '' && plateauCatalogResources.length === 0) {
        return loadPlateauResourceOptions(datasetName).then(() => undefined);
      }
      updatePlateauCatalogControls();
      return Promise.resolve();
    }

    plateauCatalogReadyPromise = (async () => {
      const dataset = await loadPlateauCatalog({ preferDefaultShimoda });
      if (dataset) await loadPlateauResourceOptions(dataset.name);
    })().finally(() => {
      plateauCatalogReadyPromise = undefined;
    });
    return plateauCatalogReadyPromise;
  };

  let defaultShimodaPlateauLoadPromise: Promise<PreparedPlateauTemplate | undefined> | undefined = undefined;
  const loadDefaultShimodaPlateauOnDemand = (
    report: (progress: PlateauImportProgress) => void,
  ): Promise<PreparedPlateauTemplate | undefined> => {
    if (preparedPlateauTemplate) return Promise.resolve(preparedPlateauTemplate);
    if (defaultShimodaPlateauLoadPromise) return defaultShimodaPlateauLoadPromise;

    defaultShimodaPlateauLoadPromise = (async () => {
      report({ message: 'Shimoda PLATEAU cache を確認中...', progress: 0.04 });
      const cached = await loadCachedPlateauSample(
        DEFAULT_SHIMODA_SAMPLE_MANIFEST_URL,
        'Shimoda PLATEAU cache',
        { onProgress: report },
      );
      if (cached) {
        report({ message: 'Shimoda PLATEAU cache 読み込み完了', progress: 0.78 });
        void ensurePlateauCatalogReady(true).catch((err) => {
          handlePlateauImportFailure('PLATEAU catalog load', err);
        });
        return cached;
      }

      report({ message: 'PLATEAU catalog を取得中...', progress: 0.08 });
      const dataset = await loadPlateauCatalog({ preferDefaultShimoda: true });
      if (!dataset) {
        setImportStatus(`${DEFAULT_PLATEAU_PREFECTURE_LABEL} ${DEFAULT_PLATEAU_MUNICIPALITY_LABEL}: dataset なし`);
        return undefined;
      }
      report({ message: `${dataset.label}: CityGML ZIP を確認中...`, progress: 0.12 });
      const resources = await loadPlateauResourceOptions(dataset.name);
      const resource = selectedPlateauCatalogResource() ?? resources[0];
      if (!resource) {
        setImportStatus(`${dataset.label}: CityGML ZIP なし`);
        return undefined;
      }

      const controller = new AbortController();
      report({ message: `${resource.datasetLabel}: ZIP source 準備中...`, progress: 0.16 });
      const zipSource = await createRemotePlateauZipSource(resource, {
        signal: controller.signal,
        onProgress: (zipProgress) => {
          report(zipProgressToModalProgressRange(zipProgress, 0.16, 0.16));
        },
      });
      report({ message: `${resource.datasetLabel}: 建物 GML を取得・変換中...`, progress: 0.32 });
      const imported = await importPlateauZipSource(zipSource, {
        entrySelection: PLATEAU_IMPORT_INITIAL_ENTRY_SELECTION,
        onProgress: (zipProgress) => {
          report(zipProgressToModalProgressRange(zipProgress, 0.32, 0.32));
        },
      });
      const template = await preparePlateauTemplate(
        {
          name: imported.manifest.name,
          bounds: imported.manifest.bounds,
          loadTerrainDrapedLayer: async (terrainHeightSampler, aoiCenter) =>
            BuildingObstacleLayer.fromGeoJson(imported.geojson, {
              aoiCenter,
              placement: 'terrain-draped',
              terrainHeightSampler,
              buildMpmObstacles: false,
            }),
        },
        { onProgress: report },
      );
      report({ message: 'PLATEAU 下田テンプレート読み込み完了', progress: 0.78 });
      return template;
    })().catch((err) => {
      defaultShimodaPlateauLoadPromise = undefined;
      throw err;
    });

    return defaultShimodaPlateauLoadPromise;
  };

  const activatePlateauTabMap = async (): Promise<void> => {
    setImportSourceMode('plateau');
    if (activeTerrainMode === 'plateau' && preparedPlateauScene) return;
    if (preparedPlateauScene) {
      await prepareForImport();
      await activatePreparedPlateauScene(preparedPlateauScene, {
        skipPrepare: true,
        preserveCamera: true,
      });
      void ensurePlateauCatalogReady(true).catch((err) => {
        handlePlateauImportFailure('PLATEAU catalog load', err);
      });
      return;
    }

    const runId = beginPlateauZipImportRun();
    const report = (progress: PlateauImportProgress): void => {
      if (runId !== plateauZipImportRun) return;
      setImportStatus(progress.message);
      setHudStatus(progress.message);
      updateLoadingModal(progress);
    };

    showLoadingModal('PLATEAU 読み込み', '下田テンプレートを準備中...');
    try {
      report({ message: '実行中の解析を停止中...', progress: 0.02 });
      await prepareForImport();
      if (runId !== plateauZipImportRun) return;

      const template = await loadDefaultShimodaPlateauOnDemand(report);
      if (runId !== plateauZipImportRun || !template) return;

      report({ message: `${template.name}: 建物配置中...`, progress: 0.8 });
      const reportBuildingProgress = (progress: PlateauImportProgress): void => {
        if (progress.progress === undefined) {
          report(progress);
          return;
        }
        report({
          ...progress,
          progress: 0.8 + Math.min(1, Math.max(0, progress.progress)) * 0.08,
        });
      };
      const state =
        preparedPlateauScene ?? await preparePlateauBuildingImport(template.source, {
          onProgress: reportBuildingProgress,
        });
      if (runId !== plateauZipImportRun) return;

      // モーダル中は途中状態を見せないため、建物込みの最終 MPM 境界だけを構成する。
      const reportSceneProgress = (progress: PlateauImportProgress): void => {
        if (progress.progress === undefined) {
          report(progress);
          return;
        }
        report({
          ...progress,
          progress: 0.9 + Math.min(1, Math.max(0, progress.progress)) * 0.08,
        });
      };
      await activatePreparedPlateauScene(state, {
        skipPrepare: true,
        preserveCamera: true,
        onProgress: reportSceneProgress,
      });
      report({ message: `${state.name}: PLATEAU 読み込み完了`, progress: 1 });
      void ensurePlateauCatalogReady(true).catch((err) => {
        handlePlateauImportFailure('PLATEAU catalog load', err);
      });
    } finally {
      if (runId === plateauZipImportRun) hideLoadingModal();
    }
  };

  importGsiTabButton.addEventListener('click', () => {
    void activateGsiMap().catch((err) => {
      console.error('[Import] GSI mode failed', err);
      const message = `GSI mode failed: ${String(err)}`;
      setImportStatus(message);
      setHudStatus(message);
    });
  });

  importPlateauTabButton.addEventListener('click', () => {
    void activatePlateauTabMap().catch((err) => {
      handlePlateauImportFailure('PLATEAU tab switch', err, {
        showModal: true,
        modalLabel: 'PLATEAU 読み込み',
      });
    });
  });

  plateauPrefectureSelect.addEventListener('change', () => {
    populatePlateauCitySelect(plateauPrefectureSelect.value, undefined, undefined);
    if (plateauYearSelect.value !== '') {
      void loadPlateauResourceOptions(plateauYearSelect.value);
    }
  });

  plateauCitySelect.addEventListener('change', () => {
    populatePlateauYearSelect(
      plateauPrefectureSelect.value,
      plateauCitySelect.value,
      undefined,
    );
    if (plateauYearSelect.value !== '') {
      void loadPlateauResourceOptions(plateauYearSelect.value);
    }
  });

  plateauYearSelect.addEventListener('change', () => {
    void loadPlateauResourceOptions(plateauYearSelect.value);
  });

  plateauResourceSelect.addEventListener('change', () => {
    updatePlateauCatalogControls();
    updatePlateauResourceStatus();
  });

  applyPlateauCatalogButton.addEventListener('click', () => {
    const resource = selectedPlateauCatalogResource();
    if (!resource) return;
    setImportSourceMode('plateau');
    void loadCatalogPlateauZip(resource).catch((err) => {
      handlePlateauImportFailure('PLATEAU catalog import', err, {
        showModal: true,
        modalLabel: `${resource.datasetLabel} の PLATEAU 読み込み`,
      });
    });
  });

  window.addEventListener('dragover', (event) => {
    if (!hasFileDrop(event.dataTransfer)) return;
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
    setImportDragActive(true);
    setImportStatus('PLATEAU ZIP drop ready');
  });

  window.addEventListener('dragleave', (event) => {
    if (event.relatedTarget !== null) return;
    setImportDragActive(false);
  });

  window.addEventListener('drop', (event) => {
    if (!hasFileDrop(event.dataTransfer)) return;
    event.preventDefault();
    setImportDragActive(false);
    const file = pickDroppedZipFile(event.dataTransfer);
    if (!file) {
      setImportStatus('PLATEAU import failed: ZIP file をドロップしてください');
      return;
    }
    void loadDroppedPlateauZip(file).catch((err) => {
      handlePlateauImportFailure('PLATEAU import', err, {
        showModal: true,
        modalLabel: `${file.name} の PLATEAU ZIP import`,
      });
    });
  });

  panel.start.addEventListener('click', () => {
    cancelRegionEdit();
    void applySimulationPanel({ resetParticles: false, stopAfterReset: false })
      .then((ok) => {
        if (ok) {
          clearResizeHover();
          startMpmCompute();
        }
      });
  });
  panel.stop.addEventListener('click', () => {
    stopMpmCompute();
  });
  panel.reset.addEventListener('click', () => {
    void applySimulationPanel({ resetParticles: true, stopAfterReset: true });
  });
  for (const input of [
    panel.viscosity,
    panel.domainFriction,
    panel.terrainFriction,
    panel.objectFriction,
    panel.domainRestitution,
    panel.terrainRestitution,
    panel.objectRestitution,
    panel.dt,
    panel.stepsPerBatch,
    panel.particleSize,
  ]) {
    input.addEventListener('change', () => {
      void applySimulationPanel({ resetParticles: false, stopAfterReset: false });
    });
  }
  panel.renderInterval.addEventListener('change', () => {
    const value = readIntegerInRange(panel.renderInterval, RENDER_INTERVAL_FRAMES, 1, 120);
    panel.renderInterval.value = String(value);
    viewer.setRenderIntervalFrames(value);
  });
  panel.waterRender.addEventListener('change', () => {
    setWaterRenderEnabled(panel.waterRender.checked);
  });
  panel.fluidDisplayType.addEventListener('change', () => {
    syncFluidDisplayControls({
      nextType: fluidDisplayTypeFromValue(panel.fluidDisplayType.value),
    });
  });
  panel.fluidDisplayMin.addEventListener('input', () => {
    syncFluidDisplayControls({ writeBack: false });
  });
  panel.fluidDisplayMin.addEventListener('change', () => {
    syncFluidDisplayControls();
  });
  panel.fluidDisplayMax.addEventListener('input', () => {
    syncFluidDisplayControls({ writeBack: false });
  });
  panel.fluidDisplayMax.addEventListener('change', () => {
    syncFluidDisplayControls();
  });
  for (const input of [
    panel.originX,
    panel.originZ,
    panel.xLength,
    panel.zLength,
  ]) {
    input.addEventListener('input', scheduleGeometryPanelApply);
    input.addEventListener('change', applyGeometryPanelChange);
  }
  panel.particleCount.addEventListener('change', () => {
    void applySimulationPanel({ resetParticles: true, stopAfterReset: true });
  });
  panel.particleSpacing.addEventListener('change', () => {
    void applySimulationPanel({ resetParticles: true, stopAfterReset: true });
  });
  panel.terrainLift.addEventListener('change', () => {
    void applySimulationPanel({ resetParticles: true, stopAfterReset: true });
  });
  panel.waterDepth.addEventListener('input', scheduleGeometryPanelApply);
  panel.waterDepth.addEventListener('change', applyGeometryPanelChange);

  // 地図上ダブルクリックで MPM 座標 (0,0,0) をクリック地点に移動
  // (originY は applySimulationPanel 内の TerrainBoundary.fromTerrain で
  // 地形から再計算されるので、ここでは originX/originZ のみ書き換える)
  canvas.addEventListener('dblclick', (e) => {
    if (
      domainMoveSession ||
      domainResizeSession ||
      fluidMoveSession ||
      performance.now() - lastResizeInteractionTime < 450 ||
      pickDomainMove(e) ||
      pickResizeFace(e) ||
      pickFluidRegion(e)
    ) {
      e.preventDefault();
      e.stopPropagation();
      return;
    }

    const rect = canvas.getBoundingClientRect();
    pointerNdc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    pointerNdc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    terrainRaycaster.setFromCamera(pointerNdc, viewer.camera);
    const hits = terrainRaycaster.intersectObject(terrain, true);
    if (hits.length === 0) return;
    const p = hits[0]!.point;
    panel.originX.value = p.x.toFixed(2);
    panel.originZ.value = p.z.toFixed(2);
    void applySimulationPanel({ resetParticles: true, stopAfterReset: true });
  });

  // デバッグ用: window.__app からシーン/カメラ/Terrain/MPM にアクセス可能に
  // 本番ビルドではこのブロックを削除する想定
  (window as unknown as { __app: unknown }).__app = {
    viewer,
    get terrain() {
      return terrain;
    },
    gsiTerrain,
    plateauTerrain,
    get mpm() {
      return mpm;
    },
    applySimulationPanel,
  };

  // 毎フレーム、Viewer から呼んでもらうフック。MPM compute は別ループで最速実行する。
  viewer.onUpdate = (camera) => {
    terrain.update(camera);
    maybeLoadPlateauBuildingsForCamera(camera);
  };

  viewer.onAfterRender = (camera) => {
    if (!waterRenderEnabled) return;

    const stats = mpm.getStats();
    const horizontalCellSizeM = horizontalScaleFromMatrixWorld(mpm);
    const fluidScale = fluidRenderScaleFromParticleSpacing(
      waterRenderParticleSpacingM,
      horizontalCellSizeM,
    );
    fluidRenderer.render({
      camera,
      modelMatrix: mpm.matrixWorld,
      activeParticleCount: stats.activeParticles,
      // water render の球半径は UI の particle spacing [m] から決める。
      // MPM grid scale は上限判定だけに使い、spacing 変更が見た目へ直接出るようにする。
      radiusWorld: fluidScale.radiusWorld,
      thicknessScale: fluidScale.thicknessScale,
      displayType: fluidDisplayType,
      displayMin: fluidDisplayRange.min,
      displayMax: fluidDisplayRange.max,
    });
  };

  // ---- HUD 表示用 ----
  // Viewer の render 実行数から、画面表示に必要な最小限の状態だけを出す。
  let lastViewerRenders = 0;
  let lastT = performance.now();

  // 1Hz で HUD を書き換える
  setInterval(() => {
    const now = performance.now();
    const dt = (now - lastT) / 1000;
    const renderStats = viewer.getRenderStats();
    const fps = (renderStats.renderedFrames - lastViewerRenders) / dt;
    lastViewerRenders = renderStats.renderedFrames;
    lastT = now;

    const s = terrain.getStats();
    const m = mpm.getStats();
    panel.activeParticleCount.value = String(m.activeParticles);
    // MPM は描画 frame とは独立に batch 実行するため、ここでは 1 秒ごとの差分で
    // 実際に走った substep 数と、数値計算上の時間がどれだけ進んだかを表示する。
    if (m.simulatedSubsteps < lastMpmSubsteps) {
      // resetDamBreak() 後は積算値が 0 に戻る。直後の HUD で負の Hz を出さないため基準も戻す。
      lastMpmSubsteps = m.simulatedSubsteps;
      lastMpmSimTime = m.simulatedTime;
    }
    const mpmSubsteps = m.simulatedSubsteps - lastMpmSubsteps;
    const mpmHz = mpmSubsteps / dt;
    const simRate = (m.simulatedTime - lastMpmSimTime) / dt;
    lastMpmSubsteps = m.simulatedSubsteps;
    lastMpmSimTime = m.simulatedTime;

    if (hud) {
      const hudLines = [
        `FPS ${fps.toFixed(1)}  tiles ${s.visible}  pending ${s.pending}`,
        `particles ${m.activeParticles}/${m.particles}  ${mpmRunning ? 'running' : 'stopped'}  sim ${m.simulatedTime.toFixed(2)}s`,
      ];
      if (mpmRunning) {
        hudLines.push(`MPM ${mpmHz.toFixed(1)}Hz  rate ${simRate.toFixed(2)}x`);
      }
      const terrainIssues = [
        m.terrainMissingSamples > 0 ? `missing ${m.terrainMissingSamples}` : undefined,
        m.terrainClampedSamples > 0 ? `clamped ${m.terrainClampedSamples}` : undefined,
        m.terrainObjectSamples > 0 ? `objects ${m.terrainObjectSamples}` : undefined,
        m.buildingSideSamples > 0 ? `side-walls ${m.buildingSideSamples}` : undefined,
      ].filter((item): item is string => item !== undefined);
      if (terrainIssues.length > 0) {
        hudLines.push(`terrain ${terrainIssues.join('  ')}`);
      }
      hud.textContent = hudLines.join('\n');
    }
  }, 1000);

  // ---- レンダループ開始 ----
  viewer.start();
}

// トップレベル await でも良いが、エラー処理を確実にしたいので catch を付ける
main().catch((err) => {
  console.error('[main] fatal error', err);
  const message = `Error: ${String(err)}`;
  if (hud) hud.textContent = message;
  document.body.classList.add('loading-blocked');
  if (loadingModal) {
    loadingModal.hidden = false;
    loadingModal.classList.add('loading-opaque');
    loadingModal.setAttribute('aria-busy', 'false');
  }
  if (loadingTitle) loadingTitle.textContent = '起動に失敗しました';
  updateLoadingModal({ message, progress: 1 });
});
