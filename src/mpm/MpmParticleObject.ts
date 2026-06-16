import {
  Box3,
  Box3Helper,
  BufferGeometry,
  DoubleSide,
  Float32BufferAttribute,
  Group,
  InstancedMesh,
  Mesh,
  MeshBasicMaterial,
  MeshBasicNodeMaterial,
  PlaneGeometry,
  TetrahedronGeometry,
  Vector3,
  type Object3D,
  type WebGPURenderer,
} from 'three/webgpu';
import {
  clamp,
  float,
  instanceIndex,
  max,
  mix,
  positionLocal,
  select,
  storage,
  uniform,
  vec3,
} from 'three/tsl';

import {
  MPM_GRID_SIZE,
  MPM_PARTICLE_SIZE_CELLS,
  MPM_RENDER_ORIGIN_M,
  MPM_RENDER_SCALE_M,
} from '../config';
import {
  MlsMpmSimulator,
  type InitialFluidRegionGrid,
  type MlsMpmStats,
} from './MlsMpmSimulator';
import type {
  BuildingSideBoundary,
  DomainBoundary,
  MpmBoxMeters,
  TerrainBoundary,
} from './boundary/MpmBoundary';
import { bindExternalStorageBuffer } from './threeStorageBuffer';

export interface MpmFluidRegionMeters {
  originX: number;
  originZ: number;
  xLength: number;
  zLength: number;
}

interface MpmDisplayFluidRegion {
  gridRegion: InitialFluidRegionGrid;
  worldRegion?: MpmFluidRegionMeters;
}

export type MpmFluidRegionHeightSampler = (
  worldX: number,
  worldZ: number,
) => number | undefined;

export interface MpmSimulationConfig {
  box: MpmBoxMeters;
  initialFluidRegion: MpmFluidRegionMeters;
  initialFluidRegionHeightSampler?: MpmFluidRegionHeightSampler;
  initialFluidBaseBoundary?: TerrainBoundary;
  viscosity: number;
  domainBoundary: DomainBoundary;
  terrainBoundary: TerrainBoundary;
  buildingSideBoundary: BuildingSideBoundary;
  terrainLiftM: number;
  waterDepthM: number;
  particleSpacingM: number;
}

export type MpmResizeFace = 'xMin' | 'xMax' | 'zMin' | 'zMax';
export type MpmParticleContourDisplayType = 'none' | 'velocity' | 'density';

const RESIZE_FACE_COLOR = 0xffbfd3;
const DOMAIN_MOVE_COLOR = 0x8bdcff;
const DOMAIN_MOVE_SELECTED_COLOR = 0x54bfe7;
const FLUID_REGION_COLOR = 0x5aa878;
const FLUID_REGION_SELECTED_COLOR = 0x4a9169;
const FLUID_REGION_DEPTH_SIDE_COLOR = 0xf2a0a0;
const FLUID_REGION_DEPTH_SIDE_SELECTED_COLOR = 0xee7f7f;
const FLUID_REGION_SIDE_VERTEX_COUNT = 8;
const FLUID_REGION_HANDLE_HEIGHT_RATIO = 0.75;
const FLUID_REGION_HANDLE_MIN_VISUAL_DEPTH_M = 40;

function writeColorHex(
  colors: ArrayLike<number> & { [index: number]: number },
  vertexIndex: number,
  hex: number,
): void {
  const offset = vertexIndex * 3;
  colors[offset] = ((hex >> 16) & 0xff) / 255;
  colors[offset + 1] = ((hex >> 8) & 0xff) / 255;
  colors[offset + 2] = (hex & 0xff) / 255;
}

interface MpmFluidRegionStateOptions {
  depthMode?: boolean;
}

interface MpmFluidRegionDisplayOptions {
  terrainLiftM?: number;
  waterDepthM?: number;
}

/**
 * MLS-MPM の計算と、粒子表示 + 直方体枠の表示をまとめた Object3D。
 *
 * 地形連携前の検証用なので、地形とは独立したローカル直方体を scene に置く。
 * 将来 DEM/SDF 衝突を足す場合も、計算座標 -> ENU 座標の変換境界はここに集約できる。
 */
export class MpmParticleObject extends Group {
  readonly simulator: MlsMpmSimulator;

  private readonly particleMesh: InstancedMesh;
  private readonly particleGeometry: TetrahedronGeometry;
  // 粒子の見かけサイズは TSL uniform で実行時に変える。
  // geometry を再生成せず、全 instance の頂点だけ shader 側で拡大縮小する。
  private readonly particleSizeScale = uniform(MPM_PARTICLE_SIZE_CELLS);
  private readonly particleContourMode = uniform(0);
  private readonly particleContourMin = uniform(0);
  private readonly particleContourMax = uniform(1);
  private readonly particleMaterial: MeshBasicNodeMaterial;
  private readonly domainDisplayGroup = new Group();
  private readonly boxHelper: Box3Helper;
  private readonly resizeFaceGeometry: PlaneGeometry;
  private readonly domainMoveMaterial: MeshBasicMaterial;
  private readonly domainMoveMesh: Mesh;
  private readonly domainMoveSideMeshes: Mesh[] = [];
  private readonly resizeFaceMeshes = new Map<MpmResizeFace, Mesh>();
  private readonly resizeFaceMaterials: MeshBasicMaterial[] = [];
  private readonly fluidRegionGeometry: BufferGeometry;
  private readonly fluidRegionMaterial: MeshBasicMaterial;
  private readonly fluidRegionMesh: Mesh;
  private fluidRegionSelected = false;
  private fluidRegionDepthMode = false;
  private fluidRegionTerrainBoundary: TerrainBoundary | undefined = undefined;
  private fluidRegionHeightSampler: MpmFluidRegionHeightSampler | undefined =
    undefined;
  private fluidRegionTerrainLiftM = 0;
  private fluidRegionWaterDepthM = 0;
  private currentSimulationBox: MpmBoxMeters | undefined = undefined;

  constructor(
    device: GPUDevice,
    renderer: WebGPURenderer,
    particleCount?: number,
  ) {
    super();
    this.name = 'MLS-MPM Dam Break';

    this.simulator = new MlsMpmSimulator(device, particleCount);
    bindExternalStorageBuffer(
      renderer,
      this.simulator.positionAttribute,
      this.simulator.positionBuffer,
    );
    bindExternalStorageBuffer(
      renderer,
      this.simulator.contourAttribute,
      this.simulator.contourAttributeBuffer,
    );

    // `Sprite.count` は軽いが、この Three.js WebGPU/node material 経路では
    // storage buffer の `instanceIndex` 読みが安定せず、粒子が表示されない環境がある。
    // 表示実績のある InstancedMesh 経路へ戻しつつ、BoxGeometry ではなく 4 面の
    // TetrahedronGeometry を使って描画三角形数を抑える。
    this.particleGeometry = new TetrahedronGeometry(1, 0);
    this.particleMaterial = new MeshBasicNodeMaterial({
      color: 0x43b7ff,
      opacity: 0.9,
      transparent: true,
      depthWrite: false,
    });
    const positionNode = storage(
      this.simulator.positionAttribute,
      'vec4',
      this.simulator.particleCount,
    ).toReadOnly().element(instanceIndex);
    const contourNode = storage(
      this.simulator.contourAttribute,
      'vec4',
      this.simulator.particleCount,
    ).toReadOnly().element(instanceIndex);
    this.particleMaterial.positionNode = positionNode.xyz.add(
      positionLocal.mul(this.particleSizeScale),
    );
    this.particleMaterial.colorNode = this.createParticleContourColorNode(contourNode);

    // instanceMatrix は全 instance で単位行列にする。実際の粒子位置は
    // storage buffer を `instanceIndex` で直接読み、vertex shader 側で加算する。
    this.particleMesh = new InstancedMesh(
      this.particleGeometry,
      this.particleMaterial,
      this.simulator.particleCount,
    );
    this.particleMesh.name = 'MLS-MPM particles';
    this.particleMesh.frustumCulled = false;
    const matrices = this.particleMesh.instanceMatrix.array;
    for (let i = 0; i < this.simulator.particleCount; i++) {
      const offset = i * 16;
      matrices[offset] = 1;
      matrices[offset + 5] = 1;
      matrices[offset + 10] = 1;
      matrices[offset + 15] = 1;
    }
    this.particleMesh.instanceMatrix.needsUpdate = true;
    this.add(this.particleMesh);

    this.domainDisplayGroup.name = 'MLS-MPM simulation domain preview';
    this.add(this.domainDisplayGroup);

    const box = new Box3(
      new Vector3(0, 0, 0),
      new Vector3(MPM_GRID_SIZE.x, MPM_GRID_SIZE.y, MPM_GRID_SIZE.z),
    );
    this.boxHelper = new Box3Helper(box, 0x9ee8ff);
    this.boxHelper.name = 'MLS-MPM simulation box';
    this.domainDisplayGroup.add(this.boxHelper);

    this.resizeFaceGeometry = new PlaneGeometry(1, 1);
    this.domainMoveMaterial = new MeshBasicMaterial({
      color: DOMAIN_MOVE_COLOR,
      opacity: 0,
      transparent: true,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -3,
      polygonOffsetUnits: -3,
      side: DoubleSide,
    });
    this.domainMoveMesh = this.createDomainMovePlane();
    this.createResizeFaces();

    this.fluidRegionGeometry = new BufferGeometry();
    this.fluidRegionMaterial = new MeshBasicMaterial({
      color: 0xffffff,
      opacity: 0,
      transparent: true,
      vertexColors: true,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
      side: DoubleSide,
    });
    this.fluidRegionMesh = new Mesh(
      this.fluidRegionGeometry,
      this.fluidRegionMaterial,
    );
    this.fluidRegionMesh.name = 'MLS-MPM initial fluid region';
    // Raycaster は visible=false の object も拾えるが、renderer は描画しない。
    // 非アクティブ時の透明 proxy draw を避け、hover/selected 中だけ表示する。
    this.fluidRegionMesh.visible = false;
    this.fluidRegionMesh.frustumCulled = false;
    this.fluidRegionMesh.renderOrder = 24;
    this.fluidRegionMesh.userData.mpmFluidRegion = true;
    this.add(this.fluidRegionMesh);

    this.setSimulationBox({
      originX: MPM_RENDER_ORIGIN_M.x,
      originY: MPM_RENDER_ORIGIN_M.y,
      originZ: MPM_RENDER_ORIGIN_M.z,
      xLength: MPM_GRID_SIZE.x * MPM_RENDER_SCALE_M,
      yLength: MPM_GRID_SIZE.y * MPM_RENDER_SCALE_M,
      zLength: MPM_GRID_SIZE.z * MPM_RENDER_SCALE_M,
    });
    this.setDisplayFluidRegionGrid({
      originX: 0,
      originZ: 0,
      xLength: MPM_GRID_SIZE.x,
      zLength: MPM_GRID_SIZE.z,
    });
  }

  stepBatch(): void {
    this.simulator.stepBatch();
  }

  update(): void {
    // 既存デバッグコード互換用。実アプリでは描画 frame と分離して stepBatch() を呼ぶ。
    this.stepBatch();
  }

  configure(config: MpmSimulationConfig): void {
    this.setSimulationBox(config.box);
    this.setInitialFluidRegion(
      config.initialFluidRegion,
      config.box,
      config.terrainBoundary,
      config.terrainLiftM,
      config.waterDepthM,
      config.initialFluidRegionHeightSampler,
    );
    this.simulator.setViscosity(config.viscosity);
    const cellSizeYM = Math.max(1e-6, config.box.yLength / MPM_GRID_SIZE.y);
    this.simulator.setInitialTerrainLiftCells(
      config.terrainLiftM / cellSizeYM,
    );
    this.simulator.setInitialWaterDepthCells(config.waterDepthM / cellSizeYM);
    this.simulator.setParticleSpacingMeters(config.particleSpacingM);
    this.simulator.setInitialFluidBaseBoundary(config.initialFluidBaseBoundary);
    this.simulator.setBoundaries(
      config.domainBoundary,
      config.terrainBoundary,
      config.buildingSideBoundary,
    );
  }

  /**
   * マウス操作中の仮表示では、枠と picker 面だけを動かす。
   * 粒子とスクリーンスペース流体は確定済み transform のまま描くため、drag 中に水面を引き伸ばさない。
   */
  setDisplayBox(box: MpmBoxMeters): void {
    const baseBox = this.currentSimulationBox;
    if (!baseBox) {
      this.setSimulationBox(box);
      return;
    }

    const baseCellX = Math.max(1e-6, baseBox.xLength / MPM_GRID_SIZE.x);
    const baseCellY = Math.max(1e-6, baseBox.yLength / MPM_GRID_SIZE.y);
    const baseCellZ = Math.max(1e-6, baseBox.zLength / MPM_GRID_SIZE.z);

    // domainDisplayGroup は MPM root の子なので、preview world box を
    // 確定済み root transform の local grid 座標へ戻して表示する。
    this.domainDisplayGroup.position.set(
      (box.originX - baseBox.originX) / baseCellX,
      (box.originY - baseBox.originY) / baseCellY,
      (box.originZ - baseBox.originZ) / baseCellZ,
    );
    this.domainDisplayGroup.scale.set(
      Math.max(1e-6, box.xLength / Math.max(1e-6, baseBox.xLength)),
      Math.max(1e-6, box.yLength / Math.max(1e-6, baseBox.yLength)),
      Math.max(1e-6, box.zLength / Math.max(1e-6, baseBox.zLength)),
    );
    this.domainDisplayGroup.updateMatrixWorld(true);
  }

  /** Raycaster に渡すための、x-/x+/z-/z+ 4 面の透明 mesh。 */
  getResizeFaceObjects(): Object3D[] {
    return Array.from(this.resizeFaceMeshes.values());
  }

  /** Ctrl 押下中に計算領域全体を掴むための、箱上面いっぱいの軽い Raycaster 対象。 */
  getDomainMoveObject(): Object3D {
    return this.domainMoveMesh;
  }

  /** Ctrl 移動は上面だけでなく側面からも掴めるよう、専用の水色 face 群を返す。 */
  getDomainMoveObjects(): Object3D[] {
    return [this.domainMoveMesh, ...this.domainMoveSideMeshes];
  }

  /** 初期流体領域の透明な操作面。粒子数に依存しない軽い Raycaster 対象にする。 */
  getFluidRegionObject(): Object3D {
    return this.fluidRegionMesh;
  }

  /**
   * 側面を raycast した場合でも、編集の基準点は volume 内の 3/4 高さの水平ハンドル面へ揃える。
   * これにより薄い PLATEAU preset でも、操作平面が下面へ張り付いたように見える経路を避ける。
   */
  snapPointToFluidRegionHandlePlane(point: Vector3): Vector3 {
    const positionAttribute = this.fluidRegionGeometry.getAttribute('position');
    if (!positionAttribute || positionAttribute.count < 12) {
      return point.clone();
    }

    const localPoint = this.fluidRegionMesh.worldToLocal(point.clone());
    localPoint.y =
      (
        positionAttribute.getY(8) +
        positionAttribute.getY(9) +
        positionAttribute.getY(10) +
        positionAttribute.getY(11)
      ) * 0.25;
    return this.fluidRegionMesh.localToWorld(localPoint);
  }

  /** 操作中の仮表示用。計算 buffer には触れず、緑の footprint 面だけを移動する。 */
  setDisplayFluidRegion(
    region: MpmFluidRegionMeters,
    box: MpmBoxMeters,
    options: MpmFluidRegionDisplayOptions = {},
  ): void {
    this.setDisplayFluidRegionGrid(
      this.fluidRegionMetersToGrid(region, box),
      region,
      box,
      options,
    );
  }

  /** Ctrl 移動中は計算領域全体を淡い水色の面で示し、通常の resize 面と視覚的に分ける。 */
  setDomainMoveState(hovered: boolean, selected: boolean): void {
    const active = hovered || selected;
    this.domainMoveMesh.visible = active;
    for (const mesh of this.domainMoveSideMeshes) {
      mesh.visible = active;
    }
    this.domainMoveMaterial.opacity = selected ? 0.26 : hovered ? 0.18 : 0;
    this.domainMoveMaterial.color.setHex(
      selected ? DOMAIN_MOVE_SELECTED_COLOR : DOMAIN_MOVE_COLOR,
    );
  }

  /** hover 中またはリサイズ中の面だけ薄いピンクで表示し、それ以外は透明に戻す。 */
  setResizeFaceState(
    hoveredFace: MpmResizeFace | undefined,
    selectedFace: MpmResizeFace | undefined,
  ): void {
    for (const [face, mesh] of this.resizeFaceMeshes) {
      const material = mesh.material;
      if (!(material instanceof MeshBasicMaterial)) continue;
      const active = face === hoveredFace || face === selectedFace;
      mesh.visible = active;
      material.opacity = active ? 0.34 : 0;
    }
  }

  /** hover 中または移動中だけ、初期流体 footprint を編集面として表示する。 */
  setFluidRegionState(
    hovered: boolean,
    selected: boolean,
    options: MpmFluidRegionStateOptions = {},
  ): void {
    const active = hovered || selected;
    this.fluidRegionSelected = selected;
    this.fluidRegionDepthMode = options.depthMode ?? false;
    this.fluidRegionMesh.visible = active;
    const opacity = selected ? 0.58 : hovered ? 0.46 : 0;
    this.fluidRegionMaterial.opacity = opacity;
    this.fluidRegionMaterial.depthWrite = false;
    this.updateFluidRegionColors();
  }

  resetDamBreak(): void {
    this.simulator.resetDamBreak();
  }

  setParticleSize(sizeCells: number): void {
    // 粒子ジオメトリは固定し、shader uniform だけで全粒子の見かけサイズを変える。
    this.particleSizeScale.value = Math.max(0.01, sizeCells);
  }

  setParticleContour(
    type: MpmParticleContourDisplayType,
    minValue: number,
    maxValue: number,
  ): void {
    // water render OFF の簡易粒子 mesh でも同じ display type / range を使う。
    // uniform だけを更新するので、粒子数が多い状態でも CPU 側の色配列更新は発生しない。
    this.particleContourMode.value = particleContourModeCode(type);
    this.particleContourMin.value = Number.isFinite(minValue) ? minValue : 0;
    this.particleContourMax.value =
      Number.isFinite(maxValue) && maxValue > this.particleContourMin.value
        ? maxValue
        : this.particleContourMin.value + 1e-6;
  }

  setParticleMeshVisible(visible: boolean): void {
    // スクリーンスペース流体 overlay を使う時は、従来の粒子 mesh を隠して二重描画を避ける。
    this.particleMesh.visible = visible;
  }

  getStats(): MlsMpmStats {
    return this.simulator.getStats();
  }

  dispose(): void {
    this.simulator.dispose();
    this.particleGeometry.dispose();
    this.particleMaterial.dispose();
    this.boxHelper.geometry.dispose();
    this.resizeFaceGeometry.dispose();
    this.domainMoveMaterial.dispose();
    this.fluidRegionGeometry.dispose();
    this.fluidRegionMaterial.dispose();
    for (const material of this.resizeFaceMaterials) {
      material.dispose();
    }
    if (Array.isArray(this.boxHelper.material)) {
      for (const material of this.boxHelper.material) material.dispose();
    } else {
      this.boxHelper.material.dispose();
    }
  }

  private createDomainMovePlane(): Mesh {
    const grid = MPM_GRID_SIZE;
    const mesh = new Mesh(this.resizeFaceGeometry, this.domainMoveMaterial);
    mesh.name = 'MLS-MPM domain move plane';
    // PlaneGeometry は XY 平面なので、X-Z 平面に倒して計算領域の上面へ置く。
    // Raycast はこの 1 枚だけで済ませ、地形 mesh 全体への高頻度判定を避ける。
    mesh.position.set(grid.x / 2, grid.y, grid.z / 2);
    mesh.rotation.x = -Math.PI / 2;
    mesh.scale.set(grid.x, grid.z, 1);
    mesh.visible = false;
    mesh.frustumCulled = false;
    mesh.renderOrder = 19;
    mesh.userData.mpmDomainMove = true;
    this.domainDisplayGroup.add(mesh);

    this.createDomainMoveSidePlanes();
    return mesh;
  }

  private createDomainMoveSidePlanes(): void {
    const grid = MPM_GRID_SIZE;
    this.addDomainMoveSidePlane({
      position: new Vector3(0, grid.y / 2, grid.z / 2),
      rotationY: -Math.PI / 2,
      scale: new Vector3(grid.z, grid.y, 1),
    });
    this.addDomainMoveSidePlane({
      position: new Vector3(grid.x, grid.y / 2, grid.z / 2),
      rotationY: Math.PI / 2,
      scale: new Vector3(grid.z, grid.y, 1),
    });
    this.addDomainMoveSidePlane({
      position: new Vector3(grid.x / 2, grid.y / 2, 0),
      rotationY: Math.PI,
      scale: new Vector3(grid.x, grid.y, 1),
    });
    this.addDomainMoveSidePlane({
      position: new Vector3(grid.x / 2, grid.y / 2, grid.z),
      rotationY: 0,
      scale: new Vector3(grid.x, grid.y, 1),
    });
  }

  private addDomainMoveSidePlane(transform: {
    position: Vector3;
    rotationY: number;
    scale: Vector3;
  }): void {
    const mesh = new Mesh(this.resizeFaceGeometry, this.domainMoveMaterial);
    mesh.name = 'MLS-MPM domain move side plane';
    mesh.position.copy(transform.position);
    mesh.rotation.y = transform.rotationY;
    mesh.scale.copy(transform.scale);
    mesh.visible = false;
    mesh.frustumCulled = false;
    mesh.renderOrder = 19;
    mesh.userData.mpmDomainMove = true;
    this.domainMoveSideMeshes.push(mesh);
    this.domainDisplayGroup.add(mesh);
  }

  private createResizeFaces(): void {
    const grid = MPM_GRID_SIZE;
    this.addResizeFace('xMin', {
      position: new Vector3(0, grid.y / 2, grid.z / 2),
      rotationY: -Math.PI / 2,
      scale: new Vector3(grid.z, grid.y, 1),
    });
    this.addResizeFace('xMax', {
      position: new Vector3(grid.x, grid.y / 2, grid.z / 2),
      rotationY: Math.PI / 2,
      scale: new Vector3(grid.z, grid.y, 1),
    });
    this.addResizeFace('zMin', {
      position: new Vector3(grid.x / 2, grid.y / 2, 0),
      rotationY: Math.PI,
      scale: new Vector3(grid.x, grid.y, 1),
    });
    this.addResizeFace('zMax', {
      position: new Vector3(grid.x / 2, grid.y / 2, grid.z),
      rotationY: 0,
      scale: new Vector3(grid.x, grid.y, 1),
    });
  }

  private addResizeFace(
    face: MpmResizeFace,
    transform: {
      position: Vector3;
      rotationY: number;
      scale: Vector3;
    },
  ): void {
    const material = new MeshBasicMaterial({
      color: RESIZE_FACE_COLOR,
      opacity: 0,
      transparent: true,
      depthWrite: false,
      side: DoubleSide,
    });
    const mesh = new Mesh(this.resizeFaceGeometry, material);
    mesh.name = `MLS-MPM resize face ${face}`;
    mesh.position.copy(transform.position);
    mesh.rotation.y = transform.rotationY;
    mesh.scale.copy(transform.scale);
    mesh.visible = false;
    mesh.frustumCulled = false;
    mesh.renderOrder = 20;
    mesh.userData.mpmResizeFace = face;

    this.resizeFaceMaterials.push(material);
    this.resizeFaceMeshes.set(face, mesh);
    this.domainDisplayGroup.add(mesh);
  }

  private setSimulationBox(box: MpmBoxMeters): void {
    this.currentSimulationBox = { ...box };
    this.position.set(box.originX, box.originY, box.originZ);
    this.scale.set(
      box.xLength / MPM_GRID_SIZE.x,
      box.yLength / MPM_GRID_SIZE.y,
      box.zLength / MPM_GRID_SIZE.z,
    );
    this.domainDisplayGroup.position.set(0, 0, 0);
    this.domainDisplayGroup.scale.set(1, 1, 1);
    this.updateMatrixWorld(true);
  }

  private setInitialFluidRegion(
    region: MpmFluidRegionMeters,
    box: MpmBoxMeters,
    terrainBoundary: TerrainBoundary,
    terrainLiftM: number,
    waterDepthM: number,
    heightSampler: MpmFluidRegionHeightSampler | undefined,
  ): void {
    this.fluidRegionTerrainBoundary = terrainBoundary;
    this.fluidRegionHeightSampler = heightSampler;
    this.fluidRegionTerrainLiftM = terrainLiftM;
    this.fluidRegionWaterDepthM = waterDepthM;

    const gridRegion = this.fluidRegionMetersToGrid(region, box);
    // active 粒子数は「流体 footprint の物理体積」に比例させる。
    // x/z の移動で計算領域からはみ出た分は simulator 側の grid clipping で inactive になる。
    const fluidVolumeM3 =
      Math.max(1e-6, region.xLength) *
      Math.max(1e-6, region.zLength) *
      Math.max(1e-6, waterDepthM);
    this.simulator.setInitialFluidRegion(gridRegion, fluidVolumeM3);
    this.setDisplayFluidRegionGrid(gridRegion, region, box);
  }

  private fluidRegionMetersToGrid(
    region: MpmFluidRegionMeters,
    box: MpmBoxMeters,
  ): InitialFluidRegionGrid {
    const xScale = MPM_GRID_SIZE.x / Math.max(1e-6, box.xLength);
    const zScale = MPM_GRID_SIZE.z / Math.max(1e-6, box.zLength);
    return {
      originX: (region.originX - box.originX) * xScale,
      originZ: (region.originZ - box.originZ) * zScale,
      xLength: Math.max(1e-6, region.xLength * xScale),
      zLength: Math.max(1e-6, region.zLength * zScale),
    };
  }

  private setDisplayFluidRegionGrid(
    gridRegion: InitialFluidRegionGrid,
    worldRegion?: MpmFluidRegionMeters,
    box?: MpmBoxMeters,
    options: MpmFluidRegionDisplayOptions = {},
  ): void {
    this.rebuildFluidRegionGeometry(gridRegion, worldRegion, box, options);
    this.fluidRegionMesh.updateMatrixWorld(true);
  }

  private rebuildFluidRegionGeometry(
    gridRegion: InitialFluidRegionGrid,
    worldRegion: MpmFluidRegionMeters | undefined,
    box: MpmBoxMeters | undefined,
    options: MpmFluidRegionDisplayOptions,
  ): void {
    const displayRegion = this.clipFluidRegionToDomain(
      gridRegion,
      worldRegion,
      box,
    );
    if (!displayRegion) {
      this.clearFluidRegionGeometry();
      return;
    }

    const displayGridRegion = displayRegion.gridRegion;
    const displayWorldRegion = displayRegion.worldRegion;
    const minX = displayGridRegion.originX;
    const maxX = displayGridRegion.originX + displayGridRegion.xLength;
    const minZ = displayGridRegion.originZ;
    const maxZ = displayGridRegion.originZ + displayGridRegion.zLength;
    const centerLocalX = (minX + maxX) * 0.5;
    const centerLocalZ = (minZ + maxZ) * 0.5;
    const centerWorldX = displayWorldRegion
      ? displayWorldRegion.originX + displayWorldRegion.xLength * 0.5
      : undefined;
    const centerWorldZ = displayWorldRegion
      ? displayWorldRegion.originZ + displayWorldRegion.zLength * 0.5
      : undefined;
    const bottomY = this.sampleFluidProxyLocalY(
      centerLocalX,
      centerLocalZ,
      centerWorldX,
      centerWorldZ,
      box,
      options,
    );
    const topY = bottomY + this.fluidProxyWaterDepthCells(box, options);
    const handleY = this.fluidProxyHandleLocalY(bottomY, topY, box);
    const vertices = [
      minX, bottomY, minZ,
      maxX, bottomY, minZ,
      maxX, bottomY, maxZ,
      minX, bottomY, maxZ,
      minX, topY, minZ,
      maxX, topY, minZ,
      maxX, topY, maxZ,
      minX, topY, maxZ,
      minX, handleY, minZ,
      maxX, handleY, minZ,
      maxX, handleY, maxZ,
      minX, handleY, maxZ,
    ];
    const centerPlaneIndices = [
      8, 9, 11, 9, 10, 11,
    ];
    const sideIndices = [
      0, 1, 4, 1, 5, 4,
      1, 2, 5, 2, 6, 5,
      2, 3, 6, 3, 7, 6,
      3, 0, 7, 0, 4, 7,
    ];
    const indices = [
      ...centerPlaneIndices,
      ...sideIndices,
    ];

    // 編集中は軽さを優先し、DEM 追従は代表点 1 回だけにする。
    // 確定後の粒子生成では simulator 側が各粒子の x/z で DEM を再サンプルする。
    this.fluidRegionGeometry.setAttribute(
      'position',
      new Float32BufferAttribute(vertices, 3),
    );
    this.fluidRegionGeometry.setIndex(indices);
    this.fluidRegionGeometry.setDrawRange(0, indices.length);
    this.fluidRegionGeometry.clearGroups();
    this.updateFluidRegionColors();
    this.fluidRegionGeometry.computeBoundingBox();
    this.fluidRegionGeometry.computeBoundingSphere();
  }

  private clipFluidRegionToDomain(
    gridRegion: InitialFluidRegionGrid,
    worldRegion: MpmFluidRegionMeters | undefined,
    box: MpmBoxMeters | undefined,
  ): MpmDisplayFluidRegion | undefined {
    const minX = Math.max(0, gridRegion.originX);
    const maxX = Math.min(
      MPM_GRID_SIZE.x,
      gridRegion.originX + gridRegion.xLength,
    );
    const minZ = Math.max(0, gridRegion.originZ);
    const maxZ = Math.min(
      MPM_GRID_SIZE.z,
      gridRegion.originZ + gridRegion.zLength,
    );

    if (maxX <= minX || maxZ <= minZ) {
      return undefined;
    }

    const clippedGridRegion = {
      originX: minX,
      originZ: minZ,
      xLength: maxX - minX,
      zLength: maxZ - minZ,
    };

    if (!box || !worldRegion) {
      return { gridRegion: clippedGridRegion };
    }

    // 表示/raycast 用の面だけを計算領域内へ切り詰める。
    // simulator へ渡す initial fluid footprint は full size のままなので、
    // 領域外へ出した分は inactive になり、戻すと再び粒子化される。
    return {
      gridRegion: clippedGridRegion,
      worldRegion: {
        originX: box.originX + (minX / MPM_GRID_SIZE.x) * box.xLength,
        originZ: box.originZ + (minZ / MPM_GRID_SIZE.z) * box.zLength,
        xLength: ((maxX - minX) / MPM_GRID_SIZE.x) * box.xLength,
        zLength: ((maxZ - minZ) / MPM_GRID_SIZE.z) * box.zLength,
      },
    };
  }

  private clearFluidRegionGeometry(): void {
    this.fluidRegionGeometry.setAttribute(
      'position',
      new Float32BufferAttribute([], 3),
    );
    this.fluidRegionGeometry.setAttribute(
      'color',
      new Float32BufferAttribute([], 3),
    );
    this.fluidRegionGeometry.setIndex([]);
    this.fluidRegionGeometry.clearGroups();
    this.fluidRegionGeometry.setDrawRange(0, 0);
    this.fluidRegionGeometry.boundingBox = null;
    this.fluidRegionGeometry.boundingSphere = null;
  }

  private sampleFluidProxyLocalY(
    localX: number,
    localZ: number,
    worldX: number | undefined,
    worldZ: number | undefined,
    box: MpmBoxMeters | undefined,
    options: MpmFluidRegionDisplayOptions = {},
  ): number {
    const terrainLiftM = options.terrainLiftM ?? this.fluidRegionTerrainLiftM;

    if (
      box &&
      worldX !== undefined &&
      worldZ !== undefined &&
      this.fluidRegionHeightSampler
    ) {
      const sampledWorldY = this.fluidRegionHeightSampler(worldX, worldZ);
      if (sampledWorldY !== undefined && Number.isFinite(sampledWorldY)) {
        const cellSizeYM = Math.max(1e-6, box.yLength / MPM_GRID_SIZE.y);
        return (sampledWorldY + terrainLiftM - box.originY) / cellSizeYM;
      }
    }

    if (this.fluidRegionTerrainBoundary) {
      const cellSizeYM =
        box === undefined
          ? 1
          : Math.max(1e-6, box.yLength / MPM_GRID_SIZE.y);
      return (
        this.fluidRegionTerrainBoundary.sampleHeightGrid(localX, localZ) +
        terrainLiftM / cellSizeYM
      );
    }

    return 0;
  }

  private fluidProxyWaterDepthCells(
    box: MpmBoxMeters | undefined,
    options: MpmFluidRegionDisplayOptions = {},
  ): number {
    const waterDepthM = Math.max(
      0,
      options.waterDepthM ?? this.fluidRegionWaterDepthM,
    );
    const cellSizeYM =
      box === undefined
        ? 1
        : Math.max(1e-6, box.yLength / MPM_GRID_SIZE.y);
    return waterDepthM / cellSizeYM;
  }

  private fluidProxyHandleLocalY(
    bottomY: number,
    topY: number,
    box: MpmBoxMeters | undefined,
  ): number {
    const physicalDepthCells = Math.max(0, topY - bottomY);
    const minVisualDepthM =
      box === undefined
        ? 0
        : Math.min(FLUID_REGION_HANDLE_MIN_VISUAL_DEPTH_M, box.yLength * 0.5);
    const cellSizeYM =
      box === undefined
        ? 1
        : Math.max(1e-6, box.yLength / MPM_GRID_SIZE.y);
    const visualDepthCells = Math.max(
      physicalDepthCells,
      minVisualDepthM / cellSizeYM,
    );
    const handleY = bottomY + visualDepthCells * FLUID_REGION_HANDLE_HEIGHT_RATIO;

    // 表示用ハンドルだけを持ち上げる。物理的な水深や上面は topY のままなので、
    // 薄い PLATEAU preset でも掴みやすくしつつ粒子初期化は変えない。
    return Math.max(bottomY, Math.min(MPM_GRID_SIZE.y, handleY));
  }

  private createParticleContourColorNode(contourNode: any) {
    const mode = this.particleContourMode;
    const rangeMin = this.particleContourMin;
    const rangeMax = this.particleContourMax;
    const range = max(rangeMax.sub(rangeMin), float(0.000001));
    const attributeValue = select(mode.lessThan(1.5), contourNode.x, contourNode.y);
    const t = clamp(attributeValue.sub(rangeMin).div(range), 0, 1);

    // screen-space water render と同じ色相の青→シアン→黄→赤 ramp。
    const blue = vec3(0.08, 0.28, 1.0);
    const cyan = vec3(0.0, 0.72, 1.0);
    const yellow = vec3(1.0, 0.88, 0.35);
    const red = vec3(1.0, 0.15, 0.10);
    const blueToCyan = mix(blue, cyan, t.div(0.34));
    const cyanToYellow = mix(cyan, yellow, t.sub(0.34).div(0.34));
    const yellowToRed = mix(yellow, red, t.sub(0.68).div(0.32));
    const contourColor = select(
      t.lessThan(0.34),
      blueToCyan,
      select(t.lessThan(0.68), cyanToYellow, yellowToRed),
    );

    const defaultColor = vec3(0.263, 0.718, 1.0);
    return select(mode.lessThan(0.5), defaultColor, contourColor);
  }

  private updateFluidRegionColors(): void {
    const positionAttribute = this.fluidRegionGeometry.getAttribute('position');
    const vertexCount = positionAttribute?.count ?? 0;
    if (vertexCount === 0) return;

    let colorAttribute = this.fluidRegionGeometry.getAttribute('color') as
      | Float32BufferAttribute
      | undefined;
    if (!colorAttribute || colorAttribute.count !== vertexCount) {
      colorAttribute = new Float32BufferAttribute(
        new Float32Array(vertexCount * 3),
        3,
      );
      this.fluidRegionGeometry.setAttribute('color', colorAttribute);
    }

    const baseColor = this.fluidRegionSelected
      ? FLUID_REGION_SELECTED_COLOR
      : FLUID_REGION_COLOR;
    const sideColor = this.fluidRegionDepthMode
      ? this.fluidRegionSelected
        ? FLUID_REGION_DEPTH_SIDE_SELECTED_COLOR
        : FLUID_REGION_DEPTH_SIDE_COLOR
      : baseColor;
    const colors = colorAttribute.array as ArrayLike<number> & {
      [index: number]: number;
    };
    for (let index = 0; index < vertexCount; index++) {
      writeColorHex(
        colors,
        index,
        index < FLUID_REGION_SIDE_VERTEX_COUNT ? sideColor : baseColor,
      );
    }
    colorAttribute.needsUpdate = true;
  }
}

function particleContourModeCode(type: MpmParticleContourDisplayType): number {
  if (type === 'velocity') return 1;
  if (type === 'density') return 2;
  return 0;
}
