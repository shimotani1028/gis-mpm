import {
  StorageInstancedBufferAttribute,
} from 'three/webgpu';

import {
  MPM_BOUNDARY_RESTITUTION,
  MPM_DEFAULT_BOX_M,
  MPM_DOMAIN_FRICTION,
  MPM_DOMAIN_RESTITUTION,
  MPM_DT,
  MPM_DYNAMIC_VISCOSITY,
  MPM_FIXED_POINT_MULTIPLIER,
  MPM_GRAVITY_Y,
  MPM_GRID_SIZE,
  MPM_INITIAL_TERRAIN_LIFT_M,
  MPM_INITIAL_WATER_DEPTH_M,
  MPM_OBJECT_FRICTION,
  MPM_OBJECT_RESTITUTION,
  MPM_PARTICLE_COUNT,
  MPM_PARTICLE_SPACING_M,
  MPM_REST_DENSITY,
  MPM_STIFFNESS,
  MPM_STEPS_PER_BATCH,
  MPM_TERRAIN_BOUNDARY_RESOLUTION,
  MPM_TERRAIN_GRID_COLLISION_THICKNESS_CELLS,
  MPM_TERRAIN_PARTICLE_CLEARANCE_CELLS,
  MPM_TERRAIN_FRICTION,
} from '../config';
import {
  BuildingSideBoundary,
  DomainBoundary,
  TerrainBoundary,
} from './boundary/MpmBoundary';

import p2gMassWgsl from './shaders/p2gMass.wgsl?raw';
import p2gMomentumStressWgsl from './shaders/p2gMomentumStress.wgsl?raw';
import updateGridWgsl from './shaders/updateGrid.wgsl?raw';
import g2pWgsl from './shaders/g2p.wgsl?raw';
import copyPositionWgsl from './shaders/copyPosition.wgsl?raw';

const WORKGROUP_SIZE = 128;
const CELL_BYTES = 16;
const PARTICLE_FLOATS = 20;
const PARTICLE_BYTES = PARTICLE_FLOATS * Float32Array.BYTES_PER_ELEMENT;
const POSITION_FLOATS = 4;
const POSITION_BYTES = POSITION_FLOATS * Float32Array.BYTES_PER_ELEMENT;
const CONTOUR_ATTRIBUTE_FLOATS = 4;
const CONTOUR_ATTRIBUTE_BYTES =
  CONTOUR_ATTRIBUTE_FLOATS * Float32Array.BYTES_PER_ELEMENT;
const UNIFORM_FLOATS = 20;
const DOMAIN_BOUNDARY_PLANE_COUNT = 6;
const BOUNDARY_PLANE_FLOATS = 8;
const BOUNDARY_PLANE_BYTES = BOUNDARY_PLANE_FLOATS * Float32Array.BYTES_PER_ELEMENT;
const TERRAIN_BOUNDARY_CELL_FLOATS = 8;
const TERRAIN_BOUNDARY_CELL_BYTES =
  TERRAIN_BOUNDARY_CELL_FLOATS * Float32Array.BYTES_PER_ELEMENT;
const BUILDING_SIDE_BOUNDARY_CELL_FLOATS = 8;
const BUILDING_SIDE_BOUNDARY_CELL_BYTES =
  BUILDING_SIDE_BOUNDARY_CELL_FLOATS * Float32Array.BYTES_PER_ELEMENT;
const GPU_BUFFER_USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
} as const;
const GPU_MAP_MODE = {
  READ: 0x0001,
} as const;
const PARTICLE_FLOAT_COMPONENT_NAMES = [
  'position.x',
  'position.y',
  'position.z',
  'position.w',
  'velocity.x',
  'velocity.y',
  'velocity.z',
  'velocity.w',
  'c0.x',
  'c0.y',
  'c0.z',
  'c0.w',
  'c1.x',
  'c1.y',
  'c1.z',
  'c1.w',
  'c2.x',
  'c2.y',
  'c2.z',
  'c2.w',
] as const;

const DEFAULT_INITIAL_FLUID_VOLUME_M3 = Math.max(
  1e-6,
  MPM_DEFAULT_BOX_M.xLength *
    MPM_DEFAULT_BOX_M.zLength *
    MPM_INITIAL_WATER_DEPTH_M,
);

export interface MlsMpmStats {
  particles: number;
  activeParticles: number;
  particleMass: number;
  grid: string;
  gridCells: number;
  stepsPerBatch: number;
  simulatedSubsteps: number;
  dt: number;
  simulatedTime: number;
  dynamicViscosity: number;
  domainFriction: number;
  terrainFriction: number;
  objectFriction: number;
  domainRestitution: number;
  terrainRestitution: number;
  objectRestitution: number;
  terrainOriginY: number;
  terrainMissingSamples: number;
  terrainClampedSamples: number;
  terrainObjectSamples: number;
  buildingSideSamples: number;
}

export interface MpmDivergenceIssue {
  particleIndex: number;
  kind: 'non-finite' | 'speed';
  component: string;
  value: number;
  speed?: number;
  maxSpeed: number;
}

export interface MpmDivergenceCheckResult {
  ok: boolean;
  checkedParticles: number;
  issue?: MpmDivergenceIssue;
}

export interface InitialFluidRegionGrid {
  originX: number;
  originZ: number;
  xLength: number;
  zLength: number;
}

interface ParticleInitData {
  particles: ArrayBuffer;
  positions: Float32Array;
  contourAttributes: Float32Array;
  count: number;
  particleMass: number;
}

const INITIAL_PARTICLE_DOMAIN_MARGIN_CELLS = 1.001;

/**
 * Zenn 記事 / WebGPU-Ocean と同じ流れで MLS-MPM を実行する raw WebGPU シミュレータ。
 *
 * 計算ステップ:
 *   clearGrid -> p2gMass -> p2gMomentumStress -> updateGrid -> g2p -> copyPosition
 *
 * copyPosition の出力 buffer は Three.js の StorageInstancedBufferAttribute と共有し、
 * CPU readback なしで粒子描画に使う。
 */
export class MlsMpmSimulator {
  readonly particleCount: number;
  readonly gridSize = MPM_GRID_SIZE;
  readonly gridCellCount =
    MPM_GRID_SIZE.x * MPM_GRID_SIZE.y * MPM_GRID_SIZE.z;
  readonly positionAttribute: StorageInstancedBufferAttribute;
  readonly positionBuffer: GPUBuffer;
  readonly contourAttribute: StorageInstancedBufferAttribute;
  readonly contourAttributeBuffer: GPUBuffer;
  readonly particleBuffer: GPUBuffer;

  private readonly device: GPUDevice;
  private readonly cellBuffer: GPUBuffer;
  private readonly uniformsBuffer: GPUBuffer;
  private readonly domainBoundaryBuffer: GPUBuffer;
  private readonly terrainBoundaryBuffer: GPUBuffer;
  private readonly buildingSideBoundaryBuffer: GPUBuffer;

  private readonly p2gMassPipeline: GPUComputePipeline;
  private readonly p2gMomentumStressPipeline: GPUComputePipeline;
  private readonly updateGridPipeline: GPUComputePipeline;
  private readonly g2pPipeline: GPUComputePipeline;
  private readonly copyPositionPipeline: GPUComputePipeline;

  private readonly p2gMassBindGroup: GPUBindGroup;
  private readonly p2gMomentumStressBindGroup: GPUBindGroup;
  private readonly updateGridBindGroup: GPUBindGroup;
  private readonly g2pBindGroup: GPUBindGroup;
  private readonly copyPositionBindGroup: GPUBindGroup;

  private simulatedSubsteps = 0;
  private activeParticleCount = 0;
  private particleMass = 1;
  private initialTerrainLiftCells =
    MPM_INITIAL_TERRAIN_LIFT_M / (MPM_DEFAULT_BOX_M.yLength / MPM_GRID_SIZE.y);
  private initialWaterDepthCells =
    MPM_INITIAL_WATER_DEPTH_M / (MPM_DEFAULT_BOX_M.yLength / MPM_GRID_SIZE.y);
  // 初期流体 footprint は、計算領域に対する grid 座標で保持する。
  // origin が負値や grid 外でもよく、reset 時に計算領域と交差する粒子だけを active にする。
  private initialFluidRegion: InitialFluidRegionGrid = {
    originX: 0,
    originZ: 0,
    xLength: MPM_GRID_SIZE.x,
    zLength: MPM_GRID_SIZE.z,
  };
  private initialFluidVolumeM3 = DEFAULT_INITIAL_FLUID_VOLUME_M3;
  private particleVolumeM3 = MPM_PARTICLE_SPACING_M ** 3;
  private dynamicViscosity = MPM_DYNAMIC_VISCOSITY;
  private dt = MPM_DT;
  private stepsPerBatch = MPM_STEPS_PER_BATCH;
  private domainBoundary: DomainBoundary;
  private terrainBoundary: TerrainBoundary;
  private initialFluidBaseBoundary: TerrainBoundary | undefined = undefined;
  private buildingSideBoundary: BuildingSideBoundary;

  constructor(device: GPUDevice, particleCount = MPM_PARTICLE_COUNT) {
    this.device = device;
    this.particleCount = Math.max(1, Math.floor(particleCount));

    this.positionAttribute = new StorageInstancedBufferAttribute(
      new Float32Array(this.particleCount * POSITION_FLOATS),
      POSITION_FLOATS,
    );
    this.contourAttribute = new StorageInstancedBufferAttribute(
      new Float32Array(this.particleCount * CONTOUR_ATTRIBUTE_FLOATS),
      CONTOUR_ATTRIBUTE_FLOATS,
    );

    this.particleBuffer = device.createBuffer({
      label: 'MLS-MPM particles',
      size: PARTICLE_BYTES * this.particleCount,
      usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST | GPU_BUFFER_USAGE.COPY_SRC,
    });

    this.cellBuffer = device.createBuffer({
      label: 'MLS-MPM grid cells',
      size: CELL_BYTES * this.gridCellCount,
      usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST,
    });

    this.positionBuffer = device.createBuffer({
      label: 'MLS-MPM render positions',
      size: POSITION_BYTES * this.particleCount,
      usage:
        GPU_BUFFER_USAGE.STORAGE |
        GPU_BUFFER_USAGE.VERTEX |
        GPU_BUFFER_USAGE.COPY_DST |
        GPU_BUFFER_USAGE.COPY_SRC,
    });
    this.contourAttributeBuffer = device.createBuffer({
      label: 'MLS-MPM render contour attributes',
      size: CONTOUR_ATTRIBUTE_BYTES * this.particleCount,
      usage:
        GPU_BUFFER_USAGE.STORAGE |
        GPU_BUFFER_USAGE.VERTEX |
        GPU_BUFFER_USAGE.COPY_DST |
        GPU_BUFFER_USAGE.COPY_SRC,
    });

    this.domainBoundary = new DomainBoundary(
      this.gridSize,
      {
        friction: MPM_DOMAIN_FRICTION,
        restitution: MPM_DOMAIN_RESTITUTION,
      },
    );
    this.terrainBoundary = TerrainBoundary.flat(
      this.gridSize,
      {
        friction: MPM_TERRAIN_FRICTION,
        restitution: MPM_BOUNDARY_RESTITUTION,
      },
      0,
      'wall',
      MPM_TERRAIN_BOUNDARY_RESOLUTION,
    );
    this.buildingSideBoundary = BuildingSideBoundary.empty(
      this.gridSize,
      {
        friction: MPM_OBJECT_FRICTION,
        restitution: MPM_OBJECT_RESTITUTION,
      },
      'wall',
      MPM_TERRAIN_BOUNDARY_RESOLUTION,
    );

    this.domainBoundaryBuffer = device.createBuffer({
      label: 'MLS-MPM domain boundary planes',
      size: DOMAIN_BOUNDARY_PLANE_COUNT * BOUNDARY_PLANE_BYTES,
      usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST,
    });

    this.terrainBoundaryBuffer = device.createBuffer({
      label: 'MLS-MPM terrain boundary heightfield',
      size:
        MPM_TERRAIN_BOUNDARY_RESOLUTION *
        MPM_TERRAIN_BOUNDARY_RESOLUTION *
        TERRAIN_BOUNDARY_CELL_BYTES,
      usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST,
    });

    this.buildingSideBoundaryBuffer = device.createBuffer({
      label: 'MLS-MPM building side boundary',
      size:
        MPM_TERRAIN_BOUNDARY_RESOLUTION *
        MPM_TERRAIN_BOUNDARY_RESOLUTION *
        BUILDING_SIDE_BOUNDARY_CELL_BYTES,
      usage: GPU_BUFFER_USAGE.STORAGE | GPU_BUFFER_USAGE.COPY_DST,
    });

    this.uniformsBuffer = device.createBuffer({
      label: 'MLS-MPM uniforms',
      size: UNIFORM_FLOATS * Float32Array.BYTES_PER_ELEMENT,
      usage: GPU_BUFFER_USAGE.UNIFORM | GPU_BUFFER_USAGE.COPY_DST,
    });
    this.writeBoundaryBuffers();
    this.writeUniforms();

    this.p2gMassPipeline = this.createPipeline('p2g mass', p2gMassWgsl, {
      fixed_point_multiplier: MPM_FIXED_POINT_MULTIPLIER,
    });
    this.p2gMomentumStressPipeline = this.createPipeline(
      'p2g momentum stress',
      p2gMomentumStressWgsl,
      {
        fixed_point_multiplier: MPM_FIXED_POINT_MULTIPLIER,
        stiffness: MPM_STIFFNESS,
        rest_density: MPM_REST_DENSITY,
      },
    );
    this.updateGridPipeline = this.createPipeline('update grid', updateGridWgsl, {
      fixed_point_multiplier: MPM_FIXED_POINT_MULTIPLIER,
      gravity_y: MPM_GRAVITY_Y,
    });
    this.g2pPipeline = this.createPipeline('g2p', g2pWgsl, {
      fixed_point_multiplier: MPM_FIXED_POINT_MULTIPLIER,
    });
    this.copyPositionPipeline = this.createPipeline('copy position', copyPositionWgsl);

    // clearGrid 専用 dispatch は encoder.clearBuffer に置き換えたため、bind group も不要。
    this.p2gMassBindGroup = this.createParticleGridUniformBindGroup(
      this.p2gMassPipeline,
      'MLS-MPM p2g mass bind group',
    );
    this.p2gMomentumStressBindGroup = this.createParticleGridUniformBindGroup(
      this.p2gMomentumStressPipeline,
      'MLS-MPM p2g momentum stress bind group',
    );
    this.updateGridBindGroup = device.createBindGroup({
      label: 'MLS-MPM update grid bind group',
      layout: this.updateGridPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.cellBuffer } },
        { binding: 1, resource: { buffer: this.uniformsBuffer } },
        { binding: 2, resource: { buffer: this.domainBoundaryBuffer } },
        { binding: 3, resource: { buffer: this.terrainBoundaryBuffer } },
        { binding: 4, resource: { buffer: this.buildingSideBoundaryBuffer } },
      ],
    });
    this.g2pBindGroup = device.createBindGroup({
      label: 'MLS-MPM g2p bind group',
      layout: this.g2pPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleBuffer } },
        { binding: 1, resource: { buffer: this.cellBuffer } },
        { binding: 2, resource: { buffer: this.uniformsBuffer } },
        { binding: 3, resource: { buffer: this.terrainBoundaryBuffer } },
        { binding: 4, resource: { buffer: this.buildingSideBoundaryBuffer } },
      ],
    });
    this.copyPositionBindGroup = device.createBindGroup({
      label: 'MLS-MPM copy position bind group',
      layout: this.copyPositionPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleBuffer } },
        { binding: 1, resource: { buffer: this.positionBuffer } },
        { binding: 2, resource: { buffer: this.contourAttributeBuffer } },
      ],
    });

    this.resetDamBreak();
  }

  setViscosity(value: number): void {
    this.dynamicViscosity = Math.max(0, value);
    this.writeUniforms();
  }

  setDt(value: number): void {
    // dt は数値発散を避けるため最低 1e-5 を保証。シェーダは uniforms.boundary_params.z 経由で参照する。
    this.dt = Math.max(1e-5, value);
    this.writeUniforms();
  }

  setStepsPerBatch(value: number): void {
    // 1 回の command buffer にまとめる substep 数。整数 1 以上。
    // UI の max=3200 と同じ上限を実行側にも置き、手入力や devtools 経由の過大値で
    // 1 batch が長くなりすぎてタブ/GPU が固まるのを防ぐ。
    this.stepsPerBatch = Math.min(3200, Math.max(1, Math.floor(value)));
  }

  setInitialTerrainLiftCells(value: number): void {
    this.initialTerrainLiftCells = Math.max(0, value);
  }

  setInitialWaterDepthCells(value: number): void {
    this.initialWaterDepthCells = Math.max(MPM_TERRAIN_PARTICLE_CLEARANCE_CELLS, value);
  }

  setParticleSpacingMeters(value: number): void {
    // UI の particle spacing は物理空間での代表間隔。
    // active 数は volume / spacing^3 で求め、particleCount を buffer 上限として扱う。
    const spacing = Math.max(1e-6, Number.isFinite(value) ? value : MPM_PARTICLE_SPACING_M);
    this.particleVolumeM3 = spacing ** 3;
  }

  setInitialFluidRegion(region: InitialFluidRegionGrid, volumeM3: number): void {
    this.initialFluidRegion = {
      originX: Number.isFinite(region.originX) ? region.originX : 0,
      originZ: Number.isFinite(region.originZ) ? region.originZ : 0,
      xLength: Math.max(1e-6, region.xLength),
      zLength: Math.max(1e-6, region.zLength),
    };
    this.initialFluidVolumeM3 = Math.max(
      1e-6,
      Number.isFinite(volumeM3) ? volumeM3 : DEFAULT_INITIAL_FLUID_VOLUME_M3,
    );
  }

  setInitialFluidBaseBoundary(boundary: TerrainBoundary | undefined): void {
    // 衝突用 terrainBoundary は DEM 床のままにし、初期配置だけ別 surface を使えるようにする。
    // PLATEAU 建物上面など、流体を最初に載せたい高さが DEM より高い場合に使う。
    this.initialFluidBaseBoundary = boundary;
  }

  setBoundaries(
    domainBoundary: DomainBoundary,
    terrainBoundary: TerrainBoundary,
    buildingSideBoundary?: BuildingSideBoundary,
  ): void {
    this.domainBoundary = domainBoundary;
    this.terrainBoundary = terrainBoundary;
    this.buildingSideBoundary =
      buildingSideBoundary ??
      BuildingSideBoundary.empty(
        this.gridSize,
        {
          friction: MPM_OBJECT_FRICTION,
          restitution: MPM_OBJECT_RESTITUTION,
        },
        'wall',
        { width: terrainBoundary.width, depth: terrainBoundary.depth },
      );
    this.writeBoundaryBuffers();
    this.writeUniforms();
  }

  resetDamBreak(): void {
    const data = this.createDamBreakParticles();
    if (data.count > this.particleCount) {
      throw new Error(`Lattice tried to place ${data.count} > buffer ${this.particleCount}`);
    }

    this.activeParticleCount = data.count;
    this.particleMass = data.particleMass;
    this.writeUniforms();
    this.device.queue.writeBuffer(this.particleBuffer, 0, data.particles);
    this.device.queue.writeBuffer(this.positionBuffer, 0, data.positions);
    this.device.queue.writeBuffer(this.contourAttributeBuffer, 0, data.contourAttributes);
    this.simulatedSubsteps = 0;
  }

  stepBatch(): void {
    // 旧実装はステージ毎に encoder/pass/submit を分けて 1 substep あたり 6 回 queue.submit していたため、
    // driver 同期の固定コストが substep 数に比例してフレーム時間を圧迫していた。
    // ここでは 1 batch 分の全 substep を単一 commandEncoder にまとめ、submit を 1 回に減らす。
    //
    // ・ clearGrid は専用 dispatch を廃止し、encoder.clearBuffer(cellBuffer) で 0 埋めする。
    //   clearBuffer は同一 encoder 内で次の compute pass との順序が保証される。
    // ・ p2gMass → p2gMomentumStress → updateGrid → g2p は同一 compute pass に並べる。
    //   WebGPU では同一 pass 内の連続 dispatch について、storage buffer の R/W 順序が保証される。
    // ・ copyPosition は描画用 positionBuffer の更新であり、ユーザーが見るのは batch 最後の結果だけ。
    //   stepsPerBatch > 1 のとき中間 substep のコピーは捨てられる無駄なので、ループ外で 1 回だけ実行する。
    const encoder = this.device.createCommandEncoder({ label: 'MLS-MPM batch encoder' });

    for (let i = 0; i < this.stepsPerBatch; i++) {
      // 毎 substep の最初に grid を 0 クリア。
      // clearBuffer は driver 内蔵経路で、shader dispatch より低オーバーヘッド。
      encoder.clearBuffer(this.cellBuffer);

      const pass = encoder.beginComputePass({ label: `MLS-MPM substep ${i}` });
      this.dispatch(pass, this.p2gMassPipeline, this.p2gMassBindGroup, this.particleCount);
      this.dispatch(
        pass,
        this.p2gMomentumStressPipeline,
        this.p2gMomentumStressBindGroup,
        this.particleCount,
      );
      this.dispatch(pass, this.updateGridPipeline, this.updateGridBindGroup, this.gridCellCount);
      this.dispatch(pass, this.g2pPipeline, this.g2pBindGroup, this.particleCount);
      pass.end();

      this.simulatedSubsteps++;
    }

    // 描画 buffer への反映は 1 batch あたり 1 回で十分。
    const copyPass = encoder.beginComputePass({ label: 'MLS-MPM copy position' });
    this.dispatch(
      copyPass,
      this.copyPositionPipeline,
      this.copyPositionBindGroup,
      this.particleCount,
    );
    copyPass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  step(): void {
    // Console や既存デバッグコードからの呼び出し互換用。
    // アプリ本体は「描画 frame」ではなく batch として stepBatch() を呼ぶ。
    this.stepBatch();
  }

  async checkDivergence(maxSpeed: number): Promise<MpmDivergenceCheckResult> {
    const checkedParticles = this.activeParticleCount;
    if (checkedParticles <= 0) {
      return { ok: true, checkedParticles };
    }

    const readbackBytes = checkedParticles * PARTICLE_BYTES;
    const readbackBuffer = this.device.createBuffer({
      label: 'MLS-MPM divergence readback',
      size: readbackBytes,
      usage: GPU_BUFFER_USAGE.MAP_READ | GPU_BUFFER_USAGE.COPY_DST,
    });
    let mapped = false;
    try {
      // particleBuffer は GPU storage buffer なので CPU から直接読めない。
      // 診断時だけ MAP_READ buffer へコピーし、active 粒子の生値を確認する。
      const encoder = this.device.createCommandEncoder({
        label: 'MLS-MPM divergence readback encoder',
      });
      encoder.copyBufferToBuffer(
        this.particleBuffer,
        0,
        readbackBuffer,
        0,
        readbackBytes,
      );
      this.device.queue.submit([encoder.finish()]);
      await readbackBuffer.mapAsync(GPU_MAP_MODE.READ);
      mapped = true;
      const particles = new Float32Array(readbackBuffer.getMappedRange());
      return scanParticleStateForDivergence(particles, checkedParticles, maxSpeed);
    } finally {
      if (mapped) {
        readbackBuffer.unmap();
      }
      readbackBuffer.destroy();
    }
  }

  getStats(): MlsMpmStats {
    return {
      particles: this.particleCount,
      activeParticles: this.activeParticleCount,
      particleMass: this.particleMass,
      grid: `${this.gridSize.x}x${this.gridSize.y}x${this.gridSize.z}`,
      gridCells: this.gridCellCount,
      stepsPerBatch: this.stepsPerBatch,
      simulatedSubsteps: this.simulatedSubsteps,
      // ここでの時刻は物理秒に校正した値ではなく、MPM の数値計算上の時間。
      dt: this.dt,
      simulatedTime: this.simulatedSubsteps * this.dt,
      dynamicViscosity: this.dynamicViscosity,
      domainFriction: this.domainBoundary.material.friction,
      terrainFriction: this.terrainBoundary.material.friction,
      objectFriction: this.buildingSideBoundary.material.friction,
      domainRestitution: this.domainBoundary.material.restitution,
      terrainRestitution: this.terrainBoundary.material.restitution,
      objectRestitution: this.buildingSideBoundary.material.restitution,
      terrainOriginY: this.terrainBoundary.originY,
      terrainMissingSamples: this.terrainBoundary.missingSamples,
      terrainClampedSamples: this.terrainBoundary.clampedSamples,
      terrainObjectSamples: this.terrainBoundary.objectSamples,
      buildingSideSamples: this.buildingSideBoundary.sideSamples,
    };
  }

  dispose(): void {
    this.particleBuffer.destroy();
    this.cellBuffer.destroy();
    this.positionBuffer.destroy();
    this.contourAttributeBuffer.destroy();
    this.uniformsBuffer.destroy();
    this.domainBoundaryBuffer.destroy();
    this.terrainBoundaryBuffer.destroy();
    this.buildingSideBoundaryBuffer.destroy();
  }

  private createPipeline(
    label: string,
    code: string,
    constants?: Record<string, number>,
  ): GPUComputePipeline {
    const module = this.device.createShaderModule({
      label: `MLS-MPM ${label} shader`,
      code,
    });

    const compute: GPUProgrammableStage =
      constants === undefined
        ? { module, entryPoint: 'main' }
        : { module, entryPoint: 'main', constants };

    return this.device.createComputePipeline({
      label: `MLS-MPM ${label} pipeline`,
      layout: 'auto',
      compute,
    });
  }

  private createParticleGridUniformBindGroup(
    pipeline: GPUComputePipeline,
    label: string,
  ): GPUBindGroup {
    return this.device.createBindGroup({
      label,
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleBuffer } },
        { binding: 1, resource: { buffer: this.cellBuffer } },
        { binding: 2, resource: { buffer: this.uniformsBuffer } },
      ],
    });
  }

  private dispatch(
    pass: GPUComputePassEncoder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    itemCount: number,
  ): void {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(itemCount / WORKGROUP_SIZE));
  }

  private writeUniforms(): void {
    const uniforms = new Float32Array([
      this.gridSize.x,
      this.gridSize.y,
      this.gridSize.z,
      this.gridCellCount,
      this.gridSize.x,
      this.gridSize.y,
      this.gridSize.z,
      MPM_TERRAIN_PARTICLE_CLEARANCE_CELLS,
      this.dynamicViscosity,
      this.domainBoundary.planes.length,
      this.terrainBoundary.width,
      this.terrainBoundary.depth,
      // boundary_params: x=terrain enabled, y=terrain grid thickness, z=dt, w=particle mass
      1,
      MPM_TERRAIN_GRID_COLLISION_THICKNESS_CELLS,
      this.dt,
      this.particleMass,
      // collision_params: x=domain restitution, yzw=reserved
      this.domainBoundary.material.restitution,
      0,
      0,
      0,
    ]);
    this.device.queue.writeBuffer(this.uniformsBuffer, 0, uniforms);
  }

  private writeBoundaryBuffers(): void {
    this.device.queue.writeBuffer(
      this.domainBoundaryBuffer,
      0,
      this.domainBoundary.toGpuData(),
    );
    this.device.queue.writeBuffer(
      this.terrainBoundaryBuffer,
      0,
      this.terrainBoundary.toGpuData(),
    );
    this.device.queue.writeBuffer(
      this.buildingSideBoundaryBuffer,
      0,
      this.buildingSideBoundary.toGpuData(),
    );
  }

  private createDamBreakParticles(): ParticleInitData {
    const particleBytes = new ArrayBuffer(PARTICLE_BYTES * this.particleCount);
    const particles = new Float32Array(particleBytes);
    const positions = new Float32Array(POSITION_FLOATS * this.particleCount);
    const contourAttributes = new Float32Array(
      CONTOUR_ATTRIBUTE_FLOATS * this.particleCount,
    );

    // 粒子格子の密度調整により、buffer capacity より active 粒子が少ないことがある。
    // inactive 粒子は copyPosition.wgsl と同じ sentinel に置き、Start 前や Reset 直後の
    // copyPosition 実行前でも原点に余剰キューブが見えないようにする。
    for (let i = 0; i < this.particleCount; i++) {
      const posBase = i * POSITION_FLOATS;
      positions[posBase] = 0;
      positions[posBase + 1] = -1000000;
      positions[posBase + 2] = 0;
      positions[posBase + 3] = 0;
      const contourBase = i * CONTOUR_ATTRIBUTE_FLOATS;
      contourAttributes[contourBase] = 0;
      contourAttributes[contourBase + 1] = 0;
      contourAttributes[contourBase + 2] = 0;
      contourAttributes[contourBase + 3] = 0;
    }

    const region = this.initialFluidRegion;
    const fluidBaseBoundary = this.initialFluidBaseBoundary ?? this.terrainBoundary;
    const waterLayout = this.computeWaterLayout(fluidBaseBoundary);
    const min = { x: region.originX, y: 0, z: region.originZ };
    const max = {
      x: region.originX + region.xLength,
      y: waterLayout.height,
      z: region.originZ + region.zLength,
    };
    const terrainLiftY = waterLayout.terrainLiftY;
    const fullArea = Math.max(
      1e-6,
      (max.x - min.x) * (max.z - min.z),
    );
    const clippedMin = {
      x: clamp(min.x, 0, this.gridSize.x),
      y: min.y,
      z: clamp(min.z, 0, this.gridSize.z),
    };
    const clippedMax = {
      x: clamp(max.x, 0, this.gridSize.x),
      y: max.y,
      z: clamp(max.z, 0, this.gridSize.z),
    };
    const clippedSize = {
      x: Math.max(0, clippedMax.x - clippedMin.x),
      y: Math.max(0, clippedMax.y - clippedMin.y),
      z: Math.max(0, clippedMax.z - clippedMin.z),
    };
    const clippedAreaFraction =
      clippedSize.x > 0 && clippedSize.z > 0
        ? (clippedSize.x * clippedSize.z) / fullArea
        : 0;
    const waterVolume =
      clippedSize.x *
      clippedSize.y *
      clippedSize.z;
    // particles は buffer capacity / upper limit として扱う。
    // 実際の active 数は、物理空間での初期流体体積に基づく密度から算出し、
    // capacity を超えた場合だけ上限で打ち切る。
    // 計算領域からはみ出した footprint はここで体積比として除外する。
    const targetParticleCount =
      this.computeTargetParticleCount(clippedAreaFraction);
    if (
      targetParticleCount === 0 ||
      clippedSize.x <= 0 ||
      clippedSize.y <= 0 ||
      clippedSize.z <= 0
    ) {
      return {
        particles: particleBytes,
        positions,
        contourAttributes,
        count: 0,
        particleMass: 0,
      };
    }
    // target count を上限で打ち切った場合も同じ水量を表現できるよう、
    // 1 粒子あたりの質量は「初期水量 / 配置目標数」で決める。
    const particleMass = MPM_REST_DENSITY * waterVolume / targetParticleCount;
    const lattice = chooseLatticeCounts(targetParticleCount, {
      x: clippedSize.x,
      y: clippedSize.y,
      z: clippedSize.z,
    });
    const step = {
      x: clippedSize.x / lattice.x,
      y: clippedSize.y / lattice.y,
      z: clippedSize.z / lattice.z,
    };
    const jitterScale = 0.20 * Math.min(step.x, step.y, step.z);
    const random = createDeterministicRandom(0x601_2025);

    let count = 0;

    // 流体 footprint は full size のまま保持するが、粒子を置くのは計算領域との交差部分だけ。
    // 外へ出した分はこの reset では active 粒子にならず、再び内側へ戻すと交差体積に応じて復活する。
    const latticeCapacity = lattice.x * lattice.y * lattice.z;
    for (let sample = 0; sample < targetParticleCount; sample++) {
      const latticeIndex = Math.floor((sample + 0.5) * latticeCapacity / targetParticleCount);
      const iz = latticeIndex % lattice.z;
      const ix = Math.floor(latticeIndex / lattice.z) % lattice.x;
      const iy = Math.floor(latticeIndex / (lattice.x * lattice.z));
      const jx = (random() - 0.5) * jitterScale;
      const jy = (random() - 0.5) * jitterScale;
      const jz = (random() - 0.5) * jitterScale;

      const rawPx = clippedMin.x + (ix + 0.5) * step.x + jx;
      const rawPz = clippedMin.z + (iz + 0.5) * step.z + jz;

      const px = clamp(
        rawPx,
        INITIAL_PARTICLE_DOMAIN_MARGIN_CELLS,
        this.gridSize.x - INITIAL_PARTICLE_DOMAIN_MARGIN_CELLS,
      );
      const pz = clamp(
        rawPz,
        INITIAL_PARTICLE_DOMAIN_MARGIN_CELLS,
        this.gridSize.z - INITIAL_PARTICLE_DOMAIN_MARGIN_CELLS,
      );
      const baseY = fluidBaseBoundary.sampleHeightGrid(px, pz);
      const waterColumnY = clippedMin.y + (iy + 0.5) * step.y;
      const baseFloorY = baseY + MPM_TERRAIN_PARTICLE_CLEARANCE_CELLS;
      const py = clamp(
        baseY + terrainLiftY + waterColumnY + jy,
        baseFloorY,
        this.gridSize.y - MPM_TERRAIN_PARTICLE_CLEARANCE_CELLS,
      );
      const downhill = this.sampleDownhillDirection(px, pz);
      const initialSpeed = 0;

      const base = count * PARTICLE_FLOATS;
      particles[base] = px;
      particles[base + 1] = py;
      particles[base + 2] = pz;
      particles[base + 3] = 1;
      // 地形上での動作確認用に、初期水柱へ小さな下り方向速度を与える。
      // C は 0 初期化のままにする。ArrayBuffer は作成時点で 0 埋め。
      particles[base + 4] = downhill.x * initialSpeed;
      particles[base + 5] = 0;
      particles[base + 6] = downhill.z * initialSpeed;
      // velocity.w は描画コンター用の最新密度比として使う。最初の step 前でも
      // 密度比表示が全て 0 にならないよう、初期値は基準状態の 1.0 にしておく。
      particles[base + 7] = 1;

      const posBase = count * POSITION_FLOATS;
      positions[posBase] = px;
      positions[posBase + 1] = py;
      positions[posBase + 2] = pz;
      positions[posBase + 3] = 1;
      const contourBase = count * CONTOUR_ATTRIBUTE_FLOATS;
      contourAttributes[contourBase] = initialSpeed;
      contourAttributes[contourBase + 1] = 1;
      contourAttributes[contourBase + 2] = 1;
      contourAttributes[contourBase + 3] = 0;

      count++;
    }

    return {
      particles: particleBytes,
      positions,
      contourAttributes,
      count,
      particleMass,
    };
  }

  private computeTargetParticleCount(activeVolumeFraction: number): number {
    const activeVolumeM3 =
      this.initialFluidVolumeM3 *
      Math.min(1, Math.max(0, activeVolumeFraction));
    if (activeVolumeM3 <= 0) return 0;

    const targetCount = Math.ceil(activeVolumeM3 / this.particleVolumeM3);
    return Math.min(this.particleCount, Math.max(1, targetCount));
  }

  private computeWaterLayout(
    fluidBaseBoundary: TerrainBoundary,
  ): { terrainLiftY: number; height: number } {
    const terrainLiftY = this.initialTerrainLiftCells;
    // 初期層は各粒子の x/z にある base surface へ追従させる。
    // base surface は通常 DEM、PLATEAU 建物がある場合は DEM と建物上面の高い方。
    // 最高 base + gap でも上端に収まる範囲へ制限し、高所側で大量の粒子が
    // 上端 clamp される状態を避ける。
    const highestBaseY = fluidBaseBoundary.maxHeightGrid + terrainLiftY;
    const availableHeightAtHighestBase =
      this.gridSize.y - MPM_TERRAIN_PARTICLE_CLEARANCE_CELLS - highestBaseY;
    const waterHeight = Math.max(
      MPM_TERRAIN_PARTICLE_CLEARANCE_CELLS,
      availableHeightAtHighestBase,
    );
    return {
      terrainLiftY,
      height: Math.min(this.initialWaterDepthCells, waterHeight),
    };
  }

  private sampleDownhillDirection(x: number, z: number): { x: number; z: number } {
    const normal = this.terrainBoundary.sampleNormalGrid(x, z);
    const len = Math.hypot(normal.x, normal.z);
    if (len < 1e-5) {
      return { x: -1, z: 0 };
    }
    return { x: normal.x / len, z: normal.z / len };
  }
}

function createDeterministicRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
    return state / 0x1_0000_0000;
  };
}

function scanParticleStateForDivergence(
  particles: Float32Array,
  checkedParticles: number,
  maxSpeed: number,
): MpmDivergenceCheckResult {
  for (let i = 0; i < checkedParticles; i++) {
    const base = i * PARTICLE_FLOATS;
    for (let componentOffset = 0; componentOffset < PARTICLE_FLOATS; componentOffset++) {
      const value = particles[base + componentOffset]!;
      if (Number.isFinite(value)) continue;

      return {
        ok: false,
        checkedParticles,
        issue: {
          particleIndex: i,
          kind: 'non-finite',
          component:
            PARTICLE_FLOAT_COMPONENT_NAMES[componentOffset] ??
            `particle[${componentOffset}]`,
          value,
          maxSpeed,
        },
      };
    }

    const vx = particles[base + 4]!;
    const vy = particles[base + 5]!;
    const vz = particles[base + 6]!;
    const speed = Math.hypot(vx, vy, vz);
    if (Number.isFinite(speed) && speed <= maxSpeed) continue;

    return {
      ok: false,
      checkedParticles,
      issue: {
        particleIndex: i,
        kind: 'speed',
        component: 'velocity.xyz',
        value: speed,
        speed,
        maxSpeed,
      },
    };
  }

  return { ok: true, checkedParticles };
}

function chooseLatticeCounts(
  targetCount: number,
  size: { x: number; y: number; z: number },
): { x: number; y: number; z: number } {
  const volume = Math.max(1e-6, size.x * size.y * size.z);
  const samplesPerUnit = Math.cbrt(targetCount / volume);
  const lattice = {
    x: Math.max(1, Math.ceil(size.x * samplesPerUnit)),
    y: Math.max(1, Math.ceil(size.y * samplesPerUnit)),
    z: Math.max(1, Math.ceil(size.z * samplesPerUnit)),
  };

  while (lattice.x * lattice.y * lattice.z < targetCount) {
    const stepX = size.x / lattice.x;
    const stepY = size.y / lattice.y;
    const stepZ = size.z / lattice.z;
    if (stepX >= stepY && stepX >= stepZ) {
      lattice.x++;
    } else if (stepY >= stepZ) {
      lattice.y++;
    } else {
      lattice.z++;
    }
  }

  return lattice;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
