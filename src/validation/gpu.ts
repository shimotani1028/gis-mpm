/**
 * 診断専用 entry point。本体から import しない・ソルバーや GPU shader を変更しない。
 * 合成境界、初期状態、GPU readback をここで構成して実 WGSL を試験する。
 */
import massCode from '../mpm/shaders/p2gMass.wgsl?raw';
import stressCode from '../mpm/shaders/p2gMomentumStress.wgsl?raw';
import gridCode from '../mpm/shaders/updateGrid.wgsl?raw';
import g2pCode from '../mpm/shaders/g2p.wgsl?raw';
import { DomainBoundary, TerrainBoundary, BuildingSideBoundary } from '../mpm/boundary/MpmBoundary';
import { MPM_FIXED_POINT_MULTIPLIER, MPM_STIFFNESS, MPM_REST_DENSITY,
  MPM_GRAVITY_Y, MPM_DYNAMIC_VISCOSITY } from '../config';

const S = MPM_FIXED_POINT_MULTIPLIER;
const U = { map: 1, src: 4, dst: 8, uniform: 64, storage: 128, query: 512 };
type Kind = 'freefall' | 'closed' | 'floor-support' | 'overflow' | 'wall' | 'column' | 'dam';
interface Case { kind: Kind; n: number; dt: number; steps: number; side: number }
const stages = ['p2gMass', 'p2gMomentumStress', 'updateGridIncludingBoundary', 'g2pIncludingBoundary'];

// 既存コードと同じ 20 floats / particle。固定 seed の規則格子で再現性を優先する。
function initialState(c: Case): { state: Float32Array; mass: number } {
  const count = c.kind === 'overflow' ? 2 : c.kind === 'freefall' || c.kind === 'floor-support' || c.kind === 'wall' ? 1 : c.side ** 3;
  const state = new Float32Array(count * 20);
  const width = c.n / 4;
  // 水柱は全水平断面、ダムは一部だけ。均一密度で静止開始する緩和試験であり、
  // EOS に整合する静水圧密度勾配を与えた平衡解ではない。
  const horizontalWidth = c.kind === 'column' ? c.n - 2.002 : width;
  const horizontalOrigin = c.kind === 'column' ? 1.001 : c.n / 4;
  for (let i = 0; i < count; i++) {
    const ix = i % c.side, iy = Math.floor(i / c.side) % c.side, iz = Math.floor(i / (c.side ** 2));
    let x = horizontalOrigin + (ix + 0.5) * horizontalWidth / c.side;
    let y = c.n / 4 + (iy + 0.5) * width / c.side;
    let z = horizontalOrigin + (iz + 0.5) * horizontalWidth / c.side;
    if (c.kind === 'column' || c.kind === 'dam') y = 0.02 + (iy + 0.5) * width / c.side;
    if (c.kind === 'floor-support') y = 0.02;
    if (c.kind === 'wall') x = 1.01;
    if (c.kind === 'overflow') x = y = z = c.n / 2 + 0.5;
    state.set([x, y, z, 1, c.kind === 'wall' ? -1 : c.kind === 'closed' ? 0.2 : 0,
      0, c.kind === 'wall' ? 0.25 : 0, 1], i * 20);
  }
  return { state, mass: c.kind === 'overflow' ? 300 : count === 1 ? 1 : MPM_REST_DENSITY * width * horizontalWidth ** 2 / count };
}

function summarize(state: Float32Array, mass: number) {
  const position = [0, 0, 0], momentum = [0, 0, 0];
  let count = 0, nonFinite = 0, maxSpeed = 0, kineticEnergy = 0;
  let minY = Infinity, minX = Infinity, maxX = -Infinity, densityMin = Infinity, densityMax = -Infinity;
  for (let b = 0; b < state.length; b += 20) {
    for (let j = 0; j < 20; j++) if (!Number.isFinite(state[b + j])) nonFinite++;
    if (state[b + 3]! <= 0.5) continue;
    count++;
    const speed = Math.hypot(state[b + 4]!, state[b + 5]!, state[b + 6]!);
    maxSpeed = Math.max(maxSpeed, speed); kineticEnergy += 0.5 * mass * speed ** 2;
    minY = Math.min(minY, state[b + 1]!); minX = Math.min(minX, state[b]!); maxX = Math.max(maxX, state[b]!);
    densityMin = Math.min(densityMin, state[b + 7]!); densityMax = Math.max(densityMax, state[b + 7]!);
    for (let a = 0; a < 3; a++) { position[a]! += state[b + a]!; momentum[a]! += mass * state[b + a + 4]!; }
  }
  return { count, particleMassTotal: count * mass, nonFinite, maxSpeed, kineticEnergy,
    center: position.map(v => v / count), momentum, minY, minX, maxX, densityMin, densityMax };
}

async function runCase(device: GPUDevice, timestamps: boolean, c: Case) {
  const { state, mass } = initialState(c);
  const count = state.length / 20, gridCount = c.n ** 3;
  // バッファ上限を確認してから確保。128³ は本体で利用可能という宣言ではない。
  for (const bytes of [state.byteLength, gridCount * 16]) {
    if (bytes > device.limits.maxStorageBufferBindingSize || bytes > device.limits.maxBufferSize)
      return { ...c, status: 'skipped-device-limit', bytes };
  }
  const buffers: GPUBuffer[] = [];
  function buffer(size: number, usage: number, data?: Float32Array) {
    const b = device.createBuffer({ size, usage }); buffers.push(b);
    if (data) device.queue.writeBuffer(b, 0, data as Float32Array<ArrayBuffer>);
    return b;
  }
  const errors: string[] = [];
  const onError = (e: GPUUncapturedErrorEvent) => errors.push(e.error.message);
  device.addEventListener('uncapturederror', onError);
  device.pushErrorScope('validation');
  let scopeOpen = true;
  const query = timestamps ? device.createQuerySet({ type: 'timestamp', count: 8 }) : undefined;
  try {
    const gridSize = { x: c.n, y: c.n, z: c.n };
    const material = { friction: 0, restitution: 0 };
    const domain = new DomainBoundary(gridSize, material);
    const terrain = TerrainBoundary.flat(gridSize, material, 0, 'wall', 2);
    const sides = BuildingSideBoundary.empty(gridSize, material, 'wall', 2);
    const particles = buffer(state.byteLength, U.storage | U.dst | U.src, state);
    const cells = buffer(gridCount * 16, U.storage | U.dst | U.src);
    const uniformData = new Float32Array([
      c.n, c.n, c.n, gridCount, c.n, c.n, c.n, 0.02,
      MPM_DYNAMIC_VISCOSITY, 6, 2, 2, 1, 0.05, c.dt, mass, 0, 0, 0, 0,
    ]);
    const uniforms = buffer(80, U.uniform | U.dst, uniformData);
    const domainData = domain.toGpuData(), terrainData = terrain.toGpuData(), sideData = sides.toGpuData();
    const planes = buffer(domainData.byteLength, U.storage | U.dst, domainData);
    const heights = buffer(terrainData.byteLength, U.storage | U.dst, terrainData);
    const walls = buffer(sideData.byteLength, U.storage | U.dst, sideData);
    const readParticles = buffer(state.byteLength, U.map | U.dst);
    const readCells = buffer(gridCount * 16, U.map | U.dst);
    const resolved = query ? buffer(256, U.query | U.src) : undefined;
    const readTime = query ? buffer(64, U.map | U.dst) : undefined;
    const gravity = c.kind === 'closed' || c.kind === 'wall' || c.kind === 'floor-support' || c.kind === 'overflow' ? 0 : MPM_GRAVITY_Y;
    const constants: Record<string, number>[] = [
      { fixed_point_multiplier: S },
      { fixed_point_multiplier: S, stiffness: MPM_STIFFNESS, rest_density: MPM_REST_DENSITY },
      { fixed_point_multiplier: S, gravity_y: gravity }, { fixed_point_multiplier: S },
    ];
    const codes = [massCode, stressCode, gridCode, g2pCode];
    const bindings = [[particles, cells, uniforms], [particles, cells, uniforms],
      [cells, uniforms, planes, heights, walls], [particles, cells, uniforms, heights, walls]];
    const pipelines: GPUComputePipeline[] = [], groups: GPUBindGroup[] = [];
    for (let i = 0; i < 4; i++) {
      const module = device.createShaderModule({ code: codes[i]!, label: stages[i]! });
      const info = await module.getCompilationInfo();
      for (const m of info.messages) if (m.type === 'error') throw new Error(`${stages[i]}:${m.lineNum}: ${m.message}`);
      const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: {
        module, entryPoint: 'main', constants: constants[i]!,
      } });
      pipelines.push(pipeline);
      groups.push(device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:
        bindings[i]!.map((b, binding) => ({ binding, resource: { buffer: b } })),
      }));
    }
    async function read(b: GPUBuffer) {
      await b.mapAsync(1);
      const result = b.getMappedRange().slice(0); b.unmap(); return result;
    }
    function dispatch(encoder: GPUCommandEncoder, stage: number, timed: boolean) {
      const descriptor: GPUComputePassDescriptor = { label: stages[stage]! };
      if (timed && query) descriptor.timestampWrites = { querySet: query, beginningOfPassWriteIndex: stage * 2, endOfPassWriteIndex: stage * 2 + 1 };
      const pass = encoder.beginComputePass(descriptor);
      pass.setPipeline(pipelines[stage]!); pass.setBindGroup(0, groups[stage]!);
      pass.dispatchWorkgroups(Math.ceil((stage === 2 ? gridCount : count) / 128)); pass.end();
    }
    // 初期 P2G 後の grid mass を測定。粒子数保存だけで質量保存と呼ばない。
    const massEncoder = device.createCommandEncoder(); massEncoder.clearBuffer(cells);
    dispatch(massEncoder, 0, false); massEncoder.copyBufferToBuffer(cells, 0, readCells, 0, gridCount * 16);
    device.queue.submit([massEncoder.finish()]);
    const gridInitial = new Int32Array(await read(readCells));
    function gridMass(data: Int32Array) {
      let sum = 0, negativeCells = 0;
      for (let i = 3; i < data.length; i += 4) { sum += data[i]! / S; if (data[i]! < 0) negativeCells++; }
      return { sum, relativeError: (sum - count * mass) / (count * mass), negativeCells };
    }
    const initialGridMass = gridMass(gridInitial);
    // warm-up は状態を破棄してから計測。時間測定に readback/copy は含めない。
    const warm = device.createCommandEncoder(); warm.clearBuffer(cells);
    for (let j = 0; j < 4; j++) dispatch(warm, j, false);
    device.queue.submit([warm.finish()]); await device.queue.onSubmittedWorkDone();
    device.queue.writeBuffer(particles, 0, state);
    const encoder = device.createCommandEncoder();
    for (let i = 0; i < c.steps; i++) {
      encoder.clearBuffer(cells);
      for (let j = 0; j < 4; j++) dispatch(encoder, j, i === c.steps - 1);
    }
    // GPU timestamp は最後の1 step。代表値と断定せず、量子化/計測passの影響も残す。
    const commands = encoder.finish(), t0 = performance.now();
    device.queue.submit([commands]); await device.queue.onSubmittedWorkDone();
    const submitAndWaitMs = performance.now() - t0;
    const copy = device.createCommandEncoder();
    copy.copyBufferToBuffer(particles, 0, readParticles, 0, state.byteLength);
    // このmassは最終G2P直前の状態から scatter したもの。
    copy.copyBufferToBuffer(cells, 0, readCells, 0, gridCount * 16);
    if (query && resolved && readTime) { copy.resolveQuerySet(query, 0, 8, resolved, 0); copy.copyBufferToBuffer(resolved, 0, readTime, 0, 64); }
    device.queue.submit([copy.finish()]);
    const finalState = new Float32Array(await read(readParticles));
    const finalGridMass = gridMass(new Int32Array(await read(readCells)));
    const final = summarize(finalState, mass), initial = summarize(state, mass);
    let gpuLastStepMs: Record<string, number> | null = null;
    if (readTime) {
      const times = new BigUint64Array(await read(readTime));
      gpuLastStepMs = Object.fromEntries(stages.map((name, i) => [name, Number(times[i * 2 + 1]! - times[i * 2]!) / 1e6]));
    }
    const checks: { name: string; pass: boolean; actual: number; tolerance: number }[] = [];
    checks.push({ name: 'finite components', pass: final.nonFinite === 0, actual: final.nonFinite, tolerance: 0 });
    checks.push({ name: 'active count conserved (not grid mass)', pass: final.count === count, actual: Math.abs(final.count - count), tolerance: 0 });
    if (c.kind === 'freefall') {
      const expectedV = gravity * c.dt * c.steps;
      const expectedY = state[1]! + gravity * c.dt ** 2 * c.steps * (c.steps + 1) / 2;
      checks.push({ name: 'semi-implicit freefall velocity', pass: Math.abs(finalState[5]! - expectedV) < 2e-4, actual: Math.abs(finalState[5]! - expectedV), tolerance: 2e-4 });
      checks.push({ name: 'semi-implicit freefall position', pass: Math.abs(finalState[1]! - expectedY) < 2e-4, actual: Math.abs(finalState[1]! - expectedY), tolerance: 2e-4 });
    }
    if (c.kind === 'closed' || c.kind === 'freefall') {
      for (const [name, m] of [['initial', initialGridMass], ['final pre-G2P', finalGridMass]] as const) {
        checks.push({ name: `${name} interior grid mass`, pass: Math.abs(m.relativeError) < 1e-4, actual: Math.abs(m.relativeError), tolerance: 1e-4 });
        checks.push({ name: `${name} no negative grid mass`, pass: m.negativeCells === 0, actual: m.negativeCells, tolerance: 0 });
      }
    }
    if (c.kind === 'closed') {
      const relativeMomentumError = Math.hypot(...final.momentum.map((v, a) => v - initial.momentum[a]!)) / Math.hypot(...initial.momentum);
      checks.push({ name: 'moving interior momentum', pass: relativeMomentumError < 1e-3, actual: relativeMomentumError, tolerance: 1e-3 });
    }
    if (c.kind === 'wall') {
      checks.push({ name: 'domain inset x=1.001 non-penetration', pass: final.minX >= 1.001 - 1e-6, actual: Math.max(0, 1.001 - final.minX), tolerance: 1e-6 });
      checks.push({ name: 'plane normal velocity stopped', pass: Math.abs(finalState[4]!) < 1e-4, actual: Math.abs(finalState[4]!), tolerance: 1e-4 });
      checks.push({ name: 'frictionless plane tangent retained', pass: Math.abs(finalState[6]! - 0.25) < 2e-4, actual: Math.abs(finalState[6]! - 0.25), tolerance: 2e-4 });
    }
    if (c.kind === 'floor-support') checks.push({ name: 'known defect: truncated floor mass fraction', pass: Math.abs(initialGridMass.sum - 0.5198) < 2e-6, actual: Math.abs(initialGridMass.sum - 0.5198), tolerance: 2e-6 });
    if (c.kind === 'overflow') checks.push({ name: 'known defect: negative grid mass from overflow', pass: initialGridMass.negativeCells > 0, actual: initialGridMass.negativeCells, tolerance: 0 });
    const scoped = await device.popErrorScope(); scopeOpen = false; if (scoped) errors.push(scoped.message);
    return { ...c, status: errors.length || checks.some(x => !x.pass) ? 'failed-check' : 'diagnostic-completed',
      physicalValidation: 'UNVERIFIED', count, mass, modelTime: c.steps * c.dt, initial, final,
      metricDefinitions: { kineticEnergy: 'translational particle energy only; excludes APIC affine energy',
        densityMin: 'minimum rho/rho0 (not SI density)', densityMax: 'maximum rho/rho0 (not SI density)' },
      parameters: { fixedPointMultiplier: S, stiffness: MPM_STIFFNESS, restDensity: MPM_REST_DENSITY,
        viscosity: MPM_DYNAMIC_VISCOSITY, gravityY: gravity, friction: 0, restitution: 0,
        floorHeight: 0, clearance: 0.02, gridCollisionThickness: 0.05, boundaryResolution: 2,
        uniformData: Array.from(uniformData) },
      initialGridMass, finalGridMass, checks, errors,
      expectedKnownDefect: c.kind === 'floor-support' ? 'y=0.02 truncated support: grid mass fraction ≈0.5198; not repaired' : c.kind === 'overflow' ? 'Two mass=300 particles overflow the central cell; not repaired' : null,
      timing: { submitAndWaitMs, stepsPerSubmitWaitSecond: 1000 * c.steps / submitAndWaitMs,
        gpuLastStepMs, renderingFps: null, note: 'CPU submit+GPU completion wait; not pure GPU time. No rendering. Each stage uses a separate diagnostic pass. Timestamp may be quantized; zero does not mean zero cost.' },
      allocatedBufferBytes: buffers.reduce((s, b) => s + b.size, 0),
      memoryNote: 'Explicit buffers including readback only; excludes shader/driver allocations and rendering',
    };
  } finally {
    if (scopeOpen) await device.popErrorScope();
    query?.destroy(); for (const b of buffers) b.destroy(); device.removeEventListener('uncapturederror', onError);
  }
}

const run = document.querySelector<HTMLButtonElement>('#run')!;
const download = document.querySelector<HTMLButtonElement>('#download')!;
const output = document.querySelector<HTMLElement>('#result')!;
const status = document.querySelector<HTMLElement>('#status')!;
let resultJson = '';
run.addEventListener('click', async () => {
  run.disabled = true; download.disabled = true; output.textContent = ''; status.textContent = 'GPU を確認中';
  let device: GPUDevice | undefined;
  try {
    if (!navigator.gpu) throw new Error('WebGPU が利用できません（未実施）');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('GPU adapter がありません（未実施）');
    const timestamps = adapter.features.has('timestamp-query');
    device = await adapter.requestDevice({ requiredFeatures: timestamps ? ['timestamp-query'] : [] });
    let deviceLost: string | null = null;
    void device.lost.then(info => { deviceLost = info.message; });
    const cases: Case[] = [];
    for (const kind of ['freefall', 'closed', 'floor-support', 'overflow', 'wall'] as Kind[]) cases.push({ kind, n: 32, dt: 0.05, steps: kind === 'floor-support' || kind === 'overflow' ? 1 : 20, side: 8 });
    for (const kind of ['dam', 'column'] as Kind[]) for (const n of [32, 64]) for (const dt of [0.05, 0.025])
      cases.push({ kind, n, dt, steps: Math.round(2 / dt), side: n / 2 });
    // 粒子数を変えた診断。同じ初期体積で粒子質量を変更する。
    cases.push({ kind: 'closed', n: 32, dt: 0.05, steps: 20, side: 16 });
    const results = [];
    for (const c of cases) {
      status.textContent = `${results.length + 1}/${cases.length}: ${c.kind} ${c.n}³ dt=${c.dt}`;
      results.push(await runCase(device, timestamps, c));
      if (deviceLost) throw new Error(`Device lost: ${deviceLost}`);
    }
    const shaderSha256: Record<string, string> = {};
    for (const [i, code] of [massCode, stressCode, gridCode, g2pCode].entries()) {
      const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(code));
      shaderSha256[stages[i]!] = Array.from(new Uint8Array(hash), b => b.toString(16).padStart(2, '0')).join('');
    }
    const report = { schemaVersion: 1, generatedAt: new Date().toISOString(), userAgent: navigator.userAgent,
      adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description },
      timestampQuery: timestamps, limits: { maxBufferSize: device.limits.maxBufferSize, maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize },
      shaderSha256, units: 'legacy grid/model time; not SI',
      limitations: ['No external dam-break reference', 'Initially-resting column is not hydrostatic equilibrium', 'Grid changes are legacy sensitivity, not SI convergence', 'No rendering/DEM/PLATEAU in this synthetic harness', 'No per-add GPU overflow instrumentation'], results };
    resultJson = JSON.stringify(report, null, 2); output.textContent = resultJson;
    const skipped = results.filter(r => r.status === 'skipped-device-limit').length;
    status.textContent = results.some(r => r.status === 'failed-check') ? '完了：失敗した検査あり。物理妥当性は未検証。' : `診断完了（未実施 ${skipped} 件、物理妥当性は未検証）`;
    download.disabled = false;
  } catch (e) { status.textContent = '検証を完了できませんでした'; output.textContent = String(e); }
  finally { device?.destroy(); run.disabled = false; }
});
download.addEventListener('click', () => {
  const url = URL.createObjectURL(new Blob([resultJson], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = 'gis-mpm-validation.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
