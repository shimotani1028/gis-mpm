/**
 * Independent CPU mathematical oracles, NOT a validated fluid solver.
 *
 * These use double precision and a distance-based cardinal B-spline, rather
 * than copying WGSL's three-weight expressions. They establish identities and
 * counterexamples against which the real shaders can be tested separately.
 * Passing these tests does not execute WGSL, prove GPU stability, validate
 * PLATEAU collisions, or establish agreement with physical water experiments.
 * Matrices below are row-major; WGSL particle c0/c1/c2 are columns.
 */

export const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
export const matvec = (a, v) => a.map(row => dot(row, v));
export const zeroMatrix = () => Array.from({ length: 3 }, () => [0, 0, 0]);

// Centered quadratic cardinal B-spline with support [-1.5, 1.5].
export function quadraticKernel(distance) {
  const r = Math.abs(distance);
  if (r < 0.5) return 0.75 - r * r;
  if (r < 1.5) return 0.5 * (1.5 - r) ** 2;
  return 0;
}

export function stencil(position, grid = [Infinity, Infinity, Infinity]) {
  const base = position.map(Math.floor);
  const nodes = [];
  for (let x = base[0] - 1; x <= base[0] + 1; x++) {
    for (let y = base[1] - 1; y <= base[1] + 1; y++) {
      for (let z = base[2] - 1; z <= base[2] + 1; z++) {
        const index = [x, y, z];
        // Matches the production stencil's omission of out-of-domain nodes.
        if (index.some((value, axis) => value < 0 || value >= grid[axis])) continue;
        const distance = index.map((value, axis) => value + 0.5 - position[axis]);
        const weight = distance.reduce((w, value) => w * quadraticKernel(value), 1);
        nodes.push({ index, distance, weight, key: index.join(',') });
      }
    }
  }
  return nodes;
}

// No stress, boundaries, or fixed-point rounding: conservative APIC transfer.
export function p2g(particles, grid) {
  const cells = new Map();
  for (const particle of particles) {
    for (const node of stencil(particle.position, grid)) {
      const cell = cells.get(node.key) ?? { mass: 0, momentum: [0, 0, 0] };
      const dm = node.weight * particle.mass;
      const affine = matvec(particle.affine ?? zeroMatrix(), node.distance);
      cell.mass += dm;
      cell.momentum = cell.momentum.map((value, axis) =>
        value + dm * (particle.velocity[axis] + affine[axis]));
      cells.set(node.key, cell);
    }
  }
  return cells;
}

// D_p = sum_i w_ip d_ip d_ip^T = I/4 for the complete unit-grid kernel.
export function g2p(position, velocityAtNode, grid) {
  const velocity = [0, 0, 0];
  const affine = zeroMatrix();
  for (const node of stencil(position, grid)) {
    const value = velocityAtNode(node);
    for (let a = 0; a < 3; a++) {
      velocity[a] += node.weight * value[a];
      for (let b = 0; b < 3; b++) {
        affine[a][b] += 4 * node.weight * value[a] * node.distance[b];
      }
    }
  }
  return { velocity, affine };
}

// Production gathers nodal mass without dividing by physical cell volume.
// Thus the default result is mass per unit computational-cell volume.
export function gatherDensity(position, cells, grid, cellVolume = 1) {
  return stencil(position, grid).reduce((sum, node) =>
    sum + node.weight * (cells.get(node.key)?.mass ?? 0), 0) / cellVolume;
}

// Exactly the model being audited: compressive-only Tait-like EOS, gamma = 5.
// This is not a claim that stiffness=3/restDensity=4 represent SI water.
export function pressure(density, restDensity, stiffness) {
  return Math.max(0, stiffness * ((Math.max(density, 1e-4) / restDensity) ** 5 - 1));
}

export function stress(gradient, pressureValue, viscosity) {
  return gradient.map((row, i) => row.map((value, j) =>
    viscosity * (value + gradient[j][i]) - (i === j ? pressureValue : 0)));
}

// Viscous mechanical power is -tau:C; tau:C = 2 mu ||sym(C)||^2 >= 0.
export function viscousDissipation(gradient, viscosity) {
  const tau = stress(gradient, 0, viscosity);
  return tau.reduce((sum, row, i) => sum + dot(row, gradient[i]), 0);
}

// x = H q and t_SI = T tau imply grad_x(v_SI) = H C H^-1 / T.
export function gradientToSI(gradient, cellWidths, secondsPerTimeUnit = 1) {
  return gradient.map((row, i) => row.map((value, j) =>
    value * cellWidths[i] / cellWidths[j] / secondsPerTimeUnit));
}

// Physical quadratic moment inverse is diag(4/h_i^2), not 4 I.
// Physical stress impulse: -dt V sigma D^-1 (x_i - x_p) w_ip.
export function stressImpulseSI(stressSI, volumeSI, dtSI, node, cellWidths) {
  const momentApplied = node.distance.map((value, axis) => 4 * value / cellWidths[axis]);
  return matvec(stressSI, momentApplied).map(value => -dtSI * volumeSI * node.weight * value);
}

/** Emulate the individual finite-f32 operations of the shader encoder.
 * NaN/Inf cases intentionally rejected: their conversion behavior is not
 * established by this oracle. GPU tests must check nonfinite diagnostics.
 */
export function encodeFixedPoint(value, multiplier = 1e7) {
  if (!Number.isFinite(value)) throw new RangeError('Only finite input is modeled');
  const scaled = Math.fround(Math.fround(value) * Math.fround(multiplier));
  const limit = Math.fround(2147483000);
  // JS has -0, whereas the WGSL i32 conversion has just one integer zero.
  return Math.trunc(Math.max(-limit, Math.min(limit, scaled))) || 0;
}

// WGSL atomic<i32> addition wraps modulo 2^32; it does not saturate the sum.
export function addI32(a, b) {
  return Number(BigInt.asIntN(32, BigInt(a) + BigInt(b)));
}

export function accumulateFixedPoint(values, multiplier = 1e7) {
  let integer = 0;
  let wrapped = false;
  let clampedContributions = 0;
  for (const value of values) {
    if (Math.abs(Math.fround(Math.fround(value) * Math.fround(multiplier))) > Math.fround(2147483000)) {
      clampedContributions++;
    }
    const encoded = encodeFixedPoint(value, multiplier);
    const exactSum = integer + encoded;
    wrapped ||= exactSum > 2147483647 || exactSum < -2147483648;
    integer = addI32(integer, encoded);
  }
  return { integer, decoded: integer / multiplier, wrapped, clampedContributions };
}

// Analytic reference used by the separate GPU cases; no fluid solver here.
export function hydrostaticPressure(depth, density = 1000, gravity = 9.80665) {
  return density * gravity * Math.max(0, depth);
}

export function freeFall({ position = 0, velocity = 0, gravity = -9.80665, dt, steps }) {
  const initialPosition = position;
  const initialVelocity = velocity;
  for (let step = 0; step < steps; step++) {
    velocity += gravity * dt;
    position += velocity * dt;
  }
  const time = dt * steps;
  const exactPosition = initialPosition + initialVelocity * time + 0.5 * gravity * time ** 2;
  return { position, velocity, exactPosition, positionError: Math.abs(position - exactPosition) };
}
