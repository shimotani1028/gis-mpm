import test from 'node:test';
import assert from 'node:assert/strict';
import {
  stencil, p2g, g2p, gatherDensity, pressure, stress, matvec,
  viscousDissipation, gradientToSI, stressImpulseSI,
  encodeFixedPoint, accumulateFixedPoint, freeFall, hydrostaticPressure,
} from '../scripts/validation/reference.mjs';

// Tests assert analytic mathematical properties / documented counterexamples.
// They deliberately DO NOT label the production shaders as validated.
const close = (actual, expected, tolerance = 2e-12) =>
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `${actual} differs from ${expected} by more than ${tolerance}`);
const closeVector = (actual, expected, tolerance) =>
  actual.forEach((value, i) => close(value, expected[i], tolerance));

test('quadratic kernel: partition of unity, zero first moment, D = I/4', () => {
  for (const fraction of [0, 1e-8, 0.1, 0.499, 0.5, 0.9, 0.999999]) {
    const nodes = stencil([4 + fraction, 5.321, 6.123]);
    close(nodes.reduce((sum, n) => sum + n.weight, 0), 1);
    for (let i = 0; i < 3; i++) {
      close(nodes.reduce((sum, n) => sum + n.weight * n.distance[i], 0), 0);
      for (let j = 0; j < 3; j++) {
        close(nodes.reduce((sum, n) => sum + n.weight * n.distance[i] * n.distance[j], 0), i === j ? 0.25 : 0);
      }
    }
  }
});

test('APIC P2G conserves mass and linear momentum for complete stencils', () => {
  const particles = [
    { position: [3.1, 4.8, 5.2], velocity: [2, -1, 3], mass: 1.7, affine: [[1, 2, 3], [-2, 0, 1], [3, -1, 2]] },
    { position: [3.9, 4.1, 5.6], velocity: [-1, 2, -0.5], mass: 2.4 },
  ];
  const cells = [...p2g(particles, [16, 16, 16]).values()];
  close(cells.reduce((sum, cell) => sum + cell.mass, 0), 4.1);
  for (let axis = 0; axis < 3; axis++) {
    close(cells.reduce((sum, cell) => sum + cell.momentum[axis], 0),
      particles.reduce((sum, p) => sum + p.mass * p.velocity[axis], 0));
  }
});

test('G2P reproduces constant and general affine velocity fields', () => {
  const position = [4.12, 5.73, 6.49];
  const affine = [[1, 2, -3], [4, -1, 2], [0.5, -2, 0.3]];
  const velocity = [2, -5, 7];
  const sampled = g2p(position, node => matvec(affine, node.distance).map((v, i) => v + velocity[i]));
  closeVector(sampled.velocity, velocity);
  sampled.affine.forEach((row, i) => closeVector(row, affine[i]));
  const constant = g2p(position, () => velocity);
  closeVector(constant.velocity, velocity);
  constant.affine.flat().forEach(value => close(value, 0));
});

test('known defect: truncated bottom stencil loses P2G grid mass and PIC velocity', () => {
  const p = { position: [4.5, 0.02, 4.5], mass: 1, velocity: [1, 0, 0] };
  const cells = p2g([p], [16, 16, 16]);
  const totalMass = [...cells.values()].reduce((sum, cell) => sum + cell.mass, 0);
  // Missing node centered at -0.5: weight = 0.5*(1.5-0.52)^2.
  close(totalMass, 0.5198);
  const recovered = g2p(p.position, () => [1, 0, 0], [16, 16, 16]);
  close(recovered.velocity[0], 0.5198);
  assert.notEqual(totalMass, p.mass); // Particle count alone cannot detect this loss.
});

test('density gather gives mass per cell; SI density needs cell-volume division', () => {
  const cells = new Map(stencil([4.1, 5.3, 6.8]).map(node => [node.key, { mass: 12 }]));
  close(gatherDensity([4.1, 5.3, 6.8], cells), 12);
  close(gatherDensity([4.1, 5.3, 6.8], cells, undefined, 2 * 3 * 4), 0.5);
});

test('EOS has zero tensile pressure, compressive slope gamma*K/rho0 and hydrostatic density bias', () => {
  close(pressure(2, 4, 3), 0);
  close(pressure(4, 4, 3), 0);
  close(pressure(8, 4, 3), 93);
  const epsilon = 1e-6;
  close((pressure(4 + epsilon, 4, 3) - pressure(4, 4, 3)) / epsilon, 15 / 4, 3e-6);
  // A weakly compressible EOS needs compression to support hydrostatic load.
  const requiredPressure = hydrostaticPressure(0.5, 4, 0.12);
  const requiredDensity = 4 * (1 + requiredPressure / 3) ** 0.2;
  close(pressure(requiredDensity, 4, 3), requiredPressure);
  assert.ok(requiredDensity > 4);
});

test('viscous stress dissipates strain, vanishes for rigid rotation', () => {
  const shear = [[0, 2, 0], [0, 0, 0], [0, 0, 0]];
  close(viscousDissipation(shear, 0.01), 0.04);
  const rotation = [[0, -2, 1], [2, 0, -3], [-1, 3, 0]];
  close(viscousDissipation(rotation, 0.01), 0);
  const gradient = [[1, 2, 3], [4, -2, 6], [7, 8, 0.2]];
  const expected = gradient.reduce((sum, row, i) => sum + row.reduce((s, value, j) =>
    s + 2 * 0.01 * ((value + gradient[j][i]) / 2) ** 2, 0), 0);
  close(viscousDissipation(gradient, 0.01), expected);
  assert.ok(expected >= 0);
});

test('isotropic stress gives zero net internal impulse on a full stencil', () => {
  const sigma = stress([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 100, 0);
  const total = [0, 0, 0];
  for (const node of stencil([5.1, 6.2, 4.3])) {
    const impulse = stressImpulseSI(sigma, 0.001, 0.01, node, [0.1, 0.2, 0.3]);
    impulse.forEach((value, axis) => { total[axis] += value; });
  }
  closeVector(total, [0, 0, 0]);
});

test('anisotropic rendering changes rigid rotation into artificial strain if raw C is used', () => {
  const widths = [2, 1, 3];
  const physicalRotation = [[0, -1, 0], [1, 0, 0], [0, 0, 0]];
  // Inverse mapping C_q = H^-1 C_x H for T=1.
  const computational = physicalRotation.map((row, i) => row.map((value, j) => value * widths[j] / widths[i]));
  const physical = gradientToSI(computational, widths);
  physical.forEach((row, i) => closeVector(row, physicalRotation[i]));
  close(viscousDissipation(physical, 0.01), 0);
  assert.ok(viscousDissipation(computational, 0.01) > 0);
});

test('SI pressure impulse requires axis-specific inverse cell widths', () => {
  const sigma = stress([[0, 0, 0], [0, 0, 0], [0, 0, 0]], 10, 0);
  const node = { weight: 0.1, distance: [0.5, 0.5, 0.5] };
  const impulse = stressImpulseSI(sigma, 1, 0.01, node, [2, 1, 4]);
  closeVector(impulse, [0.01, 0.02, 0.005]);
  // Multiplying unit-grid velocity by H would instead make x stronger than y.
  assert.ok(impulse[0] < impulse[1]);
});

test('fixed-point sub-quantum contributions disappear before accumulation', () => {
  const values = Array(1000).fill(0.4e-7);
  close(accumulateFixedPoint(values).decoded, 0);
  assert.ok(values.reduce((sum, v) => sum + v, 0) > 3.9e-5);
  assert.equal(encodeFixedPoint(-0.4e-7), 0);
});

test('per-contribution clamp cannot prevent atomic sum overflow', () => {
  const result = accumulateFixedPoint([120, 120]);
  assert.equal(result.clampedContributions, 0);
  assert.equal(result.wrapped, true);
  assert.ok(result.decoded < 0); // Positive mass becomes negative.
  close(result.decoded, -189.4967296);
});

test('intermediate signed overflow can cancel modulo 2^32 and leave a correct final sum', () => {
  const result = accumulateFixedPoint([120, 120, -120]);
  assert.equal(result.wrapped, true);
  close(result.decoded, 120);
  // Intermediate wrap alone is not necessarily a final error for integer
  // additive-only passes; range diagnostics should distinguish the two.
});

test('clamped contributions break cancellation and f32 precision exceeds nominal quantum', () => {
  const result = accumulateFixedPoint([300, -100]);
  assert.equal(result.clampedContributions, 1);
  assert.ok(Math.abs(result.decoded - 200) > 80);
  // Near momentum=100, f32 product ULP is 64, i.e. 6.4e-6 after decode.
  assert.equal(encodeFixedPoint(100), encodeFixedPoint(100 + 1e-6));
});

test('semi-implicit gravity integration converges at first order at a fixed physical time', () => {
  const coarse = freeFall({ dt: 0.02, steps: 50 });
  const fine = freeFall({ dt: 0.01, steps: 100 });
  close(coarse.velocity, -9.80665);
  close(fine.velocity, -9.80665);
  close(coarse.positionError / fine.positionError, 2, 1e-10);
});
