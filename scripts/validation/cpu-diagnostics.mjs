/**
 * Reproducible analytic diagnostics, run with:
 *   node scripts/validation/cpu-diagnostics.mjs
 *
 * This is not a CPU fluid simulation, a GPU timing benchmark, or a substitute
 * for running the four real-WGSL scenarios. The refinement diagnostic holds
 * physical block size and density fixed while scaling particle mass with h^3.
 */
import { p2g, gatherDensity, freeFall, accumulateFixedPoint } from './reference.mjs';

const refinement = [];
for (const resolution of [16, 32, 64]) {
  const h = 1 / resolution;
  const densitySI = 1000;
  const particleMassSI = densitySI * (h / 2) ** 3;
  const particles = [];
  // Fixed 0.25 m cube, embedded safely inside a 1 m closed-domain grid.
  // Two particle samples per cell axis (8 per cell) in all refinements.
  for (let ix = 0; ix < resolution / 2; ix++) {
    for (let iy = 0; iy < resolution / 2; iy++) {
      for (let iz = 0; iz < resolution / 2; iz++) {
        particles.push({
          position: [ix, iy, iz].map(i => resolution / 4 + (i + 0.5) / 2),
          velocity: [0, 0, 0], mass: particleMassSI,
        });
      }
    }
  }
  const cells = p2g(particles, [resolution, resolution, resolution]);
  const mass = [...cells.values()].reduce((sum, cell) => sum + cell.mass, 0);
  const center = Array(3).fill(0.375 * resolution);
  const interiorDensitySI = gatherDensity(center, cells, [resolution, resolution, resolution], h ** 3);
  refinement.push({ resolution, cellWidthM: h, particleCount: particles.length,
    particleMassKg: particleMassSI, gridMassKg: mass, analyticMassKg: 15.625,
    relativeMassError: Math.abs(mass - 15.625) / 15.625,
    interiorDensityKgM3: interiorDensitySI,
    relativeInteriorDensityError: Math.abs(interiorDensitySI - densitySI) / densitySI });
}

const timeRefinement = [0.04, 0.02, 0.01, 0.005].map(dt => ({ dtSeconds: dt,
  ...freeFall({ dt, steps: Math.round(1 / dt) }) }));

console.log(JSON.stringify({
  schemaVersion: 1,
  validationStatus: 'CPU analytic diagnostics only; production GPU fluid/collision stability remains unverified by this script',
  refinementScope: 'One P2G and density gather; same physical block, no time evolution or boundary collisions',
  refinement,
  timeRefinementScope: 'Semi-implicit Euler point free fall without grid transfers or boundaries, time = 1 second',
  timeRefinement,
  fixedPointCounterexamples: {
    positiveMassOverflow: accumulateFixedPoint([120, 120]),
    intermediateWrapFinalCancellation: accumulateFixedPoint([120, 120, -120]),
    contributionClamp: accumulateFixedPoint([300, -100]),
    subQuantum: accumulateFixedPoint(Array(1000).fill(0.4e-7)),
  },
  gpuPerformance: 'NOT_MEASURED',
}, null, 2));
