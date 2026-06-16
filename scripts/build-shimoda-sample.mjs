import { mkdir, mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import { build } from 'esbuild';

const repoRoot = resolve(import.meta.dirname, '..');
const outDir = resolve(repoRoot, 'public/samples/shimoda');
const outGeoJsonPath = resolve(outDir, 'buildings.geojson');
const outManifestPath = resolve(outDir, 'manifest.json');
const defaultPrefecture = '静岡県';
const defaultMunicipality = '下田市';

const tmpDir = await mkdtemp(join(tmpdir(), 'local-mpm-shimoda-'));
await build({
  entryPoints: [
    resolve(repoRoot, 'src/import/plateauCatalog.ts'),
    resolve(repoRoot, 'src/import/plateauZip.ts'),
  ],
  outdir: tmpDir,
  bundle: true,
  format: 'esm',
  target: 'es2022',
  logLevel: 'silent',
});

const catalogModule = await import(pathToFileURL(resolve(tmpDir, 'plateauCatalog.js')).href);
const zipModule = await import(pathToFileURL(resolve(tmpDir, 'plateauZip.js')).href);

const catalog = await catalogModule.loadPlateauCatalogDatasets();
const dataset = catalog.datasets.find(
  (item) =>
    item.prefectureLabel === defaultPrefecture &&
    item.municipalityLabel === defaultMunicipality,
);
if (!dataset) {
  throw new Error(`${defaultPrefecture} ${defaultMunicipality}: PLATEAU dataset not found`);
}

const resources = await catalogModule.loadPlateauCatalogResources(dataset.name);
const resource = resources[0];
if (!resource) {
  throw new Error(`${dataset.label}: CityGML ZIP resource not found`);
}

const size = await fetchZipSize(resource);
let fetchedBytes = 0;
const source = {
  fileName: resource.fileName,
  size,
  readRange: async (offset, length) => {
    const buffer = await fetchZipRange(resource.url, offset, length);
    fetchedBytes += buffer.byteLength;
    process.stderr.write(
      `\r${resource.fileName}: fetched ${formatBytes(fetchedBytes)} via Range requests`,
    );
    return buffer;
  },
};

const imported = await zipModule.importPlateauZipSource(source, {
  onProgress: (progress) => {
    if (progress.total && progress.loaded !== undefined) {
      process.stderr.write(
        `\r${progress.message} (${progress.loaded}/${progress.total})`.padEnd(100),
      );
    } else {
      process.stderr.write(`\r${progress.message}`.padEnd(100));
    }
  },
});
process.stderr.write('\n');

const manifest = {
  ...imported.manifest,
  name: 'Shimoda PLATEAU building cache',
  description:
    `Cached PLATEAU CityGML building import generated from ${resource.datasetLabel} / ${resource.name}.`,
  obstacles: imported.manifest.obstacles.map((obstacle) =>
    obstacle.type === 'geojson'
      ? {
          ...obstacle,
          path: 'buildings.geojson',
        }
      : obstacle,
  ),
};

const geojson = {
  ...imported.geojson,
  name: 'shimoda-buildings',
  metadata: {
    ...imported.geojson.metadata,
    sourceDatasetName: resource.datasetName,
    sourceDatasetTitle: resource.datasetTitle,
    sourceResourceName: resource.name,
    sourceResourceUrl: resource.url,
    generatedAt: new Date().toISOString(),
  },
};

await mkdir(dirname(outGeoJsonPath), { recursive: true });
await writeFile(outGeoJsonPath, `${JSON.stringify(geojson)}\n`, 'utf8');
await writeFile(outManifestPath, `${JSON.stringify(manifest, null, 2)}\n`, 'utf8');

console.log(`Wrote ${geojson.features.length} buildings`);
console.log(outGeoJsonPath);
console.log(outManifestPath);

async function fetchZipSize(resource) {
  const res = await fetch(resource.url, { method: 'HEAD' });
  if (res.ok) {
    const size = Number(res.headers.get('content-length'));
    if (Number.isFinite(size) && size > 0) return size;
  }
  if (resource.sizeBytes !== undefined) return resource.sizeBytes;
  throw new Error(`${resource.fileName}: failed to determine ZIP size`);
}

async function fetchZipRange(url, offset, length) {
  const end = offset + length - 1;
  const res = await fetch(url, {
    headers: {
      range: `bytes=${offset}-${end}`,
      'user-agent': 'local-mpm shimoda cache builder',
    },
  });
  if (!res.ok) {
    throw new Error(`Range fetch failed: ${res.status} ${res.statusText}`);
  }
  const buffer = await res.arrayBuffer();
  if (buffer.byteLength !== length) {
    throw new Error(
      `Range fetch returned ${buffer.byteLength} bytes; expected ${length} bytes`,
    );
  }
  return buffer;
}

function formatBytes(value) {
  if (value >= 1000 * 1000 * 1000) return `${(value / (1000 * 1000 * 1000)).toFixed(2)}GB`;
  if (value >= 1000 * 1000) return `${(value / (1000 * 1000)).toFixed(1)}MB`;
  if (value >= 1000) return `${(value / 1000).toFixed(0)}KB`;
  return `${value}B`;
}
