// =============================================================================
// Terrain tile source configuration
// -----------------------------------------------------------------------------
// GSI mode uses the existing DEM5/DEM10 fallback pipeline. Import/PLATEAU mode
// can point the same tile loader at common raster-dem encodings.
// =============================================================================

export type RasterDemEncoding = 'terrarium' | 'mapbox';
export type TerrainMeshMode = 'cell-boundary' | 'pixel-center-bridged';

export type TerrainTileSource =
  | {
      id: 'gsi';
      kind: 'gsi';
      skirtDepthM?: number;
      fixedZoom?: number;
      meshMode?: TerrainMeshMode;
    }
  | {
      id: string;
      kind: 'raster-dem';
      encoding: RasterDemEncoding;
      urlTemplate: string;
      skirtDepthM?: number;
      fixedZoom?: number;
      meshMode?: TerrainMeshMode;
    };

export const GSI_TERRAIN_SOURCE: TerrainTileSource = {
  id: 'gsi',
  kind: 'gsi',
  skirtDepthM: 0,
  meshMode: 'pixel-center-bridged',
};

export const PLATEAU_TERRARIUM_TP_SOURCE: TerrainTileSource = {
  id: 'plateau-terrarium-tp',
  kind: 'raster-dem',
  encoding: 'terrarium',
  urlTemplate: 'https://tile.plateauview.mlit.go.jp/terrarium/{z}/{x}/{y}.png?geoid=none',
  // 通常操作では GSI と同じ LOD ストリーミングにする。z=15 固定は
  // 近景品質は高いが、旋回中も原寸タイルを大量に維持して重くなるため診断用途に限定する。
  skirtDepthM: 0,
  meshMode: 'pixel-center-bridged',
};
