import { parseCityGmlBuildingFeatures } from './plateauCityGmlParser';
import type {
  PlateauCityGmlWorkerRequest,
  PlateauCityGmlWorkerResponse,
} from './plateauCityGmlWorkerTypes';

const textDecoder = new TextDecoder('utf-8');

// CityGML の decode / regex 解析は大きな GML entry で main thread を長く止める。
// worker では 1 entry ずつ GeoJSON feature へ変換し、結果だけを main thread へ返す。
self.addEventListener('message', (event: MessageEvent<PlateauCityGmlWorkerRequest>) => {
  const { id, sourceEntry, featureIndexOffset, gmlBuffer } = event.data;

  try {
    const gml = textDecoder.decode(gmlBuffer);
    const parsed = parseCityGmlBuildingFeatures(gml, sourceEntry, featureIndexOffset);
    const response: PlateauCityGmlWorkerResponse = {
      id,
      type: 'success',
      sourceEntry,
      features: parsed.features,
      budget: parsed.budget,
    };
    self.postMessage(response);
  } catch (error) {
    const response: PlateauCityGmlWorkerResponse = {
      id,
      type: 'error',
      sourceEntry,
      message: error instanceof Error ? error.message : String(error),
    };
    self.postMessage(response);
  }
});
