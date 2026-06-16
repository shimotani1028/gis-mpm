import type { BuildingGeoJson } from './BuildingObstacleLayer';
import type { CityGmlImportBudget } from './plateauCityGmlParser';

export interface PlateauCityGmlWorkerRequest {
  id: number;
  sourceEntry: string;
  featureIndexOffset: number;
  gmlBuffer: ArrayBuffer;
}

export interface PlateauCityGmlWorkerSuccess {
  id: number;
  type: 'success';
  sourceEntry: string;
  features: BuildingGeoJson['features'];
  budget: CityGmlImportBudget;
}

export interface PlateauCityGmlWorkerFailure {
  id: number;
  type: 'error';
  sourceEntry: string;
  message: string;
}

export type PlateauCityGmlWorkerResponse =
  | PlateauCityGmlWorkerSuccess
  | PlateauCityGmlWorkerFailure;
