export const ALLOWED_PLATEAU_ZIP_HOSTS = [
  'assets.cms.plateau.reearth.io',
  'gic-plateau.s3.ap-northeast-1.amazonaws.com',
  'www.geospatial.jp',
] as const;

const ALLOWED_PLATEAU_ZIP_HOST_SET = new Set<string>(ALLOWED_PLATEAU_ZIP_HOSTS);

export function isAllowedPlateauZipHost(hostname: string): boolean {
  return ALLOWED_PLATEAU_ZIP_HOST_SET.has(hostname);
}

export function validatePlateauZipUrl(raw: string | null | undefined): string | undefined {
  if (!raw) return undefined;
  try {
    const url = new URL(raw);
    if (url.protocol !== 'https:') return undefined;
    if (!isAllowedPlateauZipHost(url.hostname)) return undefined;
    if (!/\.zip(?:$|[?#])/i.test(url.href)) return undefined;
    return url.toString();
  } catch {
    return undefined;
  }
}
