import { useEffect, useCallback } from 'react';

export function useTabLogger(tabName: string) {
  useEffect(() => {
    console.debug(`[Tab:${tabName}] mount`);
    return () => console.debug(`[Tab:${tabName}] unmount`);
  }, [tabName]);

  return useCallback(
    (event: string, data?: unknown) => {
      if (data !== undefined) {
        console.debug(`[Tab:${tabName}] ${event}`, data);
      } else {
        console.debug(`[Tab:${tabName}] ${event}`);
      }
    },
    [tabName]
  );
}
