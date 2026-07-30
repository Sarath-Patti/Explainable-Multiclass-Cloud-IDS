import { useState, useEffect, useCallback } from 'react';
import { checkHealth } from '../services/api';
import { HealthResponse } from '../types/api';

export interface UseHealthReturn {
  health: HealthResponse | null;
  loading: boolean;
  error: string | null;
  latencyMs: number | null;
  refetch: () => Promise<void>;
}

export const useHealth = (pollIntervalMs: number = 30000): UseHealthReturn => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);

  const fetchHealth = useCallback(async () => {
    setLoading(true);
    const start = performance.now();
    try {
      const data = await checkHealth();
      const end = performance.now();
      setLatencyMs(Math.round(end - start));
      setHealth(data);
      setError(null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to connect to backend server');
      setHealth(null);
      setLatencyMs(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHealth();
    if (pollIntervalMs > 0) {
      const interval = setInterval(fetchHealth, pollIntervalMs);
      return () => clearInterval(interval);
    }
  }, [fetchHealth, pollIntervalMs]);

  return { health, loading, error, latencyMs, refetch: fetchHealth };
};
