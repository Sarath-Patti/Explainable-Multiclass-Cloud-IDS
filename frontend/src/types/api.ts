/** TypeScript interface definitions for API request/response payloads. */

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  timestamp: string;
}

export interface ApiError {
  message: string;
  status?: number;
}
