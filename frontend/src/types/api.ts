/** TypeScript interface definitions for API request/response payloads. */

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  timestamp: string;
}

export interface PredictionItem {
  row: number;
  prediction: string;
  confidence: number;
}

export interface PredictionSummary {
  total_samples: number;
  predicted_attacks: number;
  predicted_benign: number;
}

export interface PredictionResponse {
  summary: PredictionSummary;
  predictions: PredictionItem[];
}

export interface FeatureContribution {
  feature: string;
  value: number;
  shap_value: number;
}

export interface ExplainRequest {
  row: number;
  features: Record<string, number | string>;
}

export interface ExplainResponse {
  prediction: string;
  confidence: number;
  base_value: number;
  top_features: FeatureContribution[];
}

export interface MissingFeaturesErrorDetail {
  error: string;
  message: string;
  missing_features: string[];
}

export interface ApiError {
  message: string;
  status?: number;
  missing_features?: string[];
}
