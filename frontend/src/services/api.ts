import axios, { AxiosProgressEvent } from 'axios';
import { HealthResponse, PredictionResponse, PredictionItem } from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60s timeout for batch predictions
});

/**
 * Fetch system health status from FastAPI backend endpoint GET /api/v1/health
 */
export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await apiClient.get<HealthResponse>('/health');
  return response.data;
};

/**
 * Upload a CSV file containing network flow records for batch multiclass classification.
 * POST /api/v1/predict
 */
export const predictBatchCSV = async (
  file: File,
  onUploadProgress?: (progressEvent: AxiosProgressEvent) => void
): Promise<PredictionResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post<PredictionResponse>('/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress,
  });

  return response.data;
};

/**
 * Helper utility to export prediction results as a downloadable CSV file.
 */
export const exportPredictionsToCSV = (predictions: PredictionItem[], filename: string = 'prediction_results.csv') => {
  if (!predictions || predictions.length === 0) return;

  const headers = ['row', 'prediction', 'confidence'];
  const csvRows = [headers.join(',')];

  for (const item of predictions) {
    csvRows.push(`${item.row},"${item.prediction}",${item.confidence}`);
  }

  const csvString = csvRows.join('\n');
  const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};
