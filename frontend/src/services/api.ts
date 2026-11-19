export type StationId = 'site_113' | 'site_115' | 'site_117'

export interface PredictionRequest {
  station_id: StationId
  pm25_at_origin: number
  pm25_1h_before: number
  pm25_2h_before: number
  pm25_23h_before: number
  pm25_rolling_mean_3h: number
  pm25_rolling_mean_6h: number
  pm25_rolling_mean_24h: number
  pm25_rolling_observation_count_24h: number
  hour: number
  day_of_week: number
  month: number
  hour_sin: number
  hour_cos: number
  day_of_week_sin: number
  day_of_week_cos: number
  month_sin: number
  month_cos: number
}

export interface Prediction {
  station_id: string
  predicted_pm25_next_hour: number
  anchor_pm25_at_origin: number
  residual: number
  model: string
  forecast_horizon: string
  timestamp_utc: string
}

export interface Health {
  status: string
  model_loaded: boolean
  model_artifact: string
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init)
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // Keep the HTTP status as the surfaced error when the response is not JSON.
    }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export function getHealth(): Promise<Health> {
  return request<Health>('/health')
}

export function getPrediction(payload: PredictionRequest): Promise<Prediction> {
  return request<Prediction>('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}
