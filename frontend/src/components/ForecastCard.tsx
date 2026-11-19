import { AlertCircle, Clock } from 'lucide-react'
import type { Prediction } from '../services/api'

interface ForecastCardProps {
  prediction: Prediction | null
  loading: boolean
  error: string | null
  onPredict: () => void
}

function ForecastCard({ prediction, loading, error, onPredict }: ForecastCardProps) {
  return (
    <section className="panel" aria-labelledby="forecast-heading">
      <div className="panel-heading">
        <Clock aria-hidden="true" size={20} />
        <h2 id="forecast-heading">Next-hour PM2.5 forecast</h2>
      </div>

      {loading && <p role="status">Fetching prediction…</p>}
      {error && (
        <p className="error-message" role="alert">
          <AlertCircle aria-hidden="true" size={16} />
          {error}
        </p>
      )}
      {!loading && prediction && (
        <>
          <div className="forecast-values">
            <article className="forecast-value">
              <span>Predicted PM2.5</span>
              <strong>{prediction.predicted_pm25_next_hour.toFixed(2)} µg/m³</strong>
            </article>
            <article className="forecast-value">
              <span>Current (anchor)</span>
              <strong>{prediction.anchor_pm25_at_origin.toFixed(2)} µg/m³</strong>
            </article>
            <article className="forecast-value">
              <span>Residual correction</span>
              <strong>
                {prediction.residual >= 0 ? '+' : ''}
                {prediction.residual.toFixed(2)} µg/m³
              </strong>
            </article>
          </div>
          <p className="status-detail">
            {prediction.model} · {prediction.forecast_horizon}
          </p>
        </>
      )}
      {!loading && (
        <button className="primary-button" type="button" onClick={onPredict}>
          {prediction ? 'Refresh prediction' : 'Get next-hour prediction'}
        </button>
      )}
    </section>
  )
}

export default ForecastCard
