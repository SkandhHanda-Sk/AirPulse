import { useEffect, useState } from 'react'
import { Wifi } from 'lucide-react'
import ForecastCard from '../components/ForecastCard'
import HistoricalPanel from '../components/HistoricalPanel'
import ModelStatus from '../components/ModelStatus'
import StationSelector from '../components/StationSelector'
import { getHealth, getPrediction } from '../services/api'
import type { Health, Prediction, PredictionRequest, StationId } from '../services/api'

function buildDemoRequest(station: StationId): PredictionRequest {
  const now = new Date()
  const parts = new Intl.DateTimeFormat('en-GB', {
    timeZone: 'Asia/Kolkata',
    hour: 'numeric',
    weekday: 'short',
    month: 'numeric',
    hourCycle: 'h23',
  }).formatToParts(now)
  const hour = Number(parts.find((part) => part.type === 'hour')?.value)
  const month = Number(parts.find((part) => part.type === 'month')?.value)
  const weekday = parts.find((part) => part.type === 'weekday')?.value
  const dayOfWeek = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].indexOf(weekday ?? '')

  return {
    station_id: station,
    pm25_at_origin: 50,
    pm25_1h_before: 48,
    pm25_2h_before: 46,
    pm25_23h_before: 44,
    pm25_rolling_mean_3h: 47,
    pm25_rolling_mean_6h: 46,
    pm25_rolling_mean_24h: 45,
    pm25_rolling_observation_count_24h: 24,
    hour,
    day_of_week: dayOfWeek,
    month,
    hour_sin: Math.sin((2 * Math.PI * hour) / 24),
    hour_cos: Math.cos((2 * Math.PI * hour) / 24),
    day_of_week_sin: Math.sin((2 * Math.PI * dayOfWeek) / 7),
    day_of_week_cos: Math.cos((2 * Math.PI * dayOfWeek) / 7),
    month_sin: Math.sin((2 * Math.PI * (month - 1)) / 12),
    month_cos: Math.cos((2 * Math.PI * (month - 1)) / 12),
  }
}

function Dashboard() {
  const [station, setStation] = useState<StationId>('site_113')
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  async function requestPrediction() {
    setLoading(true)
    setError(null)
    try {
      setPrediction(await getPrediction(buildDemoRequest(station)))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Prediction request failed.')
    } finally {
      setLoading(false)
    }
  }

  function changeStation(nextStation: StationId) {
    setStation(nextStation)
    setPrediction(null)
    setError(null)
  }

  return (
    <main className="dashboard">
      <header className="page-header">
        <Wifi aria-hidden="true" size={32} />
        <div>
          <h1>AirPulse</h1>
          <p>Hyperlocal PM2.5 forecasting</p>
        </div>
      </header>

      <ModelStatus health={health} />
      <StationSelector selected={station} onSelect={changeStation} />
      <ForecastCard
        prediction={prediction}
        loading={loading}
        error={error}
        onPredict={requestPrediction}
      />
      <HistoricalPanel station={station} />

      <aside className="disclaimer">
        <strong>Disclaimer:</strong> Predictions are machine-learning estimates, not official CPCB
        forecasts or health advice. The request currently uses demonstration feature values and
        historical 2024 data; it is not a live forecast.
      </aside>
    </main>
  )
}

export default Dashboard
