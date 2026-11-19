import { Activity } from 'lucide-react'
import type { StationId } from '../services/api'

const STATION_NAMES: Record<StationId, string> = {
  site_113: 'Shadipur',
  site_115: 'NSIT Dwarka',
  site_117: 'ITO',
}

interface HistoricalPanelProps {
  station: StationId
}

function HistoricalPanel({ station }: HistoricalPanelProps) {
  return (
    <section className="panel" aria-labelledby="history-heading">
      <div className="panel-heading">
        <Activity aria-hidden="true" size={20} />
        <h2 id="history-heading">Historical PM2.5 trend</h2>
      </div>
      <div className="chart-placeholder">
        Historical chart for {STATION_NAMES[station]} (2024 data)
      </div>
      <p className="status-detail">
        Historical demonstration data, not live readings.
      </p>
    </section>
  )
}

export default HistoricalPanel
