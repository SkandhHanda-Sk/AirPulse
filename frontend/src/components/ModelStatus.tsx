import { CheckCircle, Wifi } from 'lucide-react'
import type { Health } from '../services/api'

interface ModelStatusProps {
  health: Health | null
}

function ModelStatus({ health }: ModelStatusProps) {
  const ready = health?.model_loaded === true

  return (
    <section className="status-grid" aria-label="API and model status">
      <article className="status-card">
        <Wifi aria-hidden="true" size={20} />
        <div>
          <p className="eyebrow">API status</p>
          <strong className={ready ? 'status-ready' : 'status-offline'}>
            {health?.status ?? 'Unavailable'}
          </strong>
        </div>
      </article>
      <article className="status-card">
        <CheckCircle aria-hidden="true" size={20} />
        <div>
          <p className="eyebrow">Model artifact</p>
          <strong className={ready ? 'status-ready' : 'status-offline'}>
            {ready ? 'Loaded' : 'Not loaded'}
          </strong>
          <p className="status-detail">{health?.model_artifact ?? 'Waiting for API'}</p>
        </div>
      </article>
    </section>
  )
}

export default ModelStatus
