import type { StationId } from '../services/api'

const STATIONS: { id: StationId; name: string }[] = [
  { id: 'site_113', name: 'Shadipur' },
  { id: 'site_115', name: 'NSIT Dwarka' },
  { id: 'site_117', name: 'ITO' },
]

interface StationSelectorProps {
  selected: StationId
  onSelect: (station: StationId) => void
}

function StationSelector({ selected, onSelect }: StationSelectorProps) {
  return (
    <section className="station-selector" aria-labelledby="station-heading">
      <h2 id="station-heading">Select station</h2>
      <div className="station-options">
        {STATIONS.map((station) => (
          <button
            key={station.id}
            type="button"
            className={selected === station.id ? 'station-button selected' : 'station-button'}
            aria-pressed={selected === station.id}
            onClick={() => onSelect(station.id)}
          >
            {station.name}
          </button>
        ))}
      </div>
    </section>
  )
}

export default StationSelector
