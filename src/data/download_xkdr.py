import csv
import io
import calendar
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ENV_FILE = PROJECT_ROOT / ".env"

# Dataset configuration
BASE_URL = "https://airquality.xkdr.org/v1/measurements"

STATIONS = ["site_113", "site_115", "site_117"]
PARAMETER = "PM2.5"
YEAR = 2024

EXPECTED_COLUMNS = [
    "station_id",
    "parameter_name",
    "unit",
    "period_start",
    "mean",
    "min",
    "max",
    "n",
]


def load_api_token():
    """Read XKDR token from the local .env file."""

    if not ENV_FILE.exists():
        raise FileNotFoundError(".env file not found.")

    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith("XKDR_API_TOKEN="):
            token = line.split("=", 1)[1].strip()

            if token:
                return token

    raise ValueError("XKDR_API_TOKEN is missing from .env.")


def download_month(token, month):
    """Download one month of hourly PM2.5 measurements."""

    last_day = calendar.monthrange(YEAR, month)[1]

    start_date = f"{YEAR}-{month:02d}-01"
    end_date = f"{YEAR}-{month:02d}-{last_day:02d}"

    params = [
        ("station", station) for station in STATIONS
    ]

    params.extend([
        ("parameter", PARAMETER),
        ("start", start_date),
        ("end", end_date),
        ("agg", "hourly"),
        ("format", "csv"),
    ])

    url = BASE_URL + "?" + urlencode(params)

    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "curl/8.10.1",
        },
    )

    with urlopen(request, timeout=60) as response:

        if response.status != 200:
            raise RuntimeError(
                f"HTTP error for month {month}: {response.status}"
            )

        content_type = response.headers.get("Content-Type", "")

        if "text/csv" not in content_type:
            raise RuntimeError(
                f"Unexpected response format: {content_type}"
            )

        row_count = response.headers.get("X-Row-Count")
        truncated = response.headers.get("X-Truncated")

        if truncated != "false":
            raise RuntimeError(
                f"Month {month} may be truncated: {truncated}"
            )

        if row_count is None:
            raise RuntimeError("Missing X-Row-Count header.")

        raw_bytes = response.read()

    # Validate CSV structure before saving.
    text = raw_bytes.decode("utf-8-sig")

    reader = csv.DictReader(io.StringIO(text))

    if reader.fieldnames != EXPECTED_COLUMNS:
        raise RuntimeError(
            f"Unexpected CSV columns: {reader.fieldnames}"
        )

    rows = list(reader)

    if len(rows) != int(row_count):
        raise RuntimeError(
            f"Row-count mismatch for month {month}."
        )

    # Verify station distribution.
    station_counts = {station: 0 for station in STATIONS}

    for row in rows:
        station_id = row["station_id"]

        if station_id not in station_counts:
            raise RuntimeError(
                f"Unexpected station ID: {station_id}"
            )

        station_counts[station_id] += 1

    return raw_bytes, len(rows), station_counts


def main():

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Avoid accidentally overwriting existing raw files.
    output_files = [
        RAW_DIR / f"xkdr_delhi_pm25_hourly_{YEAR}_{month:02d}.csv"
        for month in range(1, 13)
    ]

    existing = [path.name for path in output_files if path.exists()]

    if existing:
        raise FileExistsError(
            "Some output files already exist. "
            "No files were overwritten:\n" + "\n".join(existing)
        )

    token = load_api_token()

    total_records = 0

    print("\nAirPulse - XKDR Dataset Acquisition")
    print("-----------------------------------")

    for month, output_path in enumerate(output_files, start=1):

        print(f"\nDownloading {YEAR}-{month:02d}...")

        raw_bytes, count, station_counts = download_month(
            token, month
        )

        # Preserve original CSV response bytes.
        temporary_path = output_path.with_suffix(".csv.part")
        temporary_path.write_bytes(raw_bytes)
        temporary_path.replace(output_path)

        total_records += count

        print(f"Saved: {output_path.name}")
        print(f"Records: {count}")

        for station, station_count in station_counts.items():
            print(f"  {station}: {station_count}")

    print("\n-----------------------------------")
    print("Acquisition completed.")
    print(f"Total records: {total_records}")
    print(f"Files saved: {len(output_files)}")
    print(f"Location: {RAW_DIR}")


if __name__ == "__main__":
    main()