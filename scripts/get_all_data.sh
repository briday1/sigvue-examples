#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${1:-${repository_root}/data}"

if [[ "$data_root" != /* ]]; then
    data_root="$(pwd)/${data_root}"
fi

if ! python -c 'from sigvue_examples.plugins.sigmf import write_sigmf_recording' \
    >/dev/null 2>&1; then
    echo "error: the Sigvue Examples package is not installed in this Python environment." >&2
    echo "Install it first with:" >&2
    echo "  python -m pip install -e \"${repository_root}\"" >&2
    exit 1
fi

run() {
    echo
    echo "==> $*"
    "$@"
}

run python "${repository_root}/scripts/generate_comms.py" \
    --output "${data_root}/comms"
run python "${repository_root}/scripts/generate_segmented_results.py" \
    --output "${data_root}/acoustic-events-segmented/acoustic-events.json"
run python "${repository_root}/scripts/generate_lfm_collection.py" \
    --output "${data_root}/lfm-sigmf"
run python "${repository_root}/scripts/download_mit_bih_ecg.py" \
    --output "${data_root}/ecg/mit-bih"
run python "${repository_root}/scripts/download_weather_radar.py" \
    --output "${data_root}/weather-radar"
run python "${repository_root}/scripts/download_lte_sigmf.py" \
    --output "$data_root"
run python "${repository_root}/scripts/download_radio_astronomy.py" \
    --output "${data_root}/radio-astronomy"

echo
echo "All example data is available under ${data_root}"
