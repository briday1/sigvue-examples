#!/usr/bin/env bash

set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${1:-${repository_root}/data}"

if [[ "$data_root" != /* ]]; then
    data_root="$(pwd)/${data_root}"
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
run "${repository_root}/scripts/generate_lte_sigmf.sh" "$data_root"
run python "${repository_root}/scripts/download_radio_astronomy.py" \
    --output "${data_root}/radio-astronomy"

echo
echo "All example data is available under ${data_root}"
