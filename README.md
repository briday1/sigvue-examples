# Scientific Workspace Browser Examples

External, file-backed examples for [Scientific Workspace Browser](https://github.com/briday1/Scientific-Workspace-Browser). This repository deliberately keeps the small examples independent from the browser framework.

## Examples

- **QPSK Windowed Analysis** — drag or resize a short interval over a received-power overview; the selected data drives a constellation tab followed by an eye-diagram tab.
- **Acoustic Event Review** — navigate irregular markers and display waveform and spectrum products already stored in a JSON results file; the workspace performs no raw-audio processing.
- **Multi-Tone Seek** — play or seek through a time-varying tone recording; each selected buffer is shown as an average PSD above a waterfall.
- **LFM Live** — the calibrated four-channel LFM workflow using live-tail playback with historical seeking and buffered reads.
- **LFM Static** — the same calibrated LFM analysis receiving the complete OTA files with no playback controls.

Every workspace is backed by files, but generated data is not committed. Generate the compact recordings and precomputed acoustic results with:

```bash
python scripts/generate_minimal_sigmf.py
python scripts/generate_segmented_results.py
```

The LFM collection is much larger and stays local. Generate it with:

```bash
python scripts/generate_lfm_collection.py
```

## Run

Install the browser framework and this repository in one environment, then launch the included profile:

```bash
python -m pip install -e ../Scientific-Workspace-Browser
python -m pip install -e .
workspace-browser --config browser.toml
```

During development, installation of this repository is optional because `browser.toml` points directly at its root. The browser watches the repository and reloads changed workspace code when the page is refreshed.

## Test

```bash
PYTHONPATH=src:../Scientific-Workspace-Browser/src python -m unittest discover -s tests -q
```
