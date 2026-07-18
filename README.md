# Sigvue Examples

External, file-backed examples for [Sigvue](https://github.com/briday1/sigvue). This repository deliberately keeps the small examples independent from the browser framework.

The package is organized by analysis domain: `waterfall.py` contains the LTE
and radio-astronomy spectrum/waterfall workflows, `comms.py` contains QPSK and
16-QAM, and `radar_collection.py` contains the shared LFM time/frequency
workflow. `events.py` retains the independent acoustic result viewer.

The shared modules do not define browser behavior. `sigmf.py` only parses SigMF
metadata and reads requested sample ranges, while `style.py` only applies Plotly
styling to figures. Each domain module owns its browser source, delivery mode,
parameters, analysis, tabs, and plots. This keeps the dependency direction
simple: a workspace may use the SigMF reader, but the SigMF reader never knows
about a workspace.

Each domain delivery explicitly implements the framework's typed interface—for
example, `DataDelivery[SigMFRecording, WaterfallWindow]`. The first type is what
the source opens, the second is exactly what the analysis function receives.
This makes the source/delivery/analysis boundary visible without putting browser
semantics into the shared SigMF I/O module.

`capabilities.py` is the explicit bridge from that neutral I/O to Sigvue's
optional annotation and export contracts. These workspaces write standard
SigMF sample start/count, comment, generator, and UUID annotation fields. Waterfall
workspaces also read and write standard lower/upper RF-frequency edges, populate editable
bounds from the visible Plotly axes, and show in-view annotations as hoverable regions.
They also let the plugin serialize the current buffer or full recording as JSON
or MAT. If a workspace does not pass either capability, Sigvue shows neither
menu.

## Examples

- **Digital Communications** — choose a synthetic QPSK or 16-QAM SigMF recording, then drag or resize a short interval; both use a constellation tab followed by an eye-diagram tab.
- **Acoustic Event Review** — navigate irregular markers and display waveform and spectrum products already stored in a JSON results file; the workspace performs no raw-audio processing.
- **Radio Astronomy RFI Survey** — inspect real SigMF recordings from an Allen Telescope Array site survey using a sparse full-record power overview and a windowed spectrum/waterfall view.
- **LTE Recordings** — choose the 806 MHz downlink or 847 MHz uplink dataset, then drag a window over its sliding-median power overview and inspect the selected time-frequency region.
- **LFM Live View** — choose the original 10 MHz single-return collection or a 2 MHz collection with three delayed/Doppler-shifted returns; both use the same live-tail, historical-seek, and calibration interface.

Every workspace is backed by files, but generated data is not committed. Generate the compact recordings and precomputed acoustic results with:

```bash
python scripts/generate_minimal_sigmf.py
python scripts/generate_segmented_results.py
```

Both LFM collections stay local. Generate the original 10 MHz and newer 2 MHz
multi-target recordings together with:

```bash
python scripts/generate_lfm_collection.py
```

Pass `--profile 10mhz` or `--profile 2mhz` to generate only one collection.

Download the LTE downlink and uplink SigMF metadata and recordings from Daniel Estévez's
[LTE data directory](http://nas.destevez.net/~daniel/LTE/) with:

```bash
./scripts/generate_lte_sigmf.sh
```

Pass a directory as the first argument to override the default `data` root. Each
recording is placed in its workspace-specific subdirectory.

Download the six real SigMF archives from the
[Quick RFI Survey at the Allen Telescope Array](https://zenodo.org/records/8242048)
(about 3.4 GB total), verify their published MD5 checksums, and unpack them with:

```bash
python scripts/download_radio_astronomy.py
```

Use `--first` for a single roughly 559 MB recording, `--list` to inspect the
remote manifest without downloading, or `--keep-archives` to retain the source
archives after extraction. Downloaded and generated files remain under the
ignored `data/` directory and are never packaged.

The survey dataset was created by Daniel Estévez and is used under CC BY 4.0;
cite [DOI 10.5281/zenodo.8242048](https://doi.org/10.5281/zenodo.8242048) when
redistributing or publishing results derived from it.

## Run

Install the browser framework and this repository in one environment, then launch the included profile:

```bash
python -m pip install -e ../sigvue
python -m pip install -e .
sigvue --config browser.toml
```

During development, installation of this repository is optional because `browser.toml` points directly at its root. The browser watches the repository and reloads changed workspace code when the page is refreshed.

## Test

```bash
PYTHONPATH=src:../sigvue/src python -m unittest discover -s tests -q
```
