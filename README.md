# Sigvue Examples

[![Test examples](https://github.com/briday1/sigvue-examples/actions/workflows/test.yml/badge.svg)](https://github.com/briday1/sigvue-examples/actions/workflows/test.yml)

External, file-backed examples for [Sigvue](https://github.com/briday1/sigvue). This repository deliberately keeps the small examples independent from the browser framework.

Each pipeline is a directory that can be copied as a unit:

```text
src/sigvue_examples/
├── comms/
│   ├── source.py
│   ├── delivery.py
│   ├── analysis.py
│   ├── plots.py
│   ├── models.py
│   └── workspace.py
├── events/
├── waterfall/
├── radar/
├── io/sigmf/
│   ├── recording.py
│   └── capabilities.py
└── style/
    └── plotly.py
```

Every `workspace.py` is the small framework boundary: it assembles a
`Workspace` from that pipeline's `source`, `delivery`, `analysis`,
`plots`, and optional `capabilities` modules. The copyable communications and
waterfall examples keep their typed lifecycle values in a small `models.py`.
Cross-pipeline code is limited to
`io/sigmf` for ranged file access and SigMF capabilities, and
`style` for Plotly appearance. Neither shared package chooses tabs, playback
mode, parameters, or analysis behavior.

Each domain delivery explicitly subclasses the framework's typed `Delivery`—for
example, `Delivery[SigMFRecording, WaterfallWindow]`. The first type is what
the source opens, and the second is the value passed into workspace configuration
and processing. This makes the source/delivery/processing boundary visible
without putting browser semantics into the shared SigMF I/O module. Workspace
assembly uses framework objects for every behavioral slot: `Workspace`,
`Source`, optional `Delivery`, `Analysis`, `Presentation`, optional `Annotator`,
and optional `Exporter`. There are no alternate constructor names or
structurally inferred lifecycle objects.

`io/sigmf/capabilities.py` is an optional bridge from the shared I/O to Sigvue's
optional `Annotator` and `Exporter` base objects. Capability-enabled workspaces write standard
SigMF sample start/count, comment, generator, and UUID annotation fields. Waterfall
workspaces also read and write standard lower/upper RF-frequency edges, populate editable
bounds from the visible Plotly axes, and show in-view annotations as hoverable regions.
They also let the plugin serialize the current buffer or full recording as JSON
or MAT. If a workspace does not pass either capability, Sigvue shows neither
menu.

## Examples

- **Digital Communications** — choose a synthetic QPSK or 16-QAM SigMF recording, then drag or resize a short interval; both use a constellation tab followed by an eye-diagram tab.
- **Acoustic Event Review** — navigate irregular markers and display waveform and spectrum products already stored in a JSON results file; the workspace performs no raw-audio processing.
- **Radio Astronomy RFI Survey** — inspect real SigMF recordings from an Allen Telescope Array site survey with the same small, reusable windowed waterfall pipeline.
- **LTE Recordings** — choose the 806 MHz downlink or 847 MHz uplink dataset and inspect a selected interval with the bundled example's average-spectrum and waterfall presentation.
- **LFM Live View** — choose the original 10 MHz single-return collection or a 2 MHz collection with three delayed/Doppler-shifted returns; both use the same live-tail, historical-seek, and calibration interface. Analysis stays at full slow-time, fast-time, and frequency resolution while a separate Raster rendering box controls only the browser image resolution and exact block statistic.

Every workspace is backed by files, but generated data is not committed.

The interactive Plotly waterfall keeps box selection and zooming. Its Details panel
groups raster width, height, and max/mean/median aggregation in a dedicated settings
box. Every source cell contributes to the displayed raster; analysis products and the
average PSD stay at full resolution. When pan or zoom settles, Sigvue requests a new
raster of the visible source region so detail increases progressively without sending
the full heatmap matrix to the browser.

To generate or download every example dataset in one command, including the
roughly 3.4 GB radio-astronomy survey and the large LTE recordings, run:

```bash
./scripts/get_all_data.sh
```

Pass a directory as the first argument to override the default `data` root.
The script calls each of the individual generators and downloaders described
below and stops immediately if any one of them fails.

Generate only the compact recordings and precomputed acoustic results with:

```bash
python scripts/generate_comms.py
python scripts/generate_segmented_results.py
```

The LFM SigMF workspace reads both field captures and generated calibrated
collections from `data/lfm-sigmf`. Generate the four-channel 10 MHz, sixteen-channel
10 MHz, and four-channel 2 MHz multi-target collections together with:

```bash
python scripts/generate_lfm_collection.py
```

Pass `--profile 10mhz`, `--profile 10mhz-16ch`, or `--profile 2mhz` to generate
only one collection. The sixteen-channel profile is displayed as a 4×4 plot grid.
The generated manifests use standard SigMF `core:streams` entries plus the
`lfm` extension metadata needed to identify calibration, terminated-noise, and
OTA roles.

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

All included workspaces set `lazy_views=True`, so only the selected tab and
view-switcher branch is generated. A workspace that should perform its expensive
open/process work once can omit that option (the eager default) or explicitly set
`lazy_views=False`; all of its views are then created in the initial request and
later view selection stays in the browser.

## Test

```bash
python -m pip install -e ".[test,release]"
PYTHONPATH=src:../sigvue/src python -m pytest -q tests
```

The repository workflow installs Sigvue from its `main` branch, installs this
examples package without replacing that framework checkout, and runs the same
suite on every push and pull request. This keeps the example implementations
checked against the current public plugin contract.
