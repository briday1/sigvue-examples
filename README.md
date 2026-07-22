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
│   ├── domain.py
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
`plots`, and optional `capabilities` modules. `domain.py` keeps the typed domain
models and lower-level helpers that those focused modules expose. Cross-pipeline code is limited to
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

`io/sigmf/capabilities.py` is the explicit bridge from that neutral I/O to Sigvue's
optional `Annotator` and `Exporter` base objects. These workspaces write standard
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
- **LFM Live View** — choose the original 10 MHz single-return collection or a 2 MHz collection with three delayed/Doppler-shifted returns; both use the same live-tail, historical-seek, and calibration interface. Analysis stays at full slow-time, fast-time, and frequency resolution while a separate Raster rendering box controls only the browser image resolution and exact block statistic.
- **Radar Data · Generic Waterfall** — point the reusable waterfall workspace at those same SigMF collection manifests, then choose any calibration, terminated-noise, or OTA channel from a dropdown without adding radar-specific plotting code.

Every workspace is backed by files, but generated data is not committed.

The reusable waterfall workspace also demonstrates background batch actions. From
either its workspace card or discovered-item table, build exact first-2-ms interactive
waterfall reports without opening a data view, export an item's SigMF metadata, or
compile all discovered recordings into one downloadable report archive. Yellow and
green launcher states show running and completed jobs.

The same jobs can be inspected and dispatched from a terminal:

```bash
sigvue batch --config browser.toml --list
sigvue batch --config browser.toml \
  --workspace lte-recordings --action report-all --output reports
```

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
python -m pip install -e ".[test,release]"
PYTHONPATH=src:../sigvue/src python -m pytest -q tests
```

The repository workflow installs Sigvue from its `main` branch, installs this
examples package without replacing that framework checkout, and runs the same
suite on every push and pull request. This keeps the example implementations
checked against the current public plugin contract.
