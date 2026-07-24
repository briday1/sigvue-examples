# Sigvue Examples

[![Test examples](https://github.com/briday1/sigvue-examples/actions/workflows/test.yml/badge.svg)](https://github.com/briday1/sigvue-examples/actions/workflows/test.yml)

External, file-backed examples for [Sigvue](https://github.com/briday1/sigvue),
covering signal recordings, annotated physiology, and weather radar.
The repository keeps domain-specific analysis and presentation code small by
building on a packaged layer of reusable concrete plugin components.

Each domain pipeline stays focused, while shared plugin implementations are
written once:

```text
src/sigvue_examples/
├── comms/
│   ├── analysis.py
│   ├── plots.py
│   ├── models.py
│   └── workspace.py
├── events/
├── ecg/
├── weather_radar/
├── waterfall/
├── radar/
│   ├── workspace.py
│   ├── source.py
│   ├── delivery.py
│   ├── analysis.py
│   ├── models.py
│   ├── processing.py
│   ├── plots.py
│   ├── presentation.py
│   └── capabilities.py
├── plugins/
│   ├── lifecycle.py
│   ├── discovery.py
│   ├── plotly.py
│   ├── nexrad/
│   ├── wfdb/
│   └── sigmf/
│       ├── recording.py
│       ├── source.py
│       ├── delivery.py
│       ├── annotations.py
│       ├── exports.py
│       └── writer.py
└── style/
    └── plotly.py
```

Every `workspace.py` is the small framework boundary: it assembles a
`Workspace` from reusable infrastructure and the pipeline's ordinary analysis
and presentation functions. Framework-neutral configuration parsing, byte
formatting, and atomic download utilities remain in `sigvue.helpers`. Concrete
components that implement `sigvue.plugin` contracts live with these examples
in `sigvue_examples.plugins`, so they can evolve and ship as a reusable plugin
layer without becoming framework internals.

The `sigvue_examples.plugins.sigmf` package supplies drop-in discovery, ranged
reading, window delivery, power overviews, annotation persistence, JSON/MAT
export, and fixture writing. The sibling WFDB and NEXRAD packages provide the
same kind of copyable format boundary for annotated physiology and weather-radar
products. The parent `plugins` package supplies
callable lifecycle adapters, standard discovery columns, and exact Plotly
annotation-region rendering.

For example, the communications and waterfall workspaces compose
`sigmf_source`, `WindowedSigMFDelivery`, `CallableAnalysis`,
`CallablePresentation`, and `SigMFExporter` directly. Their modules only retain
the settings, calculations, plots, and UI choices that are specific to that
domain. The event example similarly wraps its custom JSON selection functions
with the callable lifecycle adapters instead of defining one-method framework
classes. `WorkspaceConfig` gives every workspace the same typed, profile-relative
path handling.

Domain packages use relative imports from `..plugins` and `..plugins.sigmf`;
scripts and external consumers use `sigvue_examples.plugins`. This keeps the
concrete source, delivery, annotation, and export implementations independent
of any one domain while avoiding parallel copies across domains in this
package. The smaller examples bundled with the Sigvue repository carry the
same public helper surface under `example_pipelines.plugins`; that copy is
deliberately local and copyable, while this one is installed as part of the
standalone examples distribution.

Capability-enabled workspaces write standard SigMF sample start/count, comment,
generator, and UUID annotation fields. Waterfall workspaces also read and write
standard lower/upper RF-frequency edges, populate editable bounds from the
visible Plotly axes, and show in-view annotations as hoverable regions. The
packaged exporter serializes the current buffer or full recording as JSON or MAT.
If a workspace does not configure a capability, Sigvue omits its menu.

## Examples

- **Digital Communications** — choose a synthetic QPSK or 16-QAM SigMF recording, then drag or resize a short interval; both use a constellation tab followed by an eye-diagram tab.
- **Annotated ECG** — choose among MIT-BIH records 100, 101, 200, and 207, then inspect both native leads alongside cardiologist beat annotations, exact RR intervals, morphology summaries with the arithmetic mean kept above every beat trace, and record metadata.
- **Acoustic Event Review** — navigate irregular markers and display waveform and spectrum products already stored in a JSON results file; the workspace performs no raw-audio processing.
- **Radio Astronomy RFI Survey** — inspect real SigMF recordings from an Allen Telescope Array site survey with the same small, reusable windowed waterfall pipeline.
- **LTE Recordings** — choose the 806 MHz downlink or 847 MHz uplink dataset and inspect a selected interval with the bundled example's average-spectrum and waterfall presentation.
- **LFM SigMF View** — choose the original 10 MHz single-return collection or a 2 MHz collection with three delayed/Doppler-shifted returns; both use the same live-tail, historical-seek, and calibration interface. Analysis stays at full slow-time, fast-time, and frequency resolution while a separate Raster rendering box controls only the browser image resolution and exact block statistic.
- **Weather Radar** — choose the TLX or FDR radar and step through 141 real NOAA NEXRAD Level III super-resolution base-reflectivity scans spanning a dense two-hour storm window using segmented previous/next time navigation. The plan-position display offers a visual eleven-option colormap picker—including the custom NEXRAD scale—alongside a native-gate distribution summary and exact product metadata.

Every workspace is backed by files, but generated data is not committed.

The interactive Plotly waterfall keeps box selection and zooming. Its Details panel
groups raster width, height, and max/mean/median aggregation in a dedicated settings
box. Every source cell contributes to the displayed raster; analysis products and the
average PSD stay at full resolution. When pan or zoom settles, Sigvue requests a new
raster of the visible source region so detail increases progressively without sending
the full heatmap matrix to the browser.

The generator and downloader scripts import the packaged plugin helpers. Before
running any command below, install this repository in editable mode (after
installing Sigvue):

```bash
python -m pip install -e ../sigvue
python -m pip install -e .
```

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

Download the ECG and dense two-hour weather-radar collections—about 58 MB
combined—with:

```bash
python scripts/download_mit_bih_ecg.py
python scripts/download_weather_radar.py
```

The ECG downloader pins records 100, 101, 200, and 207 from the
[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
and verifies every header, two-lead format-212 waveform, and reference
annotation file. Pass `--records 100 200` to download only a subset. The
dataset is distributed under the Open Data Commons Attribution License; cite
PhysioNet and DOI
[10.13026/C2F305](https://doi.org/10.13026/C2F305).

The weather-radar downloader pins every available TLX and FDR N0B
base-reflectivity scan from 03:00 through 04:59 UTC on 2024-05-20: 72 TLX
scans and 69 FDR scans. Eight concurrent downloads keep the 141-file
collection quick to materialize, while the bundled SHA-256 manifest verifies
every file from the
[NOAA NEXRAD Open Data archive](https://registry.opendata.aws/noaa-nexrad/).
Pass `--radars TLX` or `--radars FDR` to download only one sequence.
Use `--workers 1` for sequential downloading.
NOAA data disseminated through NODD is open to public use, with attribution
requested.

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
python scripts/download_lte_sigmf.py
```

Pass `--output /path/to/data` to override the default `data` root. The Python
downloader uses Sigvue's framework-neutral atomic download helper to verify each file's
expected size and SHA-256 digest, retries transient failures, and preserves
existing SigMF metadata that may contain local annotations. Each recording is
placed in its workspace-specific subdirectory.

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

Sigvue 2026.38 or newer supplies the public plugin contracts plus the
framework-neutral configuration, formatting, and download utilities. This
package supplies the concrete reusable plugin implementations used by the
examples.

Install the browser framework and this repository in one environment, then
launch the included profile:

```bash
python -m pip install -e ../sigvue
python -m pip install -e .
sigvue --config browser.toml
```

The browser itself can load this repository directly through `browser.toml`
without an editable install and watches workspace code for changes. The
generator and downloader scripts still require the editable install described
above.

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
