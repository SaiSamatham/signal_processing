# ECG QRS Detection & Beat Classification Pipeline

An end-to-end biomedical signal processing pipeline that loads ECG records from the MIT-BIH Arrhythmia Database, cleans the signal, detects heartbeats using a full implementation of the **Pan-Tompkins algorithm** (including adaptive thresholding and searchback), classifies each beat, computes heart rate / HRV, and exports results to CSV.

Built and validated against two MIT-BIH records with meaningfully different QRS morphology (records **100** and **108**) using `numpy`, `scipy`, `wfdb`, and `pandas`.

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Setup](#setup)
- [Stage-by-Stage Breakdown](#stage-by-stage-breakdown)
- [Understanding the Plots](#understanding-the-plots)
- [Understanding the Terminal Output](#understanding-the-terminal-output)
- [Output: CSV Export](#output-csv-export)
- [Bugs Found & Fixed](#bugs-found--fixed-during-development)
- [Known Limitations / Simplifications](#known-limitations--simplifications)
- [Possible Next Steps](#possible-next-steps)

---

## Pipeline Overview

```
Raw ECG (wfdb)
   │
   ├─► Bandpass filter (0.5–45 Hz)  ──► clean signal for visualization
   │
   └─► QRS-band filter (5–15 Hz)
          │
          ▼
   Differentiate ─► Square ─► Moving-window Integrate
          │
          ▼
   Adaptive thresholding (SPKI / NPKI) + prominence-filtered candidates
          │
          ▼
   Searchback (recover missed beats)
          │
          ▼
   Beat classification (Normal / Premature / Delayed, via R-R interval)
          │
          ▼
   Heart rate + HRV (SDNN)  +  CSV export
```

## Setup

```bash
pip install wfdb scipy numpy matplotlib pandas
```

Records are pulled directly from PhysioNet the first time you run the script (requires an internet connection) and cached locally afterward.

---

## Stage-by-Stage Breakdown

### 1. Load record (`wfdb`)
Loads a specified duration of a MIT-BIH record (e.g. record `'100'` or `'108'`) at its native 360 Hz sampling rate. Channel 0 (**lead MLII** — Modified Limb Lead II) is used, since it's the standard lead most QRS-detection literature and benchmarks are built around, typically giving the clearest, most consistent QRS deflection.

### 2. Bandpass filter (0.5–45 Hz)
A 4th-order Butterworth bandpass filter, applied with `scipy.signal.filtfilt` (zero-phase — no time-shift of peaks). This is the **general-purpose "clean ECG"** signal, used for visualization and for pulling final peak amplitudes:
- **0.5 Hz high-pass edge** removes baseline wander (breathing, electrode drift).
- **45 Hz low-pass edge** removes muscle noise (EMG) and powerline interference, while preserving the full diagnostic content of the P-wave, QRS complex, and T-wave.
- **Butterworth** was chosen specifically for its maximally flat passband — it doesn't distort the relative amplitude/shape of the waveform the way a Chebyshev or elliptic filter's ripple would.

### 3. QRS-band filter (5–15 Hz)
A second, separate Butterworth bandpass, tuned to the frequency range where QRS-complex energy is concentrated (per the original Pan-Tompkins paper). This narrower band actively suppresses P-waves and T-waves, which are lower-frequency and could otherwise be mistaken for beats. This signal is **detection-only** — not meant to visually resemble a normal ECG.

### 4. Differentiate → Square → Integrate
- **Differentiation** (`np.diff`) emphasizes the steep slopes characteristic of a QRS complex.
- **Squaring** makes all values positive and nonlinearly emphasizes large values over small ones, widening the gap between real QRS energy and background noise.
- **Moving-window integration** (150 ms window, via `np.convolve` with an all-ones kernel) smooths the squared signal into one wide, well-defined energy "hump" per heartbeat, rather than several jagged sub-peaks.

### 5. Adaptive thresholding (SPKI / NPKI)
Implements the actual Pan-Tompkins peak-classification logic:
- **SPKI** — running estimate of "what a real signal peak looks like."
- **NPKI** — running estimate of "what noise looks like."
- **Threshold** = `NPKI + 0.25 * (SPKI − NPKI)`, recomputed after every candidate.
- Each candidate peak is compared against the threshold; classified as signal or noise; whichever running estimate matches its classification is nudged via an exponential moving average (`0.125 × new + 0.875 × old`).
- Initialized from the first 2 seconds of the recording (`SPKI = 0.25 × max`, `NPKI = 0.5 × mean`).
- Candidate peaks are found with `scipy.signal.find_peaks`, constrained by:
  - **`distance`** (200 ms) — the heart's physiological refractory period; no two real beats can be closer than this.
  - **`prominence`** (`0.3 × SPKI`) — ensures a candidate genuinely stands out from its local surroundings, not just clears a height/distance bar. This was added specifically to prevent a single QRS complex with a notched/biphasic shape from being counted as two separate beats.

### 6. Searchback
If the gap since the last confirmed beat exceeds **166% of the recent average R-R interval** (average of the last 8 intervals), the algorithm re-scans that specific gap using a **halved threshold**, to recover a real peak that may have been narrowly rejected. The re-search window is trimmed by the 200 ms refractory distance on both ends, to avoid "rediscovering" the ringing tails of the two beats bracketing the gap.

### 7. Beat classification
Each beat (after the first two, which default to "Normal") is labeled by comparing its R-R interval to the running 8-beat average:
- **< 80% of average** → `Premature Beat`
- **> 120% of average** → `Delayed Beat`
- **otherwise** → `Normal Beat`

### 8. Heart rate & HRV
- **Heart rate (bpm)** = `60 / mean(R-R interval in seconds)`
- **HRV (SDNN)** = standard deviation of all R-R intervals, in milliseconds — the simplest standard heart-rate-variability metric.

### 9. Export
Final detected beats are written to `ecg_beat_results.csv` via `pandas`, one row per beat, with columns: `beat_index`, `sample_index`, `time_sec`, `amplitude_mV`, `label`.

---

## Understanding the Plots

The script produces several `matplotlib` figures, in this order:

| # | Plot | What it shows | What to look for |
|---|------|----------------|-------------------|
| 1 | **Raw ECG Signal** | The unprocessed signal straight from `wfdb`. | Visible baseline wander (slow drift) and high-frequency jitter riding on top of the heartbeats — the two problems the bandpass filter is about to solve. |
| 2 | **Filtered ECG Signal (0.5–45 Hz)** | The general-purpose clean signal. | A flat baseline between beats, with the QRS/P/T-wave shape preserved — used later for the final peak-amplitude overlay. |
| 3 | **Filtered QRS Signal (5–15 Hz)** | The detection-only signal. | P-waves and T-waves are largely suppressed; only sharp, isolated QRS spikes remain, each often showing more internal "ringing" than in Plot 2. |
| 4 | **4-panel: QRS-band → Differentiated → Squared → Integrated** | The full Pan-Tompkins transform, stacked so you can trace one heartbeat through each stage. | Each panel should look progressively "cleaner" — from a multi-lobed spike, to a sharp derivative burst, to an isolated positive bump, to one smooth, wide hump per heartbeat in the bottom (integrated) panel. |
| 5 | **Detected R-Peaks on Filtered ECG** | Your final detections (red dots) overlaid on the clean 0.5–45 Hz ECG trace. | Every red dot should land consistently on the same part of each heartbeat. **Note:** on record 108 specifically, the dots land on the *downward* deflection, not an upward spike — this is correct, not a bug (see note below). |
| 6 (diagnostic) | **Zoomed QRS-band + Integrated Signal, around a specific gap** | A zoomed-in look at one specific pair of beats, with vertical markers at their sample locations and the current threshold as a horizontal dashed line. | Used during debugging to confirm whether two "detected peaks" were genuinely separate heartbeats or a single QRS complex being counted twice. |

**Why the red dots sit on a downward spike for record 108:** the shape of the QRS complex isn't universal — it depends on the electrical axis of a given patient's heart relative to where the electrode is placed. Record 108's lead configuration produces an S-wave-dominant (negative) complex rather than the R-wave-dominant (positive) shape seen in record 100. The detector correctly locks onto whichever deflection carries the most energy each beat — confirmed by checking that the signal's 5th/95th percentiles are asymmetric (skewed negative) rather than assuming "peaks are always positive."

---

## Understanding the Terminal Output

Running the full script prints the following, in order:

```
['MLII', 'V1']
```
The two lead names available in this record (channel 0 = MLII is the one actually used).

```
SPKI=0.0080, NPKI=0.0031, Threshold=0.0043
```
The initial adaptive-threshold estimates, computed from the first 2 seconds of the integrated signal, before any beat classification begins.

```
99
```
Number of raw *candidate* peaks found in the integrated signal (after the `distance` + `prominence` constraints, before adaptive signal/noise classification).

```
Searchback Needed: gap of 566 samples between index 10878 and 11444
Recovered 0 peaks: []
```
Printed each time a gap exceeds 166% of the recent average R-R interval. Reports the gap size and the two bracketing beat indices, then reports how many peaks (if any) the lowered-threshold re-search found in that gap. `0` recovered means the algorithm concluded the long gap was a real, naturally longer heartbeat interval rather than a missed beat.

```
87
```
Final total number of detected beats, after adaptive thresholding and any searchback recoveries.

```
Searchback Triggered 1 times, Recovered 0 peaks
```
Summary of how many times the searchback mechanism activated across the whole recording, and how many total beats it successfully recovered.

```
Detected peaks: 87, Beat labels: 87
Counter({'Normal Beat': 76, 'Delayed Beat': 5, 'Premature Beat': 4, 'Normal': 1})
```
Sanity check confirming every detected beat received exactly one classification label, followed by the breakdown of how many beats fell into each category. (`'Normal'` vs `'Normal Beat'` appear as separate entries here due to two slightly different literal strings used for two different edge cases — the very first detected beat, and the second detected beat before any R-R average exists yet.)

```
Average Heart Rate: 57.7 BPM
```
`60 / mean(R-R interval in seconds)` — the average heart rate across the whole recording.

```
Short RR interval: 200.0 ms between detected_peaks index 28 and 29 (sample 10253 -> 10325)
```
Flags any R-R interval under 300 ms — used during debugging to catch cases where a single QRS complex was being counted as two separate beats. One such case remained after the `prominence` fix and was accepted as a minor known outlier.

```
SDNN (Heart Rate Variability): 159.42 ms
```
Standard deviation of all R-R intervals, in milliseconds — the HRV metric. (Note: short-recording SDNN isn't directly comparable to standard 24-hour clinical reference ranges, which are typically longer-duration benchmarks.)

```
[200.0, 472.22, 530.56, ...]
```
The full sorted list of every R-R interval in milliseconds — used during debugging to visually scan for outliers before diagnosing their cause.

---

## Output: CSV Export

`ecg_beat_results.csv` — one row per detected beat:

| Column | Description |
|---|---|
| `beat_index` | Sequential row counter (0, 1, 2, ...) |
| `sample_index` | Raw sample position in the original signal array |
| `time_sec` | Beat timestamp, in seconds from the start of the recording |
| `amplitude_mV` | Signal amplitude (from the 0.5–45 Hz filtered signal) at that beat |
| `label` | `Normal Beat`, `Premature Beat`, or `Delayed Beat` |

---

## Bugs Found & Fixed During Development

These were real issues caught and resolved through debugging and validation, not hypothetical:

1. **Indentation bug (threshold never updated on noise candidates).** The `else` branch handling noise-classified peaks was nested one level too deep, so `NPKI` and the recomputed `threshold` never updated correctly on every iteration — causing the threshold to drift and accept far too many false peaks.
2. **Double-detection from missing `prominence` constraint.** A single QRS complex with a biphasic/notched shape (specific to record 108's morphology) produced one wide hump in the integrated signal with enough internal structure to register as two separate `find_peaks` detections. Diagnosed by zooming into the raw and integrated signals around a specific flagged short R-R interval, and fixed by adding a `prominence` requirement.
3. **Missing edge-case `else` in beat classification.** The second-ever detected beat (before any R-R average existed) silently produced no label at all, causing `beat_labels` to be one shorter than `detected_peaks`. Caught by explicitly comparing array lengths before trusting downstream results.
4. **Stale/cached plot images during debugging.** Several iterations appeared visually identical despite code changes; resolved by printing a concrete count (`len(detected_peaks)`) alongside each plot to confirm a genuinely fresh run before interpreting the image.

---

## Known Limitations / Simplifications

- **Searchback-recovered peaks bypass full bookkeeping.** Recovered peaks are inserted into `detected_peaks`, but their heights don't update `SPKI`, and they don't get their own R-R interval or classification label — an accepted simplification given they're a small minority of total beats in this dataset.
- **One residual short R-R interval (~200 ms)** remains after the `prominence` fix, accepted as a minor outlier rather than fully investigated further.
- **Classification is R-R-interval-only** — it doesn't use QRS morphology (shape), so it can't distinguish, for example, a genuine premature ventricular contraction from a normal beat that simply arrived early for another reason.
- **SDNN over a short (90 s) window** isn't directly comparable to standard 24-hour clinical HRV reference ranges.

---

## Possible Next Steps

- Compare detections against MIT-BIH's own expert reference annotations for a true accuracy score (sensitivity / positive predictivity).
- Refactor stages into reusable functions so the full pipeline can run on any record with one call.
- Add morphology-based features (QRS width, polarity, shape) to beat classification instead of relying on R-R timing alone.
- Extend searchback-recovered peaks to fully participate in SPKI/RR/label bookkeeping.
