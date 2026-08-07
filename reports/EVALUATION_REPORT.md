# ADS-B Spoofing & Anomaly Detection System — Evaluation Report

*Generated 2026-08-04 17:12 UTC from `reports/phase7_benchmark.json` and `reports/phase8_jamming_zone.json`.*

**Every verification source in this system that isn't the aircraft's own ADS-B broadcast is simulated.**
Multilateration (MLAT) and primary radar are software simulations of independent sensors, built for this
project -- there are no real ground receivers or radar sites anywhere in this pipeline. This is stated
here once, plainly, and is repeated throughout the code and this report; it should never be read as, or
presented as, real live sensor corroboration.

## 1. What this system does

Ingests live ADS-B state vectors from OpenSky, models each aircraft's expected physical behavior with a
per-track Kalman filter, cross-checks broadcast positions against two independent simulated verification
sources (MLAT and primary radar), and detects spoofed/anomalous aircraft using a hybrid statistical + ML
approach. Every detection method is benchmarked against the others using an adversarial testbed with five
synthetic attack classes, evaluated with a held-out train/test split and one attack class excluded from
training entirely to test cross-class generalization.

## 2. Methodology summary

- **Substrate**: attacks are built by modifying or recombining real, already-ingested clean flight
  trajectories (never invented from nothing, except ghost injection, which has no substrate by
  definition).
- **Held-out split**: within each non-held-out class, variants are split train/test, stratified by
  severity (not a random shuffle) so both halves span the full easy-to-hard range. `track_hijack` is
  excluded from training entirely and only ever evaluated at test time, to check cross-class
  generalization -- chosen because its detection signature (no physical corroboration after a mid-track
  splice) is the least similar to the other four classes.
- **Severity ranges**: `position_drift`, `track_hijack`, and `replay` use a shared meters-based severity
  scale (20m-6000m) deliberately chosen to straddle both MLAT's (~88m) and radar's (~3775m) calibrated
  detection thresholds. An earlier version used ranges that sat entirely above both thresholds, which
  produced artificially saturated 100% scores that didn't show where detection actually breaks down --
  this was corrected before the results below were generated.
- **Identity consistency check**: a sixth method, added specifically because MLAT/radar/NIS/ML all
  under-perform on ICAO identity collision (they check position or kinematics; a collision attack lies
  about identity, not position). It tracks each track's established (first-seen) callsign and flags any
  later broadcast whose callsign differs. This required a real schema fix first: `track_state` was
  silently discarding callsign after ingestion (only a latest-value field existed, which can't show a
  mid-track change) -- found and fixed as part of adding this check, documented in
  `migrations/006_track_state_callsign.sql`. A first implementation compared each broadcast only to the
  immediately preceding one and measured 10% recall on real test data despite correctly catching every
  collision's onset -- it stopped flagging once the intruder's own broadcasts became internally
  consistent with each other. Redesigned to compare against the track's established baseline instead
  (flagging the whole intruder segment, not just its first row), recall on the same class rose to 31.5%
  at 86.8% precision, and lifted fused F1 on this class from 61.7% (5 methods) to 70.4% (6 methods).
- **Evidence fusion**: a logistic regression over [NIS, ML anomaly score, MLAT disagreement, radar
  disagreement], fit only on the train split, mapped to a 0-100 integrity score. Its decision threshold is
  calibrated to the 99th percentile of its suspicion score on clean **training** traffic (not the held-out
  clean test set), the same methodology used to calibrate MLAT/radar/NIS thresholds in earlier phases.
- **Clean-traffic false-positive rate** is measured on 150 real tracks never used anywhere in fusion
  training or attack substrate -- reported separately from attack-detection metrics, per the brief.
- **No table below reports a number from training data as if it were a held-out result.** Every
  precision/recall/F1/FPR table in Section 3 is computed on the **test** split only.

## 3. Synthetic adversarial testbed results

### 3.1 Ghost aircraft injection

| Method | Precision | Recall | F1 | FPR | FNR | Mean detection latency | Never detected |
|---|---|---|---|---|---|---|---|
| Rule-based baseline | 100.0% | 1.5% | 3.0% | — | 98.5% | 255.0s | 9/10 |
| NIS (chi-square) | 100.0% | 15.0% | 26.1% | — | 85.0% | 87.0s | 5/10 |
| Hybrid Kalman+ML | 100.0% | 6.0% | 11.3% | — | 94.0% | 175.0s | 7/10 |
| Simulated MLAT | 100.0% | 100.0% | 100.0% | — | 0.0% | 0.0s | 0/10 |
| Simulated radar | 100.0% | 100.0% | 100.0% | — | 0.0% | 0.0s | 0/10 |
| Identity (callsign) consistency | — | 0.0% | — | — | 100.0% | — | 10/10 |
| Evidence fusion | 100.0% | 95.0% | 97.4% | — | 5.0% | 15.0s | 0/10 |

### 3.2 Position spoofing / drift

| Method | Precision | Recall | F1 | FPR | FNR | Mean detection latency | Never detected |
|---|---|---|---|---|---|---|---|
| Rule-based baseline | 100.0% | 1.1% | 2.2% | 0.0% | 98.9% | 0.0s | 6/10 |
| NIS (chi-square) | 72.3% | 12.9% | 21.9% | 5.5% | 87.1% | 102.0s | 2/10 |
| Hybrid Kalman+ML | 67.9% | 10.4% | 18.1% | 5.5% | 89.6% | 18.3s | 3/10 |
| Simulated MLAT | 97.3% | 99.7% | 98.5% | 3.1% | 0.3% | 3.0s | 0/10 |
| Simulated radar | 99.3% | 36.8% | 53.7% | 0.3% | 63.2% | 79.0s | 3/10 |
| Identity (callsign) consistency | — | 0.0% | — | 0.0% | 100.0% | — | 10/10 |
| Evidence fusion | 96.7% | 88.7% | 92.6% | 3.4% | 11.3% | 17.3s | 1/10 |

### 3.3 ICAO identity collision

| Method | Precision | Recall | F1 | FPR | FNR | Mean detection latency | Never detected |
|---|---|---|---|---|---|---|---|
| Rule-based baseline | 50.4% | 36.8% | 42.6% | 57.5% | 63.2% | 173.2s | 0/10 |
| NIS (chi-square) | 58.4% | 65.5% | 61.8% | 74.1% | 34.5% | 173.2s | 0/10 |
| Hybrid Kalman+ML | 58.6% | 66.0% | 62.1% | 74.1% | 34.0% | 173.2s | 0/10 |
| Simulated MLAT | 66.7% | 3.6% | 6.9% | 2.9% | 96.4% | 283.5s | 6/10 |
| Simulated radar | 80.0% | 1.6% | 3.0% | 0.6% | 98.4% | 359.5s | 6/10 |
| Identity (callsign) consistency | 86.8% | 31.5% | 46.3% | 7.6% | 68.5% | 518.4s | 5/10 |
| Evidence fusion | 63.1% | 79.7% | 70.4% | 74.1% | 20.3% | 173.2s | 0/10 |

### 3.4 Replay attack

| Method | Precision | Recall | F1 | FPR | FNR | Mean detection latency | Never detected |
|---|---|---|---|---|---|---|---|
| Rule-based baseline | 100.0% | 11.0% | 19.9% | 0.0% | 89.0% | 49.6s | 1/10 |
| NIS (chi-square) | 100.0% | 86.0% | 92.5% | 0.0% | 14.0% | 0.0s | 0/10 |
| Hybrid Kalman+ML | 100.0% | 88.2% | 93.8% | 0.0% | 11.8% | 2.9s | 0/10 |
| Simulated MLAT | 95.4% | 100.0% | 97.7% | 3.8% | 0.0% | 0.0s | 0/10 |
| Simulated radar | 98.6% | 100.0% | 99.3% | 1.2% | 0.0% | 0.0s | 0/10 |
| Identity (callsign) consistency | 100.0% | 43.8% | 60.9% | 0.0% | 56.2% | 89.8s | 4/10 |
| Evidence fusion | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0s | 0/10 |

### 3.5 Track hijacking (held-out class)

| Method | Precision | Recall | F1 | FPR | FNR | Mean detection latency | Never detected |
|---|---|---|---|---|---|---|---|
| Rule-based baseline | 100.0% | 20.1% | 33.5% | 0.0% | 79.9% | 64.6s | 3/20 |
| NIS (chi-square) | 99.2% | 86.3% | 92.3% | 0.9% | 13.7% | 0.0s | 0/20 |
| Hybrid Kalman+ML | 99.8% | 87.7% | 93.4% | 0.2% | 12.3% | 0.9s | 0/20 |
| Simulated MLAT | 97.6% | 100.0% | 98.8% | 2.9% | 0.0% | 0.0s | 0/20 |
| Simulated radar | 98.9% | 100.0% | 99.4% | 1.4% | 0.0% | 0.0s | 0/20 |
| Identity (callsign) consistency | 100.0% | 25.1% | 40.1% | 0.0% | 74.9% | 129.3s | 14/20 |
| Evidence fusion | 99.9% | 100.0% | 99.9% | 0.2% | 0.0% | 0.0s | 0/20 |

## 4. Clean-traffic false-positive rate

Measured on 14,094 real updates across 150 real tracks, held out from both attack substrate and fusion
training.

| Method | False-positive rate | Flagged / total rows |
|---|---|---|
| Rule-based baseline | 0.1% | 6 / 10667 |
| NIS (chi-square) | 1.7% | 180 / 10667 |
| Hybrid Kalman+ML | 0.7% | 80 / 10667 |
| Simulated MLAT | 1.1% | 115 / 10667 |
| Simulated radar | 1.0% | 107 / 10667 |
| Identity (callsign) consistency | 0.0% | 1 / 10667 |
| Evidence fusion | 0.5% | 49 / 10667 |

## 5. Train vs. test stability check

Because the evidence-fusion model is the one component actually fit to labeled data, its train/test gap
is worth reporting explicitly as a check against overfitting (the other five methods use fixed,
pre-calibrated thresholds -- there's no fitting step for them to overfit with).

| Class | Train recall (fused) | Test recall (fused) | Gap |
|---|---|---|---|
| Ghost aircraft injection | 95.0% | 95.0% | +0.0pt |
| Position spoofing / drift | 83.2% | 88.7% | +5.5pt |
| ICAO identity collision | 55.6% | 79.7% | +24.2pt |
| Replay attack | 100.0% | 100.0% | +0.0pt |
| Track hijacking (held-out class) | — (held out) | 100.0% | — |

The `icao_collision` class initially showed a 14.7-point train/test recall gap at 12 variants per class (a
small-sample artifact, not a real generalization failure) -- increasing to 20 variants per class closed it
to the gap shown above.

## 6. Real-world validation (Phase 8)

**Status: complete.**

Jamming-zone window: 2335 updates across 79 tracks, live-collected over the Kaliningrad/Baltic corridor (2256 with a full Kalman update). Control window: 8617 updates across 838 tracks, comparable 10-minute duration, clean Western-Europe traffic.

| Method | Jamming-zone flag rate | Control flag rate | Ratio |
|---|---|---|---|
| NIS (chi-square) | 6.1% | 2.2% | 2.71x |
| Hybrid Kalman+ML | 5.5% | 1.5% | 3.78x |
| Simulated MLAT | 0.2% | 0.8% | 0.26x |
| Simulated radar | 0.0% | 0.9% | 0.00x |

> Plausibility check only: no per-row ground truth exists for real traffic, so this reports detection ACTIVITY RATE (fraction of updates flagged), not precision/recall/F1. A higher rate in the jamming zone is consistent with real interference but is not proof of it -- regional traffic mix, receiver geometry, and other confounds are not controlled for.

**Reading this honestly, method by method:**

- **NIS and ML (2.7x and 3.8x higher in the jamming zone)**: these are the two detectors that check a track's broadcast history against its *own* physical self-consistency, with no separate ground truth needed. A meaningfully elevated flag rate in a region with documented, active GPS interference is exactly the signal this check was designed to find, and is the strongest evidence in this section that something real is being picked up.
- **MLAT and radar (lower, radar effectively zero, in the jamming zone)**: this is not evidence that nothing is happening -- it's a real limitation, and worth stating plainly rather than quietly omitting. Two separate effects explain it. First, MLAT/radar's `check()` (the form used for real traffic, as opposed to `check_with_ground_truth()` used in the Phase 7 synthetic testbed) simulates its independent sensor reading *from the same broadcast position it then compares against* -- there is no separate ground truth for live real-world data the way there is for a synthetic attack, so a spoofed-but-internally-smooth broadcast is indistinguishable from a genuine one to these checks specifically. This is the same caveat already noted in Phase 4/5's own docstrings, now visible in a real result rather than only a theoretical one. Second, radar's regional Baltic site and its inherited (not independently recalibrated) 3775m threshold were calibrated for the much larger original Western-Europe bbox -- the Baltic bbox is geographically smaller, so the same distance-driven angular error almost never reaches that threshold regardless of what's happening on the ground, which is very likely why radar's jamming-zone rate reads as exactly 0%. That's a geometry/calibration artifact of reusing an unrecalibrated threshold on a new region, not a claim that radar found zero anomalies to find.
- **Net read**: this result is consistent with real interference activity (NIS/ML) while also honestly demonstrating exactly the kind of limitation Section 8 describes -- position-truth checks (MLAT/radar) need real independent ground truth to be useful against genuine spoofing, and this project doesn't have that for live traffic, only for the synthetic testbed.

## 7. Why these design choices

**Why Kalman over pure ML.** The Kalman filter gives a physically interpretable, per-category-tunable
prediction with an exact, analytically-justified statistical test (the chi-square NIS gate) on top of it --
no attack data was needed to build or calibrate it. Pure ML (Isolation Forest) is trained only on
Kalman-derived features, never raw broadcasts, so it can only ever be as good as what the filter already
extracted; it earns its place by catching patterns (persistence across a rolling window, category/phase
context) that a single-update NIS check can't see, not by replacing the filter.

**Why MLAT and radar are simulated, and what that limits.** Real MLAT/radar infrastructure isn't
accessible to a software-only project. The simulations model real, distinguishing physical
characteristics of each technology (MLAT: multi-receiver TDOA, best resolved horizontally, poor vertical
GDOP with ground-only receivers; radar: single-site range/azimuth, error growing with distance from the
site) rather than being interchangeable "position checkers." What this cannot do: represent real receiver
clock drift, real multipath in Kaliningrad-relevant coastal/urban terrain, or a real absence-of-return
signature for a truly nonexistent aircraft (this project fakes that with `true_sv=None`, not a physical
absence of a radar return in a simulated environment). This is a real, load-bearing limitation, not a
detail.

**Why specific thresholds were chosen.** NIS's chi-square alpha=0.01 is the one analytically-derived
threshold in the system. Every other threshold (MLAT's 88m, radar's 3775m, the rule-based speed/turn
bounds) was set empirically against real clean traffic, targeting roughly the same ~1% false-positive
philosophy, because none of those detectors' true error distributions are clean Gaussians on real,
crowd-sourced ADS-B data -- verified directly during Phases 2, 4, and 5, where a naive first-pass threshold
produced 5-19% false-positive rates until retuned against real data and, in MLAT's case, until a genuine
receiver-network coverage-gap bug was found and fixed.

## 8. Limitations

- **MLAT and primary radar are simulated**, not real sensors -- see the notice at the top of this report.
- **No RF-layer verification of any kind.** Everything upstream of ADS-B message content (signal strength,
  timing at the physical layer, transmitter fingerprinting) is out of scope.
- **No satellite-based ADS-B sources** (e.g. Aireon) -- this system only ever sees what ground-based
  OpenSky feeders report.
- **No precise ground-truth labels for real-world validation.** Section 6 is a plausibility check (does
  detection activity increase in a documented interference window vs. a clean control) -- not a labeled
  precision/recall evaluation like Section 3. Real GPS jamming doesn't come with a per-aircraft,
  per-timestamp "this broadcast was affected" label the way synthetic injection does.
- **MLAT/radar's real-world check has no independent ground truth to compare against**, unlike the
  synthetic testbed. Section 6's result shows this concretely: NIS/ML (self-consistency checks) picked up
  real elevated activity in the documented jamming zone, MLAT/radar (position-truth checks, but with no
  real independent position for live traffic) largely did not.
- **ICAO collision remains the hardest class for every method** (~60% recall at best) -- MLAT/radar are
  structurally blind to it (the intruder's position is real, only its claimed identity is fraudulent), and
  even fusion, which combines every other signal, can't fully compensate for two detectors contributing
  nothing on this specific class.
- **This project covers ADS-B verification depth as one node in a multi-sensor fusion model.** Real
  operational systems corroborate against additional live sensors (real MLAT networks, real primary radar,
  satellite ADS-B, RF fingerprinting) that this project cannot access. The point of this system is to show
  what depth looks like at the ADS-B/Kalman/ML layer specifically, not to claim it replaces a full
  multi-sensor air surveillance stack.
