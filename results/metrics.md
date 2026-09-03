# Metrics

Two evaluation sets are used throughout, and they are kept strictly separate.

| set | n | what it is |
|---|---|---|
| **held-out val** | 479 | 10% of the 4,785 training pairs, `SPLIT_SEED=42`, excluded from training **and** from model selection |
| **released test** | 297 | the organisers' test pairs, released after the submitted model was already trained and frozen |

Every delta below carries a paired bootstrap 95% CI. A CI spanning zero is not a
result. Reproduce any table with:

```bash
python score.py <gt_dir> <baseline_pred_dir> [<other_pred_dir> ...]
```

---

## 1. Headline — released test set (297 pairs, never trained on)

| | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| floor — bicubic ×2 on the noisy input | 20.455 | 0.5191 | 0.4655 |
| **submitted model** | **23.982** | **0.6455** | **0.1509** |
| reference — perfect denoise, then bicubic ×2 | 27.326 | 0.7673 | 0.3246 |

The reference row is *not* an upper bound. It is what a flawless denoiser scores
if it then does nothing clever about resolution: downsample the ground truth,
bicubic it back. A learned model can legitimately beat it, and on LPIPS ours
does — by more than 2×, because that row never sees the noisy image and so
never learns to resolve texture. Between the two, the model closes **52% of
the achievable PSNR range and 48% of the SSIM range** (54% and 47% on the
held-out val split).

## 2. Headline — held-out val (479 pairs)

| | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| floor — bicubic ×2 on the noisy input | 20.380 | 0.5085 | 0.4869 |
| **submitted model** | **23.706** | **0.6230** | **0.1577** |
| reference — perfect denoise, then bicubic ×2 | 26.579 | 0.7530 | 0.3346 |

**Caveat we state rather than hide.** The dataset is ordered by source, and the
random split leaks: 23.4% of held-out images have a near-duplicate (descriptor
similarity > 0.95) in training. 23.706 dB is a *same-source, in-distribution*
number. The 297-pair test result above does not have this problem, which is why
it leads.

| held-out set | n | mean NN similarity | near-duplicates |
|---|---|---|---|
| random (the split we use) | 479 | 0.907 | 23.4% |
| grouped, blocks of 40 | 484 | 0.898 | 19.4% |
| rarest content clusters | 484 | 0.809 | **0.6%** |

---

## 3. The noise floor, measured first

Two runs identical but for `SEED`:

| | PSNR | SSIM | LPIPS |
|---|---|---|---|
| seed 42 | 23.607 | 0.6160 | 0.1737 |
| seed 1337 | 23.613 | 0.6165 | 0.1732 |
| **gap at convergence** | **0.006** | **0.0005** | **0.0005** |

Mid-training the same two runs diverge by up to **0.388 dB** — larger than almost
every effect measured in this project. Two consequences, both enforced:

- only fully annealed final checkpoints are ever compared;
- anything under 0.006 dB / 0.0005 SSIM / 0.0005 LPIPS is noise, and is reported
  as noise.

---

## 4. What was accepted

| change | PSNR | SSIM | LPIPS | verdict |
|---|---|---|---|---|
| learned global skip (vs fixed bicubic) | +0.002 [ns] | +0.0002 [ns] | −0.0013* | kept — LPIPS only |
| **full-resolution fine-tuning phase** | **+0.377** | **+0.0127** | **−0.0110** | **the biggest win in the project** |
| `FULLRES_EPOCHS` 4 → 8 | +0.038* | +0.0023* | −0.0022* | kept, promoted to `best.pth` |
| LPIPS in the loss at weight 0.10 | — | — | large | kept; see §6 |

The full-resolution phase — the last *N* epochs train on whole 128px inputs
instead of 64px crops — had been in `train.py` from the first commit and no
time-capped run had ever reached it. It moves all three metrics the same way,
which nothing else does. Almost the entire effect lands in **one epoch**:

| | last crop epoch | first full-res epoch | plateau sd |
|---|---|---|---|
| 4 full-res epochs | 23.297 | 23.675 (+0.378) | 0.016 |
| 8 full-res epochs | 23.286 | 23.629 (+0.343) | 0.035 |

It is a step, not a ramp. `FULLRES_EPOCHS=12` is ruled out by that trace rather
than by burning another run on it.

---

## 5. What was measured and rejected

Every row is a full matched training run, bootstrapped against the submitted
model on the 479 held-out pairs. All are rejected for the same reason: each wins
one metric by trading another, and the organisers' metric weighting is
undisclosed.

| candidate | PSNR | SSIM | LPIPS | why rejected |
|---|---|---|---|---|
| synthetic training data from GT | −0.048* | −0.0063* | +0.0092* | hurts all three |
| noise-level conditioning | −0.007* | −0.0007* | +0.0003 [ns] | closes 11% of the clean-input deficit, costs perceptual quality |
| + degradation augmentation | +0.015* | −0.0007* | +0.0030* | buys PSNR with LPIPS |
| + augmentation *and* conditioning | +0.031* | −0.0011* | +0.0048* | LPIPS cost is 10× its seed floor, PSNR gain only 5× |
| SSIM matched to the scored implementation | −0.050* | +0.0005* | +0.0002 [ns] | +0.0005 SSIM is exactly the seed floor |
| frequency-domain (FFT) loss | −0.111* | −0.0009* | −0.0005 [ns] | HF energy 0.47 → 0.59, and PSNR paid for all of it |
| wide-kind augmentation | −0.014* | −0.0032* | +0.0072* | large robustness gain, real in-distribution cost — see §7 |
| the 297 released test pairs as extra training data | −0.025* | −0.0001 [ns] | −0.0012* | +6.9% data on a model flat since epoch 12 |
| 8× dihedral test-time augmentation | + | + | − | 8× compute; inference time is scored |
| two-model ensemble | +0.36 | + | 2.2× worse | halves throughput, and LPIPS is our weakest metric |

`*` = bootstrap CI excludes zero. `[ns]` = not significant.

**The submitted model is the only one that is best on all three metrics at
once.** With the metric weighting undisclosed, that is the position we can
defend without guessing.

---

## 6. The loss

```
L = 0.50·Charbonnier + 0.35·(1 − SSIM) + 0.15·gradient-L1 + 0.10·LPIPS
```

PSNR and SSIM both *reward blur*, so a distortion-only objective cannot tell a
good restoration from a smeared one. We track a fourth number for this —
`hf_ratio`, high-frequency energy above 0.25 cycles/px relative to ground truth,
where 1.0 matches GT. The submitted model sits at **0.470**.

LPIPS converges much later than the others (~epoch 36, against ~16 for PSNR and
~26 for SSIM), so any LPIPS comparison from a short run is provisional. This
invalidated one earlier reading and is now a standing rule.

---

## 7. Robustness — three axes, one weakness

**Content novelty:** no measurable effect. **Scale:** 0.264 dB across a 4×
change in inference size, and it *favours larger inputs*, so the 512←256 half of
the problem is low risk.

**Noise level: the real weakness.** Hand the model a *clean* low-res input and it
denoises anyway, erasing detail that was never noise:

| | PSNR | SSIM | LPIPS |
|---|---|---|---|
| bicubic ×2 on a clean input | 26.579 | 0.7530 | 0.3346 |
| our model on a clean input | 25.522 | 0.6645 | 0.3060 |
| **SR gain over bicubic** | **−1.057** | | |

It loses to plain bicubic on 99% of clean images. `failure_clean_input.png` shows
it. This is the honest limit of a model trained on one noise level.

**A retraction that shaped the method.** Degradation augmentation was first
reported as fixing this, on the strength of a stress test. That test was
circular: `stress.py` and `augment.py` imported the *same* constants from
`degrade.py`, so the probe drew from the distribution the model had trained on.
Re-run against a held-out degradation family — uniform noise, box blur,
quantisation, none of which any augmenter produces — the picture inverts:

| LPIPS degradation, held-out family | Δ from t=0 to hardest |
|---|---|
| submitted model | +0.4214 |
| + augmentation (more of our own fitted noise) | +0.4235 |
| + augmentation and conditioning | +0.4184 |
| **+ augmentation of *kind*** (`AUG_MODE=wide`) | **+0.1241** |

> Training on variety of **kind** transfers to unseen degradations.
> Training on variety of **amount** does not.

The wide-augmented model degrades **3.4× less** on LPIPS against degradations it
has never seen. It is not the submitted model, because it costs 0.0072 LPIPS
in-distribution — 14× the seed floor — and how far the test set's OOD half
actually sits is the one thing we cannot measure. It is the first thing we would
ship if that were known.

**The rule this earned:** an augmentation and the probe used to validate it must
never share a generator.

---

## 8. Throughput

| | |
|---|---|
| end-to-end, 297 images | **9.1 s** (24.97 ms/img inference) |
| hardware | NVIDIA RTX 3050 Laptop GPU, 4 GiB, CUDA 13.2 |
| precision | fp16 autocast (identical to fp32 to 4 dp) |
| timing method | `time.perf_counter()` from the first line of `run.py` to the last file written |

Startup is **~1.8 s** of `import torch` plus CUDA context, before a single image
is touched. On an H100, where inference for a few hundred images falls to
~0.5–1.4 s, that fixed cost is **56–78% of end-to-end**. Halving the network
would buy ~10% of wall time and cost real quality, so "shrink the model for
speed" is dead and the right move is to spend inference time on quality.

`cudnn.benchmark` is chosen from the input count, not pinned:

| batch | `benchmark=True` | `benchmark=False` |
|---|---|---|
| 8 | 40.4 ms/img, 952 MiB | **24.2 ms/img, 365 MiB** |
| 16 | 61.4 ms/img, 1865 MiB | **23.7 ms/img, 696 MiB** |

Autotuning finds ~11% faster steady-state kernels but costs ~26.5 s up front and
only repays that at ~9,650 images. Hence the 10,000-image threshold in `run.py`.

> **Near-miss worth recording.** The first bisect ran every variant inside one
> *warm* process, where the autotune cache was already populated, and showed no
> difference at all. The effect exists only in a fresh process — which is exactly
> what the graders run.

---

## 9. The degradation forward model

Fitted empirically over 300 real pairs rather than assumed:

```
y = downsample₂(x) · speckle  +  shot  +  read
```

Gamma speckle, Poisson shot noise and Gaussian read noise, in that order, with
the variance law fitted at R² = 0.99. Speckle level varies about **1.8×** across
images, and the per-image spread within the real corpus is 3.65× end to end.

It is used by `augment.py` and by the diagnostics. It is **not** used to
synthesise training data: three rounds of improving the fit never flipped the
sign of that experiment (−0.048 dB, −0.0063 SSIM against a matched control), and
`SYNTH_PROB` ships at 0.

---

## 10. Data quality — the low-PSNR tail is real

The images we score worst on look like salt-and-pepper noise, so we checked
whether they are junk. Lag-1 autocorrelation of the ground truth across all
4,785 images: **exactly three are genuinely degenerate.** Everything else at low
autocorrelation is real, very fine-grained SEM.

Dropping the worst 24 val images moves the mean 23.706 → **24.054 dB**. If the
official test set has the same tail, a meaningful share of the final score is
decided by images nobody can restore — and deleting that tail from training
would remove the strongest signal we have against over-smoothing.

---

## Figures

| file | what it shows |
|---|---|
| `test_examples.png` | released test pairs: input, bicubic, ours, ground truth |
| `failure_clean_input.png` | the clean-input failure, §7 |
| `headroom.png` | how much of the achievable PSNR/SSIM range we close |
| `robustness_heldout.png` | the held-out degradation family, §7 |
| `augmentation_rejected.png` | all four augmentation variants, measured |
| `fullres_phase.png` | the full-resolution phase as a step, §4 |
| `training_curve.png` | per-epoch PSNR / SSIM / LPIPS |
| `dataset_analysis.png` | intensity, noise level and content spread of the corpus |
| `architecture.png` | MiniRestormer |
