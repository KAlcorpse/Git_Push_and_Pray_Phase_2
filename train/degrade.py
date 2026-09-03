"""Forward degradation model, fitted from the data.

This is the third version. The history matters, because two plausible-sounding
diagnoses were both wrong and the measurements are what settled it.

WHAT THE VARIANCE SAYS
----------------------
The old model was Var(y|x) = s^2 x^2 + g^2 -- speckle plus read noise. Fitting a
third term over 300 pairs and 40 intensity bins:

    s^2 x^2        + g^2      R2 = 0.996339
    s^2 x^2 + a x  + g^2      R2 = 0.999545     <- residual error 8x smaller
              a x  + g^2      R2 = 0.967256

The linear term is Poisson shot noise and it is not optional. At dark pixels it
DOMINATES: at x=0.18 the shot variance is 1.44e-3 against 0.82e-3 for speckle
and 0.86e-3 for read noise. The two-term fit had no way to express it and
absorbed it by inflating both other terms (s 0.159 -> 0.182, g 0.029 -> 0.049).

WHAT THE HIGHER MOMENTS SAY
---------------------------
Variance alone does not identify the distribution, and the old model was wrong
about the shape. Real residuals are skewed and heavy-tailed, increasingly so at
low intensity -- the signature of shot noise at low counts. Gaussian speckle
cannot produce it. Measured, dark decile / bright decile:

                              std      ac    skew_dark  kurt_dark  skew_brt  kurt_brt
    REAL                     0.102  -0.052      0.796      2.549     0.377     0.622
    old model (v2)           0.103  -0.056      0.160      0.162     0.233     0.212
    this model               0.099  -0.055      0.334      0.646     0.380     0.524

Three changes get there: Gamma-distributed speckle instead of Gaussian (a Gamma
multiplier is the standard model for coherent speckle and carries the right
positive skew), the Poisson term above, and per-image level variation -- the
real noise level varies about 1.3x across images, and pooling images of
different variance is itself a source of heavy tails.

WHERE IT IS STILL WRONG
-----------------------
The dark decile is still too light-tailed: kurtosis 0.65 against a real 2.55.
Something at low signal is not captured. This is the first place to look if
synthetic data still underperforms.

WHAT DID NOT MATTER
-------------------
v2 fixed the residual autocorrelation (0.000 -> -0.055 against a real -0.052) on
the theory that white noise was the problem. Trained head to head at matched
budget, that changed the result by 0.00002 SSIM -- nothing. Autocorrelation was
a real defect but a cosmetic one. Kept here because it is free, not because it
earns anything.

THE MODEL
---------
    per image:  L ~ level jitter, applied to every noise term
    at HR:      speckle (Gamma) -> shot (Poisson) -> read (Gaussian)
    resize:     bicubic 2x, no anti-aliasing
    at LR:      speckle (Gamma) -> shot (Poisson) -> read (Gaussian)

VAR_AT_HR is the fraction of noise VARIANCE injected before the resize. It is
0.36, fitted to the residual autocorrelation: 0 gives white noise (-0.000) and 1
gives -0.154, against a real -0.052.
"""
import numpy as np
import torch
import torch.nn.functional as F

SPECKLE_STD = 0.1590      # s   multiplicative, Gamma
SHOT_GAIN = 0.008018      # a   Var contribution a*x; photon count is x/a
READ_STD = 0.0293         # g   additive Gaussian
VAR_AT_HR = 0.36          # fraction of noise variance injected before the resize
RESIZE_NOISE_GAIN = 0.721 # K: std of bicubic_downsample_2x(white noise)
LEVEL_JITTER = 0.10       # per-image spread of the overall level (measured ~0.10)


def bicubic_downsample(x):
    """(..., 2H, 2W) -> (..., H, W), matching the fitted degradation kernel."""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    squeeze = t.ndim == 2
    if squeeze:
        t = t[None]
    out = F.interpolate(t[:, None], scale_factor=0.5, mode="bicubic",
                        align_corners=False, antialias=False)[:, 0].numpy()
    return out[0] if squeeze else out


class Degrader:
    """gt_hr (2H,2W) float32 in [0,1] -> (H,W) float32 degraded.

    `jitter` widens the per-image level distribution beyond the measured 0.10.
    The test set contains out-of-distribution samples, and a model trained at
    one exact noise level has no reason to hold up at another, so the training
    distribution is deliberately made wider than the fitted one.
    """

    def __init__(self, speckle_std=SPECKLE_STD, shot_gain=SHOT_GAIN,
                 read_std=READ_STD, var_at_hr=VAR_AT_HR, jitter=0.2, seed=None):
        self.s, self.a, self.g = speckle_std, shot_gain, read_std
        self.var_at_hr = var_at_hr
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)

    def _stage(self, x, s, a, g):
        if s > 0:
            k = 1.0 / (s * s)                        # Gamma(k, 1/k): mean 1, std s
            x = x * self.rng.gamma(k, 1.0 / k, x.shape).astype(np.float32)
        if a > 0:
            x = (a * self.rng.poisson(np.maximum(x, 0.0) / a)).astype(np.float32)
        if g > 0:
            x = x + (g * self.rng.standard_normal(x.shape)).astype(np.float32)
        return x.astype(np.float32)

    def __call__(self, gt_hr):
        gt_hr = np.asarray(gt_hr, dtype=np.float32)
        lvl = 1.0
        if self.jitter:
            lvl = max(0.05, 1.0 + self.rng.normal(0.0, self.jitter))

        pv, K = self.var_at_hr, RESIZE_NOISE_GAIN
        s, a, g = self.s * lvl, self.a * lvl * lvl, self.g * lvl
        hr = self._stage(gt_hr, s * np.sqrt(pv) / K, a * pv / (K * K),
                         g * np.sqrt(pv) / K)
        lo = bicubic_downsample(hr)
        return self._stage(lo, s * np.sqrt(1 - pv), a * (1 - pv),
                           g * np.sqrt(1 - pv))
