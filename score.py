#!/usr/bin/env python3
"""Score restored images against ground truth on the three official metrics.

    python score.py <gt_dir> <pred_dir> [<pred_dir2> ...]

Reports mean PSNR / SSIM / LPIPS over the images common to every directory, and
-- for every directory after the first -- a paired bootstrap 95% CI on the
difference from that first one.

The bootstrap exists because it is the only thing that distinguishes a real
improvement from val-split noise, and it cuts both ways. The three architecture
variants here land within 0.07 dB of each other, which looks like a tie by eye;
pairing removes the image-to-image variance and shows the ordering
learned > base > noskip is real, every CI excluding zero. Small and real is a
different thing from noise -- but so is large and unpaired. Judge a delta by
this test, not by the gap between two means.
"""
import argparse
import os
import sys

import numpy as np
import torch
from skimage.metrics import structural_similarity


def load(path):
    a = np.load(path)
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] <= 4 else a[0]
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float32) / float(np.iinfo(a.dtype).max)
    return np.clip(a.astype(np.float32), 0.0, 1.0)


_LPIPS = None


def lpips_batch(pred, gt, device):
    """pred/gt: (B,H,W) float32 in [0,1] -> (B,) LPIPS distances."""
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in _LPIPS.parameters():
            p.requires_grad_(False)
    with torch.no_grad():
        p = torch.from_numpy(pred).to(device)[:, None].repeat(1, 3, 1, 1) * 2 - 1
        g = torch.from_numpy(gt).to(device)[:, None].repeat(1, 3, 1, 1) * 2 - 1
        return _LPIPS(p, g).flatten().cpu().numpy()


def score_dir(gt_dir, pred_dir, names, device, batch=16):
    ps, ss, lp = [], [], []
    for i in range(0, len(names), batch):
        chunk = names[i:i + batch]
        g = np.stack([load(os.path.join(gt_dir, n)) for n in chunk])
        p = np.stack([load(os.path.join(pred_dir, n)) for n in chunk])
        if g.shape != p.shape:
            sys.exit(f"shape mismatch in {pred_dir}: got {p.shape[1:]}, "
                     f"ground truth is {g.shape[1:]}")
        mse = ((g - p) ** 2).reshape(len(chunk), -1).mean(1)
        ps.append(10 * np.log10(1.0 / np.maximum(mse, 1e-12)))
        ss.append(np.array([structural_similarity(a, b, data_range=1.0)
                            for a, b in zip(g, p)]))
        lp.append(lpips_batch(p, g, device))
        print(f"\r  {pred_dir}: {min(i + batch, len(names))}/{len(names)}",
              end="", file=sys.stderr, flush=True)
    print("\r" + " " * 60 + "\r", end="", file=sys.stderr)
    return dict(psnr=np.concatenate(ps), ssim=np.concatenate(ss),
                lpips=np.concatenate(lp))


def paired_ci(a, b, n_boot=10000, seed=0):
    """95% CI on mean(b - a), resampling images (not values) with replacement."""
    d = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return d.mean(), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt_dir")
    ap.add_argument("pred_dirs", nargs="+")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sets = [set(f for f in os.listdir(d) if f.endswith(".npy"))
            for d in [args.gt_dir] + args.pred_dirs]
    names = sorted(set.intersection(*sets))
    if not names:
        sys.exit("no filenames common to all directories")
    for d, s in zip([args.gt_dir] + args.pred_dirs, sets):
        if len(s) != len(names):
            print(f"note: {d} has {len(s)} files, scoring the {len(names)} in common",
                  file=sys.stderr)

    res = {d: score_dir(args.gt_dir, d, names, device) for d in args.pred_dirs}

    print(f"\n{len(names)} images\n")
    print(f"{'run':28s} {'PSNR':>8s} {'SSIM':>8s} {'LPIPS':>8s}")
    for d in args.pred_dirs:
        r = res[d]
        print(f"{d[-28:]:28s} {r['psnr'].mean():8.3f} {r['ssim'].mean():8.4f} "
              f"{r['lpips'].mean():8.4f}")

    if len(args.pred_dirs) > 1:
        base = args.pred_dirs[0]
        print(f"\npaired bootstrap vs {base}  (95% CI; CI spanning 0 = not a result)")
        for d in args.pred_dirs[1:]:
            print(f"  {d[-28:]:28s}", end="")
            for m, fmt in (("psnr", "+.3f"), ("ssim", "+.4f"), ("lpips", "+.4f")):
                mean, lo, hi = paired_ci(res[base][m], res[d][m], args.boot)
                sig = " " if lo <= 0 <= hi else "*"
                print(f"  {m} {mean:{fmt}} [{lo:{fmt}},{hi:{fmt}}]{sig}", end="")
            print()
        print("\n  * = CI excludes zero.  PSNR/SSIM higher is better, LPIPS lower.")


if __name__ == "__main__":
    main()
