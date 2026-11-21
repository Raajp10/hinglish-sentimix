"""Full experiment driver (resumable: finished runs are skipped).

Stage 1 (tune):  seed 42, each LR in LR_GRID, train MAX_EPOCHS, log val wF1 per epoch.
                 Pick (lr, epoch) with the best val weighted F1. Test is NOT touched.
Stage 2 (final): best (lr, epochs) x SEEDS -> test predictions in results/runs/<model>/seedN.
Stage 3 (norm):  same best config with --normalize, for the model(s) given by --norm_models
                 (proposal: the weakest transformer; pick it after stage 2).

Usage:
  python -m src.sweep --models mbert xlmr indicbert hingbert          # stage 1+2
  python -m src.sweep --models indicbert --norm_models indicbert      # + stage 3
  python -m src.sweep --models xlmr --stage final --save_model        # weights for demo
"""
from __future__ import annotations

import argparse
import gc
import json

from src.common import ROOT, RUNS, safe_name
from src.models import CORE

LR_GRID = [1e-5, 2e-5, 3e-5, 5e-5]
MAX_EPOCHS = 4
SEEDS = [42, 43, 44]
SWEEP_DIR = ROOT / "results" / "sweep"


def free():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass


def tune(model: str, lrs, max_epochs: int) -> dict:
    from src.train import train
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    f = SWEEP_DIR / f"{safe_name(model)}.json"
    log = json.loads(f.read_text()) if f.exists() else {"runs": {}}
    for lr in lrs:
        key = f"{lr:g}"
        if key in log["runs"]:
            print(f"skip tune {model} lr={key}")
            continue
        r = train(model, lr, max_epochs, seed=42, predict_test=False)
        log["runs"][key] = r["config"]["history"]
        f.write_text(json.dumps(log, indent=2))
        free()
    best = max(((float(lr), h["epoch"], h["weighted_f1"]) for lr, hist in log["runs"].items()
                for h in hist), key=lambda x: x[2])
    log["best"] = {"lr": best[0], "epochs": best[1], "val_weighted_f1": best[2]}
    f.write_text(json.dumps(log, indent=2))
    print(f"BEST {model}: {log['best']}")
    return log["best"]


def final(model: str, best: dict, seeds, normalize=False, save_model=False) -> None:
    from src.train import train
    for i, s in enumerate(seeds):
        d = RUNS / (safe_name(model) + ("+norm" if normalize else "")) / f"seed{s}"
        if (d / "metrics.json").exists():
            print(f"skip final {d}")
            continue
        train(model, best["lr"], best["epochs"], s, normalize=normalize,
              save_model=save_model and i == 0)
        free()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=CORE)
    ap.add_argument("--norm_models", nargs="*", default=[])
    ap.add_argument("--stage", choices=["all", "tune", "final"], default="all")
    ap.add_argument("--lrs", nargs="+", type=float, default=LR_GRID)
    ap.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--save_model", action="store_true")
    a = ap.parse_args()

    for m in a.models:
        if a.stage in ("all", "tune"):
            best = tune(m, a.lrs, a.max_epochs)
        else:
            best = json.loads((SWEEP_DIR / f"{safe_name(m)}.json").read_text())["best"]
        if a.stage in ("all", "final"):
            final(m, best, a.seeds, save_model=a.save_model)
    for m in a.norm_models:
        best = json.loads((SWEEP_DIR / f"{safe_name(m)}.json").read_text())["best"]
        final(m, best, a.seeds, normalize=True)


if __name__ == "__main__":
    main()
