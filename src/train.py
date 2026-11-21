"""Fine-tune one transformer for 3-class Hinglish sentiment.

Plain PyTorch loop (no HF Trainer) so it works on transformers 4.x and 5.x.
Evaluates on val after every epoch and logs per-epoch val metrics (used by the
sweep to pick epochs). Test is predicted once, with the final-epoch model.

Usage:
  python -m src.train --model xlmr --lr 2e-5 --epochs 3 --seed 42
  python -m src.train --model mbert --lr 3e-5 --epochs 3 --seed 42 --normalize
  python -m src.train --model xlmr ... --save_model      # keep weights for the demo
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

from src.common import LABELS, ROOT, load_split, metrics, run_dir, safe_name, save_run, set_seed
from src.models import resolve


class Enc(torch.utils.data.Dataset):
    def __init__(self, enc, y):
        self.enc, self.y = enc, list(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.y[i])
        return item


@torch.no_grad()
def predict(model, loader, device, amp) -> np.ndarray:
    model.eval()
    out = []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(**b).logits
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


def train(model_name: str, lr: float, epochs: int, seed: int, normalize: bool = False,
          batch_size: int = 32, max_len: int = 128, warmup: float = 0.1,
          weight_decay: float = 0.01, save_model: bool = False, out_dir=None,
          predict_test: bool = True, limit: int | None = None) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    hf_id, tok_kw = resolve(model_name)

    tok = AutoTokenizer.from_pretrained(hf_id, **tok_kw)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=3, id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)}).to(device)

    splits = {s: load_split(s, normalize) for s in ("train", "val", "test")}
    if limit:  # smoke tests only
        splits = {s: d.head(limit) for s, d in splits.items()}
    coll = DataCollatorWithPadding(tok)

    def loader(df, shuffle):
        enc = tok(list(df["text"]), truncation=True, max_length=max_len)
        g = torch.Generator().manual_seed(seed)
        return DataLoader(Enc(enc, df["y"]), batch_size=batch_size if shuffle else 128,
                          shuffle=shuffle, collate_fn=coll, generator=g)

    tr_dl, va_dl = loader(splits["train"], True), loader(splits["val"], False)
    te_dl = loader(splits["test"], False)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=lr)
    steps = len(tr_dl) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)  # works on all torch 2.x

    history = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for b in tr_dl:
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = model(**b).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        vp = predict(model, va_dl, device, amp)
        vm = metrics(splits["val"]["y"], vp.argmax(1))
        history.append({"epoch": ep, "train_loss": tot / len(tr_dl), **vm})
        print(f"[{model_name} lr={lr} seed={seed}{' norm' if normalize else ''}] "
              f"ep{ep} loss={tot/len(tr_dl):.4f} val wF1={vm['weighted_f1']:.4f} "
              f"macroF1={vm['macro_f1']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    config = {"model": model_name, "hf_id": hf_id, "lr": lr, "epochs": epochs, "seed": seed,
              "normalize": normalize, "batch_size": batch_size, "max_len": max_len,
              "warmup": warmup, "weight_decay": weight_decay, "train_seconds": time.time() - t0,
              "history": history}
    result = {"val": history[-1], "config": config}

    if predict_test:
        out = out_dir or run_dir(model_name, seed, normalize)
        tp = predict(model, te_dl, device, amp)
        tm = metrics(splits["test"]["y"], tp.argmax(1))
        save_run(out, splits["test"], tp, history[-1], tm, config)
        result["test"] = tm
        print(f"TEST wF1={tm['weighted_f1']:.4f} macroF1={tm['macro_f1']:.4f} -> {out}")
        if save_model:
            mdir = ROOT / "results" / "models" / (safe_name(model_name) + ("+norm" if normalize else ""))
            model.save_pretrained(mdir)
            tok.save_pretrained(mdir)
            print(f"saved model -> {mdir}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--save_model", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: use first N rows")
    a = ap.parse_args()
    r = train(a.model, a.lr, a.epochs, a.seed, a.normalize, a.batch_size, a.max_len,
              save_model=a.save_model, limit=a.limit)
    print(json.dumps(r.get("test", r["val"]), indent=2))


if __name__ == "__main__":
    main()
