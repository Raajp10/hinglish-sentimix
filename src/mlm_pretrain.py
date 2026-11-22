"""Build OUR Hinglish BERT: continued masked-language-model (MLM) pre-training of
mBERT on Romanized Hinglish text, then fine-tune it like every other model.

Corpus sources (combine any):
  sentimix    SentiMix TRAIN tweets only (labels unused). Alone = "task-adaptive" (TAPT).
  hinglid     L3Cube-HingLID (~44k Roman Hinglish tweets, GitHub, CC BY-NC-SA 4.0).
  hingcorpus  First N lines of L3Cube-HingCorpus (52.9M Roman Hinglish tweets, Google Drive).

Decontamination: any corpus line equal to, or sharing an 8-gram with, a SentiMix
val/test tweet is dropped. Val/test texts are never trained on.

Usage (repo root, GPU):
  python -m src.mlm_pretrain --name ourbert-tapt --sources sentimix --epochs 3
  python -m src.mlm_pretrain --name ourbert --sources sentimix hinglid --epochs 3
  python -m src.mlm_pretrain --name ourbert --sources sentimix hinglid hingcorpus --hingcorpus_lines 500000
then fine-tune:  python -m src.sweep --models ourbert --save_model
Resumable: checkpoints every --ckpt_every steps to results/pretrained/<name>/ckpt.pt.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import random
import re
import subprocess
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling,
                          get_linear_schedule_with_warmup)

from src.common import PROC, ROOT, set_seed
from src.models import resolve
from src.prepare import fix_mojibake

EXT = ROOT / "data" / "external"
MLM_DIR = ROOT / "data" / "mlm"
OUT_ROOT = ROOT / "results" / "pretrained"
HINGLID_URL = "https://raw.githubusercontent.com/l3cube-pune/code-mixed-nlp/{br}/L3Cube-HingLID/all.txt"
HINGCORPUS_GDRIVE_ID = "1s_6eHO9zDhxQ-xVN1TyNguszV1meZkl9"  # from github.com/l3cube-pune/code-mixed-nlp

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_WS = re.compile(r"\s+")


# ---------------------------------------------------------------- corpus
def clean_line(s: str) -> str | None:
    s = fix_mojibake(s)
    s = _URL.sub(" ", s)
    s = _MENTION.sub(" ", s)
    s = s.replace("#", " ")
    s = _WS.sub(" ", s).strip().lower()
    s = re.sub(r"^rt\b:?\s*", "", s)
    toks = s.split()
    if not 4 <= len(toks) <= 80:
        return None
    letters = [c for c in s if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) < 0.9:
        return None  # keep Roman-script text only
    return s


def _key(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def _grams(words: list[str], n: int = 8) -> set[str]:
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def heldout_index():
    exact, grams = set(), set()
    for sp in ("val", "test"):
        for t in pd.read_csv(PROC / f"{sp}.csv")["text"].fillna(""):
            w = _key(t)
            exact.add(" ".join(w))
            grams |= _grams(w)
    return exact, grams


def src_sentimix():
    yield from pd.read_csv(PROC / "train.csv")["text"].fillna("").astype(str)


def src_hinglid():
    EXT.mkdir(parents=True, exist_ok=True)
    f = EXT / "hinglid_all.txt"
    if not f.exists():
        for br in ("main", "master"):
            try:
                urllib.request.urlretrieve(HINGLID_URL.format(br=br), f)
                break
            except Exception as e:  # noqa: BLE001
                last = e
        else:
            raise RuntimeError(f"could not download HingLID: {last}")
    cur = []
    with open(f, encoding="utf-8") as fh:  # CoNLL: token<TAB>lang, blank line between tweets
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                if cur:
                    yield " ".join(cur)
                    cur = []
            else:
                cur.append(line.split("\t")[0])
    if cur:
        yield " ".join(cur)


def _iter_text_stream(path: Path):
    """Yield lines from a plain/gz/zip/tar(.gz) file without extracting everything."""
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic[:2] == b"\x1f\x8b" and not tarfile.is_tarfile(path):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
            yield from fh
    elif magic[:2] == b"PK":
        with zipfile.ZipFile(path) as z:
            members = sorted((m for m in z.infolist() if not m.is_dir()), key=lambda m: -m.file_size)
            for m in members:
                with z.open(m) as raw:
                    yield from io.TextIOWrapper(raw, encoding="utf-8", errors="ignore")
    elif tarfile.is_tarfile(path):
        with tarfile.open(path, "r|*") as t:
            for m in t:
                if m.isfile():
                    raw = t.extractfile(m)
                    yield from io.TextIOWrapper(raw, encoding="utf-8", errors="ignore")
    else:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            yield from fh


def src_hingcorpus(path: str | None):
    EXT.mkdir(parents=True, exist_ok=True)
    p = Path(path) if path else EXT / "hingcorpus_raw"
    if not p.exists():
        print("downloading HingCorpus from Google Drive (large; may take a while)...", flush=True)
        r = subprocess.run(["gdown", "--fuzzy", f"https://drive.google.com/uc?id={HINGCORPUS_GDRIVE_ID}",
                            "-O", str(p)])
        if r.returncode != 0 or not p.exists():
            raise RuntimeError(
                "gdown failed (Drive quota?). Fix: open the Drive link from "
                "github.com/l3cube-pune/code-mixed-nlp, 'Add shortcut to Drive', then pass "
                "--hingcorpus_path /content/drive/MyDrive/<file>")
    for line in _iter_text_stream(p):
        # tolerate CSV/TSV dumps: take the longest field
        parts = re.split(r"\t|,(?=\S)", line.strip()) if ("\t" in line) else [line]
        yield max(parts, key=len)


def build_corpus(sources, hingcorpus_lines, hingcorpus_path, seed) -> tuple[Path, dict]:
    MLM_DIR.mkdir(parents=True, exist_ok=True)
    exact, grams = heldout_index()
    seen, lines, stats = set(), [], {}
    for src in sources:
        it = {"sentimix": src_sentimix, "hinglid": src_hinglid,
              "hingcorpus": lambda: src_hingcorpus(hingcorpus_path)}[src]()
        kept = dropped_contam = 0
        cap = hingcorpus_lines if src == "hingcorpus" else None
        for raw in it:
            s = clean_line(raw)
            if s is None or s in seen:
                continue
            w = _key(s)
            if " ".join(w) in exact or (_grams(w) & grams):
                dropped_contam += 1
                continue
            seen.add(s)
            lines.append(s)
            kept += 1
            if cap and kept >= cap:
                break
        stats[src] = {"kept": kept, "dropped_overlap_with_val_test": dropped_contam}
        print(f"{src}: kept {kept} lines, dropped {dropped_contam} overlapping val/test", flush=True)
    random.Random(seed).shuffle(lines)
    n_eval = min(2000, max(200, len(lines) // 100))
    tag = "_".join(sources)
    train_f, eval_f = MLM_DIR / f"{tag}.train.txt", MLM_DIR / f"{tag}.eval.txt"
    train_f.write_text("\n".join(lines[n_eval:]) + "\n", encoding="utf-8")
    eval_f.write_text("\n".join(lines[:n_eval]) + "\n", encoding="utf-8")
    stats.update({"total_train_lines": len(lines) - n_eval, "eval_lines": n_eval,
                  "total_words": sum(len(x.split()) for x in lines)})
    return train_f, {"eval_file": str(eval_f), **stats}


# ---------------------------------------------------------------- training
class Lines(torch.utils.data.Dataset):
    def __init__(self, path):
        self.x = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


class make_collate:  # a class (not a closure) so DataLoader workers can pickle it on macOS/Windows
    def __init__(self, tok, max_len, mlm_prob):
        self.tok, self.max_len = tok, max_len
        self.mlm = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=mlm_prob)

    def __call__(self, batch):
        enc = self.tok(batch, truncation=True, max_length=self.max_len)
        return self.mlm([{k: enc[k][i] for k in enc} for i in range(len(batch))])


@torch.no_grad()
def mlm_loss(model, texts, tok, device, amp, max_len, seed=0, bs=128) -> float:
    """Average MLM loss with a FIXED masking seed so before/after are comparable."""
    model.eval()
    g = torch.get_rng_state()
    torch.manual_seed(seed)
    coll = make_collate(tok, max_len, 0.15)
    tot = n = 0
    for i in range(0, len(texts), bs):
        b = {k: v.to(device) for k, v in coll(texts[i:i + bs]).items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = model(**b).loss
        k = int((b["labels"] != -100).sum())
        tot += float(loss) * k
        n += k
    torch.set_rng_state(g)
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ourbert")
    ap.add_argument("--base", default="mbert")
    ap.add_argument("--sources", nargs="+", default=["sentimix", "hinglid"],
                    choices=["sentimix", "hinglid", "hingcorpus"])
    ap.add_argument("--hingcorpus_lines", type=int, default=500_000)
    ap.add_argument("--hingcorpus_path", default=None)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max_steps", type=int, default=None, help="overrides --epochs")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--mlm_prob", type=float, default=0.15)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N corpus lines")
    a = ap.parse_args()

    set_seed(a.seed)
    out = OUT_ROOT / a.name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        print(f"{out} already finished; delete it to retrain.")
        return

    train_f, cstats = build_corpus(a.sources, a.hingcorpus_lines, a.hingcorpus_path, a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    hf_id, tok_kw = resolve(a.base)
    tok = AutoTokenizer.from_pretrained(hf_id, **tok_kw)
    model = AutoModelForMaskedLM.from_pretrained(hf_id).to(device)

    ds = Lines(train_f)
    if a.limit:
        ds.x = ds.x[:a.limit]
    eval_corpus = Path(cstats["eval_file"]).read_text(encoding="utf-8").splitlines()[:2000]
    # SentiMix val TEXT (no labels) is used only to report MLM loss before/after, never for training/selection.
    eval_sentimix = pd.read_csv(PROC / "val.csv")["text"].fillna("").astype(str).tolist()[:2000]
    if a.limit:
        eval_corpus, eval_sentimix = eval_corpus[:200], eval_sentimix[:200]

    g = torch.Generator().manual_seed(a.seed)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, generator=g,
                    collate_fn=make_collate(tok, a.max_len, a.mlm_prob), num_workers=2,
                    drop_last=True)
    steps = a.max_steps or max(1, int(math.ceil(len(dl) * a.epochs)))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)  # works on all torch 2.x

    step, log = 0, {"history": []}
    ck = out / "ckpt.pt"
    if ck.exists():
        state = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        step, log = state["step"], state["log"]
        print(f"resumed at step {step}/{steps}")
    else:
        log["before"] = {"corpus_eval_loss": mlm_loss(model, eval_corpus, tok, device, amp, a.max_len),
                         "sentimix_val_loss": mlm_loss(model, eval_sentimix, tok, device, amp, a.max_len)}
        print("before:", log["before"], flush=True)

    model.train()
    t0, run = time.time(), 0.0
    done = step >= steps
    epoch = 0
    while not done:
        g.manual_seed(a.seed + epoch)  # deterministic order per epoch -> resume skips correctly
        skip = step - epoch * len(dl) if step > epoch * len(dl) else 0
        for i, b in enumerate(dl):
            if i < skip:
                continue
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
            step += 1
            run = 0.98 * run + 0.02 * loss.item() if run else loss.item()
            if step % 100 == 0 or step == steps:
                el = time.time() - t0
                print(f"step {step}/{steps} loss {run:.3f} ({el/60:.1f} min)", flush=True)
                log["history"].append({"step": step, "loss": run})
            if step % a.ckpt_every == 0 and step < steps:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                            "step": step, "log": log}, ck)
            if step >= steps:
                done = True
                break
        epoch += 1

    log["after"] = {"corpus_eval_loss": mlm_loss(model, eval_corpus, tok, device, amp, a.max_len),
                    "sentimix_val_loss": mlm_loss(model, eval_sentimix, tok, device, amp, a.max_len)}
    for k in ("corpus_eval_loss", "sentimix_val_loss"):
        log.setdefault("perplexity", {})[k] = {"before": math.exp(log["before"][k]),
                                               "after": math.exp(log["after"][k])}
    print("after:", log["after"])
    print("perplexity:", json.dumps(log["perplexity"], indent=1))

    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "pretrain_info.json").write_text(json.dumps(
        {"args": vars(a), "base_hf_id": hf_id, "steps": steps, "corpus": cstats, **log}, indent=2))
    ck.unlink(missing_ok=True)
    (out / "DONE").write_text("ok")
    print(f"saved -> {out}\nnext: python -m src.sweep --models {a.name} --save_model")


if __name__ == "__main__":
    main()
