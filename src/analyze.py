"""Aggregate all runs -> tables, figures, error-review sheet.

  python -m src.analyze                      # tables + figures + error sample
  python -m src.analyze --tally FILE.csv     # summarize a manually annotated review sheet

Outputs (results/):
  tables/main_results.{csv,md}       mean ± std over seeds, test set
  tables/by_cmi.csv, by_length.csv   weighted F1 per slice (mean over seeds)
  tables/neutral_negative.csv        neu<->neg confusion rates
  tables/normalization.csv           model vs model+norm, with paired bootstrap CI
  figures/*.png                      bar charts + confusion matrices
  error_review/to_annotate.csv       ~100 misclassified test tweets for manual review
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from src.common import LABELS, PROC, ROOT, RUNS  # noqa: E402
from src.prepare import CMI_BUCKETS, LEN_BUCKETS  # noqa: E402

OUT = ROOT / "results"
TAB, FIG, ERR = OUT / "tables", OUT / "figures", OUT / "error_review"
# Fixed display order -> fixed color per model (color follows the entity).
ORDER = ["tfidf_lr", "mbert", "xlmr", "indicbert", "hingbert", "hingroberta", "muril"]
PRETTY = {"tfidf_lr": "TF-IDF + LR", "mbert": "mBERT", "xlmr": "XLM-R", "indicbert": "IndicBERT",
          "hingbert": "HingBERT", "hingroberta": "HingRoBERTa", "muril": "MuRIL"}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#2b2b2a", "#6b6b66", "#e6e5e0"
ERROR_CATEGORIES = ["sarcasm/irony", "non-standard spelling", "heavy code-mixing",
                    "neutral-vs-negative boundary", "needs world/political context",
                    "mixed sentiment", "label noise (gold label looks wrong)",
                    "emoji/hashtag-only signal", "other"]


def pretty(tag: str) -> str:
    base, norm = tag.replace("+norm", ""), tag.endswith("+norm")
    return PRETTY.get(base, base) + (" + norm" if norm else "")


def color(tag: str) -> str:
    base = tag.replace("+norm", "")
    i = ORDER.index(base) if base in ORDER else len(ORDER)
    return PALETTE[i % len(PALETTE)]


def sort_tags(tags):
    return sorted(tags, key=lambda t: (ORDER.index(t.replace("+norm", "")) if t.replace("+norm", "")
                                       in ORDER else 99, t.endswith("+norm"), t))


def load_runs(include_tiny=False) -> dict[str, list[dict]]:
    runs = {}
    for d in sorted(RUNS.glob("*/seed*")):
        tag = d.parent.name
        if not include_tiny and tag.startswith("tiny"):
            continue
        if not (d / "test_predictions.csv").exists():
            continue
        runs.setdefault(tag, []).append({
            "seed": d.name, "pred": pd.read_csv(d / "test_predictions.csv", dtype={"id": str}),
            "metrics": json.loads((d / "metrics.json").read_text())})
    return {t: runs[t] for t in sort_tags(runs)}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main_table(runs) -> pd.DataFrame:
    rows = []
    keys = ["weighted_f1", "macro_f1", "accuracy", "f1_negative", "f1_neutral", "f1_positive"]
    for tag, rs in runs.items():
        row = {"model": pretty(tag), "tag": tag, "n_seeds": len(rs)}
        for k in keys:
            v = np.array([r["metrics"]["test"][k] for r in rs]) * 100
            row[k], row[k + "_std"] = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
        row["val_weighted_f1"] = np.mean([r["metrics"]["val"]["weighted_f1"] for r in rs]) * 100
        cfg = rs[0]["metrics"]["config"]
        row["lr"], row["epochs"] = cfg.get("lr", ""), cfg.get("epochs", "")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "main_results.csv", index=False)

    lines = ["| Model | Seeds | Weighted F1 | Macro F1 | Acc | F1 neg | F1 neu | F1 pos | Val wF1 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        f = lambda k: f"{r[k]:.1f} ± {r[k+'_std']:.1f}" if r["n_seeds"] > 1 else f"{r[k]:.1f}"
        lines.append(f"| {r['model']} | {r['n_seeds']} | {f('weighted_f1')} | {f('macro_f1')} | "
                     f"{f('accuracy')} | {f('f1_negative')} | {f('f1_neutral')} | {f('f1_positive')} "
                     f"| {r['val_weighted_f1']:.1f} |")
    lines += ["", "Reference (SemEval-2020 Task 9, official test): mBERT baseline 65.4, best system 75.0 (weighted F1)."]
    (TAB / "main_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return df


def slice_table(runs, test, col, buckets, fname) -> pd.DataFrame:
    order = [b[2] for b in buckets]
    rows = []
    for tag, rs in runs.items():
        for r in rs:
            m = test.merge(r["pred"][["id", "pred"]], on="id")
            for b, g in m.groupby(col):
                rows.append({"tag": tag, "bucket": b, "n": len(g),
                             "weighted_f1": 100 * f1_score(g["label"], g["pred"], average="weighted")})
    df = (pd.DataFrame(rows).groupby(["tag", "bucket"])
          .agg(weighted_f1=("weighted_f1", "mean"), std=("weighted_f1", "std"), n=("n", "first"))
          .reset_index())
    df["bucket"] = pd.Categorical(df["bucket"], order, ordered=True)
    df = df.sort_values(["tag", "bucket"])
    df.to_csv(TAB / fname, index=False)
    return df


def grouped_bars(df, tags, title, xlabel, fname):
    buckets = [b for b in df["bucket"].cat.categories if b in set(df["bucket"])]
    n = len(tags)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(buckets) + 0.15 * n * len(buckets), 4.2))
    x = np.arange(len(buckets))
    for i, t in enumerate(tags):
        sub = df[df["tag"] == t].set_index("bucket").reindex(buckets)
        ax.bar(x + (i - (n - 1) / 2) * w, sub["weighted_f1"], w * 0.92, color=color(t),
               label=pretty(t), edgecolor="white", linewidth=1)
    ns = df.drop_duplicates("bucket").set_index("bucket").reindex(buckets)["n"]
    ax.set_xticks(x, [f"{b}\nn={int(k)}" for b, k in zip(buckets, ns)])
    ax.set_ylabel("Weighted F1 (%)", color=INK)
    ax.set_xlabel(xlabel, color=INK)
    lo = max(0, np.floor(df["weighted_f1"].min() / 10) * 10 - 10)
    ax.set_ylim(lo, min(100, df["weighted_f1"].max() + 6))
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    style(ax)
    ax.legend(frameon=False, fontsize=8, ncol=min(n, 4), loc="upper left", bbox_to_anchor=(0, -0.2))
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summary_bar(df):
    fig, ax = plt.subplots(figsize=(1.5 + 1.0 * len(df), 4))
    x = np.arange(len(df))
    ax.bar(x, df["weighted_f1"], 0.6, yerr=df["weighted_f1_std"], capsize=3,
           color=[color(t) for t in df["tag"]], edgecolor="white", error_kw={"ecolor": MUTED})
    for xi, v in zip(x, df["weighted_f1"]):
        ax.text(xi, v + 1.2, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    ax.axhline(65.4, color=MUTED, ls="--", lw=1)
    ax.text(len(df) - 0.5, 65.9, "official mBERT baseline 65.4", ha="right", fontsize=8, color=MUTED)
    ax.set_xticks(x, [pretty(t) for t in df["tag"]], rotation=20, ha="right")
    ax.set_ylabel("Test weighted F1 (%)", color=INK)
    ax.set_ylim(max(0, df["weighted_f1"].min() - 15), min(100, df["weighted_f1"].max() + 6))
    ax.set_title("Test weighted F1 (mean ± std over seeds)", color=INK, loc="left", fontsize=11)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "main_results.png", dpi=200)
    plt.close(fig)


def confusion(runs):
    tags = list(runs)
    cols = min(3, len(tags))
    rws = int(np.ceil(len(tags) / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(3.4 * cols, 3.2 * rws), squeeze=False)
    rates = []
    for ax, t in zip(axes.flat, tags):
        cm = sum(confusion_matrix(r["pred"]["label"], r["pred"]["pred"], labels=LABELS) for r in runs[t])
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if cmn[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), ["neg", "neu", "pos"], fontsize=8)
        ax.set_yticks(range(3), ["neg", "neu", "pos"], fontsize=8)
        ax.set_title(pretty(t), fontsize=10, color=INK)
        ax.set_xlabel("predicted", fontsize=8, color=MUTED)
        ax.set_ylabel("gold", fontsize=8, color=MUTED)
        rates.append({"model": pretty(t), "tag": t,
                      "neg_predicted_as_neu": cmn[0, 1], "neu_predicted_as_neg": cmn[1, 0],
                      "neg_predicted_as_pos": cmn[0, 2], "pos_predicted_as_neg": cmn[2, 0],
                      "neu_predicted_as_pos": cmn[1, 2], "pos_predicted_as_neu": cmn[2, 1]})
    for ax in list(axes.flat)[len(tags):]:
        ax.axis("off")
    fig.suptitle("Row-normalized confusion matrices (test, summed over seeds)", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrices.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(rates).round(3).to_csv(TAB / "neutral_negative.csv", index=False)


def bootstrap_delta(y, pa, pb, n=2000, seed=0):
    """Paired bootstrap CI for weighted-F1(b) - weighted-F1(a) on the same test items."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    d = []
    for _ in range(n):
        s = rng.choice(idx, len(idx))
        d.append(f1_score(y[s], pb[s], average="weighted") - f1_score(y[s], pa[s], average="weighted"))
    d = np.array(d) * 100
    return np.percentile(d, 2.5), np.percentile(d, 97.5), (d <= 0).mean()


def ensemble_pred(rs) -> pd.DataFrame:
    """Average probabilities over seeds -> one prediction per tweet."""
    base = rs[0]["pred"][["id", "label"]].copy()
    p = np.mean([r["pred"][[f"p_{l}" for l in LABELS]].values for r in rs], axis=0)
    base["pred"] = [LABELS[i] for i in p.argmax(1)]
    return base


def normalization(runs):
    rows = []
    for tag in runs:
        if tag.endswith("+norm") and tag[:-5] in runs:
            a, b = runs[tag[:-5]], runs[tag]
            ea, eb = ensemble_pred(a), ensemble_pred(b)
            m = ea.merge(eb, on=["id", "label"], suffixes=("_a", "_b"))
            lo, hi, p = bootstrap_delta(m["label"].values, m["pred_a"].values, m["pred_b"].values)
            fa = np.mean([r["metrics"]["test"]["weighted_f1"] for r in a]) * 100
            fb = np.mean([r["metrics"]["test"]["weighted_f1"] for r in b]) * 100
            rows.append({"model": pretty(tag[:-5]), "wF1_without": round(fa, 2), "wF1_with_norm": round(fb, 2),
                         "delta": round(fb - fa, 2), "bootstrap_95ci_low": round(lo, 2),
                         "bootstrap_95ci_high": round(hi, 2), "p_delta_le_0": round(p, 3)})
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "normalization.csv", index=False)
    if len(df):
        print("\nNormalization:\n" + df.to_string(index=False))
    return df


def error_sample(runs, test, main_df, n=100, seed=0):
    tf = [t for t in main_df.sort_values("weighted_f1", ascending=False)["tag"] if t != "tfidf_lr"]
    best = tf[0] if tf else main_df["tag"].iloc[0]
    preds = {t: ensemble_pred(rs).set_index("id")["pred"] for t, rs in runs.items() if not t.endswith("+norm")}
    df = test.set_index("id").copy()
    for t, p in preds.items():
        df[f"pred_{t}"] = p
    wrong = df[df[f"pred_{best}"] != df["label"]].copy()
    wrong["n_models_wrong"] = sum((wrong[f"pred_{t}"] != wrong["label"]).astype(int) for t in preds)
    # stratify by gold label x CMI bucket so the sample isn't all one kind of error
    frac = min(1.0, n / len(wrong))
    parts = [g.sample(max(1, round(len(g) * frac)), random_state=seed)
             for _, g in wrong.groupby(["label", "cmi_bucket"], observed=True)]
    samp = pd.concat(parts)
    samp = samp.sample(min(n, len(samp)), random_state=seed).reset_index()
    samp["error_category"] = ""
    samp["notes"] = ""
    cols = ["id", "text", "label", f"pred_{best}"] + [f"pred_{t}" for t in preds if t != best] + \
           ["n_models_wrong", "cmi", "cmi_bucket", "n_tokens", "error_category", "notes"]
    samp[cols].to_csv(ERR / "to_annotate.csv", index=False)
    (ERR / "CATEGORIES.md").write_text(
        f"Sample: {len(samp)} test tweets misclassified by the best transformer ({pretty(best)}),\n"
        "stratified by gold label x CMI bucket. Fill `error_category` with ONE of\n"
        "(use `a; b` only if two clearly apply):\n\n" + "\n".join(f"- {c}" for c in ERROR_CATEGORIES) +
        "\n\nThen run: python -m src.analyze --tally results/error_review/to_annotate.csv\n")
    print(f"\nError sample: {len(samp)} rows from {len(wrong)} errors of {pretty(best)} -> {ERR/'to_annotate.csv'}")


def tally(path):
    df = pd.read_csv(path)
    cats = df["error_category"].fillna("").str.split(";").explode().str.strip()
    cats = cats[cats != ""]
    if cats.empty:
        print("No categories filled in yet.")
        return
    t = cats.value_counts().rename_axis("category").reset_index(name="count")
    t["percent_of_tweets"] = (100 * t["count"] / len(df)).round(1)
    t.to_csv(TAB / "error_categories.csv", index=False)
    print(t.to_string(index=False))
    by = df.assign(cat=df["error_category"].fillna("").str.split(";")).explode("cat")
    by["cat"] = by["cat"].str.strip()
    by[by["cat"] != ""].pivot_table(index="cat", columns="cmi_bucket", values="id", aggfunc="count",
                                   fill_value=0).to_csv(TAB / "error_categories_by_cmi.csv")
    fig, ax = plt.subplots(figsize=(6.5, 0.4 * len(t) + 1))
    ax.barh(t["category"][::-1], t["count"][::-1], color=PALETTE[0], height=0.6)
    for y, v in enumerate(t["count"][::-1]):
        ax.text(v + 0.3, y, str(v), va="center", fontsize=8, color=INK)
    ax.set_xlabel("tweets", color=INK)
    ax.set_title(f"Manual error review (n={len(df)})", color=INK, loc="left", fontsize=11)
    style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID)
    fig.tight_layout()
    fig.savefig(FIG / "error_categories.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tally", default=None)
    ap.add_argument("--n_errors", type=int, default=100)
    a = ap.parse_args()
    for d in (TAB, FIG, ERR):
        d.mkdir(parents=True, exist_ok=True)
    if a.tally:
        tally(a.tally)
        return
    runs = load_runs()
    if not runs:
        raise SystemExit("no runs in results/runs yet")
    test = pd.read_csv(PROC / "test.csv", dtype={"id": str})
    main_df = main_table(runs)
    summary_bar(main_df)
    plain = [t for t in runs if not t.endswith("+norm")]
    cmi = slice_table(runs, test, "cmi_bucket", CMI_BUCKETS, "by_cmi.csv")
    ln = slice_table(runs, test, "len_bucket", LEN_BUCKETS, "by_length.csv")
    grouped_bars(cmi, plain, "Weighted F1 by degree of code-mixing (CMI)", "CMI bucket", "f1_by_cmi.png")
    grouped_bars(ln, plain, "Weighted F1 by tweet length (tokens)", "length bucket", "f1_by_length.png")
    confusion(runs)
    normalization(runs)
    error_sample(runs, test, main_df[~main_df["tag"].str.endswith("+norm")], n=a.n_errors)
    print(f"\nTables -> {TAB}\nFigures -> {FIG}")


if __name__ == "__main__":
    main()
