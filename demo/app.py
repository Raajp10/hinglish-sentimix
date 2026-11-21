"""Gradio demo: type a Hinglish sentence, see every saved model's prediction side by side.

Models are loaded from results/models/<name>/ (created by `--save_model`).
Run:  python -m demo.app        (add --share on Colab for a public link)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import LABELS  # noqa: E402
from src.normalize import load_normalizer  # noqa: E402

MODEL_DIR = ROOT / "results" / "models"
EXAMPLES = [
    "yaar ye movie was totally bakwaas",
    "kya mast gaana hai, loved it 😍",
    "bhai delivery 3 din late aayi, worst service ever",
    "kal meeting 10 baje hai office mein",
    "wah kya baat hai, phir se light chali gayi 👏",  # sarcasm
]


def load_all():
    models = {}
    for d in sorted(p for p in MODEL_DIR.glob("*") if (p / "config.json").exists()):
        tok = AutoTokenizer.from_pretrained(d)
        m = AutoModelForSequenceClassification.from_pretrained(d).eval()
        models[d.name] = (tok, m)
    if not models:
        raise SystemExit(f"No models in {MODEL_DIR}. Run sweep/train with --save_model first.")
    return models


def main():
    import gradio as gr

    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    models = load_all()
    try:
        norm = load_normalizer()
    except FileNotFoundError:
        norm = None

    @torch.no_grad()
    def classify(text):
        rows = []
        for name, (tok, m) in models.items():
            inp = text.lower()
            if name.endswith("+norm") and norm:
                inp = norm(inp)
            p = torch.softmax(m(**tok(inp, return_tensors="pt", truncation=True, max_length=128)).logits, -1)[0]
            i = int(p.argmax())
            rows.append([name, LABELS[i], *[f"{x:.2f}" for x in p.tolist()]])
        return rows

    demo = gr.Interface(
        fn=classify,
        inputs=gr.Textbox(label="Hinglish text", lines=2),
        outputs=gr.Dataframe(headers=["model", "prediction", "p(neg)", "p(neu)", "p(pos)"]),
        examples=EXAMPLES,
        title="Lost in Code-Mixing? Hinglish sentiment, model by model",
    )
    demo.launch(share=a.share)


if __name__ == "__main__":
    main()
