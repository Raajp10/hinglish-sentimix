"""Build a tiny random BERT + WordPiece tokenizer from train.csv, saved to
results/tiny_bert. Lets you smoke-test src.train / src.sweep on CPU without
downloading weights:  python -m src.train --model results/tiny_bert --epochs 1 --limit 200
"""
from pathlib import Path

import pandas as pd
from tokenizers import BertWordPieceTokenizer
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "results" / "tiny_bert"
out.mkdir(parents=True, exist_ok=True)
texts = pd.read_csv(ROOT / "data" / "processed" / "train.csv")["text"].fillna("").tolist()
wp = BertWordPieceTokenizer(lowercase=True)
wp.train_from_iterator(texts, vocab_size=3000)
wp.save_model(str(out))
tok = BertTokenizerFast(vocab_file=str(out / "vocab.txt"))
tok.save_pretrained(out)
cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=64, num_hidden_layers=2,
                 num_attention_heads=2, intermediate_size=128, num_labels=3)
BertForSequenceClassification(cfg).save_pretrained(out)
print("tiny model ->", out)
