# Lost in Code-Mixing? Benchmarking Multilingual Transformers on Hinglish Sentiment

CS 52570 (NLP), Purdue University Northwest, Group 7: Raaj Patel, Amartya Mishra.

Benchmarks TF-IDF+LR, mBERT, XLM-R, IndicBERT and HingBERT on SemEval-2020 Task 9 (SentiMix Hinglish),
with error analysis by Code-Mixing Index, length and class, plus a spelling-normalization experiment.

## Quickstart
```bash
pip install torch -r requirements.txt
python -m src.prepare        # clean data, dedupe leakage, CMI
python -m src.normalize      # build spelling-normalization dictionary
python -m src.baseline       # TF-IDF + LR (CPU, ~1 min)
python -m src.baseline --normalize
```
GPU part: open `notebooks/run_gpu.ipynb` in Colab (T4) or Kaggle and run all cells (~3 h total, resumable). Or:
```bash
python -m src.sweep --models mbert xlmr indicbert hingbert --save_model
python -m src.sweep --models <weakest> --stage final --norm_models <weakest>
python -m src.analyze
```
Demo: `python -m demo.app`

## Protocol
- Splits: official train / validation / test. 2,783 train tweets duplicated in val/test (or within train) are removed →
  12,204 / 3,000 / 3,000.
- Tuning on validation only: LR ∈ {1, 2, 3, 5}e-5, up to 4 epochs, seed 42. Best config re-run with seeds 42/43/44, mean ± std on test.
- Metrics: weighted F1 (official), macro F1, per-class F1. Normalization Δ with a paired bootstrap 95% CI.

## Results
See `results/tables/main_results.md` and `results/figures/`. Baseline TF-IDF+LR: **68.0** test weighted F1
(official mBERT baseline 65.4; best 2020 system 75.0).

## Layout
See `CLAUDE.md`.

## Data
SentiMix (Patwa et al., 2020), CC BY 4.0. See `data/README.md`.
