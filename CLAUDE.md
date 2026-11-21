# CLAUDE.md — Hinglish SentiMix benchmark (CS 52570 NLP final project, Group 7)

You are helping finish a course project: **"Lost in Code-Mixing? Benchmarking Multilingual
Transformers on Hinglish Sentiment Analysis."** Team: Raaj Patel (lead), Amartya Mishra.
The proposal is the spec. Do not add scope beyond it unless asked.

## Goals (from the proposal — these are graded)
1. Benchmark mBERT, XLM-R-base, IndicBERT, HingBERT vs TF-IDF+LR on SentiMix Hinglish, identical settings.
2. Best model ≥ 70% weighted F1 (official mBERT baseline 65.4, best system 75.0). Report honestly either way.
3. Error analysis: slice by Code-Mixing Index, length, class (esp. neutral vs negative); manual review of ~100 errors.
4. One improvement: spelling normalization (nhi/nahin → nahi) on the weakest transformer, with vs without.
Deliverables: GitHub repo (reproducible), results table, error-analysis charts, final report, slides. Optional Gradio demo.

## Layout
```
data/raw/                 SentiMix CoNLL files + test labels (do not edit)
data/processed/           generated: train/val/test.csv + stats.json
data/normalization_dict.json  generated
src/prepare.py            parse, clean, dedupe leakage, CMI + buckets
src/normalize.py          seed dict + train-mined variant dict; load_normalizer()
src/baseline.py           TF-IDF (word 1-2 + char 2-5) + LogisticRegression, C tuned on val
src/models.py             registry: mbert, xlmr, indicbert, hingbert (+ hingroberta, muril optional)
src/train.py              one fine-tuning run (plain PyTorch loop, AMP on GPU)
src/sweep.py              LR sweep on val -> best (lr, epochs) -> 3 seeds on test; resumable
src/analyze.py            tables, figures, bootstrap CI for normalization, error-review sheet, --tally
demo/app.py               Gradio side-by-side demo (loads results/models/*)
notebooks/run_gpu.ipynb   Colab/Kaggle runner for all GPU work
scripts/make_tiny_model.py  tiny random BERT for CPU smoke tests
results/                  runs/<model>[+norm]/seed<N>/{metrics.json,test_predictions.csv}, sweep/, tables/, figures/, error_review/
docs/report_outline.md    report + slides skeleton
```

## Commands
```bash
pip install -r requirements.txt            # + torch
python -m src.prepare                      # always first
python -m src.normalize
python -m src.baseline && python -m src.baseline --normalize
python -m src.sweep --models mbert xlmr indicbert hingbert --save_model   # GPU
python -m src.sweep --models <weakest> --stage final --norm_models <weakest>
python -m src.analyze
python -m src.analyze --tally results/error_review/to_annotate.csv        # after manual labels
python -m demo.app
# CPU smoke test (no downloads):
python scripts/make_tiny_model.py && python -m src.train --model results/tiny_bert --epochs 1 --limit 300
```

## Hard rules
- **Never tune anything on test.** LR/epochs/C are chosen on val only. Test is scored once per final run.
- All models use the same processed splits and the same cleaning. Normalization is the only allowed difference, and only in `+norm` runs.
- Normalization dict is mined from **train only**.
- Keep the leakage fix in `prepare.py`: the released train file contains ~2.7k tweets that also appear
  in val. They are removed from train (final: 12,204 train / 3,000 val / 3,000 test). Mention this in the report.
- Don't commit model weights (`results/models/` is gitignored). Commit `results/runs`, `tables`, `figures`.
- Seeds: 42, 43, 44. Report mean ± std.
- Use `python -m src.<module>` from the repo root (package imports).

## Current status (done and verified on CPU)
- [x] Data pipeline, mojibake fix (emoji), mention/URL/RT removal, leakage dedupe, CMI + buckets
- [x] Normalization dictionary (65 seed + ~300 mined; ~3% of tokens changed)
- [x] Baseline: TF-IDF+LR **test wF1 68.0** (val 61.8); +norm 68.4 (Δ +0.4, 95% CI [-0.2, +1.0], not significant)
- [x] train.py / sweep.py smoke-tested with tiny model (CPU); analyze.py runs end-to-end on baseline runs
- [ ] GPU: sweep + 3 seeds for 4 transformers (use notebooks/run_gpu.ipynb on Colab T4 / Kaggle P100)
- [ ] Normalization experiment on weakest transformer
- [ ] `python -m src.analyze` with all runs
- [ ] Manual error review: humans fill `error_category` in results/error_review/to_annotate.csv (≈100 rows), then `--tally`
- [ ] Final report + slides (docs/report_outline.md)
- [ ] Optional: demo

Observation to explain in the report: test scores run ~6 points above val for the baseline. Val and test
come from different collection batches; report both and don't treat the gap as a bug.

## Tasks for you (Claude Code), in order
1. Run the CPU smoke test above. Fix anything broken before touching GPU work.
2. `git init`, commit, help push to GitHub (user creates the empty repo). Put the URL into `notebooks/run_gpu.ipynb` (REPO_URL).
3. The user runs the notebook on Colab. When they pull results back (`git pull` or copy `results/` from Drive),
   run `python -m src.analyze` and summarize: best model, whether the 70% target was met, CMI/length trends,
   neutral↔negative confusion, normalization delta + CI.
4. If a model fails to load:
   - IndicBERT: needs `sentencepiece`; tokenizer uses `use_fast=False, keep_accents=True` (already set).
     If HF access/loading fails, try `ai4bharat/IndicBERTv2-MLM-only` and note the substitution in the report.
   - HingBERT is BERT-base; if lr sweep diverges at 5e-5, it's fine, the sweep picks the best LR.
   - OOM on T4: `--batch_size 16` (sweep passes defaults; edit `train()` call or run src.train directly).
5. Pre-fill the error review to save the team time: for each row in to_annotate.csv suggest an `error_category`
   in a **separate column `suggested_category`**. Humans make the final call in `error_category`. Don't fill it yourself.
6. Draft the report in `docs/report.md` from `docs/report_outline.md` using only numbers in `results/tables/`.
   Never invent or round-up numbers. Every figure referenced must exist in `results/figures/`.
7. Draft slides outline (10–12 slides) in `docs/slides.md`.

## Style
- User (Raaj) prefers terse, direct answers and shipping over explanation. Show diffs/commands, not essays.
- Keep code simple and readable; it's graded by a professor who will skim the repo.
