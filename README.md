# Yelp complaint tagger

A small multi-label model that tags *why* a restaurant got a bad review (wait time, staff attitude, food quality, order accuracy, price/value, hygiene/safety, ambience, billing/policy). Model: [nrr26/yelp-complaint-tagger](https://huggingface.co/nrr26/yelp-complaint-tagger). Average F1 0.71 vs 0.58 for a keyword baseline on 354 held-out reviews.

> Learning project, not production software. The model is licensed CC BY-NC 4.0 (non-commercial).

Data: [Yelp/yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full), 1-2 star reviews only. Yelp data is for academic, non-commercial use, so the model is released for non-commercial research.

## Layout
| Path | What |
|---|---|
| `notebooks/yelp_complaint_tagger.ipynb` | The full build: data, LLM labelling, baseline, fine-tuning, evaluation, publishing (open in Colab) |
| `labels.yaml` | Label definitions and rules (v1.1) |

**Data is not included** because Yelp's license doesn't allow redistribution; the notebook re-downloads `Yelp/yelp_review_full` and rebuilds the labels.

## Progress
- [x] 1. Label design (v1.1, see `labels.yaml`)
- [x] 2-3. Colab setup, load data
- [x] 4. LLM labelling (Qwen3-4B-Instruct, prompt v2; private dataset `nrr26/yelp-complaint-labels`)
- [x] 5. Hand checks (10-review pilot, 20-review spot check)
- [x] 6. Baseline (TF-IDF + logistic regression)
- [x] 7. Fine-tune DistilBERT (weighted loss, per-label thresholds)
- [x] 8. Evaluate, model card, publish
- [x] Notebook in `notebooks/`
- [ ] Demo video, Gradio demo
