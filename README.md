# Yelp complaint tagger

A small multi-label model that tags *why* a restaurant got a bad review (wait time, staff attitude, food quality, order accuracy, price/value, hygiene/safety, ambience, billing/policy). Model: [nrr26/yelp-complaint-tagger](https://huggingface.co/nrr26/yelp-complaint-tagger). Average F1 0.71 vs 0.58 for a keyword baseline on 354 held-out reviews.

> Learning project, not production software. The model is licensed CC BY-NC 4.0 (non-commercial).

Data: [Yelp/yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full), 1-2 star reviews only. Yelp data is for academic, non-commercial use, so the model is released for non-commercial research.

## Layout
| Path | What | In git? |
|---|---|---|
| `labels.yaml` | Label definitions (change the version when labels change) | yes |
| `notebooks/` | Colab notebooks, one per step | yes |
| `src/` | Reusable code (labelling prompt, training script) | yes |
| `data/raw/` | Downloaded reviews (reproducible from the Hub) | no |
| `data/labeled/` | LLM-labelled training set (small CSV) | yes |
| `data/gold/` | Hand-checked test set: the source of truth | yes |
| `models/` | Trained weights (versioned on the Hugging Face Hub instead) | no |
| `launch/` | LinkedIn carousel and posts, video slides (PDF + PPTX), demo script, recording guide, teleprompter | yes |

## Progress
- [x] 1. Label design (v1.1, see `labels.yaml`)
- [x] 2-3. Colab setup, load data
- [x] 4. LLM labelling (Qwen3-4B-Instruct, prompt v2; private dataset `nrr26/yelp-complaint-labels`)
- [x] 5. Hand checks (10-review pilot, 20-review spot check)
- [x] 6. Baseline (TF-IDF + logistic regression)
- [x] 7. Fine-tune DistilBERT (weighted loss, per-label thresholds)
- [x] 8. Evaluate, model card, publish
- [ ] Notebook in `notebooks/`, demo video, Gradio demo
