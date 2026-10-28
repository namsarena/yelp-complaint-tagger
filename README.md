# Yelp complaint tagger

A small multi-label model that tags *why* a restaurant got a bad review (wait time, staff attitude, food quality, order accuracy, price/value, hygiene/safety, ambience, billing/policy). Target: published on the Hugging Face Hub as `nrr26/yelp-complaint-tagger`.

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

## Progress
- [x] 1. Label design (v0.1)
- [ ] 2-3. Colab setup, load data
- [ ] 4. LLM labelling
- [ ] 5. Hand-check gold set
- [ ] 6. Baseline
- [ ] 7. Fine-tune DistilBERT
- [ ] 8. Evaluate
- [ ] 9. Model card + publish
- [ ] 10. Gradio demo
