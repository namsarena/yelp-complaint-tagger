# Yelp complaint tagger

A small multi-label model that tags *why* a restaurant got a bad review (wait time, staff attitude, food quality, order accuracy, price/value, hygiene/safety, ambience, billing/policy). Model: [nrr26/yelp-complaint-tagger](https://huggingface.co/nrr26/yelp-complaint-tagger). Average F1 0.71 vs 0.58 for a keyword baseline on 354 held-out reviews.

**Try it in your browser:** [huggingface.co/spaces/nrr26/complaint-tagger](https://huggingface.co/spaces/nrr26/complaint-tagger): a tab for diners (what to expect) and a tab for owners (best next action). No code needed.

> Learning project, not production software. The model is licensed CC BY-NC 4.0 (non-commercial).

Data: [Yelp/yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full), 1-2 star reviews only. Yelp data is for academic, non-commercial use, so the model is released for non-commercial research.

## For developers

### Use it
~~~python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "nrr26/yelp-complaint-tagger"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
th = model.config.label_thresholds          # one cut-off per label; use these, not 0.5

def tag(text):
    with torch.no_grad():
        p = torch.sigmoid(model(**tok(text, return_tensors="pt", truncation=True)).logits)[0]
    return {l: round(float(x), 2) for l, x in zip(model.config.id2label.values(), p) if x > th[l]}

tag("We waited 45 minutes and the burger was cold.")   # {'wait_time': 0.99, 'food_quality': 0.83}
~~~
Input: one negative (1-2 star) review in English. Output: the complaint types above their thresholds, with scores.

### What the model is
| | |
|---|---|
| Base | `distilbert-base-uncased` (66M parameters), multi-label head with 8 outputs |
| Training data | 2,000 Yelp 1-2 star restaurant reviews labelled by `Qwen/Qwen3-4B-Instruct-2507` (prompt v2: label rules + 5 few-shot examples) |
| Training | 8 epochs, lr 5e-5, weighted BCE (`pos_weight` capped at 10), best epoch by validation macro-F1 |
| Thresholds | Tuned per label on 200 validation reviews; stored in `config.label_thresholds` |
| Results | Macro-F1 0.71 vs 0.58 for TF-IDF + logistic regression, on 354 LLM-labelled test reviews |
| Known limits | Complaints only; weak on negation; inherits the labelling LLM's habits; see the model card |

### Extend it
The notebook runs every step in order; each step has a heading.

1. **Change the labels**: edit `labels.yaml` and the `LABELS` dict in Step 3. Keep definitions short and add rules for confusable pairs.
2. **Use your own data**: replace the Step 2 loading cell with your reviews (a `text` column). Use data you have the rights to.
3. **Relabel**: run Step 3. Pilot on 10-20 hand-labelled examples first; aim for about 70% precision and recall before the full run.
4. **Hand-check**: label 100-200 reviews yourself as a gold test set. This is the most valuable improvement to the current project.
5. **Retrain**: Steps 4-6 (split, baseline, fine-tune). About 15 minutes on a free Colab T4.
6. **Swap the base model** (optional): try `microsoft/deberta-v3-small` or `answerdotai/ModernBERT-base` in Step 6.
7. **Serve on CPU** (optional): export to ONNX, then run with ONNX Runtime (for example on AWS Lambda):
   ~~~bash
   pip install "optimum[onnxruntime]"
   optimum-cli export onnx --model nrr26/yelp-complaint-tagger --task text-classification onnx/
   ~~~
   Keep applying `label_thresholds` from `config.json` after a sigmoid.

### Ideas to contribute
- A hand-checked gold test set (200+ reviews)
- Happy and neutral reviews, so the model can say "no complaint"
- Highlighting the sentence behind each tag
- Versions for hotels, clinics, retail or support tickets
- A simple web page (Hugging Face Space) for non-technical users

Open an issue or a pull request, or post in the model's Community tab.

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

## License
| Part | License |
|---|---|
| Code in this repo (notebook, `labels.yaml`, app code) | [MIT](LICENSE) |
| Model weights on Hugging Face (`nrr26/yelp-complaint-tagger`) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): non-commercial, because they were trained on Yelp data |
| Training data | Not included. `Yelp/yelp_review_full` is under Yelp's dataset terms (non-commercial) |

This is a learning project, not production software.
