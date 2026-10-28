# LinkedIn post: Yelp complaint tagger

**Attach:** `carousel.pdf` as a document post (LinkedIn: Start a post › "+" › Add a document; title it "I built a model that reads bad reviews").
**Link:** put the Hugging Face link in the first comment; LinkedIn shows posts with outside links to fewer people.
**Before posting:** make the model public on Hugging Face (Settings › Change model visibility).

---

A 1-star review tells a restaurant that something went wrong. It doesn't say what.

So I built a small AI model that reads bad reviews and tags why: wait time, rude staff, food quality, wrong orders, price, hygiene, ambience or billing.

It's my first model on Hugging Face, built in one day with free tools:

→ Wrote a labelling guide: 8 complaint types and a few rules ("praise isn't a complaint", "slow isn't rude")
→ A free open-source LLM labelled 3,500 Yelp reviews; I hand-checked a sample (about 18 of 20 right)
→ Fine-tuned DistilBERT, a 66M-parameter model, on a free Colab GPU
→ Average F1 went from 0.58 (keyword baseline) to 0.71, better on all 8 labels

What surprised me most: the model copies its teacher. It learned the LLM labeller's habits, good and bad. So the model card lists where it fails, not only where it works.

This is a learning project, not a product, and the model is non-commercial because it's trained on Yelp data.

If your team reads a lot of customer feedback, which complaint types would you track? Link in the comments.

#MachineLearning #HuggingFace #AI #ProductManagement #LearningInPublic

---

**First comment:**
Try the model: https://huggingface.co/nrr26/yelp-complaint-tagger
The model card covers the data, the results and the limitations.

**Alt text for the document:** An 8-page carousel: the problem, what the model does with an example review, the 5 build steps, a bar chart comparing F1 scores for 8 complaint types (baseline vs model), tools and time, lessons and next steps, and the Hugging Face link.
