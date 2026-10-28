"""Restaurant Complaint Tagger: a simple page for owners on top of nrr26/yelp-complaint-tagger."""
import gradio as gr
import matplotlib
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "nrr26/yelp-complaint-tagger"
MAX_REVIEWS = 200
MAX_CHARS = 2000

tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()
THRESHOLDS = model.config.label_thresholds
LABELS = list(model.config.id2label.values())

NAMES = {
    "wait_time": "Wait time",
    "staff_attitude": "Staff attitude",
    "food_quality": "Food quality",
    "order_accuracy": "Order accuracy",
    "price_value": "Price and value",
    "hygiene_safety": "Hygiene and safety",
    "ambience": "Ambience",
    "billing_policy": "Billing and policy",
}

# What an owner can do about each problem; a starting checklist, not professional advice.
PLAYBOOK = {
    "wait_time": ["Check staffing against your busiest hours", "Have a host greet and seat guests within 2 minutes",
                  "Time orders from ticket to table and set a target"],
    "staff_attitude": ["Share the exact review quotes at the next team huddle", "Run a 15-minute refresher on greeting and handling complaints",
                       "Give a manager the job of checking in with every table"],
    "food_quality": ["Taste-check the dishes named in reviews before each service", "Check holding times and temperatures for hot food",
                     "Ask the kitchen lead which recipe or supplier changed recently"],
    "order_accuracy": ["Read orders back to guests and double-check takeout bags", "Add a final check at the pass before food leaves",
                       "Mark sold-out items on the menu and POS right away"],
    "price_value": ["Compare portions and prices with 2-3 nearby competitors", "Make sure portions match the menu photos and descriptions",
                    "Highlight a good-value option on the menu"],
    "hygiene_safety": ["Treat any illness report seriously: follow up with the guest and review food-safety logs",
                       "Add hourly checks for tables and restrooms", "Book a deep clean if several reviews mention dirt or pests"],
    "ambience": ["Turn music down during dinner and test it from the tables", "Check temperature and seating comfort at peak times",
                 "Fix the specific issue reviews mention (noise, smell, lighting)"],
    "billing_policy": ["Have a manager review every charge complaint within 24 hours", "Write down your refund and reservation policies and train staff on them",
                       "Check that the POS discounts and splits work correctly"],
}

EXAMPLE = """Waited 50 minutes for two burgers on a Saturday night.
Our server was rude and rolled her eyes when we asked for water.
The pasta was cold and the sauce tasted like it came from a can.
They forgot the fries and charged us for them anyway.
The hostess ignored us for 20 minutes even though the place was half empty.
$22 for a tiny salad. Not worth it.
Found a hair in my soup and the table was sticky.
Food took forever and arrived lukewarm."""


def score(texts):
    """Probability per label for each text."""
    out = []
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i + 16], return_tensors="pt", truncation=True, max_length=384, padding=True)
        with torch.no_grad():
            out.extend(torch.sigmoid(model(**enc).logits).tolist())
    return out


def tag_review(review: str) -> dict:
    """Tag the complaint types in one negative restaurant review.

    Args:
        review: The text of one 1-2 star restaurant review, in English.

    Returns:
        A dict of complaint types found (wait_time, staff_attitude, food_quality, order_accuracy,
        price_value, hygiene_safety, ambience, billing_policy) mapped to the model's confidence (0-1).
    """
    review = (review or "").strip()[:MAX_CHARS]
    if not review:
        return {}
    p = score([review])[0]
    return {l: round(p[i], 2) for i, l in enumerate(LABELS) if p[i] > THRESHOLDS[l]}


def weekly_report(reviews: str):
    """Tag many negative restaurant reviews (one per line), count the complaints and suggest what to fix first.

    Args:
        reviews: Several 1-2 star restaurant reviews, one review per line.

    Returns:
        A table of reviews with their complaint types, a chart of complaint counts, and a plain-English
        list of what to fix first.
    """
    lines = [l.strip()[:MAX_CHARS] for l in (reviews or "").splitlines() if l.strip()][:MAX_REVIEWS]
    if not lines:
        return pd.DataFrame(columns=["Review", "Problems"]), None, "Paste at least one review, one per line."
    probs = score(lines)
    found = [[l for i, l in enumerate(LABELS) if p[i] > THRESHOLDS[l]] for p in probs]
    table = pd.DataFrame({"Review": lines, "Problems": [", ".join(NAMES[l] for l in f) or "None found" for f in found]})
    counts = pd.Series([l for f in found for l in f]).value_counts()

    fig = None
    if len(counts):
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(counts) + 1))
        order = counts.sort_values()
        ax.barh([NAMES[l] for l in order.index], order.values, color="#C8323C")
        ax.set_xlabel("Reviews mentioning it")
        ax.xaxis.get_major_locator().set_params(integer=True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.tight_layout()

    if not len(counts):
        advice = "No complaints found. This tool is built for 1-2 star reviews; happy reviews usually show nothing."
    else:
        parts = [f"**{len(lines)} reviews checked.** Fix these first:\n"]
        for rank, (label, n) in enumerate(counts.head(3).items(), 1):
            parts.append(f"### {rank}. {NAMES[label]} ({n} of {len(lines)} reviews)")
            parts.extend(f"- {a}" for a in PLAYBOOK[label])
            parts.append("")
        parts.append("_Suggestions are a starting checklist, not professional advice. Run it again next week to see if the bar got smaller._")
        advice = "\n".join(parts)
    return table, fig, advice


with gr.Blocks(title="Restaurant Complaint Tagger") as demo:
    gr.Markdown(
        "# Restaurant Complaint Tagger\n"
        "Paste your **1-2 star reviews**, one per line. See what customers complain about most, and what to fix first.\n\n"
        "A learning project · [model](https://huggingface.co/nrr26/yelp-complaint-tagger) (non-commercial) · "
        "[code](https://github.com/namsarena/yelp-complaint-tagger) · reviews you paste are not stored by this app"
    )
    with gr.Tab("This week's reviews"):
        box = gr.Textbox(label="Reviews (one per line, up to 200)", lines=10, placeholder="Paste reviews here...")
        with gr.Row():
            run = gr.Button("Find the problems", variant="primary")
            gr.Button("Load 8 example reviews").click(lambda: EXAMPLE, outputs=box, api_visibility="private")
        advice = gr.Markdown()
        chart = gr.Plot(label="Complaints this week")
        table = gr.Dataframe(label="Each review", wrap=True)
        run.click(weekly_report, inputs=box, outputs=[table, chart, advice])
    with gr.Tab("One review"):
        one = gr.Textbox(label="One review", lines=3, value="We waited 45 minutes and the burger was cold.")
        res = gr.JSON(label="Problems found (with confidence)")
        gr.Button("Check", variant="primary").click(tag_review, inputs=one, outputs=res)
    gr.Markdown(
        "**Limits:** built for complaints only (4-5 star reviews may get false tags), can misread negation "
        "(\"not rude\"), and learned from AI-labelled examples. English only."
    )

if __name__ == "__main__":
    demo.launch(mcp_server=True)
