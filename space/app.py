"""Restaurant Complaint Tagger: a page for diners and owners on top of nrr26/yelp-complaint-tagger."""
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


# Heads-up tips for someone deciding where to eat, based only on the complaints pasted in.
DINER_TIPS = {
    "wait_time": "Expect waits. Book ahead or go outside peak hours.",
    "staff_attitude": "Some diners complain about service. If service matters for your occasion, go at a quieter time or consider another spot.",
    "food_quality": "Reviewers mention food problems. Look at recent photos and order the dishes others praise.",
    "order_accuracy": "Orders sometimes come out wrong. Check takeout and delivery orders before you leave.",
    "price_value": "Some diners find it overpriced. Check the menu prices before you go.",
    "hygiene_safety": "There are cleanliness or food-safety complaints. Check the latest local health inspection results first.",
    "ambience": "Noise or comfort complaints. Ask for a quieter table, or pick another place for a long conversation.",
    "billing_policy": "Billing complaints. Check your bill, and ask about deposits or cancellation rules up front.",
}

# Fictional restaurants with review sets written for the demo.
SAMPLES = {
"Tamarind & Tandoor · San Jose, CA · Indian": """The butter chicken was bland and the naan was burnt on one side.
Biryani was dry and the rice tasted reheated.
Waited 40 minutes for our food on a quiet Tuesday.
The paneer tikka was cold in the middle.
Samosas were greasy and stale.""",
"Pacific Patty Shack · Los Angeles, CA · American": """Waited over an hour for two burgers and fries.
Nobody took our order for 25 minutes after we sat down.
The line was out the door and only one cashier was working.
Our milkshakes took 20 minutes and the burger was overcooked.
We were seated quickly but then forgotten for half an hour.""",
"Cascade Pasta Co. · Seattle, WA · Italian": """$34 for a small plate of spaghetti. Not worth it.
Tiny portions for the price, we left hungry.
The tiramisu was $16 and the size of a matchbox.
Overpriced wine and the lasagna was just okay.
Paid $90 for two people and still went for pizza after.""",
"Pike Spice House · Seattle, WA · Indian": """Ordered lamb vindaloo for delivery and got chicken korma instead.
Our order was missing the garlic naan and the raita.
They forgot the rice with both curries.
Asked for mild and got extra spicy. Wrong order again.
Half the menu was sold out on a Friday night.""",
"Mulberry Row Trattoria · New York, NY · Italian": """The waiter rolled his eyes when we asked about gluten-free pasta.
Hostess was rude and told us to hurry up and order.
They charged us for bread we never ordered and the manager refused to remove it.
Our server was dismissive and never smiled once.
We were overcharged on the wine and had to argue to get it fixed.""",
"Hudson Corner Diner · Brooklyn, NY · American": """Found a hair in my pancakes and the table was sticky.
The restroom was filthy and had no soap.
Saw a cockroach near the booth by the window.
Music was so loud we could not hear each other.
I got food poisoning after the chicken sandwich.""",
}


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


def _analyse(reviews):
    """Split pasted text into reviews and tag each one."""
    lines = [l.strip()[:MAX_CHARS] for l in (reviews or "").splitlines() if l.strip()][:MAX_REVIEWS]
    probs = score(lines) if lines else []
    found = [[l for i, l in enumerate(LABELS) if p[i] > THRESHOLDS[l]] for p in probs]
    counts = pd.Series([l for f in found for l in f], dtype=object).value_counts()
    return lines, found, counts


def _chart(counts, title):
    if not len(counts):
        return None
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(counts) + 1.2))
    order = counts.sort_values()
    ax.barh([NAMES[l] for l in order.index], order.values, color="#C8323C")
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlabel("Reviews mentioning it")
    ax.xaxis.get_major_locator().set_params(integer=True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def _table(lines, found):
    return pd.DataFrame({"Review": lines, "Problems": [", ".join(NAMES[l] for l in f) or "None found" for f in found]})


EMPTY = "Paste at least one review, one per line, or pick a sample restaurant."


def owner_report(reviews: str):
    """For restaurant owners: tag 1-2 star reviews (one per line), count the complaints and give the best next action.

    Args:
        reviews: Several 1-2 star reviews of one restaurant, one review per line.

    Returns:
        A plain-English action plan (best next action plus the top 3 problems with steps), a chart of
        complaint counts, and a table of each review with its complaint types.
    """
    lines, found, counts = _analyse(reviews)
    if not lines:
        return EMPTY, None, pd.DataFrame(columns=["Review", "Problems"])
    if not len(counts):
        return ("No complaints found. This tool is built for 1-2 star reviews; happy reviews usually show nothing.",
                None, _table(lines, found))
    top, n = counts.index[0], int(counts.iloc[0])
    parts = [f"## Best next action\n**{PLAYBOOK[top][0]}**\n\n"
             f"Why: {NAMES[top].lower()} comes up in {n} of {len(lines)} reviews, more than anything else.\n",
             "## Top problems and what to do"]
    for rank, (label, c) in enumerate(counts.head(3).items(), 1):
        parts.append(f"### {rank}. {NAMES[label]} ({c} of {len(lines)} reviews)")
        parts.extend(f"- {a}" for a in PLAYBOOK[label])
        parts.append("")
    parts.append("_A starting checklist, not professional advice. Run it again next week to see if the top bar got smaller._")
    return "\n".join(parts), _chart(counts, "What customers complain about"), _table(lines, found)


def diner_tips(reviews: str):
    """For diners: summarise what people complain about at a restaurant and give practical tips before visiting.

    Args:
        reviews: Several 1-2 star reviews of one restaurant, one review per line.

    Returns:
        Plain-English tips for a visitor, based only on the complaints provided, and a chart of complaint counts.
    """
    lines, found, counts = _analyse(reviews)
    if not lines:
        return EMPTY, None
    if not len(counts):
        return "No clear complaints in these reviews.", None
    parts = [f"## Before you go\nBased on **{len(lines)} negative reviews**, here is what people complain about most:\n"]
    for label, c in counts.head(3).items():
        parts.append(f"- **{NAMES[label]}** ({c} of {len(lines)}): {DINER_TIPS[label]}")
    parts.append("\n_These tips only reflect the complaints you pasted. Read recent positive reviews too before deciding._")
    return "\n".join(parts), _chart(counts, "What diners complain about")


def load_sample(name):
    return SAMPLES.get(name, "")


with gr.Blocks(title="Restaurant Complaint Tagger") as demo:
    gr.Markdown(
        "# Restaurant Complaint Tagger\n"
        "Paste a restaurant's **1-2 star reviews**, one per line, and see what people complain about.\n\n"
        "A learning project · [model](https://huggingface.co/nrr26/yelp-complaint-tagger) (non-commercial) · "
        "[code](https://github.com/namsarena/yelp-complaint-tagger) · reviews you paste are not stored by this app"
    )
    with gr.Tab("For diners"):
        gr.Markdown("Thinking of eating somewhere? Paste its bad reviews and get a quick heads-up.")
        d_pick = gr.Dropdown(list(SAMPLES), label="Or try a sample restaurant (fictional)", value=None)
        d_box = gr.Textbox(label="Reviews (one per line, up to 200)", lines=8, placeholder="Paste reviews here...")
        d_run = gr.Button("Show me what to expect", variant="primary")
        d_tips = gr.Markdown()
        d_chart = gr.Plot(label="Complaints")
        d_pick.change(load_sample, inputs=d_pick, outputs=d_box, api_visibility="private")
        d_run.click(diner_tips, inputs=d_box, outputs=[d_tips, d_chart])
    with gr.Tab("For restaurant owners"):
        gr.Markdown("Paste this week's 1-2 star reviews and get your best next action.")
        o_pick = gr.Dropdown(list(SAMPLES), label="Or try a sample restaurant (fictional)", value=None)
        o_box = gr.Textbox(label="Reviews (one per line, up to 200)", lines=8, placeholder="Paste reviews here...")
        o_run = gr.Button("Find my best next action", variant="primary")
        o_plan = gr.Markdown()
        o_chart = gr.Plot(label="Complaints this week")
        o_table = gr.Dataframe(label="Each review", wrap=True)
        o_pick.change(load_sample, inputs=o_pick, outputs=o_box, api_visibility="private")
        o_run.click(owner_report, inputs=o_box, outputs=[o_plan, o_chart, o_table])
    gr.api(tag_review)
    gr.Markdown(
        "**Limits:** built for complaints only (4-5 star reviews may get false tags), can misread negation "
        "(\"not rude\"), may tag food quality when the real problem is a wrong order, and learned from "
        "AI-labelled examples. English only. Sample restaurants and their reviews are fictional."
    )

if __name__ == "__main__":
    demo.launch(mcp_server=True)
