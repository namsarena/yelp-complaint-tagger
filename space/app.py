"""Restaurant Complaint Tagger: a page for diners and owners on top of nrr26/yelp-complaint-tagger.

A learning project. Outputs are automated estimates; see DISCLAIMER."""
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

# Demo restaurants: names and reviews made up for the demo; not real businesses.
SAMPLES = {
"Tandoor Telescope · San Jose, CA": """The butter chicken was bland and the naan was burnt on one side.
Biryani was dry and the rice tasted reheated.
Waited 40 minutes for our food on a quiet Tuesday.
The paneer tikka was cold in the middle.
Samosas were greasy and stale.""",
"Patty Parallax Burgers · Los Angeles, CA": """Waited over an hour for two burgers and fries.
Nobody took our order for 25 minutes after we sat down.
The line was out the door and only one cashier was working.
Our milkshakes took 20 minutes and the burger was overcooked.
We were seated quickly but then forgotten for half an hour.""",
"Rigatoni Rainfall · Seattle, WA": """$34 for a small plate of spaghetti. Not worth it.
Tiny portions for the price, we left hungry.
The tiramisu was $16 and the size of a matchbox.
Overpriced wine and the lasagna was just okay.
Paid $90 for two people and still went for pizza after.""",
"Curry Compass Takeout · Seattle, WA": """Ordered lamb vindaloo for delivery and got chicken korma instead.
Our order was missing the garlic naan and the raita.
They forgot the rice with both curries.
Asked for mild and got extra spicy. Wrong order again.
Half the menu was sold out on a Friday night.""",
"Gnocchi Gallery Trattoria · New York, NY": """The waiter rolled his eyes when we asked about gluten-free pasta.
Hostess was rude and told us to hurry up and order.
They charged us for bread we never ordered and the manager refused to remove it.
Our server was dismissive and never smiled once.
We were overcharged on the wine and had to argue to get it fixed.""",
"Skillet Satellite Diner · Brooklyn, NY": """Found a hair in my pancakes and the table was sticky.
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


EMPTY = "Pick a demo restaurant, or switch to **Use my own reviews** and paste at least one review (one per line)."

NOTE = ("\n\n---\n_Automated estimate from a small AI model. It only reads the text you pasted, can be wrong, "
        "and is not a rating, inspection or endorsement of any business. See the disclaimer below._")


def owner_report(reviews: str, restaurant: str = ""):
    """For restaurant owners: tag 1-2 star reviews (one per line), count the complaints and give the best next action.

    Args:
        reviews: Several 1-2 star reviews of one restaurant, one review per line.
        restaurant: Optional restaurant name, used only to label the report.

    Returns:
        A plain-English action plan (best next action plus the top 3 problems with steps), a chart of
        complaint counts, and a table of each review with its complaint types.
    """
    lines, found, counts = _analyse(reviews)
    if not lines:
        return EMPTY, None, pd.DataFrame(columns=["Review", "Problems"])
    if not len(counts):
        return ("No complaints found. This tool is built for 1-2 star reviews; happy reviews usually show nothing." + NOTE,
                None, _table(lines, found))
    top, n = counts.index[0], int(counts.iloc[0])
    head = f"# {restaurant.strip()[:80]}\n" if (restaurant or "").strip() else ""
    parts = [head + f"## ✅ Your best next action\n**{PLAYBOOK[top][0]}**\n\n"
             f"Why: **{NAMES[top].lower()}** comes up in {n} of {len(lines)} reviews, more than anything else.\n",
             "## Top problems and what to try"]
    for rank, (label, c) in enumerate(counts.head(3).items(), 1):
        parts.append(f"### {rank}. {NAMES[label]} ({c} of {len(lines)} reviews)")
        parts.extend(f"- {a}" for a in PLAYBOOK[label])
        parts.append("")
    parts.append("_General ideas to consider, not professional, legal or health advice. Run it again next week to see if the top bar got smaller._")
    return "\n".join(parts) + NOTE, _chart(counts, "What customers complain about"), _table(lines, found)


def diner_tips(reviews: str, restaurant: str = ""):
    """For diners: summarise what people complain about at a restaurant and give practical tips before visiting.

    Args:
        reviews: Several 1-2 star reviews of one restaurant, one review per line.
        restaurant: Optional restaurant name, used only to label the tips.

    Returns:
        Plain-English tips for a visitor, based only on the complaints provided, and a chart of complaint counts.
    """
    lines, found, counts = _analyse(reviews)
    if not lines:
        return EMPTY, None
    if not len(counts):
        return "No clear complaints in these reviews." + NOTE, None
    head = f"# {restaurant.strip()[:80]}\n" if (restaurant or "").strip() else ""
    parts = [head + f"## Before you book\nIn the **{len(lines)} negative review{'s' if len(lines) != 1 else ''}** analysed, people mention these most:\n"]
    for label, c in counts.head(3).items():
        parts.append(f"- **{NAMES[label]}** ({c} of {len(lines)}): {DINER_TIPS[label]}")
    parts.append("\n_This only reflects the reviews analysed here, not the whole restaurant. Read recent positive reviews too, "
                 "and check official health inspection records for any safety concern._")
    return "\n".join(parts) + NOTE, _chart(counts, "What diners complain about")


def load_sample(name):
    """Fill the hidden review box and the name field from a demo restaurant."""
    return SAMPLES.get(name, ""), (name or "").split(" · ")[0]


def switch_mode(mode):
    """Show the demo picker or the 'my own reviews' fields."""
    own = mode == MODES[1]
    return (gr.update(visible=not own, value=None), gr.update(visible=own, value=""),
            gr.update(visible=own, value="", label="Reviews: one per line, up to 200"),
            gr.update(visible=not own, open=False), gr.update(value=""))


MODES = ["Try a demo restaurant", "Use my own reviews"]


HOW_IT_FINDS = """**This page never fetches reviews from any website.** Demo restaurants come with made-up reviews. To try your own:

1. Open the restaurant on a site or app you already use (Google Maps, Yelp, a delivery app).
2. Filter to **1-2 star reviews** and copy the review text.
3. Paste them here, **one review per line**.

Why not search by restaurant name? Review sites don't allow other tools to copy their reviews automatically, and their
official APIs return only a few reviews, need paid keys, and limit how review text can be reused. This free learning project
stays on the safe side: you choose exactly what gets analysed.

Don't paste names, phone numbers or other personal details."""

HOW_IT_WORKS = """1. A small AI model (66 million parameters) reads each review and tags up to 8 problem types:
   wait time, staff attitude, food quality, order accuracy, price and value, hygiene and safety, ambience, billing and policy.
2. The page counts the tags across all reviews.
3. Fixed, written rules turn the top problems into tips (diners) or actions (owners). No chatbot writes them.

The model learned from about 2,000 reviews in Yelp's public research dataset, labelled by another AI, so **accuracy varies**:
it's built for complaints only, can misread negation ("not rude"), and sometimes tags food quality when the real
problem is a wrong order."""

HOW_TRAINED = """**Data.** Yelp's public research dataset of reviews ([Yelp/yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full)),
released by Yelp for non-commercial use. Nothing was scraped. From its 1-2 star reviews, 3,500 were picked at random.

**Labels.** A free, open AI model (Qwen3-4B-Instruct) tagged each review using a written guide: 8 problem types plus rules
such as "praise isn't a complaint" and "slow isn't rude". Reviews not about restaurants were dropped, leaving 2,361.
A hand check of a sample found about 18 in 20 labels right.

**Training.** DistilBERT, a small model (66 million parameters), learned from 1,800 of those reviews on a free Google Colab
GPU in about 15 minutes. Rare problems got extra weight so they aren't ignored, and each problem type got its own cut-off.

**Results.** On 354 reviews it never saw, it scored macro-F1 0.71, against 0.58 for a simple keyword method. In acceptance
tests on 110 test restaurants, the owner's best next action was right 84 times. It is weakest on wrong orders.

Full details: the [model card](https://huggingface.co/nrr26/yelp-complaint-tagger) and the
[notebook on GitHub](https://github.com/namsarena/yelp-complaint-tagger)."""

DISCLAIMER = """**Learning project, for education and demonstration only.** This page is a free, non-commercial learning project
by an individual. It is provided **"as is" and "as available", without warranties of any kind**, express or implied,
including accuracy, completeness, reliability, or fitness for a particular purpose.

**No reliance.** Results are automated estimates from a small AI model and can be incomplete or wrong. They reflect only the
text entered on this page and are **not statements of fact** about any business, and not a rating, review, health inspection,
certification, or endorsement. Don't make decisions about a business, a person, or your health based only on this page.
Tips and actions are general information, **not professional, legal, health, financial, or business advice**. For food-safety
concerns, check official local health inspection records.

**Limitation of liability.** To the fullest extent permitted by law, the author is not liable for any loss or damage arising
from use of, or reliance on, this page or its results. You use it at your own risk.

**Your input.** Only enter text you have the right to use. Don't enter personal information (names, phone numbers, emails).
Text is processed in memory and not saved by this app; the hosting platform (Hugging Face) may keep standard request logs.

**No affiliation.** Not affiliated with or endorsed by Yelp, Google, or any restaurant. Trademarks belong to their owners.
The model was trained on Yelp's public research dataset under its non-commercial terms; no reviews were scraped, and this
page does not fetch reviews from any website. The demo restaurants and their reviews were made up for this demo; any
resemblance to a real business is coincidental.

**Concerns.** If you think anything here is inaccurate or should be removed, post in this Space's Community tab and it will
be reviewed. Model license: CC BY-NC 4.0 (non-commercial). Code: MIT."""

THEME = gr.themes.Soft(primary_hue="red", neutral_hue="stone")

with gr.Blocks(title="Restaurant Complaint Tagger") as demo:
    gr.Markdown(
        "# 🍽️ Restaurant Complaint Tagger\n"
        "Find out **why** people leave bad reviews, in seconds. One tab helps you **decide where to eat**; "
        "the other helps owners **decide what to fix first**.\n\n"
        "_Free learning project · automated estimates, accuracy varies · "
        "[model](https://huggingface.co/nrr26/yelp-complaint-tagger) · [code](https://github.com/namsarena/yelp-complaint-tagger)_"
    )
    with gr.Tab("🧭 Before You Book (diners)"):
        gr.Markdown("**Planning a meal out?** See what unhappy guests complained about, with a tip for each.\n\n"
                    "**Step 1.** Choose a demo restaurant, or switch to **Use my own reviews**.  \n"
                    "**Step 2.** Click **Show me what to expect**.  \n"
                    "**Step 3.** Read the tips and the chart.")
        d_mode = gr.Radio(MODES, value=MODES[0], label="Reviews to analyse", show_label=False)
        d_pick = gr.Dropdown(list(SAMPLES), label="Demo restaurants", value=list(SAMPLES)[0])
        d_name = gr.Textbox(label="Restaurant name (optional, only used to label the results)", visible=False,
                               placeholder="e.g. The corner cafe on Main St")
        d_box = gr.Textbox(label="Reviews: one per line, up to 200", lines=8, visible=False,
                              placeholder="Paste 1-2 star reviews here, one per line.\nExample: Waited 40 minutes for our food.")
        with gr.Accordion("Show the demo reviews being analysed", open=False) as d_peek:
            d_shown = gr.Markdown()
        with gr.Accordion("❓ Where do the reviews come from?", open=False):
            gr.Markdown(HOW_IT_FINDS)
        d_run = gr.Button("Show me what to expect", variant="primary")
        gr.Markdown("<small>By using this page you agree to the disclaimer at the bottom.</small>")
        d_tips = gr.Markdown()
        d_chart = gr.Plot(label="What people complain about")
        d_mode.change(switch_mode, inputs=d_mode, outputs=[d_pick, d_name, d_box, d_peek, d_tips],
                          api_visibility="private")
        d_pick.change(load_sample, inputs=d_pick, outputs=[d_box, d_name], api_visibility="private")
        d_box.change(lambda t: "\n".join(f"- {l}" for l in (t or "").splitlines() if l.strip()), inputs=d_box,
                         outputs=d_shown, api_visibility="private")
        d_run.click(diner_tips, inputs=[d_box, d_name], outputs=[d_tips, d_chart])
    with gr.Tab("🧑‍🍳 Owner's Fix-It List"):
        gr.Markdown("**Run a restaurant?** Get your **best next action** from this week's 1-2 star reviews.\n\n"
                    "**Step 1.** Choose a demo restaurant, or switch to **Use my own reviews**.  \n"
                    "**Step 2.** Click **Find my best next action**.  \n"
                    "**Step 3.** Start with the action at the top; the chart and table show the evidence.")
        o_mode = gr.Radio(MODES, value=MODES[0], label="Reviews to analyse", show_label=False)
        o_pick = gr.Dropdown(list(SAMPLES), label="Demo restaurants", value=list(SAMPLES)[1])
        o_name = gr.Textbox(label="Restaurant name (optional, only used to label the results)", visible=False,
                               placeholder="e.g. The corner cafe on Main St")
        o_box = gr.Textbox(label="Reviews: one per line, up to 200", lines=8, visible=False,
                              placeholder="Paste 1-2 star reviews here, one per line.\nExample: Waited 40 minutes for our food.")
        with gr.Accordion("Show the demo reviews being analysed", open=False) as o_peek:
            o_shown = gr.Markdown()
        with gr.Accordion("❓ Where do the reviews come from?", open=False):
            gr.Markdown(HOW_IT_FINDS)
        o_run = gr.Button("Find my best next action", variant="primary")
        gr.Markdown("<small>By using this page you agree to the disclaimer at the bottom.</small>")
        o_plan = gr.Markdown()
        o_chart = gr.Plot(label="Complaints this week")
        o_table = gr.Dataframe(label="Each review and the problems found", wrap=True)
        o_mode.change(switch_mode, inputs=o_mode, outputs=[o_pick, o_name, o_box, o_peek, o_plan],
                          api_visibility="private")
        o_pick.change(load_sample, inputs=o_pick, outputs=[o_box, o_name], api_visibility="private")
        o_box.change(lambda t: "\n".join(f"- {l}" for l in (t or "").splitlines() if l.strip()), inputs=o_box,
                         outputs=o_shown, api_visibility="private")
        o_run.click(owner_report, inputs=[o_box, o_name], outputs=[o_plan, o_chart, o_table])
    demo.load(load_sample, inputs=d_pick, outputs=[d_box, d_name], api_visibility="private")
    demo.load(load_sample, inputs=o_pick, outputs=[o_box, o_name], api_visibility="private")
    gr.api(tag_review)
    with gr.Accordion("⚙️ How it works", open=False):
        gr.Markdown(HOW_IT_WORKS)
    with gr.Accordion("📚 How the model was trained", open=False):
        gr.Markdown(HOW_TRAINED)
    with gr.Accordion("⚖️ Disclaimer and privacy", open=False):
        gr.Markdown(DISCLAIMER)
    gr.Markdown("_Learning project, provided as is. Automated estimates; accuracy varies. Not affiliated with Yelp, Google or any restaurant. "
                "Demo restaurants and their reviews are made up. Not professional advice. Open **Disclaimer and privacy** above for details._")

if __name__ == "__main__":
    demo.launch(mcp_server=True, theme=THEME)
