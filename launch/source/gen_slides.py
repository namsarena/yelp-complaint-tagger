"""Builds LinkedIn carousel (1080x1350) and video slides (1920x1080) as HTML for Chrome print-to-PDF."""
F1 = [("wait_time", .72, .79), ("staff_attitude", .74, .78), ("food_quality", .86, .87), ("order_accuracy", .25, .60),
      ("price_value", .67, .73), ("hygiene_safety", .52, .71), ("ambience", .42, .64), ("billing_policy", .46, .61)]
URL = "huggingface.co/nrr26/yelp-complaint-tagger"
DISC = "Learning project · not production software · model licensed CC BY-NC 4.0 (non-commercial)"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@500;600&display=swap');
:root{--paper:#F5F5F1;--ink:#1C1F26;--muted:#5E636D;--red:#C8323C;--mustard:#E0A22B;--line:#D9D9D2;--chip:#FFFFFF;--dark:#1C1F26;
--display:'Bricolage Grotesque',Georgia,serif;--body:'Source Sans 3',Arial,sans-serif;--mono:'IBM Plex Mono',Menlo,monospace}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--body);color:var(--ink);background:var(--paper)}
.page{width:W;height:H;padding:PADpx;position:relative;display:flex;flex-direction:column;gap:GAPpx;overflow:hidden;page-break-after:always;background:var(--paper)}
.page.dark{background:var(--dark);color:#F5F5F1}
.eyebrow{font-family:var(--mono);font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:var(--red);font-size:EYEpx}
.dark .eyebrow{color:var(--mustard)}
h1,h2{font-family:var(--display);font-weight:800;line-height:1.02;letter-spacing:-.02em;text-wrap:balance}
h1{font-size:H1px} h2{font-size:H2px}
p,li{font-size:BODYpx;line-height:1.35}
.muted{color:var(--muted)} .dark .muted{color:#B9BCC3}
.foot{position:absolute;left:PADpx;right:PADpx;bottom:FOOTBpx;display:flex;justify-content:space-between;font-family:var(--mono);font-size:FOOTpx;color:var(--muted);border-top:2px dashed var(--line);padding-top:14px}
.dark .foot{color:#9EA2AA;border-color:#3A3E47}
.ticket{background:var(--chip);border:2px solid var(--ink);border-radius:6px;padding:28px 32px;box-shadow:8px 8px 0 var(--ink)}
.dark .ticket{color:var(--ink)}
.ticket .rv{font-size:BODYpx;line-height:1.35;font-style:italic}
.chips{display:flex;flex-wrap:wrap;gap:12px;margin-top:20px}
.chip{font-family:var(--mono);font-weight:600;font-size:CHIPpx;border:2px solid var(--ink);border-radius:999px;padding:6px 16px;background:var(--paper)}
.chip.hot{background:var(--red);color:#fff;border-color:var(--red)}
.chip b{font-weight:600;opacity:.75;margin-left:8px}
.grid{display:grid;gap:18px}
.card{border:2px solid var(--ink);border-radius:8px;padding:22px 26px;background:var(--chip)}
.card h3{font-family:var(--display);font-size:H3px;font-weight:700;margin-bottom:8px}
.card p{font-size:SMALLpx;color:var(--muted)}
.dark .card{background:#262A33;border-color:#3A3E47}
.dark .card p{color:#C7CAD0}
.step{display:grid;grid-template-columns:NUMWpx 1fr;gap:20px;align-items:start}
.num{font-family:var(--mono);font-weight:600;font-size:NUMFpx;color:var(--red);line-height:1}
.big{font-family:var(--display);font-weight:800;font-size:BIGpx;line-height:1;color:var(--red)}
.bars{display:flex;flex-direction:column;gap:14px}
.bar{display:grid;grid-template-columns:LBLWpx 1fr 70px;align-items:center;gap:14px;font-family:var(--mono);font-size:BARFpx}
.track{position:relative;height:30px;background:#E6E6E0;border-radius:3px}
.b0{position:absolute;left:0;top:0;height:13px;background:#9DA1A9;border-radius:2px}
.b1{position:absolute;left:0;bottom:0;height:15px;background:var(--red);border-radius:2px}
.legend{display:flex;gap:28px;font-family:var(--mono);font-size:BARFpx;color:var(--muted)}
.sw{display:inline-block;width:18px;height:12px;margin-right:8px;vertical-align:middle}
.disc{background:var(--mustard);color:var(--ink);font-family:var(--mono);font-weight:600;font-size:FOOTpx;padding:10px 16px;border-radius:4px;align-self:flex-start}
ul.clean{list-style:none;display:flex;flex-direction:column;gap:14px}
ul.clean li::before{content:"→ ";color:var(--red);font-weight:600}
@page{size:W H;margin:0}
"""

def css_for(kind):
    if kind == "carousel":
        v = dict(W="1080px", H="1350px", PAD=84, GAP=34, EYE=24, H1=92, H2=70, H3=30, BODY=32, SMALL=24, FOOT=19, FOOTB=44,
                 CHIP=24, NUMW=70, NUMF=40, BIG=150, LBLW=300, BARF=22)
    else:
        v = dict(W="1920px", H="1080px", PAD=110, GAP=36, EYE=26, H1=110, H2=76, H3=34, BODY=34, SMALL=26, FOOT=20, FOOTB=48,
                 CHIP=26, NUMW=80, NUMF=44, BIG=150, LBLW=300, BARF=24)
    c = CSS
    for k in sorted(v, key=len, reverse=True):
        c = c.replace(k + "px", f"{v[k]}px" if isinstance(v[k], int) else v[k]).replace(k, str(v[k])) if k in ("W", "H") else c.replace(k + "px", f"{v[k]}px")
    return c

def foot(n, total, dark=False):
    return f'<div class="foot"><span>{DISC}</span><span>{n}/{total}</span></div>'

def chips(items):
    return '<div class="chips">' + "".join(f'<span class="chip{" hot" if hot else ""}">{t}{f"<b>{s}</b>" if s else ""}</span>' for t, s, hot in items) + "</div>"

def bars():
    rows = "".join(f'<div class="bar"><span>{n}</span><div class="track"><div class="b0" style="width:{b*100:.0f}%"></div>'
                   f'<div class="b1" style="width:{m*100:.0f}%"></div></div><span>{m:.2f}</span></div>' for n, b, m in F1)
    leg = '<div class="legend"><span><i class="sw" style="background:#9DA1A9"></i>Keyword baseline</span><span><i class="sw" style="background:var(--red)"></i>My model</span><span>F1 score, 0–1</span></div>'
    return f'<div class="bars">{rows}</div>{leg}'

TOOLS = [("Google Colab", "Free T4 GPU for labelling and training"), ("Hugging Face Hub", "Private dataset + public model hosting"),
         ("Qwen3-4B-Instruct", "Free open LLM that labelled 3,500 reviews"), ("DistilBERT", "66M-parameter model I fine-tuned"),
         ("Python libraries", "transformers, datasets, scikit-learn, pandas"), ("Claude", "AI coach for each step; git for versions")]
TIME = [("Label guide + setup", "45 min"), ("Data + pilot labelling", "1.5 h"), ("Full labelling run", "40 min"),
        ("Baseline + training", "1 h"), ("Checks + model card", "1 h"), ("Publish", "15 min")]
IMPROVE = [("A bigger human-checked test set", "Score against people, not only the AI labeller"),
           ("Add happy and neutral reviews", "So it can say “no complaint”"),
           ("More rare examples", "Billing and wrong orders need more data"),
           ("Show the evidence", "Highlight the sentence behind each tag"),
           ("A weekly dashboard", "Run it on every new review automatically")]
USES = [("Owners and managers", "Weekly complaint breakdown: fix the right thing first"),
        ("Groups and franchises", "Compare locations; send help where wait times spike"),
        ("Delivery platforms", "Route each complaint to the right team"),
        ("Reputation agencies", "Hundreds of reviews into a one-page monthly report"),
        ("Restaurant software", "Review insights inside POS or reservation tools"),
        ("Food-safety teams", "Surface hygiene complaints fast for follow-up")]

def table_rows(rows):
    return "".join(f'<div class="card"><h3>{a}</h3><p>{b}</p></div>' for a, b in rows)

def carousel():
    T = 8
    s = []
    s.append(f'''<section class="page dark"><span class="eyebrow">My first model on Hugging Face</span>
<h1>I built a model that reads bad restaurant reviews and tells you <span style="color:var(--mustard)">why</span>.</h1>
<div class="ticket"><p class="rv">“We waited 45 minutes and the burger was cold.”</p>{chips([("wait_time","0.99",True),("food_quality","0.83",True)])}</div>
<p class="muted">1 day · free tools · open source</p><span class="disc">A learning project</span>{foot(1,T)}</section>''')
    s.append(f'''<section class="page"><span class="eyebrow">The problem</span>
<h2>A 1-star review tells you that you failed. Not why.</h2>
<div class="big">193,166</div><p>bad reviews in one public Yelp dataset. No manager reads all of that.</p>
<p class="muted">Was it the wait? A rude server? Cold food? A wrong order? The bill?</p>{foot(2,T)}</section>''')
    s.append(f'''<section class="page"><span class="eyebrow">What it does</span>
<h2>Give it a bad review. Get every complaint inside it.</h2>
<div class="ticket"><p class="rv">“Good ambiance, but the food smelled and it wasn’t worth the price for a buffet.”</p>{chips([("food_quality","0.89",True),("price_value","0.98",True)])}</div>
<p>8 complaint types:</p>{chips([(n,"",False) for n,_,_ in F1])}
<p class="muted">Praise like “good ambiance” is not tagged. It learned the difference.</p>{foot(3,T)}</section>''')
    steps = [("Write the rules", "8 labels plus rules: praise isn’t a complaint; slow isn’t rude."),
             ("Label with a free LLM", "Qwen3-4B tagged 3,500 Yelp reviews. My hand check: ~18/20 right."),
             ("Beat a baseline", "Simple keyword model first, as the score to beat."),
             ("Fine-tune DistilBERT", "66M parameters, weighted for rare labels, ~15 min on a free GPU."),
             ("Publish honestly", "Model card with results and limitations.")]
    st = "".join(f'<div class="step"><span class="num">{i+1:02d}</span><div><h3 style="font-family:var(--display);font-size:34px">{a}</h3><p class="muted" style="font-size:26px">{b}</p></div></div>' for i,(a,b) in enumerate(steps))
    s.append(f'''<section class="page"><span class="eyebrow">How I built it</span><h2>Five steps, one day, $0.</h2>{st}{foot(4,T)}</section>''')
    s.append(f'''<section class="page"><span class="eyebrow">Results</span><h2>Beats the keyword baseline on all 8 labels.</h2>
<p><b>Average F1: 0.58 → 0.71.</b> Biggest gains on rare complaints like wrong orders (0.25 → 0.60).</p>{bars()}
<p class="muted" style="font-size:22px">Test set: 354 held-out reviews labelled by the same LLM.</p>{foot(5,T)}</section>''')
    tl = "".join(f'<div class="bar" style="grid-template-columns:1fr 120px"><span>{a}</span><span style="text-align:right;color:var(--red)">{b}</span></div>' for a,b in TIME)
    s.append(f'''<section class="page"><span class="eyebrow">Tools and time</span><h2>All free. About 6 hours.</h2>
<div class="grid" style="grid-template-columns:1fr 1fr">{table_rows(TOOLS)}</div>
<div class="bars">{tl}</div><p class="muted" style="font-size:24px">Your own business model: plan 1–2 weeks. Most of it is collecting reviews and hand-checking labels.</p>{foot(6,T)}</section>''')
    s.append(f'''<section class="page"><span class="eyebrow">What I learned · what’s next</span><h2>The model copies its teacher.</h2>
<ul class="clean"><li>Labels need written rules, or even humans disagree.</li><li>Rare complaints need extra weight, or the model ignores them.</li><li>AI-made labels carry the AI’s habits. Check them by hand.</li></ul>
<p><b>Next:</b></p><ul class="clean">{"".join(f"<li>{a}</li>" for a,_ in IMPROVE)}</ul>{foot(7,T)}</section>''')
    s.append(f'''<section class="page dark"><span class="eyebrow">Try it</span><h2>Free on Hugging Face.</h2>
<div class="ticket"><p style="font-family:var(--mono);font-size:30px;font-weight:600">{URL}</p></div>
<p>Where it could help: restaurant managers, franchises, delivery platforms, reputation agencies, restaurant software, food-safety teams.</p>
<span class="disc">Learning project · non-commercial license · feedback welcome</span>{foot(8,T)}</section>''')
    return s

def video():
    T = 5
    s = []
    s.append(f'''<section class="page dark" style="justify-content:center"><span class="eyebrow">Yelp complaint tagger · demo</span>
<h1 style="max-width:1500px">Reads bad restaurant reviews and tells you <span style="color:var(--mustard)">why</span>.</h1>
<p class="muted">{URL}</p><span class="disc">A learning project, shared to learn in public</span>{foot(1,T)}</section>''')
    steps = [("Rules", "8 complaint types + clear rules"), ("Labels", "Free AI labelled 3,500 reviews; ~18/20 right by hand"),
             ("Baseline", "Keyword model: F1 0.58"), ("Training", "Small DistilBERT model: F1 0.71"), ("Publish", "Honest model card on Hugging Face")]
    st = "".join(f'<div class="card"><span class="num">{i+1:02d}</span><h3 style="margin-top:14px">{a}</h3><p>{b}</p></div>' for i,(a,b) in enumerate(steps))
    s.append(f'''<section class="page"><span class="eyebrow">How I built it</span><h2>Five steps, one day, free tools.</h2>
<div class="grid" style="grid-template-columns:repeat(5,1fr)">{st}</div>
<div style="display:flex;gap:80px;align-items:flex-end"><div><p class="muted">Keyword baseline</p><div class="big" style="color:#9DA1A9">0.58</div></div>
<div class="big" style="color:var(--muted)">→</div><div><p class="muted">My model</p><div class="big">0.71</div></div>
<p style="max-width:620px">Average F1 score on 354 held-out reviews. Better on all 8 complaint types.</p></div>{foot(2,T)}</section>''')
    tl = "".join(f'<div class="bar" style="grid-template-columns:1fr 130px"><span>{a}</span><span style="text-align:right;color:var(--red)">{b}</span></div>' for a,b in TIME)
    s.append(f'''<section class="page"><span class="eyebrow">Tools and time</span><h2>Everything was free. It took about 6 hours.</h2>
<div style="display:grid;grid-template-columns:1.4fr 1fr;gap:56px"><div class="grid" style="grid-template-columns:1fr 1fr">{table_rows(TOOLS)}</div>
<div style="display:flex;flex-direction:column;gap:22px"><div class="bars">{tl}</div>
<div class="card" style="border-color:var(--red)"><h3>Build your own: 1–2 weeks</h3><p>Collect your own reviews, hand-check a few hundred labels, then train. Training itself takes minutes.</p></div></div></div>{foot(3,T)}</section>''')
    s.append(f'''<section class="page"><span class="eyebrow">How it could improve</span><h2>Five upgrades for version 2.</h2>
<div class="grid" style="grid-template-columns:repeat(5,1fr)">{"".join(f'<div class="card"><span class="num">{i+1:02d}</span><h3 style="margin-top:14px">{a}</h3><p>{b}</p></div>' for i,(a,b) in enumerate(IMPROVE))}</div>
<p class="muted">Known limits today: complaints only (not 5-star reviews), trips on negation, and copies some habits of its AI labeller.</p>{foot(4,T)}</section>''')
    s.append(f'''<section class="page"><span class="eyebrow">Where this helps</span><h2>Six practical use cases.</h2>
<div class="grid" style="grid-template-columns:repeat(3,1fr)">{table_rows(USES)}</div>
<p><b>Beyond restaurants:</b> hotels, clinics, retail, winery tasting rooms. Change the labels, get a new model.</p>
<p class="muted" style="font-size:26px">Commercial use needs a version trained on your own reviews; this one is non-commercial.</p>{foot(5,T)}</section>''')
    return s

for name, kind, pages in [("carousel", "carousel", carousel()), ("video-slides", "video", video())]:
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>{name}</title><style>{css_for(kind)}</style></head><body>{''.join(pages)}</body></html>"
    open(f"{name}.html", "w").write(html)
    print("wrote", name, len(pages), "pages")
