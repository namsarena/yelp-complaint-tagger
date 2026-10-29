"""BAT + UAT for the live Space nrr26/complaint-tagger.

BAT (build acceptance): the Space is up, every endpoint answers, edge cases don't crash.
UAT (user acceptance): 110 fictional restaurants (55 US states and territories x 2). Each gets 5 reviews:
3 about one main problem + 2 about other problems. Pass = the owner's "best next action" targets the main problem.
All restaurants and reviews are synthetic, generated from templates.
Usage: python run_bat_uat.py <out_dir>
"""
import csv, json, random, sys, time
from pathlib import Path
from gradio_client import Client

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results"); OUT.mkdir(parents=True, exist_ok=True)
SPACE = "nrr26/complaint-tagger"
random.seed(42)

PLACES = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
          "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
          "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
          "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
          "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
          "West Virginia", "Wisconsin", "Wyoming", "District of Columbia", "Puerto Rico", "Guam", "U.S. Virgin Islands", "American Samoa"]
CUISINES = {"Indian": ["butter chicken", "biryani", "garlic naan"], "Italian": ["lasagna", "carbonara", "margherita pizza"],
            "American": ["cheeseburger", "BBQ ribs", "mac and cheese"], "Mexican": ["carne asada tacos", "burrito", "enchiladas"],
            "Chinese": ["kung pao chicken", "dumplings", "fried rice"], "Thai": ["pad thai", "green curry", "spring rolls"],
            "Japanese": ["ramen", "sushi rolls", "chicken katsu"], "Mediterranean": ["falafel wrap", "lamb gyro", "hummus plate"]}
WORDS = ["Copper", "Juniper", "Harbor", "Maple", "Saffron", "Cedar", "Lantern", "Willow", "Granite", "Marigold", "Pepper", "Sundial"]
PLACES_NOUN = ["Kitchen", "Table", "Bistro", "House", "Corner", "Canteen", "Grill", "Eatery"]
T = {
 "wait_time": ["Waited over an hour for the {d}.", "Nobody came to our table for 30 minutes.", "It took 45 minutes to get our {d} on a slow night.", "We sat forever before anyone took our order."],
 "staff_attitude": ["Our server was rude and rolled his eyes at us.", "The manager was dismissive when we asked about the {d}.", "The hostess snapped at us for asking for a different table.", "Staff were unfriendly and ignored our questions."],
 "food_quality": ["The {d} was bland and cold.", "My {d} was overcooked and dry.", "The {d} tasted stale, like it was made yesterday.", "Soggy {d} with no flavor at all."],
 "order_accuracy": ["We ordered the {d} and got a completely different dish.", "Our takeout bag was missing the {d}.", "They forgot half of our order.", "I asked for no onions and they still put onions on everything."],
 "price_value": ["$40 for a tiny portion of {d}. Not worth it.", "Way overpriced for such small portions.", "The {d} cost twice what it should.", "Paid a fortune and still left hungry."],
 "hygiene_safety": ["Found a hair in my {d}.", "The tables were sticky and the restroom was filthy.", "Saw a cockroach near the kitchen door.", "Got food poisoning after eating the {d}."],
 "ambience": ["The music was so loud we couldn't talk.", "It was freezing inside and the chairs were uncomfortable.", "The room smelled like old grease and the lighting was harsh.", "Tables were crammed together and the room was chaotic and noisy."],
 "billing_policy": ["They charged us twice for the {d}.", "They added a surprise service fee and refused to remove it.", "The manager refused a refund for a wrong charge.", "They wouldn't honor the coupon and overcharged us."],
}
NAMES = {"wait_time": "wait time", "staff_attitude": "staff attitude", "food_quality": "food quality", "order_accuracy": "order accuracy",
         "price_value": "price and value", "hygiene_safety": "hygiene and safety", "ambience": "ambience", "billing_policy": "billing and policy"}
LABELS = list(T)

def cases():
    out, k = [], 0
    for place in PLACES:
        for j in range(2):
            cuisine = list(CUISINES)[k % len(CUISINES)]
            main = LABELS[k % len(LABELS)]
            others = random.sample([l for l in LABELS if l != main], 2)
            dishes = CUISINES[cuisine]
            rev = [t.format(d=random.choice(dishes)) for t in random.sample(T[main], 3)]
            rev += [random.choice(T[o]).format(d=random.choice(dishes)) for o in others]
            random.shuffle(rev)
            name = f"{WORDS[k % len(WORDS)]} {PLACES_NOUN[(k // 3) % len(PLACES_NOUN)]}"
            out.append({"id": k + 1, "place": place, "restaurant": name, "cuisine": cuisine, "main": main,
                        "others": ",".join(others), "reviews": "\n".join(rev)})
            k += 1
    return out

def timed(fn, *a, **kw):
    t = time.time(); r = fn(*a, **kw); return r, round(time.time() - t, 2)

client = Client(SPACE, verbose=False)
bat = []
def check(name, ok, detail=""):
    bat.append({"check": name, "pass": bool(ok), "detail": str(detail)[:200]}); print(("PASS " if ok else "FAIL ") + name, flush=True)

try:
    r, s = timed(client.predict, "We waited 45 minutes and the burger was cold.", api_name="/tag_review")
    check("tag_review answers with known labels", isinstance(r, dict) and set(r) <= set(LABELS) and "wait_time" in r, f"{r} in {s}s")
except Exception as e: check("tag_review answers with known labels", False, e)
for api, args, marker in [("/owner_report", ("Nobody took our order for 25 minutes.\nThey charged us twice.",), "best next action"),
                          ("/diner_tips", ("Found a hair in my soup.\nThe restroom was filthy.",), "Before you book")]:
    try:
        r, s = timed(client.predict, *args, api_name=api)
        check(f"{api} returns its section", marker.lower() in str(r[0]).lower(), f"{s}s")
        check(f"{api} carries the disclaimer note", "automated estimate" in str(r[0]).lower())
    except Exception as e: check(f"{api} returns its section", False, e)
for label, text, expect in [("empty input is handled", "", "paste at least one review"),
                            ("happy review shows no crash", "Amazing food and lovely staff!", ""),
                            ("emoji and accents are handled", "Café was dirty 😡 and the crêpes were cold", ""),
                            ("very long review is accepted", "The soup was cold. " * 400, "")]:
    try:
        r, s = timed(client.predict, text, api_name="/owner_report")
        check(label, expect in str(r[0]).lower(), f"{s}s")
    except Exception as e: check(label, False, e)
try:
    r, s = timed(client.predict, "\n".join(["The soup was cold."] * 250), api_name="/owner_report")
    rows = len(r[2]["data"]) if isinstance(r[2], dict) else None
    check("caps input at 200 reviews", rows == 200, f"rows={rows} in {s}s")
except Exception as e: check("caps input at 200 reviews", False, e)

rows = []
for c in cases():
    rec = dict(c)
    try:
        (plan, _chart, _table), s1 = timed(client.predict, c["reviews"], api_name="/owner_report")
        (tips, _c2), s2 = timed(client.predict, c["reviews"], api_name="/diner_tips")
        first = plan.split("Why:", 1)[1].split("comes up", 1)[0].lower() if "Why:" in plan else ""
        rec.update(top=first.strip(" *"), owner_pass=NAMES[c["main"]] in first,
                   diner_pass=NAMES[c["main"]].split(" and ")[0] in tips.lower().split("people mention these most")[-1][:400],
                   secs_owner=s1, secs_diner=s2, error="")
    except Exception as e:
        rec.update(top="", owner_pass=False, diner_pass=False, secs_owner=None, secs_diner=None, error=str(e)[:200])
    rows.append(rec)
    print(f"{c['id']:3d} {c['place']:22s} {c['main']:15s} -> {rec['top']:20s} {'PASS' if rec['owner_pass'] else 'FAIL'}", flush=True)

with open(OUT / "uat_cases.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
by = {}
for r in rows:
    d = by.setdefault(r["main"], [0, 0]); d[0] += r["owner_pass"]; d[1] += 1
times = [r["secs_owner"] for r in rows if r["secs_owner"]]
summary = {"bat_pass": sum(b["pass"] for b in bat), "bat_total": len(bat), "bat": bat,
           "uat_owner_pass": sum(r["owner_pass"] for r in rows), "uat_diner_pass": sum(r["diner_pass"] for r in rows),
           "uat_total": len(rows), "errors": sum(bool(r["error"]) for r in rows),
           "by_main_problem": {k: f"{v[0]}/{v[1]}" for k, v in by.items()},
           "median_secs_owner": sorted(times)[len(times) // 2] if times else None,
           "max_secs_owner": max(times) if times else None}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "bat"}))
