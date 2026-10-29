"""Unit tests for the page-cleanup and filtering logic in space/app.py (run from the repo root)."""
import sys
sys.path.insert(0, "space")
import app

GOOGLE_PAGE = """Reviews
Sort by
Newest
Jamie R.
Local Guide · 54 reviews · 120 photos
2 weeks ago
We waited almost an hour for our food and nobody came to check on us once.
Helpful
Share
Priya K.
3 reviews
a month ago
Absolutely loved the brunch here, friendly staff and the best pancakes in town!
Helpful
Chris M.
5 days ago
The burger was cold and the fries were soggy, and they still charged full price for it.
Response from the owner
3 days ago
Thank you for your feedback, we are sorry about your experience.
Helpful
"""

YELP_PAGE = """Recommended Reviews
Search reviews
Sam T.
Elite 24
Oct 12, 2026
Our server rolled her eyes when we asked for water and never came back to the table.
Useful
Funny
Cool
Dana L.
Oct 10, 2026
Found a hair in my salad and the restroom was dirty, will not be returning here again.
Useful
Read more
"""

def test_cleanup_keeps_review_sentences_only():
    lines = app.clean_page(GOOGLE_PAGE)
    assert lines[0].startswith("We waited almost an hour")
    assert any("burger was cold" in l for l in lines)
    assert not any(l.lower().startswith(("helpful", "local guide", "sort by", "newest")) for l in lines)

def test_cleanup_skips_owner_replies():
    assert not any("Thank you for your feedback" in l for l in app.clean_page(GOOGLE_PAGE))

def test_cleanup_caps_at_latest_15():
    page = "\n".join(f"This is review number {i} and the food was cold and late again." for i in range(40))
    assert len(app.clean_page(page)) == 15

def test_cleanup_handles_yelp_layout():
    lines = app.clean_page(YELP_PAGE)
    assert len(lines) == 2 and "rolled her eyes" in lines[0]

def test_negative_filter_drops_praise():
    kept = app.negative_only(app.clean_page(GOOGLE_PAGE))
    assert not any("loved the brunch" in l for l in kept)
    assert any("waited almost an hour" in l for l in kept)

def test_owner_report_whole_page():
    plan = app.owner_report(GOOGLE_PAGE + YELP_PAGE, "Test Bistro", True, True)[0]
    assert plan.startswith("# Test Bistro") and "best next action" in plan.lower()

def test_demo_path_unchanged():
    name = "Patty Parallax Burgers · Los Angeles, CA"
    plan = app.owner_report(app.SAMPLES[name], "Patty Parallax Burgers")[0]
    assert "wait time" in plan.lower().split("why:")[1][:40]

def test_empty_and_happy_input():
    assert "No negative reviews found" in app.owner_report("", "", True, True)[0]
    assert "No negative reviews found" in app.diner_tips("Amazing food, lovely staff, we will definitely come back soon!", "", True, True)[0]

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
