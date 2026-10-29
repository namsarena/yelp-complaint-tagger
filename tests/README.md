# Tests for the Restaurant Complaint Tagger page

`run_bat_uat.py` tests the live Space `nrr26/complaint-tagger` through its API.

- **BAT (build acceptance):** every endpoint answers; empty input, a happy review, emoji and accents, a very long review, and 250 reviews (capped at 200) don't break it; outputs carry the disclaimer note.
- **UAT (user acceptance):** 110 test restaurants, 2 in each of 50 US states plus DC, Puerto Rico, Guam, the U.S. Virgin Islands and American Samoa. Each gets 5 template-generated reviews: 3 about one main problem and 2 about others. Owner pass = the best next action targets the main problem. Diner pass = the main problem appears in the top 3 tips.

Run: `pip install gradio_client && python run_bat_uat.py results`

## Results (2026-10-28)
| Test | Result |
|---|---|
| BAT | 10/10 passed |
| UAT, owner best next action | 84/110 (76%) |
| UAT, diner top 3 | 110/110 |
| Errors | 0 |
| Speed | median 0.7 s, max 1.7 s per owner report |

By main problem (owner): wait time 14/14, staff attitude 14/14, food quality 14/14, hygiene 14/14, ambience 10/13, billing 9/13, price 8/14, order accuracy 1/14.
The main weakness: wrong-order reviews that name a dish get tagged as food quality. Reviews are synthetic and template-based, so real-world accuracy can differ.
