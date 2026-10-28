---
title: Restaurant Complaint Tagger
emoji: 🍽️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
short_description: Paste bad restaurant reviews, see what to fix first
models:
- nrr26/yelp-complaint-tagger
---

# Restaurant Complaint Tagger

Paste your 1-2 star reviews. The page tags each complaint (wait time, staff attitude, food quality, order accuracy, price, hygiene, ambience, billing), counts them, and suggests what to fix first.

- Model: [nrr26/yelp-complaint-tagger](https://huggingface.co/nrr26/yelp-complaint-tagger) (CC BY-NC 4.0)
- Code: [github.com/namsarena/yelp-complaint-tagger](https://github.com/namsarena/yelp-complaint-tagger) (MIT)

A learning project, not production software. Use it on negative reviews only; reviews you paste are processed in memory and not stored by this app.
