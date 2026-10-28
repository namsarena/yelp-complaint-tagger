# Recording guide: demo video (about 7 minutes)

Everything here is free. Total time: about 2 hours including setup, two takes and editing.

## What you need
| Piece | Use | Where |
|---|---|---|
| Teleprompter | Read the script while looking at the camera | The teleprompter page (link in chat), opened in a browser |
| Video slides | Slides 1–5 shown during "How I built it" to "Use cases" | `video-slides.pdf`, opened in Preview in full screen |
| Demo | Type 3 reviews live | Colab demo cell below (or a Gradio Space, if built) |
| Recorder | Camera + screen | **OBS Studio** (free, no watermark, no time limit): obsproject.com |
| Editor | Trim, captions | **CapCut desktop** (free tier has auto-captions) or **iMovie** (built into Mac) |

Loom's free plan caps video length (5 minutes when last checked), and this video runs about 7, so use OBS, or split it into two Looms.

## 1. Set up the teleprompter (5 min)
1. Open the teleprompter page and drag the browser window to the **top centre of the screen, right under the webcam**, about 1/3 of the screen wide. Your eyes then stay close to the lens.
2. Set **Size** so you can read it comfortably from your seat (about 56–70) and **Speed** to about 40.
3. Do one practice read. Use ↑/↓ to adjust speed while it runs. Click the text to pause.
4. Turn **Cues** off for the real take if they distract you; they're only stage directions.
5. Using a phone as the prompter instead: open the same link on the phone and prop it just below the webcam.

## 2. Set up OBS (15 min, first time only)
1. Install OBS, open it and skip the auto-config wizard.
2. **Scene "Camera"**: Sources › + › Video Capture Device › your webcam. Resize to fill.
3. **Scene "Screen"**: + › macOS Screen Capture › your display; + › Video Capture Device (same webcam) › make it a small circle or box in the bottom-right corner.
4. Settings › Video: 1920×1080, 30 fps. Settings › Output › Recording format: **mp4**.
5. Audio: check the mic meter moves when you talk. A wired headset mic beats a laptop mic.
6. Give macOS permission when asked (System Settings › Privacy & Security › Screen & System Audio Recording, Camera, Microphone).

## 3. Prepare the demo (5 min)
Open Colab, click Runtime › Change runtime type › CPU is fine, and run this one cell before recording:

```python
!pip -q install transformers torch
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
repo = "nrr26/yelp-complaint-tagger"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()
th = model.config.label_thresholds

def tag(text):
    with torch.no_grad():
        p = torch.sigmoid(model(**tok(text, return_tensors="pt", truncation=True)).logits)[0]
    return {l: round(float(x), 2) for l, x in zip(model.config.id2label.values(), p) if x > th[l]}

tag("test")
```
During the demo, type `tag("…your review…")` into a new cell and run it. Zoom the browser to 150% (Cmd +) so text is readable on video.

## 4. Record (30–45 min)
Record in short clips, one per scene change. It's easier than one perfect take.

| Script section | OBS scene | On screen |
|---|---|---|
| 0:00 Hook, 0:20 Problem | Camera | You |
| 1:00 What it does | Screen | Model page on Hugging Face, scroll slowly |
| 1:45 Live demo | Screen | Colab, type the 3 reviews |
| 4:00 How I built it, 4:50 Tools and time | Screen | Video slides 2 and 3 |
| 5:40 Where it falls short | Camera | You |
| 6:15 How it could improve, 6:50 Use cases | Screen | Video slides 4 and 5 |
| 8:20 Close | Camera | You, then video slide 1 as the last frame |

Tips: pause 2 seconds before and after each section (easy cut points), smile on the first and last line, and if you stumble, pause and restart the sentence; cut it later.

## 5. Edit (30 min)
1. Import the clips into CapCut (or iMovie) in order and trim the pauses and mistakes.
2. Captions: CapCut › Text › Auto captions. Most people watch LinkedIn videos muted, so captions matter.
3. Add `video-slides.pdf` slide 1 as a 3-second title card at the start.
4. Export at 1080p, mp4.

## 6. Publish
- **LinkedIn:** upload the mp4 natively (not a YouTube link), with a short version of the post text and the Hugging Face link in the first comment.
- **YouTube (optional):** upload as Public or Unlisted; title "I built an AI model that reads bad restaurant reviews | learning project"; put the Hugging Face link and the disclaimer in the description.
- **Hugging Face:** add the video link to the model card under a "Demo" heading.

## Disclaimer to keep everywhere
Learning project, not production software. Model licensed CC BY-NC 4.0 (non-commercial) because it was trained on Yelp data.
