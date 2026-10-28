# Demo video script: Yelp complaint tagger

About 7 minutes at a relaxed pace. Lines starting with **[TYPE]** are typed into the demo live. *Cues* say what is on screen.
Disclaimer: this is a learning project, not production software. The model is licensed CC BY-NC 4.0 (non-commercial).

## 0:00 · Hook
*CAMERA. Smile, slow down.*

- Every restaurant gets bad reviews.
- But a one-star rating only tells you that something went wrong.
- It doesn't tell you what.
- So I built a small AI model that reads bad reviews and tells you why, in seconds.
- And I published it for free on Hugging Face.
- Quick note: this is a learning project, not a finished product. I'll show you what works and what doesn't.

## 0:20 · The problem
*CAMERA.*

- Picture a restaurant manager on a Monday morning.
- Forty new reviews came in over the weekend. Twelve are one or two stars.
- Was it the wait? A rude server? Cold food? A wrong order? A billing mistake?
- Reading every review takes time, and patterns are easy to miss.
- One public Yelp dataset alone has more than 190,000 bad reviews. Nobody reads all of that.

## 1:00 · What it does
*SCREEN: the model page on Hugging Face. Scroll slowly past the label list.*

- Here's the model. It's called Yelp Complaint Tagger.
- You give it a negative review, and it returns one or more of eight complaint types.
- Wait time. Staff attitude. Food quality. Order accuracy.
- Price and value. Hygiene and safety. Ambience. And billing or policy.
- One review can hold several complaints. The model catches each one and gives it a confidence score.

## 1:45 · Live demo
*SCREEN: demo page (Gradio Space) or the Colab 'tag' cell. Type each review live.*

- Let me show you. First review:
- **[TYPE]** We waited 45 minutes and the burger was cold.
- It finds two problems: wait time and food quality. That's exactly right.
- The next one is trickier:
- **[TYPE]** They forgot half our order and charged us twice. The manager refused a refund.
- It flags billing and policy with high confidence. It doesn't always catch the forgotten items, and I'll come back to that.
- Now one from my own experience:
- **[TYPE]** Good ambiance, but the food smelled and it wasn't worth the price for a buffet.
- Food quality and price. And look at what it did not tag: ambience.
- The reviewer praised the ambiance, so it isn't a complaint. The model learned that difference.
- Now picture running this on all forty weekend reviews at once.
- Instead of reading every one, the manager sees that half of them are about wait time.
- That points to Saturday-night staffing, not the kitchen.

## 4:00 · How I built it
*SCREEN: video slide 2, 'How I built it'.*

- How did I build it? In one day, with free tools.
- First, I wrote a short labelling guide: eight complaint types and a few clear rules.
- For example, praise is not a complaint, and slow service is not the same as rude service.
- Second, a free open-source AI model labelled 3,500 real Yelp reviews using that guide.
- I checked its work by hand. About 18 out of 20 were right.
- Third, I trained a small, fast model on those labels. Small means it's cheap to run on thousands of reviews.
- On a standard quality score, the F1 score, it went from 0.58 for a keyword approach to 0.71.
- It beats the keyword approach on all eight complaint types.

## 4:50 · Tools and time
*SCREEN: video slide 3, 'Tools and time'.*

- Everything I used is free.
- Google Colab gave me a free GPU. Hugging Face hosts the data and the model.
- An open-source AI model did the first round of labelling, and DistilBERT is the small model I trained.
- It took me about six hours, with an AI coach walking me through each step.
- If you want a model for your own business, plan on about one to two weeks.
- Most of that time goes into collecting your own reviews and checking a few hundred labels by hand. The training itself takes minutes.

## 5:40 · Where it falls short
*CAMERA. Plain and direct.*

- It isn't perfect, so here is where it struggles.
- It only knows complaints. Give it a five-star review and it will still look for problems, so use it on one- and two-star reviews.
- It can trip on negation, like 'the staff was not rude'.
- It learned from AI-made labels, so it carries some of that AI's habits.
- And this version is trained on Yelp's public data, which is licensed for non-commercial use.
- A business version would be trained on your own reviews.

## 6:15 · How it could improve
*SCREEN: video slide 4, 'How it could improve'.*

- Here is how I would make it better.
- First, a larger hand-checked test set, a few hundred reviews, so the score reflects people, not just the AI labeller.
- Second, train it on happy and neutral reviews too, so it can say 'no complaint'.
- Third, more examples for the rare complaints, like billing and wrong orders.
- Fourth, show the exact sentence behind each tag, so a manager can trust it at a glance.
- And finally, a simple dashboard that runs it on every new review, every week.

## 6:50 · Use cases
*SCREEN: video slide 5, 'Where this helps'.*

- So where could this be useful? Here are six practical use cases.
- One. Restaurant owners and managers: a weekly complaint breakdown, so you fix the right thing first.
- Two. Restaurant groups and franchises: compare locations. If one store has twice the wait-time complaints, you know where to send help.
- Three. Delivery and ordering platforms: send each complaint to the right team. Order mistakes go to the kitchen, billing goes to finance.
- Four. Reputation and marketing agencies: turn hundreds of client reviews into a one-page monthly report.
- Five. Restaurant software companies, like point-of-sale or reservation platforms: build review insights into the product.
- Six. Food-safety teams: surface hygiene complaints fast, so a person can follow up.
- The same approach works beyond restaurants: hotels, clinics, retail, or wineries with tasting rooms.
- Change the labels, and you have a new model.

## 8:20 · Close
*CAMERA. Smile.*

- The model is free to try on Hugging Face. The link is in the description.
- If you run a restaurant, or a team that reads a lot of customer feedback, tell me which complaint types you'd want to track.
- And once more: this is a learning project, shared to learn in public. Thanks for watching.
