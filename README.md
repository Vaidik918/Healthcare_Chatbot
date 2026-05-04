# 🏥 HealthBot — AI Healthcare Navigator


> **🔗 Live Demo: [https://healthcare-chatbot-0qw1.onrender.com](https://healthcare-chatbot-0qw1.onrender.com)**
> *(First load may take 30–50 seconds on free tier)*

An intelligent healthcare chatbot for India that helps users with procedure cost estimates, hospital recommendations, symptom-based specialist guidance, and clinical care pathways — all in Hinglish.

---

## ✨ Features at a Glance

| Feature | Description |
|---------|-------------|
| 🏨 Hospital Suggestions | Top 5 hospitals by city & specialty, ranked using real patient reviews (VADER NLP) |
| 💊 Procedure Cost Estimates | Detailed cost breakdown by hospital tier, room type & comorbidities |
| 💰 Budget Filtering | Filter hospitals within your budget — "3 lakh mein bypass Delhi mein" |
| 🩺 Symptom Guidance | Tells which specialist to see, urgency level, tests needed — in Hinglish |
| 👨‍⚕️ Doctor Recommendations | Top doctors per specialty extracted from patient reviews |
| 📋 Clinical Pathways | Step-by-step care journey for surgeries (pre-op → ICU → discharge) |
| 🔁 Multi-turn Sessions | Remembers context across conversation turns |

---
Ensure the questions asked are within scope! Refer to the procedures list in RATE_CARDS.xlsx prior.
## 📸 Screenshots

### 🏥 Hospital Suggestions (Review-ranked)
> Top 2 hospitals ranked by VADER review sentiment, bottom 3 by market alignment score

![Hospital Suggestions](screenshots/ss1_hospitals.png)

---

### 💰 Detailed Cost Breakdown
> Full bill breakdown — Variable + Semi-Variable buckets, tier multipliers, ICU charges

![Cost Breakdown](screenshots/ss2_breakdown.png)

---

### 🩺 Health Guidance Card
> Symptom → Specialist mapping with urgency, tests, warning signs — in Hinglish

![Health Guidance](screenshots/ss3_health_guidance.png)

---

### 👨‍⚕️ Top Doctors by Specialty
> Doctors extracted from patient reviews — ranked by mentions & sentiment score

![Top Doctors](screenshots/ss4_doctors.png)

---

### 💸 Budget-Aware Hospital Search
> "3 lakh mein bypass surgery Delhi mein" — shows budget check + affordable hospitals

![Budget Query](screenshots/ss5_budget.png)

![Budget Results](screenshots/ss6_budget_hospitals.png)

---

## 🚀 Quick Start

### Option 1 — Use Live Demo
👉 **[https://healthcare-chatbot-0qw1.onrender.com](https://healthcare-chatbot-0qw1.onrender.com)**

Try these queries:
- `"Knee replacement cost in Delhi"`
- `"Mujhe chest mein dard ho raha hai"`
- `"Best eye hospital in Indore"`
- `"Bypass surgery within 3 lakh in Delhi"`
- `"Knee replacement mein kya kya hota hai?"`

---

### Option 2 — Run Locally

**Prerequisites:** Python 3.9+, OpenRouter API key → [openrouter.ai](https://openrouter.ai)

```bash
# 1. Clone
git clone https://github.com/Vaidik918/Healthcare_Chatbot.git
cd Healthcare_Chatbot

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install
pip install -r requirements.txt

# 4. Environment
echo "OPENROUTER_API_KEY=your_key_here" > .env

# 5. Run
uvicorn api:app --reload --port 8000
```

Open: **[http://localhost:8000](http://localhost:8000)**

---

## 🧠 How Hospital Ranking Works

```
User Query: "Eye hospitals in Indore"
                ↓
All Indore hospitals scored
                ↓
┌─────────────────────────────────────────────┐
│  Position 1 & 2  →  VADER Review Pool       │
│  (Hospitals with actual patient reviews     │
│   for this specialty — Strong/Moderate/Weak)│
├─────────────────────────────────────────────┤
│  Position 3, 4 & 5  →  Market Score Pool   │
│  (Hospitals with high structural alignment  │
│   — Apollo, Medanta, big chains etc.)       │
└─────────────────────────────────────────────┘
```

**Review scoring pipeline:**
1. `reviews_output.jsonl` → `preprocess_reviews.py` (VADER NLP)
2. Sentence-level sentiment per specialty keyword
3. Confidence-weighted `specialty_score` saved to `specialty_scores.json`
4. Used at query time to rank hospitals

---

## 📁 Project Structure

```
Healthcare_Chatbot/
├── api.py                   # FastAPI server & all routes
├── chatbot.py               # Session management, LLM routing, hospital ranking
├── check.py                 # Cost estimation & procedure-based hospital lookup
├── llm_utils.py             # OpenRouter LLM integration + system prompt
├── preprocess_reviews.py    # VADER NLP pipeline → specialty_scores.json
├── healthbot.html           # Frontend UI (single file)
├── RATE_CARDS.xlsx          # Procedure pricing data (8 procedures × 9 cities)
├── Hospital_data.csv        # Hospital master database
├── hospitals_with_types.csv # Hospital specialty & tier mapping
├── specialty_scores.json    # Pre-computed VADER review scores per specialty
├── requirements.txt
└── .env.example
```

---

## ⚙️ Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | API key from [openrouter.ai](https://openrouter.ai) — uses `gpt-4o-mini` |

---

## 🔧 Tech Stack

- **Backend:** FastAPI + Python 3.9+
- **LLM:** GPT-4o-mini via OpenRouter
- **NLP:** VADER Sentiment Analysis (review scoring)
- **Frontend:** Vanilla HTML/CSS/JS (single file, no framework)
- **Data:** Excel rate cards + CSV hospital data + JSON review scores
- **Hosting:** Render (free tier)

---

## 💬 Example Queries

| Query | What happens |
|-------|-------------|
| `Knee replacement cost Delhi` | Cost estimate with full breakdown |
| `Mujhe chest mein dard ho raha hai` | Cardiologist guidance + nearby hospitals |
| `Best eye hospital in Indore` | Top 5 hospitals, review-ranked |
| `3 lakh mein bypass surgery Delhi` | Budget check + affordable hospital list |
| `Bypass surgery ke baad kitne din?` | Full 12-step clinical care pathway |
| `Doctors for cardio in Indore` | Top doctors from patient reviews |
| `Main 68 saal ka hoon, knee replacement` | Age-adjusted cost estimate |

---

## 📄 License
MIT — free to use and modify.
