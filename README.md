# Pakistan Job Market Analyzer 🇵🇰

An AI-powered web app that analyzes Pakistan's job market using real Rozee.pk data. Built as a portfolio project combining data science, machine learning, and LLM integration.

**[Live Demo →](https://pakistan-job-market-analyzer.streamlit.app)**

---

## Features

### 📈 Market Overview Dashboard
- Job count, salary stats, and city coverage at a glance
- Top 10 hiring cities and functional areas
- In-demand skills, career level breakdown, job type distribution
- Salary distribution histograms and box plots by city

### 💰 Salary Predictor
- Input your experience, city, career level, and skills
- Get an instant salary range prediction with a visual gauge
- See how your expected salary compares to the full market
- Benchmarks: median salary by city and career level

### 🤖 AI Career Advisor
- Chat with an LLM grounded in real Pakistan job market data
- Ask about skills to learn, salary expectations, city comparisons, career transitions
- Answers cite actual numbers from the dataset (not generic advice)
- Powered by Llama 3.1 via Groq (free)

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data | Pandas, NumPy |
| ML | Scikit-learn (Random Forest) |
| Visualization | Plotly |
| App | Streamlit |
| LLM | Groq API (Llama 3.1) |
| Deployment | Streamlit Community Cloud |

---

## Dataset

**RozeePK-Jobs-2024.csv** — 1,059 job listings scraped from Rozee.pk in 2024  
Fields: Job title, salary, location, functional area, career level, required skills, education, experience

After cleaning: 678 rows with salary data, 367 rows with all features for ML training.

---

## ML Model

**Random Forest Regressor** trained to predict monthly salary (PKR)

- Features: years of experience, career level (ordinal encoded), top 25 skills (multi-hot), city (one-hot, top 8 + Other)
- Train/test split: 80/20
- MAE: ~PKR 19,600 | R²: 0.22 | 5-fold CV R²: 0.23

The R² reflects genuine salary variance in the market — many roles in the same city/level pay differently based on company and negotiation, which no model can capture from job posting data alone.

---

## Run Locally

```bash
git clone https://github.com/mahnoorauf/Pakistan-Job-Market-Analyzer
cd Pakistan-Job-Market-Analyzer

# Create venv and install dependencies
uv venv && uv pip install -r requirements.txt

# Run the app
.venv/Scripts/streamlit run app.py   # Windows
# or
.venv/bin/streamlit run app.py       # Mac/Linux
```

For the AI Career Advisor, add your free Groq API key (get one at [console.groq.com](https://console.groq.com)):
```bash
# Create .streamlit/secrets.toml
echo 'GROQ_API_KEY = "gsk_..."' > .streamlit/secrets.toml
```

---

## Project Structure

```
├── app.py                   # Streamlit app (3 pages)
├── requirements.txt
├── data/
│   ├── raw/                 # Original CSV from Rozee.pk
│   └── processed/           # Cleaned data (jobs_cleaned.csv)
├── models/
│   └── salary_predictor.pkl # Trained Random Forest model bundle
├── notebooks/
│   ├── EDA.ipynb            # Exploratory data analysis
│   └── salary_predictor.ipynb  # Model training pipeline
└── src/
    └── llm_advisor.py       # Market context builder + Groq chat
```

---

## Author

**Mahnoor** — Data Science student, Pakistan  
Built in 5 days as a portfolio project.
