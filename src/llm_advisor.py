import pandas as pd
from groq import Groq


def build_market_context(df: pd.DataFrame) -> str:
    salary_df = df.dropna(subset=["salary_pkr"])
    skills_series = df["Skills"].dropna().str.split(",").explode().str.strip().str.lower()

    top_cities = (
        salary_df.groupby("city")["salary_pkr"]
        .agg(["count", "median"])
        .sort_values("count", ascending=False)
        .head(10)
    )
    top_areas = (
        salary_df.groupby("Functional Area")["salary_pkr"]
        .agg(["count", "median"])
        .sort_values("count", ascending=False)
        .head(10)
    )
    level_salary = (
        salary_df.groupby("Career Level")["salary_pkr"]
        .median()
        .sort_values(ascending=False)
    )
    top_skills = skills_series.value_counts().head(25).index.tolist()
    top_titles = df["Title"].value_counts().head(15).index.tolist()
    edu = df["Minimum Education"].value_counts().head(6)

    lines = [
        "=== PAKISTAN JOB MARKET DATA (Rozee.pk 2024) ===",
        f"Total job listings: {len(df):,}",
        f"Jobs with salary data: {len(salary_df):,}",
        f"Overall salary range: PKR {salary_df['salary_pkr'].min():,.0f} – PKR {salary_df['salary_pkr'].max():,.0f}",
        f"Median salary: PKR {salary_df['salary_pkr'].median():,.0f}",
        f"Average salary: PKR {salary_df['salary_pkr'].mean():,.0f}",
        "",
        "--- TOP CITIES (jobs | median salary) ---",
    ]
    for city, row in top_cities.iterrows():
        lines.append(f"  {city}: {int(row['count'])} jobs | PKR {row['median']:,.0f} median")

    lines += ["", "--- TOP FUNCTIONAL AREAS (jobs | median salary) ---"]
    for area, row in top_areas.iterrows():
        lines.append(f"  {area}: {int(row['count'])} jobs | PKR {row['median']:,.0f} median")

    lines += ["", "--- SALARY BY CAREER LEVEL ---"]
    for level, med in level_salary.items():
        lines.append(f"  {level}: PKR {med:,.0f} median")

    lines += ["", "--- TOP 25 IN-DEMAND SKILLS ---"]
    lines.append("  " + ", ".join(top_skills))

    lines += ["", "--- TOP JOB TITLES ---"]
    lines.append("  " + ", ".join(top_titles))

    lines += ["", "--- EDUCATION REQUIREMENTS ---"]
    for edu_level, count in edu.items():
        lines.append(f"  {edu_level}: {count} jobs")

    lines += [
        "",
        "--- JOB TYPE BREAKDOWN ---",
        "  " + " | ".join(f"{jt}: {cnt}" for jt, cnt in df["Job Type"].value_counts().items()),
    ]

    return "\n".join(lines)


SYSTEM_PROMPT = """You are an expert Pakistan job market career advisor. You have access to real data from Rozee.pk (2024) covering {total_jobs} job listings.

Use the market data below to give specific, data-driven advice. Always cite actual numbers from the data when relevant (salary figures, skill demand, city comparisons). Be honest if the data doesn't cover something.

When advising on salaries, mention ranges. When advising on skills, prioritize the most in-demand ones. Keep responses focused and practical — the user is likely a student or early-career professional in Pakistan.

MARKET DATA:
{market_context}"""


def get_response(api_key: str, market_context: str, messages: list, total_jobs: int) -> str:
    client = Groq(api_key=api_key)

    system = SYSTEM_PROMPT.format(
        market_context=market_context,
        total_jobs=f"{total_jobs:,}"
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": system}] + messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content
