import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Pakistan Job Market Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/jobs_cleaned.csv")

@st.cache_resource
def load_model():
    with open("models/salary_predictor.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
bundle = load_model()


def predict_salary(experience_years, city, skills_list, career_level):
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    top_25_skills = bundle["top_25_skills"]
    top_cities = bundle["top_cities"]
    career_map = bundle["career_map"]
    city_dummies_cols = bundle["city_dummies_cols"]

    city_clean = city if city in top_cities else "Other"
    career_num = career_map.get(career_level, 2)

    row = {"experience_years": experience_years, "career_level_num": career_num}
    for skill in top_25_skills:
        col = f"skill_{skill.replace(' ', '_')}"
        row[col] = 1 if skill in [s.lower().strip() for s in skills_list] else 0
    for col in city_dummies_cols:
        row[col] = 1 if col == f"city_{city_clean}" else 0

    X = pd.DataFrame([row])[feature_cols]
    pred = model.predict(X)[0]
    return int(pred * 0.85), int(pred * 1.15)


# --- Sidebar ---
st.sidebar.title("Pakistan Job Market")
page = st.sidebar.radio("Navigate", ["📈 Market Overview", "💰 Salary Predictor"])

st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df):,} jobs | Rozee.pk 2024")


# ===================== PAGE 1: MARKET OVERVIEW =====================
if page == "📈 Market Overview":
    st.title("📊 Pakistan Job Market Dashboard")
    st.caption("Based on Rozee.pk 2024 job listings")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Jobs", f"{len(df):,}")
    col2.metric("Avg Salary", f"PKR {df['salary_pkr'].mean():,.0f}")
    col3.metric("Median Salary", f"PKR {df['salary_pkr'].median():,.0f}")
    col4.metric("Cities Covered", df["city"].nunique())

    st.markdown("---")

    # Row 1: top cities + job type distribution
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Top 10 Hiring Cities")
        city_counts = df["city"].value_counts().head(10).reset_index()
        city_counts.columns = ["City", "Job Count"]
        fig = px.bar(city_counts, x="Job Count", y="City", orientation="h",
                     color="Job Count", color_continuous_scale="Blues",
                     template="plotly_white")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"), height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Job Type Breakdown")
        jt = df["Job Type"].value_counts().reset_index()
        jt.columns = ["Job Type", "Count"]
        fig = px.pie(jt, values="Count", names="Job Type",
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     template="plotly_white")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Functional area + career level
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("Top Functional Areas")
        fa = df["Functional Area"].value_counts().head(12).reset_index()
        fa.columns = ["Area", "Count"]
        fig = px.bar(fa, x="Count", y="Area", orientation="h",
                     color="Count", color_continuous_scale="Purples",
                     template="plotly_white")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"), height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col_right2:
        st.subheader("Career Level Distribution")
        cl = df["Career Level"].value_counts().reset_index()
        cl.columns = ["Level", "Count"]
        fig = px.bar(cl, x="Level", y="Count", color="Count",
                     color_continuous_scale="Greens", template="plotly_white")
        fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-20, height=420)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Salary distribution + salary by city
    st.subheader("Salary Distribution (PKR)")
    salary_df = df.dropna(subset=["salary_pkr"])

    col_l3, col_r3 = st.columns(2)
    with col_l3:
        fig = px.histogram(salary_df, x="salary_pkr", nbins=40,
                           labels={"salary_pkr": "Salary (PKR)"},
                           color_discrete_sequence=["#636EFA"],
                           template="plotly_white")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)

    with col_r3:
        top5_cities = salary_df["city"].value_counts().head(5).index.tolist()
        box_df = salary_df[salary_df["city"].isin(top5_cities)]
        fig = px.box(box_df, x="city", y="salary_pkr",
                     labels={"salary_pkr": "Salary (PKR)", "city": "City"},
                     color="city", template="plotly_white")
        fig.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)

    # Row 4: Top skills
    st.subheader("Top 20 In-Demand Skills")
    skills_series = df["Skills"].dropna().str.split(",").explode().str.strip().str.lower()
    top_skills = skills_series.value_counts().head(20).reset_index()
    top_skills.columns = ["Skill", "Count"]
    fig = px.bar(top_skills, x="Count", y="Skill", orientation="h",
                 color="Count", color_continuous_scale="Oranges",
                 template="plotly_white")
    fig.update_layout(showlegend=False, coloraxis_showscale=False,
                      yaxis=dict(autorange="reversed"), height=480)
    st.plotly_chart(fig, use_container_width=True)


# ===================== PAGE 2: SALARY PREDICTOR =====================
elif page == "💰 Salary Predictor":
    st.title("💰 Salary Predictor")
    st.caption("Predict your expected salary based on your profile")

    model_metrics = bundle["metrics"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Model MAE", f"PKR {model_metrics['mae']:,}")
    m2.metric("Model R²", model_metrics["r2"])
    m3.metric("CV R² (5-fold)", model_metrics["cv_r2_mean"])

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Profile")

        experience = st.slider("Years of Experience", 0, 20, 2)

        city = st.selectbox(
            "City",
            options=["Lahore", "Islamabad", "Karachi", "DHA", "Johar Town",
                     "Jhelum", "Gulberg 3", "Gulberg", "Other"],
            index=0
        )

        career_level = st.selectbox(
            "Career Level",
            options=list(bundle["career_map"].keys()),
            index=1
        )

        selected_skills = st.multiselect(
            "Your Skills (select all that apply)",
            options=[s.title() for s in bundle["top_25_skills"]],
            default=[]
        )

        predict_btn = st.button("Predict Salary", type="primary", use_container_width=True)

    with col2:
        st.subheader("Prediction Result")

        if predict_btn:
            skills_input = [s.lower() for s in selected_skills]
            low, high = predict_salary(experience, city, skills_input, career_level)
            mid = (low + high) // 2

            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 12px; padding: 28px; text-align: center; color: white;'>
                <p style='font-size:14px; margin:0; opacity:0.85;'>Predicted Salary Range</p>
                <p style='font-size:38px; font-weight:700; margin:8px 0;'>
                    PKR {mid:,}
                </p>
                <p style='font-size:16px; margin:0; opacity:0.85;'>
                    PKR {low:,} — PKR {high:,}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=mid,
                number={"prefix": "PKR ", "valueformat": ","},
                gauge={
                    "axis": {"range": [0, 150000]},
                    "bar": {"color": "#764ba2"},
                    "steps": [
                        {"range": [0, 40000], "color": "#f8d7da"},
                        {"range": [40000, 80000], "color": "#fff3cd"},
                        {"range": [80000, 150000], "color": "#d4edda"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": mid
                    }
                }
            ))
            fig.update_layout(height=260, margin=dict(t=20, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

            # Context: how does this compare to market?
            salary_df = df.dropna(subset=["salary_pkr"])
            pct = (salary_df["salary_pkr"] < mid).mean() * 100
            st.info(f"This salary is higher than **{pct:.0f}%** of jobs in the dataset.")

        else:
            st.markdown("""
            <div style='background:#f8f9fa; border-radius:12px; padding:48px;
                        text-align:center; color:#6c757d;'>
                <p style='font-size:40px; margin:0;'>🎯</p>
                <p style='font-size:16px; margin:8px 0;'>Fill in your profile and click<br><b>Predict Salary</b></p>
            </div>
            """, unsafe_allow_html=True)

    # Salary benchmarks section
    st.markdown("---")
    st.subheader("Market Salary Benchmarks")

    salary_df = df.dropna(subset=["salary_pkr"])

    bench_col1, bench_col2 = st.columns(2)

    with bench_col1:
        city_salary = salary_df.groupby("city")["salary_pkr"].median().sort_values(ascending=False).head(10).reset_index()
        city_salary.columns = ["City", "Median Salary"]
        fig = px.bar(city_salary, x="Median Salary", y="City", orientation="h",
                     color="Median Salary", color_continuous_scale="Teal",
                     template="plotly_white",
                     labels={"Median Salary": "Median Salary (PKR)"})
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"), height=380,
                          title="Median Salary by City")
        st.plotly_chart(fig, use_container_width=True)

    with bench_col2:
        cl_salary = salary_df.groupby("Career Level")["salary_pkr"].median().sort_values().reset_index()
        cl_salary.columns = ["Career Level", "Median Salary"]
        fig = px.bar(cl_salary, x="Median Salary", y="Career Level", orientation="h",
                     color="Median Salary", color_continuous_scale="Mint",
                     template="plotly_white",
                     labels={"Median Salary": "Median Salary (PKR)"})
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"), height=380,
                          title="Median Salary by Career Level")
        st.plotly_chart(fig, use_container_width=True)
