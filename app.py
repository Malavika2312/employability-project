import streamlit as st
import pandas as pd
import pickle

# Load model
model, columns = pickle.load(open("model.pkl", "rb"))

st.title("🎯 Student Employability & Skill Recommendation System")

# Upload dataset
file = st.file_uploader("Upload Student Dataset", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    # 🔥 Fix column names
    df.columns = df.columns.str.strip().str.lower()

    st.success("✅ Dataset Uploaded Successfully")

    # ======================
    # PREPROCESSING
    # ======================
    remove_cols = ["student_id", "placement_status", "employable"]
    X = df.drop(columns=[col for col in remove_cols if col in df.columns])

    if "branch" in X.columns:
        X["branch"] = X["branch"].astype("category").cat.codes

    if "college_tier" in X.columns:
        X["college_tier"] = X["college_tier"].astype("category").cat.codes

    X = X.select_dtypes(exclude=["object"])
    X = X.fillna(0)
    X = X.reindex(columns=columns, fill_value=0)

    # ======================
    # EMPLOYABILITY %
    # ======================
    probs = model.predict_proba(X)[:, 1]
    df["employability %"] = (probs * 100).round(2)

    # ======================
    # CAREER SKILLS DATABASE
    # ======================
    skills_db = {

        "data analyst": ["Python", "SQL", "Excel", "Power BI", "Tableau"],
        "data scientist": ["Python", "Machine Learning", "Deep Learning"],
        "business analyst": ["Excel", "SQL", "Power BI", "Communication"],

        "software engineer": ["Java", "DSA", "Algorithms"],
        "web developer": ["HTML", "CSS", "JavaScript", "React"],
        "frontend developer": ["HTML", "CSS", "JavaScript"],
        "backend developer": ["Python", "Node.js", "SQL"],
        "full stack developer": ["HTML", "CSS", "JavaScript", "React", "Node.js"],

        "ai engineer": ["Python", "Deep Learning", "NLP"],
        "machine learning engineer": ["Python", "ML", "TensorFlow"],

        "data engineer": ["Python", "SQL", "ETL"],
        "devops engineer": ["Docker", "Kubernetes", "AWS"],
        "cloud engineer": ["AWS", "Azure"],

        "cyber security": ["Networking", "Ethical Hacking"],
        "ui ux designer": ["Figma", "UI Design"],
        "qa tester": ["Testing", "Selenium"]
    }

    # ======================
    # SKILL RECOMMENDATION
    # ======================
    def recommend(row):
        goal = str(row.get("career_goal", "")).lower()
        required = skills_db.get(goal, [])

        # 🔥 Already known skills
        known = str(row.get("known_skills", "")).split(",")

        # Remove spaces
        known = [k.strip().lower() for k in known]

        # Recommend missing skills
        missing = [skill for skill in required if skill.lower() not in known]

        if missing:
            return ", ".join(missing)
        else:
            return "All required skills covered ✅"

    df["recommended skills"] = df.apply(recommend, axis=1)

    # ======================
    # 🎯 FINAL TABLE (ALL STUDENTS)
    # ======================
    st.subheader("📊 All Students Result")

    display_cols = [
        "student_id",
        "branch",
        "career_goal",
        "known_skills",
        "employability %",
        "recommended skills"
    ]

    # show only available columns
    display_cols = [col for col in display_cols if col in df.columns]

    st.dataframe(df[display_cols])

    # ======================
    # 📥 DOWNLOAD
    # ======================
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download Full Results", csv, "final_results.csv")

else:
    st.warning("Upload dataset to continue")