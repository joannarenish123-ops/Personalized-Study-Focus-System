import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Page Configuration
st.set_page_config(page_title="Personalized Study Focus System", page_icon="🧠", layout="centered")

st.title("🧠 Personalized Study Focus System")
st.write("An AI-driven student productivity tool predicting procrastination risk and providing tailored study strategies.")

# 1. Load Data & Train Model
@st.cache_data
def train_model():
    df = pd.read_csv('study_data.csv')
    df = df.dropna()
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])
    
    # Encode text columns to numbers
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# 2. Interactive Input Form
st.subheader("📋 Enter Planned Study Session Details")
col1, col2 = st.columns(2)

with col1:
    subj = st.selectbox("Subject Category", ["Theory / Social Science", "Mathematics", "Science", "Languages"])
    time_val = st.selectbox("Time of Day", ["Early Morning", "Morning/Afternoon", "Evening", "Night"])
    fatigue = st.selectbox("Fatigue Level", ["Fully Energized & Fresh", "Slightly Tired / Normal", "Very Tired / Exhausted"])

with col2:
    phone = st.selectbox("Phone Placement", ["In another room", "Same room (out of reach)", "On desk / in hand"])
    sleep = st.selectbox("Prior Sleep", ["Less than 5 hours", "5 to 6 hours", "7+ hours"])
    dread = st.selectbox("Task Dread / Friction", ["Excited / Ready to start", "Neutral / Standard task", "Dreading / Resisting heavily"])

# Mapping choices to numeric values
subj_map = {"Theory / Social Science": 0, "Mathematics": 1, "Science": 2, "Languages": 3}
time_map = {"Early Morning": 0, "Morning/Afternoon": 1, "Evening": 2, "Night": 3}
fatigue_map = {"Fully Energized & Fresh": 0, "Slightly Tired / Normal": 1, "Very Tired / Exhausted": 2}
phone_map = {"In another room": 0, "Same room (out of reach)": 1, "On desk / in hand": 2}
sleep_map = {"Less than 5 hours": 0, "5 to 6 hours": 1, "7+ hours": 2}
dread_map = {"Excited / Ready to start": 0, "Neutral / Standard task": 1, "Dreading / Resisting heavily": 2}

# 3. Live Prediction & Strategy
if st.button("🚀 Predict Focus & Get Strategy", use_container_width=True):
    user_data = [[
        subj_map[subj],
        time_map[time_val],
        fatigue_map[fatigue],
        phone_map[phone],
        sleep_map[sleep],
        dread_map[dread]
    ]]
    
    pred = model.predict(user_data)[0]
    
    st.markdown("---")
    st.subheader("📊 Focus Prediction Output")

    if pred == 0:
        st.success("🟢 **Low Procrastination Risk (High Focus Expected)**")
        st.info("⏰ **Recommended Session:** 50 minutes deep study / 10 minutes break.")
    elif pred == 1:
        st.warning("🟡 **Medium Procrastination Risk (Moderate Focus)**")
        st.info("⏰ **Recommended Session:** 30 minutes study / 5 minutes break.")
    else:
        st.error("🔴 **High Procrastination Risk (Distraction Likely!)**")
        st.info("⏰ **Recommended Session:** 15-minute micro-sprints with active break resets.")

    # Personalized Trigger-Based Recommendations
    st.markdown("### 💡 Tailored Action Steps:")
    recs = []

    if phone == "On desk / in hand":
        recs.append("📱 **Environment:** Relocate your phone into another room or enable Focus Mode.")
    if fatigue == "Very Tired / Exhausted":
        recs.append("💧 **Energy:** Drink a glass of water and complete a quick 2-minute physical stretch.")
    if dread == "Dreading / Resisting heavily":
        recs.append("⏱️ **Mindset:** Use the '5-Minute Rule'—commit to studying for just 5 minutes to overcome inertia.")
    if sleep == "Less than 5 hours":
        recs.append("😴 **Burnout Warning:** Keep this session shorter (max 30 minutes) to retain information.")

    if recs:
        for r in recs:
            st.write(f"• {r}")
    else:
        st.write("• Parameters look optimal! Maintain focus and eliminate micro-distractions.")