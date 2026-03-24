import streamlit as st
import pickle
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# Custom UI
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #00ffcc;
    }
    .subtitle {
        text-align: center;
        color: #aaaaaa;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #00ffcc;
        color: black;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">📧 AI Spam Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">NLP-based Email Classification</div>', unsafe_allow_html=True)

# Input
user_input = st.text_area("✍️ Enter or paste your email text here:", height=200)

# Analyze
if st.button("🚀 Analyze Email"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        vec = vectorizer.transform([user_input])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        st.subheader("🔍 Result")

        if pred == 1:
            st.error("🚫 Spam Email Detected")
        else:
            st.success("✅ This is a Safe Email")

        # Probabilities
        st.write(f"📊 Spam Probability: {prob[1]:.2f}")
        st.write(f"📊 Not Spam Probability: {prob[0]:.2f}")

        # Chart
        fig, ax = plt.subplots()
        ax.bar(["Not Spam", "Spam"], prob)
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)