# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import joblib
import torch
import numpy as np
import pandas as pd
import shap
import gdown
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================================================
# DARK THEME CSS
# =========================================================
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }
.stApp { background-color: #0e1117; }
h1, h2, h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# PATHS
# =========================================================
BASE_PATH = "."

SVM_MODEL_PATH = BASE_PATH + "/models/svm_model.pkl"
TFIDF_PATH = BASE_PATH + "/models/tfidf_vectorizer.pkl"
META_MODEL_PATH = BASE_PATH + "/models/meta_model.pkl"
#BERT_MODEL_PATH = BASE_PATH + "/models/distilbert_model"
BERT_MODEL_PATH = "microsoft/deberta-v3-small"

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():

    # Download only if not already present
    if not os.path.exists("svm_model.pkl"):
        gdown.download("https://drive.google.com/uc?id=1krz9t3rl8Y9WfJ5z7OSWjay_VRZoE_YZ", "svm_model.pkl", quiet=False)

    if not os.path.exists("tfidf.pkl"):
        gdown.download("https://drive.google.com/uc?id=1tKr1XOzux_7-qwaGKdnTvfgbGMgB1Y0L", "tfidf.pkl", quiet=False)

    if not os.path.exists("meta.pkl"):
        gdown.download("https://drive.google.com/uc?id=1N7wMZVX-PzMdiU8byZyTCeNgx6vkWFWV", "meta.pkl", quiet=False)

    # Load models
    svm_model = joblib.load("svm_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    meta_model = joblib.load("meta.pkl")

    # Load BERT from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small", use_fast = False)
    bert_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-small")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_model.to(device)
    bert_model.eval()

    return svm_model, tfidf, meta_model, tokenizer, bert_model, device

# =========================================================
# BERT PREDICTION
# =========================================================
def bert_predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    return torch.softmax(outputs.logits, dim=1).cpu().numpy()

svm_model, tfidf, meta_model, tokenizer, bert_model, device = load_models()

# =========================================================
# ENSEMBLE PREDICTION
# =========================================================
def predict_job_type(text):

    # SVM
    text_tfidf = tfidf.transform([text])
    svm_prob = svm_model.predict_proba(text_tfidf)

    # BERT
    bert_prob = bert_predict(text)

    # Ensemble
    meta_input = np.hstack((bert_prob, svm_prob))
    probs = meta_model.predict_proba(meta_input)[0]

    classes = meta_model.classes_

    pred_idx = np.argmax(probs)
    label = classes[pred_idx]
    confidence = probs[pred_idx]

    results = [(classes[i], probs[i]) for i in np.argsort(probs)[::-1]]

    return label, confidence, results

# =========================================================
# SHAP EXPLAINER
# =========================================================
@st.cache_resource
def load_explainer():
    sample_data = tfidf.transform(["sample text"])
    return shap.LinearExplainer(svm_model, sample_data)

svm_explainer = load_explainer()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("📌 About Project")
st.sidebar.write("""
This system uses:
- TF-IDF + SGD Classifier  
- Transformer (BERT)  
- Ensemble Learning  
- SHAP Explainability  
""")

st.sidebar.markdown("### 💡 Try Example")

if st.sidebar.button("Load Example"):
    st.session_state["text"] = "Looking for a part-time retail sales associate for weekend shifts"

# =========================================================
# MAIN UI
# =========================================================
st.markdown("<h1 style='text-align:center;'>💼 Job Type Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered classification with explainability</p>", unsafe_allow_html=True)

st.divider()

# =========================================================
# INPUT
# =========================================================
text = st.text_area(
    "📝 Enter Job Description",
    height=150,
    value=st.session_state.get("text", "")
)

# =========================================================
# BUTTON ACTION
# =========================================================
if st.button("🚀 Predict"):

    if text.strip() == "":
        st.warning("⚠️ Please enter job description")
    else:

        # =================================================
        # PREDICTION
        # =================================================
        label, conf, results = predict_job_type(text)

        # =================================================
        # RESULT DISPLAY
        # =================================================
        col1, col2 = st.columns(2)

        with col1:
            st.success(f"🎯 Prediction: {label}")

        with col2:
            st.metric("Confidence", f"{round(conf * 100, 2)}%")

        # Progress bar
        st.progress(float(conf))

        st.divider()

        # =================================================
        # BAR CHART
        # =================================================
        st.subheader("📊 Prediction Distribution")

        df_chart = pd.DataFrame(results, columns=["Class", "Probability"])
        st.bar_chart(df_chart.set_index("Class"))

        st.divider()

        # =================================================
        # TOP PREDICTIONS
        # =================================================
        st.subheader("🏆 Top Predictions")

        for cls, prob in results[:3]:
            st.markdown(f"**{cls}** → `{round(prob, 3)}`")

        st.divider()

        # =================================================
        # SHAP EXPLANATION
        # =================================================
        st.subheader("🧠 Important Keywords")

        sample_vec = tfidf.transform([text])
        shap_values = svm_explainer(sample_vec)

        pred_class = np.argmax(svm_model.predict_proba(sample_vec))

        feature_names = tfidf.get_feature_names_out()
        #values = shap_values.values[0][:, pred_class]
        values = shap_values.values[0]
        if values.ndim > 1:
            values = values[:, pred_class]
        sample_dense = sample_vec.toarray()[0]

        indices = np.where(sample_dense != 0)[0]

        words = feature_names[indices]
        word_values = values[indices]

        if len(words) > 0:

            top_indices = np.argsort(np.abs(word_values))[::-1][:10]

            # =============================================
            # TAG STYLE
            # =============================================
            tags = ""
            for i in top_indices:
                tags += f"<span style='background:#1f77b4;padding:6px;border-radius:8px;margin:4px;color:white;display:inline-block'>{words[i]}</span> "

            st.markdown(tags, unsafe_allow_html=True)

            # =============================================
            # TABLE
            # =============================================
            st.subheader("📊 Word Importance Table")

            df_words = pd.DataFrame({
                "Word": words[top_indices],
                "Importance": np.round(word_values[top_indices], 3)
            })

            st.dataframe(df_words, use_container_width=True)

            # =============================================
            # BAR GRAPH
            # =============================================
            st.subheader("📈 Importance Graph")

            st.bar_chart(df_words.set_index("Word"))

        else:
            st.warning("No important keywords found.")

        st.divider()

        # =================================================
        # DOWNLOAD REPORT
        # =================================================
        report = f"""
==============================
JOB CLASSIFICATION REPORT
==============================

Job Description:
{text}

------------------------------
Prediction:
------------------------------
Label: {label}
Confidence: {round(conf,3)}

------------------------------
Top Predictions:
------------------------------
"""

        for cls, prob in results[:3]:
            report += f"{cls} : {round(prob,3)}\n"

        report += "\n------------------------------\nImportant Keywords:\n------------------------------\n"

        if len(words) > 0:
            for i in top_indices:
                report += f"{words[i]} : {round(word_values[i],3)}\n"

        report += "\n==============================\nEnd of Report\n=============================="

        st.download_button(
            label="📥 Download Detailed Report",
            data=report,
            file_name="job_prediction_report.txt",
            mime="text/plain"
        )

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption("🚀 Built with Streamlit | ML + NLP + Explainable AI")