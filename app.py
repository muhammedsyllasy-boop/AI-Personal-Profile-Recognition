from pathlib import Path
import json
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "synthetic_profiles.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "models" / "model_metrics.json"

st.set_page_config(page_title="AI Personal Profile Recognition", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main {background: #f5f7fb;}
.block-container {padding-top: 1.5rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; color: white;
       background: linear-gradient(120deg,#14283d,#236078);}
.hero h1 {margin:0; font-size:2rem;}
.hero p {margin:.45rem 0 0; opacity:.9;}
.metric-card {background:white;padding:1rem;border-radius:12px;border:1px solid #e8edf3;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH) if DATA_PATH.exists() else pd.DataFrame()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

def ensure_assets():
    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        with st.spinner("Preparing the sample dataset and training models for first use..."):
            subprocess.run([sys.executable, str(ROOT/"generate_dataset.py")], check=True)
            subprocess.run([sys.executable, str(ROOT/"train_model.py")], check=True)
            load_data.clear()
            load_model.clear()

with st.sidebar:
    st.title("🧠 Profile AI")
    st.caption("Personal Profile Recognition")
    page = st.radio("Navigation", ["Dashboard", "AI Prediction", "AI Photo Analyzer", "Experiments",
                                   "Algorithms", "Simulations", "Generate Report"])
    st.divider()
    st.caption("Demo project • Synthetic data")
    if st.button("Refresh project assets", use_container_width=True):
        for fn in [DATA_PATH, MODEL_PATH, METRICS_PATH]:
            if fn.exists(): fn.unlink()
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

st.markdown('<div class="hero"><h1>AI Personal Profile Recognition</h1><p>Explore profile patterns and predict a suitable professional domain with Machine Learning.</p></div>', unsafe_allow_html=True)

try:
    ensure_assets()
    df = load_data()
    model = load_model()
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
except Exception as e:
    st.error(f"Project setup failed: {e}")
    st.info("Run: python generate_dataset.py  then  python train_model.py")
    st.stop()

domains = sorted(df["professional_domain"].unique())
features = ["age", "education_level", "experience_years", "programming_score",
            "communication_score", "creativity_score", "analytical_score",
            "leadership_score", "design_score", "business_score", "science_score"]

if page == "Dashboard":
    st.subheader("Dashboard Overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Profiles analysed", f"{len(df):,}")
    c2.metric("Professional domains", len(domains))
    c3.metric("Best model", metrics.get("best_model", "N/A"))
    c4.metric("Test accuracy", f"{metrics.get('best_accuracy', 0):.1%}")
    left,right = st.columns(2)
    with left:
        st.markdown("#### Professional domain distribution")
        st.bar_chart(df["professional_domain"].value_counts())
    with right:
        st.markdown("#### Average skill scores")
        score_cols = [c for c in df.columns if c.endswith("_score")]
        st.bar_chart(df[score_cols].mean().sort_values(ascending=False))
    st.markdown("#### Dataset preview")
    st.dataframe(df.head(12), use_container_width=True)
    st.caption("All records are synthetic and are intended for education and demonstration, not real hiring decisions.")

elif page == "AI Prediction":
    st.subheader("Predict a Professional Domain")
    st.write("Enter profile details and skill ratings from 0 to 100.")
    with st.form("profile_form"):
        a,b,c = st.columns(3)
        age = a.slider("Age", 18, 60, 23)
        education = b.selectbox("Education level", ["High School","Diploma","Bachelor","Master","PhD"])
        experience = c.slider("Experience (years)", 0.0, 30.0, 1.0, 0.5)
        skill_cols = st.columns(3)
        values = {}
        for i, (key, label) in enumerate([
            ("programming_score","Programming"),("communication_score","Communication"),
            ("creativity_score","Creativity"),("analytical_score","Analytical"),
            ("leadership_score","Leadership"),("design_score","Design"),
            ("business_score","Business"),("science_score","Science")]):
            values[key] = skill_cols[i%3].slider(label, 0, 100, 60)
        submitted = st.form_submit_button("Predict profile", type="primary", use_container_width=True)
    if submitted:
        row = pd.DataFrame([{"age":age,"education_level":education,"experience_years":experience,**values}])[features]
        pred = model.predict(row)[0]
        probs = model.predict_proba(row)[0]
        classes = model.classes_
        st.success(f"Predicted professional domain: {pred}")
        result = pd.DataFrame({"Domain":classes, "Probability":probs}).sort_values("Probability", ascending=False)
        st.bar_chart(result.set_index("Domain"))
        st.caption("This is a model estimate from synthetic training data, not a validated career assessment.")


elif page == "AI Photo Analyzer":
    st.subheader("Free AI Photo Analyzer")
    st.write(
        "Upload a photo and generate a caption using a free AI model running locally on your computer. "
        "No OpenAI API key or paid API credits are needed."
    )
    st.info(
        "Privacy and limitations: the image is processed on your computer after the model has been downloaded. "
        "This tool creates basic visual captions only. It does not identify people or infer identity, age, "
        "nationality, ethnicity, religion, health, personality, occupation, or other private traits. "
        "Only upload images you have permission to use."
    )

    uploaded = st.file_uploader(
        "Upload a photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear JPG, PNG, or WebP image. Maximum size: 10 MB."
    )

    if uploaded is not None:
        if uploaded.size > 10 * 1024 * 1024:
            st.error("The image is larger than 10 MB. Please upload a smaller image.")
        else:
            from PIL import Image
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Photo selected for analysis", use_container_width=True)
            analyze = st.button("Analyze photo for free", type="primary", use_container_width=True)
            if analyze:
                try:
                    import torch
                    from transformers import BlipProcessor, BlipForConditionalGeneration

                    @st.cache_resource(show_spinner="Downloading the free AI model (first use only) and loading it...")
                    def load_photo_model():
                        model_id = "Salesforce/blip-image-captioning-base"
                        processor = BlipProcessor.from_pretrained(model_id)
                        vision_model = BlipForConditionalGeneration.from_pretrained(model_id)
                        vision_model.eval()
                        return processor, vision_model

                    processor, vision_model = load_photo_model()
                    prompts = [
                        ("Overall visual description", "a photo of"),
                        ("Clothing and visible presentation", "a person wearing"),
                        ("Setting and visible objects", "a photo showing")
                    ]
                    captions = []
                    with st.spinner("Generating captions on your computer..."):
                        for heading, prompt in prompts:
                            inputs = processor(images=image, text=prompt, return_tensors="pt")
                            with torch.inference_mode():
                                output = vision_model.generate(
                                    **inputs,
                                    max_new_tokens=45,
                                    num_beams=3,
                                    early_stopping=True
                                )
                            caption = processor.decode(output[0], skip_special_tokens=True).strip()
                            captions.append((heading, caption))

                    st.markdown("### Photo analysis")
                    report_lines = ["# Photo Analysis Report", "", "Generated locally with BLIP.", ""]
                    for heading, caption in captions:
                        st.markdown(f"**{heading}**")
                        st.write(caption if caption else "The model could not generate a clear caption.")
                        report_lines.extend([f"## {heading}", caption or "No clear caption generated.", ""])
                    st.caption(
                        "AI captions can be inaccurate or incomplete. Review the image yourself; "
                        "the model may occasionally describe details incorrectly."
                    )
                    st.download_button(
                        "Download analysis report",
                        "\n".join(report_lines),
                        file_name="photo_analysis_report.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                except ImportError as e:
                    st.error(f"A required package is missing: {e}")
                    st.code("python -m pip install -r requirements.txt")
                except Exception as e:
                    st.error(f"Photo analysis failed: {e}")
                    st.caption(
                        "Check your internet connection for the first model download, then try again. "
                        "The model is cached on your computer after its first successful download."
                    )
    else:
        st.caption("Upload a photo, then press Analyze photo for free.")

elif page == "Experiments":
    st.subheader("Experiments")
    st.write("Compare model performance using the held-out test split.")
    model_rows = [{"Algorithm":name, "Accuracy":data["accuracy"]} for name,data in metrics.get("models",{}).items()]
    exp = pd.DataFrame(model_rows).sort_values("Accuracy", ascending=False)
    st.dataframe(exp, use_container_width=True, hide_index=True)
    if not exp.empty:
        st.bar_chart(exp.set_index("Algorithm"))
        st.info(f"Best performing model in this run: {metrics.get('best_model')}.")

elif page == "Algorithms":
    st.subheader("Algorithms Used")
    descriptions = {
        "Random Forest":"Combines many decision trees and averages their predictions to improve robustness.",
        "Extra Trees":"Uses randomized decision trees to reduce variance and provide a strong ensemble baseline.",
        "Gradient Boosting":"Builds trees sequentially, where each new tree focuses on correcting earlier errors.",
        "Logistic Regression":"A linear classification baseline that estimates class probabilities."
    }
    for name, desc in descriptions.items():
        with st.expander(name):
            st.write(desc)
            if name in metrics.get("models",{}):
                st.metric("Test accuracy", f"{metrics['models'][name]['accuracy']:.1%}")

elif page == "Simulations":
    st.subheader("Profile Simulation")
    st.write("Generate a random example profile and run it through the trained model.")
    if st.button("Generate and predict a random profile", type="primary"):
        rng = np.random.default_rng()
        education = rng.choice(["High School","Diploma","Bachelor","Master","PhD"])
        sample = {
            "age":int(rng.integers(18,46)), "education_level":education,
            "experience_years":round(float(rng.uniform(0,15)),1),
            **{c:int(rng.integers(0,101)) for c in features if c.endswith("_score")}
        }
        row = pd.DataFrame([sample])[features]
        st.dataframe(row, use_container_width=True)
        st.success(f"Simulated prediction: **{model.predict(row)[0]}**")

elif page == "Generate Report":
    st.subheader("Generate Project Report")
    st.write("Download a summary of the dataset and model evaluation.")
    report = [
        "# AI Personal Profile Recognition — Model Report",
        "",
        "## Project objective",
        "Predict a broad professional domain from synthetic education, experience, and skill-profile features.",
        "",
        "## Dataset",
        f"- Number of synthetic profiles: {len(df):,}",
        f"- Number of input features: {len(features)}",
        f"- Target classes: {', '.join(domains)}",
        "",
        "## Best model",
        f"- Algorithm: {metrics.get('best_model','N/A')}",
        f"- Test accuracy: {metrics.get('best_accuracy',0):.3f}",
        "",
        "## Model comparison",
    ]
    for name, item in metrics.get("models",{}).items():
        report.append(f"- {name}: {item['accuracy']:.3f}")
    report += ["", "## Limitations", "The data is synthetic. Results do not establish real-world career suitability and must not be used for hiring, admissions, or other consequential decisions."]
    report_text = "\n".join(report)
    st.download_button("Download report (Markdown)", report_text, "AI_Profile_Recognition_Report.md", "text/markdown")
    st.download_button("Download dataset (CSV)", df.to_csv(index=False), "synthetic_profiles.csv", "text/csv")

st.divider()
st.caption("AI Personal Profile Recognition | Educational prototype | Synthetic data only")
