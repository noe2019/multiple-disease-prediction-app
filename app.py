import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import DATASETS

st.set_page_config(page_title="Multiple Disease Prediction", page_icon="🩺", layout="wide")
st.title("🩺 Multiple Disease Prediction")
st.caption("Early-risk prediction of type 2 diabetes, heart disease and Parkinson's disease · simulated data · educational use only")


@st.cache_resource
def train(name):
    df = DATASETS[name]()
    X, y = df.drop(columns="Outcome"), df["Outcome"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    models = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Random forest": RandomForestClassifier(n_estimators=300, random_state=0),
        "Gradient boosting": GradientBoostingClassifier(random_state=0),
    }
    scores = {}
    for k, m in models.items():
        m.fit(Xtr, ytr)
        scores[k] = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    best = max(scores, key=scores.get)
    return df, models, scores, best


with st.sidebar:
    st.header("About")
    st.write("Models trained on simulated cohorts that mimic the Pima diabetes, Cleveland heart and Oxford Parkinson's voice datasets.")
    st.markdown("[Portfolio](https://noe2019.github.io/DA-Portfolio-Website/)")
    st.warning("Not a medical device. Consult a health professional.")

for tab, name in zip(st.tabs(list(DATASETS)), DATASETS):
    with tab:
        df, models, scores, best = train(name)
        X = df.drop(columns="Outcome")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Risk calculator")
            vals = {}
            cols = st.columns(2)
            for i, col in enumerate(X.columns):
                lo, hi, med = float(X[col].min()), float(X[col].max()), float(X[col].median())
                with cols[i % 2]:
                    if X[col].nunique() <= 4:
                        vals[col] = st.selectbox(col, sorted(X[col].unique()), key=f"{name}{col}")
                    else:
                        vals[col] = st.number_input(col, lo, hi, med, key=f"{name}{col}")
            model_name = st.selectbox("Model", list(models), index=list(models).index(best), key=f"m{name}")
            p = models[model_name].predict_proba(pd.DataFrame([vals])[X.columns])[0, 1]
            level = "High" if p >= 0.6 else "Moderate" if p >= 0.3 else "Low"
            st.metric(f"Predicted risk – {name}", f"{p:.0%}", level + " risk", delta_color="off")
            st.progress(float(p))
        with c2:
            st.subheader("Model performance (ROC-AUC, hold-out)")
            st.plotly_chart(px.bar(x=list(scores), y=list(scores.values()), labels={"x": "", "y": "ROC-AUC"},
                                   range_y=[0.5, 1], text=[f"{v:.3f}" for v in scores.values()]), use_container_width=True)
            rf = models["Random forest"]
            imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
            st.plotly_chart(px.bar(imp, orientation="h", title="Feature importance (Random forest)",
                                   labels={"value": "Importance", "index": ""}).update_layout(showlegend=False),
                            use_container_width=True)
        st.caption(f"Cohort size: {len(df):,} · prevalence {df.Outcome.mean():.0%} · best model: {best}")
